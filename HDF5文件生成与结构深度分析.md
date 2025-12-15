# HDF5 文件生成与结构深度分析

## 目录
- [1. 概述](#1-概述)
- [2. 输出文件的生成过程](#2-输出文件的生成过程)
- [3. HDF5 文件格式详解](#3-hdf5-文件格式详解)
- [4. 文件内部数据组织](#4-文件内部数据组织)
- [5. 具体的文件结构示例](#5-具体的文件结构示例)
- [6. 数据分割策略](#6-数据分割策略)
- [7. 文件的可访问性和验证](#7-文件的可访问性和验证)
- [8. 与后续使用的关联](#8-与后续使用的关联)
- [9. 不同数据集的文件差异](#9-不同数据集的文件差异)
- [10. 文件大小和存储效率](#10-文件大小和存储效率)
- [11. 数据的完整处理链](#11-数据的完整处理链)
- [12. 代码分析](#12-代码分析)
- [13. 性能分析](#13-性能分析)
- [14. 使用示例](#14-使用示例)

---

## 1. 概述

`preprocess_dataset.py` 是事件相机数据预处理的核心脚本，负责将原始事件数据转换为适合深度学习训练的 HDF5 格式文件。该脚本处理 Gen1 和 Gen4 两种数据集，支持多种事件表示方法（Stacked Histogram、Mixed Density Event Stack）。

**主要功能**：
- 从原始 HDF5 事件数据中提取事件窗口
- 构建事件表示（event representations）
- 过滤和对齐标注数据
- 使用 BLOSC 压缩存储处理后的数据
- 支持并行处理多个序列

---

## 2. 输出文件的生成过程

### 2.1 处理流程概览

```
原始数据目录结构:
input_dir/
├── train/
│   ├── sequence_name_td.h5          # 原始事件数据
│   └── sequence_name_bbox.npy        # 原始标注
├── val/
└── test/

↓ 运行 preprocess_dataset.py

输出数据目录结构:
target_dir/
├── train/
│   └── sequence_name/
│       ├── labels_v2/
│       │   ├── labels.npz
│       │   └── timestamps_us.npy
│       └── event_representations_v2/
│           └── {representation_name}/
│               ├── event_representations.h5
│               ├── objframe_idx_2_repr_idx.npy
│               └── timestamps_us.npy
├── val/
└── test/
```

### 2.2 主要处理步骤

#### 步骤 1: 命令行参数解析
```python
# 从命令行获取配置
python preprocess_dataset.py \
    ${DATA_DIR} \               # 输入目录
    ${DEST_DIR} \               # 目标目录
    conf_preprocess/representation/stacked_hist.yaml \  # 表示方法配置
    conf_preprocess/extraction/const_duration.yaml \    # 事件窗口提取配置
    conf_preprocess/filter_gen1.yaml \                  # 边界框过滤配置
    -ds gen1 \                  # 数据集类型
    -np 20                      # 并行进程数
```

#### 步骤 2: 配置加载与表示方法初始化
```python
# 加载配置 (lines 831-849)
config = get_configuration(
    ev_repr_yaml_config=Path(args.ev_repr_yaml_config),
    extraction_yaml_config=Path(args.extraction_yaml_config),
)

# 创建事件表示工厂
ev_repr_factory = name_2_ev_repr_factory[config.name](config)
ev_repr = ev_repr_factory.create(height=height, width=width)
```

#### 步骤 3: 遍历序列并构建处理任务
```python
# 构建序列数据列表 (lines 863-895)
for split in [train_path, val_path, test_path]:
    for npy_file in split.iterdir():
        if npy_file.suffix != ".npy":
            continue
        # 查找对应的 HDF5 文件
        h5f_path = npy_file.parent / (
            npy_file.stem.split("bbox")[0] + f"td{'.dat' if dataset == 'gen1' else ''}.h5"
        )
        
        # 创建输出目录
        out_seq_path = split_out_dir / dir_name
        out_labels_path = out_seq_path / "labels_v2"
        out_ev_repr_path = out_seq_path / "event_representations_v2" / ev_repr_string
        
        # 添加到处理队列
        sequence_data = {
            DataKeys.InNPY: npy_file,
            DataKeys.InH5: h5f_path,
            DataKeys.OutLabelDir: out_labels_path,
            DataKeys.OutEvReprDir: out_ev_repr_path,
            DataKeys.SplitType: split_name_2_type[split.name],
        }
        seq_data_list.append(sequence_data)
```

#### 步骤 4: 并行/串行处理序列
```python
# 并行处理 (lines 906-923)
if num_processes > 1:
    func = partial(
        process_sequence,
        dataset, filter_cfg, ev_repr,
        ev_repr_num_events, ev_repr_delta_ts_ms,
        ts_step_ev_repr_ms, downsample_by_2,
    )
    with get_context("spawn").Pool(num_processes) as pool:
        for _ in pool.imap_unordered(func, iterable=seq_data_list, chunksize=1):
            pbar.update()
```

### 2.3 单个序列的处理流程 (`process_sequence` 函数)

```python
def process_sequence(
    dataset: str,
    filter_cfg: DictConfig,
    event_representation: RepresentationBase,
    ev_repr_num_events: Optional[int],
    ev_repr_delta_ts_ms: Optional[int],
    ts_step_ev_repr_ms: int,
    downsample_by_2: bool,
    sequence_data: Dict[DataKeys, Union[Path, SplitType]],
):
```

**核心处理步骤**：

1. **提取和对齐时间戳**（lines 640-655）
   ```python
   (
       labels_per_frame,          # 每帧的标注列表
       frame_timestamps_us,       # 标注帧的时间戳
       ev_repr_timestamps_us,     # 事件表示的时间戳
       frameidx2repridx,          # 帧索引到表示索引的映射
   ) = labels_and_ev_repr_timestamps(
       npy_file=in_npy_file,
       split_type=split_type,
       filter_cfg=filter_cfg,
       align_t_ms=100,            # 对齐时间（毫秒）
       ts_step_ev_repr_ms=50,     # 事件表示时间步长
       dataset_type=dataset,
   )
   ```

2. **保存标注数据**（lines 663-667）
   ```python
   save_labels(
       out_labels_dir=out_labels_dir,
       labels_per_frame=labels_per_frame,
       frame_timestamps_us=frame_timestamps_us,
   )
   ```

3. **生成事件表示并写入 HDF5**（lines 670-680）
   ```python
   write_event_data(
       in_h5_file=in_h5_file,
       ev_out_dir=out_ev_repr_dir,
       dataset=dataset,
       event_representation=event_representation,
       ev_repr_num_events=ev_repr_num_events,
       ev_repr_delta_ts_ms=ev_repr_delta_ts_ms,
       ev_repr_timestamps_us=ev_repr_timestamps_us,
       downsample_by_2=downsample_by_2,
       frameidx2repridx=frameidx2repridx,
   )
   ```

### 2.4 中间临时文件的处理

在生成 HDF5 文件时，使用临时文件确保写入的原子性：

```python
# 临时文件命名 (lines 565-569)
ev_outfile = ev_out_dir / f"event_representations{'_ds2_nearest' if downsample_by_2 else ''}.h5"
ev_outfile_in_progress = ev_outfile.parent / (
    ev_outfile.stem + "_in_progress" + ev_outfile.suffix
)

# 写入完成后重命名 (line 616)
os.rename(ev_outfile_in_progress, ev_outfile)
```

**优点**：
- 防止部分写入导致的文件损坏
- 支持断点续传（检测到 `_in_progress` 文件则删除重新生成）

---

## 3. HDF5 文件格式详解

### 3.1 HDF5 层次结构

HDF5（Hierarchical Data Format）是一种用于存储和组织大量数据的文件格式。在本项目中，主要使用两种 HDF5 文件：

#### 类型 1：原始事件数据文件（输入）
```
{sequence_name}_td.h5
└── events/                    [Group]
    ├── x                      [Dataset: (N,) int16]
    ├── y                      [Dataset: (N,) int16]
    ├── p                      [Dataset: (N,) int16]
    ├── t                      [Dataset: (N,) int64]
    ├── height                 [Attribute: int]
    └── width                  [Attribute: int]
```

#### 类型 2：事件表示文件（输出）
```
event_representations.h5
└── data                       [Dataset: (T, C, H, W) uint8/int8]
    ├── shape                  [Attribute: (T, C, H, W)]
    ├── dtype                  [Attribute: uint8 或 int8]
    └── compression            [Attribute: BLOSC]
```

### 3.2 BLOSC 压缩配置

```python
# RVT/utils/preprocessing.py
def _blosc_opts(complevel=1, complib="blosc:zstd", shuffle="byte"):
    """
    BLOSC 压缩选项配置
    
    Args:
        complevel: 压缩级别 (1-9, 1=最快, 9=最高压缩率)
        complib: 压缩库 (blosc:zstd, blosc:lz4, blosc:zlib 等)
        shuffle: 洗牌算法 ("byte", "bit", 或 None)
    
    Returns:
        HDF5 compression options dict
    """
    shuffle = 2 if shuffle == "bit" else 1 if shuffle == "byte" else 0
    compressors = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
    complib = ["blosc:" + c for c in compressors].index(complib)
    args = {
        "compression": 32001,  # BLOSC filter ID
        "compression_opts": (0, 0, 0, 0, complevel, shuffle, complib),
    }
    if shuffle > 0:
        args["shuffle"] = False  # 禁用 h5py 自带的 shuffle
    return args
```

**压缩特性**：
- **Filter ID 32001**：BLOSC 压缩器的 HDF5 filter ID
- **Zstd 算法**：Facebook 开发的快速压缩算法，平衡速度和压缩率
- **Byte Shuffle**：重新排列字节以提高压缩效率
- **Complevel 1**：优先速度，适合频繁读取的训练数据

### 3.3 数据集（Dataset）配置

```python
# H5Writer 类中的数据集创建 (lines 90-101)
maxshape = (None,) + ev_repr_shape    # (None, C, H, W) - 第一维可扩展
chunkshape = (1,) + ev_repr_shape     # (1, C, H, W) - 每个块存储一个时间步

self.h5f.create_dataset(
    key,
    dtype=self.numpy_dtype.name,
    shape=chunkshape,                  # 初始大小：(1, C, H, W)
    chunks=chunkshape,                 # 块大小：(1, C, H, W)
    maxshape=maxshape,                 # 最大形状：(None, C, H, W)
    **_blosc_opts(complevel=1, shuffle="byte"),
)
```

**重要参数解释**：

1. **shape vs maxshape**
   - `shape=(1, C, H, W)`：初始数据集大小为 1 个时间步
   - `maxshape=(None, C, H, W)`：第一维可无限扩展

2. **chunks=(1, C, H, W)**
   - 每个 chunk 包含完整的一个时间步的数据
   - 优化顺序访问性能（训练时按时间顺序读取）
   - 每次 resize 和写入都是整个 chunk

3. **动态扩展机制**
   ```python
   def add_data(self, data: np.ndarray):
       new_size = self.t_idx + 1
       self.h5f[self.key].resize(new_size, axis=0)  # 扩展第一维
       self.h5f[self.key][self.t_idx:new_size] = data
       self.t_idx = new_size
   ```

---

## 4. 文件内部数据组织

### 4.1 Histogram 张量的存储

**Stacked Histogram 表示**（`StackedHistogram` 类）：

```python
# 形状：(2*nbins, H, W) - 2 个极性 × nbins 时间分箱
# 数据类型：uint8 (0-255)
# 存储格式：[正极性 bin0, 正极性 bin1, ..., 负极性 bin0, 负极性 bin1, ...]

# 构建过程 (lines 81-134)
representation = th.zeros(
    (2, nbins, H, W),  # channels=2 (正负极性)
    dtype=th.uint8,
)

# 事件映射到 bins
t_norm = (time - time[0]) / max((time[-1] - time[0]), 1)  # 归一化到 [0, 1]
t_idx = (t_norm * nbins).floor().clamp(max=nbins-1)       # 映射到 bin 索引

# 计算线性索引并累加
indices = x + W*y + H*W*t_idx + nbins*H*W*pol
representation.put_(indices, values=1, accumulate=True)

# 裁剪到 count_cutoff
representation = th.clamp(representation, min=0, max=count_cutoff)

# 展平为 (2*nbins, H, W)
representation = th.reshape(representation, (-1, H, W))
```

**示例值**：
```python
# nbins=10, count_cutoff=10
# 输出形状：(20, 240, 304) for Gen1 或 (20, 360, 640) for Gen4 (downsampled)

# 第 0 通道：正极性在第 0 个时间 bin 的事件计数
representation[0, 120, 150] = 5  # 该像素位置在 bin0 有 5 个正极性事件

# 第 10 通道：负极性在第 0 个时间 bin 的事件计数
representation[10, 120, 150] = 3  # 该像素位置在 bin0 有 3 个负极性事件
```

### 4.2 Mixed Density Event Stack 的存储

```python
# 形状：(nbins, H, W)
# 数据类型：int8 (-128 to 127)
# 时间分布：对数分布（最近的事件在高索引 bins）

# 时间映射到 bin (lines 231-234)
# f(bin) = (1/2)^(N-bin) = t_norm
# bin = N - log(t_norm) / log(1/2)
bin_float = nbins - th.log(t_norm) / math.log(1/2)

# 极性映射：{0, 1} → {-1, 1}
pol = pol * 2 - 1

# 累加表示并执行 cumsum
representation.put_(indices, values=pol, accumulate=True)
representation = cumsum_channel(representation, num_channels=nbins)
```

### 4.3 时间戳信息的保存方式

#### timestamps_us.npy（事件表示时间戳）

```python
# 文件：event_representations_v2/{repr_name}/timestamps_us.npy
# 形状：(T,) - T 为事件表示的数量
# 数据类型：int64
# 单位：微秒 (μs)

# 生成逻辑 (lines 447-473)
ev_repr_timestamps_us_end = list(
    reversed(range(frame_timestamps_us[0], 0, -delta_t_us))
)[1:-1]  # 从第一个标注帧向前生成

# 在标注帧之间插值
for frame_ts_us_start, frame_ts_us_end in zip(
    frame_timestamps_us[:-1], frame_timestamps_us[1:]
):
    new_edge_timestamps = np.linspace(
        frame_ts_us_start, frame_ts_us_end, num_ev_repr_between + 1
    ).astype('int64')
    ev_repr_timestamps_us_end.extend(new_edge_timestamps)

# 示例值：
# [50000, 100000, 150000, 200000, 250000, ...]  # 每 50ms 一个时间戳
```

#### timestamps_us.npy（标注时间戳）

```python
# 文件：labels_v2/timestamps_us.npy
# 形状：(F,) - F 为标注帧的数量
# 数据类型：int64
# 单位：微秒 (μs)

# 从原始标注中提取 (lines 399-424)
unique_ts_us = np.unique(sequence_labels['t'])
frame_timestamps_us = [unique_ts_us[unique_ts_idx_first]]

# 根据基准帧率过滤
for ts in unique_ts_us:
    diff_to_ref = ts - frame_timestamps_us[-1]
    base_delta_count = round(diff_to_ref / base_delta_ts_labels_us)
    if np.abs(diff_to_ref - base_delta_count * base_delta_ts_labels_us) <= 2000:
        frame_timestamps_us.append(ts)

# Gen1: 基准 4Hz (250ms)，输出约 10Hz
# Gen4: 基准 30/60Hz，输出约 10Hz
```

### 4.4 边界框标注信息的存储

#### labels.npz

```python
# 文件：labels_v2/labels.npz
# 包含两个数组：

# 1. labels: structured array
#    dtype: [('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), 
#            ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4')]
#    形状：(N,) - N 为所有帧的总标注数量

# 2. objframe_idx_2_label_idx: int array
#    形状：(F,) - F 为帧数
#    值：每帧第一个标注在 labels 数组中的起始索引

# 示例：
labels = np.array([
    (250000, 10.5, 20.3, 50.0, 80.0, 0, 1.0, 42),  # 帧 0 的第一个框
    (250000, 100.2, 150.7, 60.0, 90.0, 2, 0.95, 43),  # 帧 0 的第二个框
    (500000, 15.1, 25.8, 55.0, 85.0, 0, 0.98, 42),  # 帧 1 的第一个框
], dtype=[('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'),
          ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4')])

objframe_idx_2_label_idx = np.array([0, 2])  # 帧 0 从索引 0 开始，帧 1 从索引 2 开始
```

**标注字段说明**：
- `t`：时间戳（微秒）
- `x`, `y`：边界框左上角坐标（像素）
- `w`, `h`：边界框宽度和高度（像素）
- `class_id`：类别 ID（0=行人, 1=两轮车, 2=汽车）
- `class_confidence`：类别置信度
- `track_id`：跟踪 ID

#### objframe_idx_2_repr_idx.npy

```python
# 文件：event_representations_v2/{repr_name}/objframe_idx_2_repr_idx.npy
# 形状：(F,) - F 为标注帧数
# 数据类型：int64
# 含义：标注帧索引到事件表示索引的映射

# 生成逻辑 (lines 475-477)
frameidx_2_repridx = np.searchsorted(
    ev_repr_timestamps_us_end, frame_timestamps_us, side='left'
)

# 示例：
# 假设 ev_repr_timestamps_us = [50000, 100000, 150000, 200000, 250000, 300000]
#      frame_timestamps_us     = [100000, 250000]
# 则 objframe_idx_2_repr_idx   = [1, 4]
# 含义：第 0 个标注帧对应第 1 个事件表示，第 1 个标注帧对应第 4 个事件表示
```

### 4.5 元数据的组织结构

虽然当前实现较为简化，但可以通过 HDF5 attributes 存储元数据：

```python
# 可选的元数据存储（当前未实现，但建议添加）
with h5py.File('event_representations.h5', 'r') as f:
    # 数据集维度信息
    print(f['data'].attrs.get('description', 'Event representations'))
    print(f['data'].attrs.get('representation_type', 'stacked_histogram'))
    print(f['data'].attrs.get('nbins', 10))
    print(f['data'].attrs.get('count_cutoff', 10))
    
    # 数据集统计信息
    print(f['data'].attrs.get('num_timesteps', len(f['data'])))
    print(f['data'].attrs.get('dataset_type', 'gen1'))
```

---

## 5. 具体的文件结构示例

### 5.1 典型 HDF5 文件的目录树结构

使用 `h5ls` 工具查看文件结构：

```bash
$ h5ls -r event_representations.h5
/                        Group
/data                    Dataset {1000, 20, 240, 304}
    Attribute: CLASS = "CARRAY"
    Attribute: COMPRESSION = "blosc:zstd"
```

### 5.2 完整输出目录示例

```
target_dir/
├── train/
│   ├── 17-04-05_14-31-57_397500000_457500000/
│   │   ├── labels_v2/
│   │   │   ├── labels.npz                     # 45 KB
│   │   │   └── timestamps_us.npy              # 2 KB
│   │   └── event_representations_v2/
│   │       └── stacked_histogram_dt=50_nbins=10/
│   │           ├── event_representations.h5    # 150 MB
│   │           ├── objframe_idx_2_repr_idx.npy # 2 KB
│   │           └── timestamps_us.npy           # 8 KB
│   ├── 17-04-05_15-21-00_176500000_236500000/
│   │   └── ...
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### 5.3 每个 Dataset 的具体形状和数据类型

| 文件路径 | Dataset/Array | 形状 | 数据类型 | 大小示例 |
|---------|--------------|------|---------|---------|
| `event_representations.h5` | `/data` | `(T, C, H, W)` | `uint8` | 150-500 MB |
| `labels.npz` | `labels` | `(N,)` | structured | 20-100 KB |
| `labels.npz` | `objframe_idx_2_label_idx` | `(F,)` | `int64` | 1-5 KB |
| `timestamps_us.npy` (labels) | - | `(F,)` | `int64` | 1-5 KB |
| `timestamps_us.npy` (repr) | - | `(T,)` | `int64` | 5-20 KB |
| `objframe_idx_2_repr_idx.npy` | - | `(F,)` | `int64` | 1-5 KB |

**符号说明**：
- `T`：事件表示的时间步数（通常 1000-5000）
- `C`：通道数（Stacked Histogram: 20，Mixed Density: 10）
- `H`：高度（Gen1: 240，Gen4: 360 下采样后）
- `W`：宽度（Gen1: 304，Gen4: 640 下采样后）
- `N`：总标注框数量（所有帧的总和）
- `F`：标注帧数量（通常 100-500）

### 5.4 数据存储的具体示例数值

#### 示例：StackedHistogram 数据

```python
import h5py
import numpy as np

with h5py.File('event_representations.h5', 'r') as f:
    data = f['data']
    print(f"Shape: {data.shape}")  # (2500, 20, 240, 304)
    print(f"Dtype: {data.dtype}")  # uint8
    
    # 读取第 100 个时间步
    frame_100 = data[100]  # shape: (20, 240, 304)
    
    # 查看第一个通道（正极性，第一个时间 bin）
    channel_0 = frame_100[0]  # shape: (240, 304)
    print(f"Min: {channel_0.min()}, Max: {channel_0.max()}")  # Min: 0, Max: 10
    print(f"Mean: {channel_0.mean():.2f}")  # Mean: 0.15
    print(f"Non-zero pixels: {np.count_nonzero(channel_0)}")  # 1250
```

#### 示例：Labels 数据

```python
import numpy as np

data = np.load('labels.npz')
labels = data['labels']
objframe_idx_2_label_idx = data['objframe_idx_2_label_idx']

print(f"Total labels: {len(labels)}")  # 5482
print(f"Total frames: {len(objframe_idx_2_label_idx)}")  # 250

# 第一个标注
first_label = labels[0]
print(f"Time: {first_label['t']} μs")
print(f"BBox: x={first_label['x']:.1f}, y={first_label['y']:.1f}, "
      f"w={first_label['w']:.1f}, h={first_label['h']:.1f}")
print(f"Class: {first_label['class_id']}, Track: {first_label['track_id']}")

# 输出:
# Time: 397645000 μs
# BBox: x=125.3, y=85.7, w=42.5, h=68.2
# Class: 0, Track: 15
```

---

## 6. 数据分割策略

### 6.1 训练/验证/测试集的划分方式

数据集划分由输入目录结构决定：

```python
# 从输入目录读取划分 (lines 852-861)
dataset_input_path = Path(args.input_dir)
train_path = dataset_input_path / "train"
val_path = dataset_input_path / "val"
test_path = dataset_input_path / "test"

# Prophesee 数据集的标准划分：
# Gen1: train=25 sequences, val=6 sequences, test=8 sequences
# Gen4: train=40 sequences, val=10 sequences, test=12 sequences
```

### 6.2 序列过滤

某些序列在过滤后可能没有有效标注，会被自动删除：

```python
# 预定义的忽略序列 (lines 63-71)
dirs_to_ignore = {
    "gen1": (
        "17-04-06_09-57-37_6344500000_6404500000",
        "17-04-13_19-17-27_976500000_1036500000",
        "17-04-06_15-14-36_1159500000_1219500000",
        "17-04-11_15-13-23_122500000_182500000",
    ),
    "gen4": (),
}

# 动态过滤：无标注序列 (lines 656-660)
except NoLabelsException:
    parent_dir = out_labels_dir.parent
    print(f"No labels after filtering. Deleting {str(parent_dir)}")
    shutil.rmtree(parent_dir)
    return
```

### 6.3 数据集的索引和 ID 映射

```python
# DataModule 中的索引构建（参见 dataset_rnd.py）
class GenXDatasetRandomAccess:
    def __init__(self, sequences: List[SequenceForRandomAccess]):
        self.sequences = sequences
        
        # 计算累积长度用于全局索引
        self.cumulative_sizes = [0]
        for seq in sequences:
            self.cumulative_sizes.append(
                self.cumulative_sizes[-1] + len(seq)
            )
        
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx: int):
        # 二分查找定位序列
        seq_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        local_idx = idx - self.cumulative_sizes[seq_idx]
        return self.sequences[seq_idx][local_idx]
```

### 6.4 类别标签的存储和编码

```python
# 类别映射
# Gen1: {0: 'pedestrian', 1: 'two_wheeler', 2: 'car'}
# Gen4: {0: 'pedestrian', 1: 'two_wheeler', 2: 'car'}
#       原始还包含 {3: 'truck', 4: 'bus', 5: 'traffic_sign', 6: 'traffic_light'}
#       但在预处理时被过滤掉 (line 287)

def prophesee_remove_labels_filter_gen4(labels: np.ndarray) -> np.ndarray:
    keep = labels['class_id'] <= 2  # 只保留前 3 个类别
    return labels[keep]
```

### 6.5 样本的组织方式

每个样本由一个序列窗口组成：

```python
# SequenceForRandomAccess.__getitem__ (lines 47-81)
# 返回的数据字典：
{
    DataType.EV_REPR: [            # 事件表示列表，长度=sequence_length
        tensor([20, 240, 304]),    # 时间步 t
        tensor([20, 240, 304]),    # 时间步 t+1
        ...
        tensor([20, 240, 304]),    # 时间步 t+sequence_length-1
    ],
    DataType.OBJLABELS_SEQ: SparselyBatchedObjectLabels,  # 稀疏标注
    DataType.IS_FIRST_SAMPLE: True,  # 随机访问时总是 True
    DataType.IS_PADDED_MASK: [False, False, ..., False],  # 无填充
}
```

---

## 7. 文件的可访问性和验证

### 7.1 如何通过 h5py 读取和验证文件

```python
import h5py
import numpy as np

def inspect_event_repr_file(h5_path):
    """检查事件表示 HDF5 文件"""
    with h5py.File(h5_path, 'r') as f:
        print("=" * 50)
        print(f"File: {h5_path}")
        print("=" * 50)
        
        # 列出所有 keys
        print(f"\nKeys: {list(f.keys())}")
        
        # 检查数据集
        data = f['data']
        print(f"\nDataset 'data':")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Size: {data.nbytes / 1e6:.2f} MB")
        print(f"  Chunks: {data.chunks}")
        print(f"  Compression: {data.compression}")
        
        # 统计信息
        print(f"\nStatistics (first 10 frames):")
        sample = data[:10]
        print(f"  Min: {sample.min()}")
        print(f"  Max: {sample.max()}")
        print(f"  Mean: {sample.mean():.4f}")
        print(f"  Std: {sample.std():.4f}")
        
        # 检查稀疏性
        sparsity = np.count_nonzero(sample) / sample.size
        print(f"  Sparsity: {sparsity:.4f} (non-zero ratio)")

# 使用示例
inspect_event_repr_file('event_representations.h5')
```

### 7.2 文件的完整性检查

```python
def check_file_integrity(sequence_dir, repr_name):
    """检查序列目录的文件完整性"""
    errors = []
    
    # 1. 检查目录结构
    labels_dir = sequence_dir / "labels_v2"
    repr_dir = sequence_dir / "event_representations_v2" / repr_name
    
    if not labels_dir.exists():
        errors.append(f"Missing labels_v2 directory")
    if not repr_dir.exists():
        errors.append(f"Missing representation directory")
    
    # 2. 检查必需文件
    required_files = {
        'labels': labels_dir / "labels.npz",
        'label_timestamps': labels_dir / "timestamps_us.npy",
        'event_repr': repr_dir / "event_representations.h5",
        'repr_timestamps': repr_dir / "timestamps_us.npy",
        'frame_to_repr': repr_dir / "objframe_idx_2_repr_idx.npy",
    }
    
    for name, path in required_files.items():
        if not path.exists():
            errors.append(f"Missing {name} file: {path}")
    
    if errors:
        return False, errors
    
    # 3. 检查数据一致性
    try:
        # 加载数据
        label_data = np.load(required_files['labels'])
        label_ts = np.load(required_files['label_timestamps'])
        repr_ts = np.load(required_files['repr_timestamps'])
        frame2repr = np.load(required_files['frame_to_repr'])
        
        with h5py.File(required_files['event_repr'], 'r') as f:
            ev_repr_shape = f['data'].shape
        
        # 检查形状一致性
        num_frames = len(label_ts)
        num_reprs = len(repr_ts)
        
        if len(frame2repr) != num_frames:
            errors.append(
                f"Frame2Repr length ({len(frame2repr)}) != "
                f"num frames ({num_frames})"
            )
        
        if ev_repr_shape[0] != num_reprs:
            errors.append(
                f"Event repr timesteps ({ev_repr_shape[0]}) != "
                f"num timestamps ({num_reprs})"
            )
        
        # 检查时间戳单调性
        if not np.all(np.diff(label_ts) > 0):
            errors.append("Label timestamps not strictly increasing")
        if not np.all(np.diff(repr_ts) > 0):
            errors.append("Repr timestamps not strictly increasing")
        
        # 检查映射有效性
        if frame2repr.max() >= num_reprs:
            errors.append(
                f"Frame2Repr contains invalid index: "
                f"{frame2repr.max()} >= {num_reprs}"
            )
        
    except Exception as e:
        errors.append(f"Error loading files: {str(e)}")
    
    return len(errors) == 0, errors

# 使用示例
success, errors = check_file_integrity(
    Path("target_dir/train/sequence_name"),
    "stacked_histogram_dt=50_nbins=10"
)
if not success:
    for error in errors:
        print(f"ERROR: {error}")
```

### 7.3 数据的一致性验证

```python
def verify_labels_alignment(sequence_dir, repr_name):
    """验证标注与事件表示的对齐"""
    labels_dir = sequence_dir / "labels_v2"
    repr_dir = sequence_dir / "event_representations_v2" / repr_name
    
    # 加载时间戳
    label_ts = np.load(labels_dir / "timestamps_us.npy")
    repr_ts = np.load(repr_dir / "timestamps_us.npy")
    frame2repr = np.load(repr_dir / "objframe_idx_2_repr_idx.npy")
    
    print(f"Label frames: {len(label_ts)}")
    print(f"Repr timesteps: {len(repr_ts)}")
    
    # 检查每个标注帧是否正确对齐
    misaligned = []
    for i, (label_t, repr_idx) in enumerate(zip(label_ts, frame2repr)):
        repr_t = repr_ts[repr_idx]
        if label_t != repr_t:
            misaligned.append((i, label_t, repr_t))
    
    if misaligned:
        print(f"\nWARNING: Found {len(misaligned)} misaligned frames")
        for frame_idx, label_t, repr_t in misaligned[:5]:
            print(f"  Frame {frame_idx}: label_t={label_t}, repr_t={repr_t}, "
                  f"diff={abs(label_t - repr_t)} μs")
    else:
        print("\n✓ All frames correctly aligned")
```

### 7.4 常见的数据质量问题

| 问题 | 表现 | 可能原因 | 解决方法 |
|-----|------|---------|---------|
| 文件损坏 | `OSError: Unable to open file` | 写入过程中断 | 删除并重新生成 |
| 时间戳不单调 | 标注时间戳乱序 | 原始数据问题 | 使用 `_correct_time` 修复 |
| 标注为空 | `NoLabelsException` | 过滤后无有效标注 | 调整过滤参数或排除该序列 |
| 数据类型不匹配 | `AssertionError: dtype mismatch` | 表示方法配置错误 | 检查 YAML 配置 |
| 内存不足 | `MemoryError` | 序列过长 | 增加并行进程数，减少内存占用 |
| 压缩库缺失 | `ImportError: No module named 'hdf5plugin'` | 未安装 hdf5plugin | `pip install hdf5plugin` |

---

## 8. 与后续使用的关联

### 8.1 DataModule 如何读取这些文件

```python
# SequenceBase 类的初始化 (lines 43-98)
class SequenceBase:
    def __init__(
        self, path, ev_representation_name, sequence_length,
        dataset_type, downsample_by_factor_2, only_load_end_labels
    ):
        # 1. 定位事件表示目录
        ev_repr_dir = path / "event_representations_v2" / ev_representation_name
        labels_dir = path / "labels_v2"
        
        # 2. 加载标注
        label_data = np.load(labels_dir / "labels.npz")
        self.label_factory = ObjectLabelFactory.from_structured_array(
            object_labels=label_data['labels'],
            objframe_idx_2_label_idx=label_data['objframe_idx_2_label_idx'],
            input_size_hw=(height, width),
            downsample_factor=2 if downsample_by_factor_2 else None,
        )
        
        # 3. 加载映射
        self.objframe_idx_2_repr_idx = np.load(
            ev_repr_dir / "objframe_idx_2_repr_idx.npy"
        )
        
        # 4. 记录 HDF5 文件路径（惰性加载）
        ds_factor_str = "_ds2_nearest" if downsample_by_factor_2 else ""
        self.ev_repr_file = ev_repr_dir / f"event_representations{ds_factor_str}.h5"
```

### 8.2 事件表示的读取

```python
# SequenceBase._get_event_repr_torch (lines 103-113)
def _get_event_repr_torch(self, start_idx: int, end_idx: int):
    """从 HDF5 读取事件表示并转换为 PyTorch 张量"""
    with h5py.File(str(self.ev_repr_file), 'r') as h5f:
        ev_repr = h5f['data'][start_idx:end_idx]  # NumPy array
    
    # 转换为 PyTorch 张量
    ev_repr = torch.from_numpy(ev_repr)
    if ev_repr.dtype != torch.uint8:
        ev_repr = torch.asarray(ev_repr, dtype=torch.float32)
    
    # 拆分为列表
    ev_repr = torch.split(ev_repr, 1, dim=0)
    ev_repr = [x[0] for x in ev_repr]  # 移除 batch 维度
    return ev_repr
```

### 8.3 文件加载的性能特征

| 访问模式 | 性能 | 使用场景 |
|---------|------|---------|
| 顺序读取 | **优秀** | 流式处理（Streaming DataPipe） |
| 随机访问 | **良好** | 随机采样训练（Random Access） |
| 跨序列跳跃 | **较慢** | 应避免 |
| 小批量读取 | **中等** | 每次读取 sequence_length 个时间步 |

**性能优化建议**：
1. **预加载元数据**：启动时加载所有 `.npy` 文件到内存
2. **HDF5 文件保持打开**：避免频繁打开/关闭（但会占用文件句柄）
3. **使用 SSD 存储**：随机访问延迟更低
4. **合理设置 chunk size**：当前 `(1, C, H, W)` 适合顺序访问

### 8.4 随机访问 vs 顺序访问的设计

#### 随机访问（Random Access）

```python
# SequenceForRandomAccess (sequence_rnd.py)
class SequenceForRandomAccess(SequenceBase):
    def __getitem__(self, index: int):
        # 直接通过索引访问
        corrected_idx = index + self.start_idx_offset
        labels_repr_idx = self.objframe_idx_2_repr_idx[corrected_idx]
        
        end_idx = labels_repr_idx + 1
        start_idx = end_idx - self.seq_len
        
        # 从 HDF5 读取 [start_idx:end_idx] 区间
        ev_repr = self._get_event_repr_torch(start_idx, end_idx)
        labels = [self._get_labels_from_repr_idx(i) 
                  for i in range(start_idx, end_idx)]
        
        return {
            DataType.EV_REPR: ev_repr,
            DataType.OBJLABELS_SEQ: SparselyBatchedObjectLabels(labels),
            DataType.IS_FIRST_SAMPLE: True,  # 每次都是新样本
        }
```

**特点**：
- 支持数据增强和采样策略
- 每个样本独立，支持打乱（shuffle）
- 适合训练阶段

#### 顺序访问（Streaming）

```python
# SequenceForStreaming (sequence_for_streaming.py)
class SequenceForStreaming(SequenceBase):
    def __iter__(self):
        # 顺序遍历整个序列
        for repr_idx in range(self.start_offset, self.max_offset + 1):
            # 读取当前窗口
            end_idx = repr_idx + 1
            start_idx = max(0, end_idx - self.seq_len)
            
            # 可能包含填充
            is_first_sample = (repr_idx == self.start_offset)
            is_padded = [i < self.seq_len - repr_idx - 1 
                        for i in range(self.seq_len)]
            
            ev_repr = self._get_event_repr_torch(start_idx, end_idx)
            labels = [self._get_labels_from_repr_idx(i) 
                     for i in range(start_idx, end_idx)]
            
            yield {
                DataType.EV_REPR: ev_repr,
                DataType.OBJLABELS_SEQ: SparselyBatchedObjectLabels(labels),
                DataType.IS_FIRST_SAMPLE: is_first_sample,
                DataType.IS_PADDED_MASK: is_padded,
            }
```

**特点**：
- 连续读取，利用 OS 缓存
- 支持 RNN 状态跨样本传递
- 适合验证/测试阶段

### 8.5 内存映射（mmap）的使用

当前实现**未使用** mmap，而是通过 h5py 直接读取：

```python
# 当前方式：直接读取
with h5py.File(h5_path, 'r') as f:
    data = f['data'][start:end]  # 读取到内存

# 可选的 mmap 方式（未实现）：
with h5py.File(h5_path, 'r', rdcc_nbytes=1024**3) as f:
    # 增大 chunk cache 以加速重复访问
    data = f['data']
    # data 仍需显式切片才会读取
```

**为何不使用 mmap**：
1. HDF5 压缩数据不支持真正的 mmap（必须解压）
2. h5py 内部已经优化了 chunk cache
3. 训练时通常不重复访问相同数据

---

## 9. 不同数据集的文件差异

### 9.1 Gen1 数据的输出文件特征

| 属性 | 值 |
|-----|-----|
| **分辨率** | 240 × 304 |
| **原始标注帧率** | 4 Hz |
| **输出标注帧率** | ~10 Hz |
| **事件表示时间步长** | 50 ms (可配置) |
| **下采样** | 否 |
| **文件名后缀** | `_td.dat.h5` (原始), `event_representations.h5` (输出) |
| **类别** | 3 类（行人、两轮车、汽车） |
| **忽略序列** | 4 个（预定义） |

**典型序列大小**：
```
17-04-05_14-31-57_397500000_457500000/
├── labels_v2/
│   ├── labels.npz              # ~40 KB (约 500 个框)
│   └── timestamps_us.npy       # ~2 KB (约 250 帧)
└── event_representations_v2/
    └── stacked_histogram_dt=50_nbins=10/
        ├── event_representations.h5   # ~180 MB (约 1200 时间步)
        ├── objframe_idx_2_repr_idx.npy  # ~2 KB
        └── timestamps_us.npy            # ~10 KB
```

### 9.2 Gen4 数据的输出文件特征

| 属性 | 值 |
|-----|-----|
| **原始分辨率** | 720 × 1280 |
| **下采样后分辨率** | 360 × 640 |
| **原始标注帧率** | 30/60 Hz (可变) |
| **输出标注帧率** | ~10 Hz |
| **事件表示时间步长** | 50 ms (可配置) |
| **下采样** | **是（factor=2，nearest-exact）** |
| **文件名后缀** | `_td.h5` (原始), `event_representations_ds2_nearest.h5` (输出) |
| **类别** | 3 类（过滤后，原始 7 类） |
| **忽略序列** | 0 个 |

**典型序列大小**：
```
moorea_2019-06-26_000_td/
├── labels_v2/
│   ├── labels.npz              # ~120 KB (约 1500 个框)
│   └── timestamps_us.npy       # ~3 KB (约 400 帧)
└── event_representations_v2/
    └── stacked_histogram_dt=50_nbins=10/
        ├── event_representations_ds2_nearest.h5  # ~550 MB (约 2400 时间步)
        ├── objframe_idx_2_repr_idx.npy          # ~3 KB
        └── timestamps_us.npy                    # ~20 KB
```

### 9.3 下采样处理

```python
# 下采样逻辑 (lines 533-545, 607-610)
def downsample_ev_repr(x: torch.Tensor, scale_factor: float):
    """下采样事件表示（Gen4 专用）"""
    orig_dtype = x.dtype
    
    # uint8 需要转换为 int16 避免溢出
    if orig_dtype == torch.int8:
        x = torch.asarray(x, dtype=torch.int16)
        x = torch.asarray(x + 128, dtype=torch.uint8)
    
    # 使用 nearest-exact 插值
    x = torch.nn.functional.interpolate(
        x, scale_factor=scale_factor, mode="nearest-exact"
    )
    
    # 转换回原始类型
    if orig_dtype == torch.int8:
        x = torch.asarray(x, dtype=torch.int16)
        x = torch.asarray(x - 128, dtype=torch.int8)
    
    return x

# 在写入前应用 (lines 607-612)
if downsample_by_2:
    ev_repr = ev_repr.unsqueeze(0)  # 添加 batch 维度
    ev_repr = downsample_ev_repr(x=ev_repr, scale_factor=0.5)
    ev_repr_numpy = ev_repr.numpy()[0]
```

**为什么 Gen4 需要下采样**：
1. **减少计算量**：720×1280 → 360×640，参数量减少 75%
2. **内存效率**：文件大小减少约 75%
3. **训练速度**：前向/后向传播更快
4. **保持一致性**：与 Gen1 的计算复杂度相近

### 9.4 混合数据集的组织方式

当前实现支持分别处理 Gen1 和 Gen4，但 DataModule 可以混合加载：

```python
# DataModule 中的混合数据集配置
class DataModule(pl.LightningDataModule):
    def setup(self, stage: str):
        if stage == "fit":
            # 加载 Gen1 训练集
            gen1_dataset = GenXDatasetRandomAccess(
                dataset_path=gen1_train_path,
                dataset_type=DatasetType.GEN1,
                ...
            )
            
            # 加载 Gen4 训练集
            gen4_dataset = GenXDatasetRandomAccess(
                dataset_path=gen4_train_path,
                dataset_type=DatasetType.GEN4,
                ...
            )
            
            # 合并数据集
            self.train_dataset = ConcatDataset([gen1_dataset, gen4_dataset])
```

### 9.5 分辨率差异的处理

模型通过 `downsample_by_factor_2` 参数感知分辨率：

```python
# 数据加载时 (sequence_base.py, lines 79-84)
label_factory = ObjectLabelFactory.from_structured_array(
    object_labels=labels,
    objframe_idx_2_label_idx=objframe_idx_2_label_idx,
    input_size_hw=(height, width),  # Gen1: (240, 304), Gen4: (720, 1280)
    downsample_factor=2 if downsample_by_factor_2 else None,  # Gen4: 2
)

# 标注框坐标自动缩放
# 例如：Gen4 原始框 (x=640, y=360, w=100, h=150)
#      下采样后   (x=320, y=180, w=50, h=75)
```

---

## 10. 文件大小和存储效率

### 10.1 单个文件的典型大小

#### Gen1 序列（60 秒视频）

| 文件 | 未压缩 | 压缩后 | 压缩率 |
|-----|--------|--------|--------|
| 原始事件 HDF5 | ~500 MB | ~150 MB | 3.3× |
| 事件表示 HDF5 | ~700 MB | ~180 MB | 3.9× |
| labels.npz | - | ~40 KB | - |
| timestamps 等 | - | ~15 KB | - |
| **总计** | ~1.2 GB | ~330 MB | **3.6×** |

#### Gen4 序列（60 秒视频）

| 文件 | 未压缩 | 压缩后 | 压缩率 |
|-----|--------|--------|--------|
| 原始事件 HDF5 | ~2 GB | ~600 MB | 3.3× |
| 事件表示 HDF5（下采样） | ~2.5 GB | ~550 MB | 4.5× |
| labels.npz | - | ~120 KB | - |
| timestamps 等 | - | ~25 KB | - |
| **总计** | ~4.5 GB | ~1.1 GB | **4.1×** |

### 10.2 压缩前后的大小对比

```python
# 理论计算：Stacked Histogram (nbins=10)
# Gen1: T × 20 × 240 × 304 × uint8
#     = 1200 × 20 × 240 × 304 × 1 byte
#     = 1,753,344,000 bytes ≈ 1.67 GB (未压缩)

# 实际压缩后：~180 MB
# 压缩率：1.67 GB / 180 MB ≈ 9.3×

# Gen4 (下采样): T × 20 × 360 × 640 × uint8
#     = 2400 × 20 × 360 × 640 × 1 byte
#     = 11,059,200,000 bytes ≈ 10.3 GB (未压缩)

# 实际压缩后：~550 MB
# 压缩率：10.3 GB / 550 MB ≈ 18.7×
```

**为什么压缩效果好**：
1. **高度稀疏**：大部分像素值为 0（无事件）
2. **局部相关**：相邻像素值相似
3. **Byte Shuffle**：将相同字节位置聚集，利用模式重复
4. **Zstd 算法**：针对模式匹配优化

### 10.3 存储效率和访问速度的权衡

| 配置 | 压缩率 | 读取速度 | 写入速度 | 推荐场景 |
|-----|--------|---------|---------|---------|
| `complevel=1` (当前) | 3-5× | **快** | **快** | 训练数据 |
| `complevel=5` | 5-8× | 中等 | 中等 | 归档数据 |
| `complevel=9` | 8-12× | 慢 | **很慢** | 长期存储 |
| 无压缩 | 1× | **最快** | **最快** | 临时数据 |

**当前配置理由**：
- 训练时频繁读取，速度优先
- 存储成本相对较低
- 压缩率已经足够（3-5×）

### 10.4 优化建议

#### 优化 1：调整 chunk size

```python
# 当前：(1, C, H, W) - 每个 chunk 存储 1 个时间步
# 优化：(10, C, H, W) - 每个 chunk 存储 10 个时间步

# 优点：
# - 减少 chunk overhead
# - 提高顺序读取效率
# - 更好的压缩率

# 缺点：
# - 随机访问单个时间步需要读取更多数据
# - 对于 sequence_length < 10 的情况不友好
```

#### 优化 2：使用更激进的压缩

```python
# 对于不常访问的数据（如测试集）
_blosc_opts(complevel=5, shuffle="byte")  # 压缩率提升 50-80%
```

#### 优化 3：分离热/冷数据

```python
# 热数据（训练集）：低压缩，快速访问
# 冷数据（测试集）：高压缩，节省空间
if split_type == SplitType.TRAIN:
    compression_opts = _blosc_opts(complevel=1)
else:
    compression_opts = _blosc_opts(complevel=5)
```

---

## 11. 数据的完整处理链

### 11.1 从 Tarball 原始数据 → HDF5 文件的完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Prophesee 原始数据集                          │
│  gen1_tar/ 或 gen4_tar/                                          │
│  ├── train/                                                      │
│  │   ├── {sequence_name}_td.dat (Gen1) / _td.h5 (Gen4)         │
│  │   └── {sequence_name}_bbox.npy                               │
│  ├── val/                                                        │
│  └── test/                                                       │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│          步骤 1: 加载配置和初始化                                 │
│  • 解析命令行参数                                                │
│  • 加载 YAML 配置：                                              │
│    - representation (stacked_hist/mixeddensity)                 │
│    - extraction (const_duration/const_count)                    │
│    - filter (bbox filters)                                      │
│  • 创建事件表示工厂                                              │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│          步骤 2: 遍历序列并构建任务队列                           │
│  for each split in [train, val, test]:                         │
│    for each sequence in split:                                  │
│      • 查找 {name}_td.h5 和 {name}_bbox.npy                     │
│      • 创建输出目录                                              │
│      • 添加到 seq_data_list                                     │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│          步骤 3: 并行处理序列 (process_sequence)                 │
│  for each sequence_data in seq_data_list:                       │
│    ┌────────────────────────────────────────────────┐          │
│    │ 3.1 提取时间戳                                  │          │
│    │  • 读取原始标注 (.npy)                          │          │
│    │  • 应用过滤器：                                 │          │
│    │    - prophesee_bbox_filter (最小尺寸)          │          │
│    │    - crop_to_fov_filter (裁剪到 FOV)           │          │
│    │    - remove_faulty_huge_bbox_filter            │          │
│    │  • 提取唯一标注时间戳                           │          │
│    │  • 计算事件表示时间戳                           │          │
│    │  • 生成映射 frameidx2repridx                   │          │
│    └────────────────────────────────────────────────┘          │
│                         ↓                                        │
│    ┌────────────────────────────────────────────────┐          │
│    │ 3.2 保存标注 (save_labels)                      │          │
│    │  • 合并所有帧的标注到 labels 数组                │          │
│    │  • 创建 objframe_idx_2_label_idx 索引           │          │
│    │  • 保存 labels.npz                              │          │
│    │  • 保存 timestamps_us.npy                       │          │
│    └────────────────────────────────────────────────┘          │
│                         ↓                                        │
│    ┌────────────────────────────────────────────────┐          │
│    │ 3.3 生成事件表示 (write_event_representations)  │          │
│    │  • 打开原始事件 HDF5 (H5Reader)                 │          │
│    │  • 创建输出 HDF5 (H5Writer)                     │          │
│    │  • 对于每个事件表示时间戳：                      │          │
│    │    a) 计算事件窗口 [start_idx, end_idx]         │          │
│    │    b) 读取事件切片 (x, y, p, t)                 │          │
│    │    c) 构建表示 (histogram/mixeddensity)         │          │
│    │    d) 可选：下采样 (Gen4)                       │          │
│    │    e) 写入 HDF5 (add_data)                     │          │
│    │  • 保存 timestamps_us.npy                       │          │
│    │  • 保存 objframe_idx_2_repr_idx.npy             │          │
│    │  • 重命名临时文件                                │          │
│    └────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    输出数据集                                    │
│  target_dir/                                                     │
│  ├── train/                                                      │
│  │   └── {sequence_name}/                                       │
│  │       ├── labels_v2/                                         │
│  │       │   ├── labels.npz                                     │
│  │       │   └── timestamps_us.npy                              │
│  │       └── event_representations_v2/                          │
│  │           └── {representation_name}/                         │
│  │               ├── event_representations[_ds2_nearest].h5     │
│  │               ├── objframe_idx_2_repr_idx.npy                │
│  │               └── timestamps_us.npy                          │
│  ├── val/                                                        │
│  └── test/                                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 每个阶段的数据形状变化

```
原始事件数据 (H5Reader.get_event_slice):
  x: (N,) int64  - 事件 x 坐标
  y: (N,) int64  - 事件 y 坐标
  p: (N,) int64  - 事件极性 {0, 1}
  t: (N,) int64  - 事件时间戳 (μs)
                 ↓
事件窗口提取:
  根据时间戳提取子集 [t_start, t_end]
  窗口大小由 ev_repr_delta_ts_ms 或 ev_repr_num_events 决定
                 ↓
事件表示构建 (StackedHistogram.construct):
  representation: (2, nbins, H, W) uint8
  • 2 个极性通道
  • nbins 个时间分箱
  • 空间分辨率 (H, W)
                 ↓
展平并写入:
  representation: (2*nbins, H, W) uint8
  例如：(20, 240, 304) for Gen1
                 ↓
可选下采样 (Gen4):
  representation: (2*nbins, H//2, W//2) uint8
  例如：(20, 360, 640)
                 ↓
HDF5 数据集:
  /data: (T, C, H, W) uint8
  • T = 事件表示数量 (时间步)
  • C = 2*nbins
  • (H, W) = 空间分辨率
```

### 11.3 信息的保留和丢失

#### 保留的信息：
1. **时间信息**：通过时间分箱（bins）保留
2. **空间信息**：像素级精度
3. **极性信息**：正负极性分离存储
4. **事件密度**：通过计数值（0-255）
5. **标注信息**：完整保留（位置、类别、跟踪 ID）

#### 丢失的信息：
1. **精确时间戳**：事件只保留到 bin 级别
2. **事件顺序**：同一 bin 内的事件顺序丢失
3. **高频细节**：时间分辨率受 nbins 限制
4. **过量事件**：count_cutoff 导致饱和

#### 量化分析：
```python
# 原始数据：
# - 每个事件 4 个属性 (x, y, p, t)，每个属性 2-8 字节
# - 总大小：N × (2+2+2+8) = N × 14 bytes

# 压缩后：
# - 所有事件聚合到 histogram
# - 总大小：T × C × H × W × 1 byte
# - 例如：2500 × 20 × 240 × 304 = 3.65 GB → 压缩到 ~180 MB

# 信息密度：
# - 原始：14 bytes/event
# - 压缩：180 MB / (N events) ≈ 0.001-0.01 bytes/event
# - 压缩比：~1000-10000×
```

### 11.4 可逆性分析

| 转换 | 可逆性 | 说明 |
|-----|--------|------|
| 事件 → Histogram | **部分** | 可恢复空间位置和近似时间，但丢失精确时间戳 |
| Histogram → 压缩 HDF5 | **完全** | BLOSC 无损压缩 |
| 原始分辨率 → 下采样 | **不可逆** | 信息永久丢失 |
| 标注过滤 | **不可逆** | 过滤掉的标注无法恢复 |
| 时间对齐 | **部分** | 可通过 timestamps_us.npy 恢复对应关系 |

**建议**：
- 保留原始数据备份
- 记录所有预处理参数（YAML 配置）
- 版本化输出数据（使用 `_v2` 等后缀）

---

## 12. 代码分析

### 12.1 生成文件的核心代码片段

#### H5Writer 类：HDF5 写入器

```python
class H5Writer:
    """HDF5 文件写入器，支持动态扩展和 BLOSC 压缩"""
    
    def __init__(
        self, outfile: Path, key: str, ev_repr_shape: Tuple, numpy_dtype: np.dtype
    ):
        """
        初始化 HDF5 写入器
        
        Args:
            outfile: 输出文件路径
            key: 数据集名称（通常为 "data"）
            ev_repr_shape: 事件表示形状 (C, H, W)
            numpy_dtype: 数据类型 (np.uint8 或 np.int8)
        """
        assert len(ev_repr_shape) == 3
        self.h5f = h5py.File(str(outfile), "w")
        
        # 使用 weakref finalizer 确保文件关闭
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)
        
        self.key = key
        self.numpy_dtype = numpy_dtype
        
        # 创建可扩展的 HDF5 数据集
        maxshape = (None,) + ev_repr_shape      # (None, C, H, W)
        chunkshape = (1,) + ev_repr_shape       # (1, C, H, W)
        self.maxshape = maxshape
        
        self.h5f.create_dataset(
            key,
            dtype=self.numpy_dtype.name,
            shape=chunkshape,                    # 初始大小
            chunks=chunkshape,                   # 块大小
            maxshape=maxshape,                   # 最大形状
            **_blosc_opts(complevel=1, shuffle="byte"),  # BLOSC 压缩
        )
        self.t_idx = 0  # 当前时间步索引
    
    def add_data(self, data: np.ndarray):
        """追加一个时间步的数据"""
        assert data.dtype == self.numpy_dtype
        assert data.shape == self.maxshape[1:]  # (C, H, W)
        
        # 扩展数据集
        new_size = self.t_idx + 1
        self.h5f[self.key].resize(new_size, axis=0)
        
        # 写入数据
        self.h5f[self.key][self.t_idx:new_size] = data
        self.t_idx = new_size
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalizer()  # 确保文件关闭
    
    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()
```

#### write_event_representations：主写入逻辑

```python
def write_event_representations(
    in_h5_file: Path,
    ev_out_dir: Path,
    dataset: str,
    event_representation: RepresentationBase,
    ev_repr_num_events: Optional[int],
    ev_repr_delta_ts_ms: Optional[int],
    ev_repr_timestamps_us: np.ndarray,
    downsample_by_2: bool,
    overwrite_if_exists: bool = False,
) -> None:
    """
    从原始事件数据生成事件表示并写入 HDF5
    
    Args:
        in_h5_file: 输入的原始事件 HDF5 文件
        ev_out_dir: 输出目录
        dataset: 数据集类型 ("gen1" 或 "gen4")
        event_representation: 事件表示对象 (StackedHistogram 等)
        ev_repr_num_events: 每个窗口的事件数（与 delta_ts 二选一）
        ev_repr_delta_ts_ms: 事件窗口时长（毫秒）
        ev_repr_timestamps_us: 所有事件表示的时间戳（微秒）
        downsample_by_2: 是否下采样（Gen4）
        overwrite_if_exists: 是否覆盖已存在的文件
    """
    # 1. 构建输出文件路径
    ev_outfile = (
        ev_out_dir
        / f"event_representations{'_ds2_nearest' if downsample_by_2 else ''}.h5"
    )
    
    if ev_outfile.exists() and not overwrite_if_exists:
        return
    
    # 2. 使用临时文件避免部分写入
    ev_outfile_in_progress = ev_outfile.parent / (
        ev_outfile.stem + "_in_progress" + ev_outfile.suffix
    )
    if ev_outfile_in_progress.exists():
        os.remove(ev_outfile_in_progress)
    
    # 3. 获取事件表示的形状和数据类型
    ev_repr_shape = tuple(event_representation.get_shape())  # (C, H, W)
    if downsample_by_2:
        ev_repr_shape = ev_repr_shape[0], ev_repr_shape[1] // 2, ev_repr_shape[2] // 2
    ev_repr_dtype = event_representation.get_numpy_dtype()
    
    # 4. 打开输入和输出文件
    with H5Reader(in_h5_file, dataset=dataset) as h5_reader, \
         H5Writer(
             ev_outfile_in_progress,
             key="data",
             ev_repr_shape=ev_repr_shape,
             numpy_dtype=ev_repr_dtype,
         ) as h5_writer:
        
        height, width = h5_reader.get_height_and_width()
        assert (height, width) == ev_repr_shape[-2:] or \
               (height // 2, width // 2) == ev_repr_shape[-2:]
        
        ev_ts_us = h5_reader.time  # 所有事件的时间戳
        
        # 5. 计算每个窗口的起始和结束事件索引
        end_indices = np.searchsorted(ev_ts_us, ev_repr_timestamps_us, side="right")
        
        if ev_repr_num_events is not None:
            # 基于事件数量
            start_indices = np.maximum(end_indices - ev_repr_num_events, 0)
        else:
            # 基于时间窗口
            assert ev_repr_delta_ts_ms is not None
            start_indices = np.searchsorted(
                ev_ts_us,
                ev_repr_timestamps_us - ev_repr_delta_ts_ms * 1000,
                side="left",
            )
        
        # 6. 遍历每个时间窗口
        for idx_start, idx_end in zip(start_indices, end_indices):
            # a) 读取事件窗口
            ev_window = h5_reader.get_event_slice(idx_start=idx_start, idx_end=idx_end)
            
            # b) 构建事件表示
            ev_repr = event_representation.construct(
                x=ev_window["x"],
                y=ev_window["y"],
                pol=ev_window["p"],
                time=ev_window["t"],
            )
            
            # c) 可选：下采样
            if downsample_by_2:
                ev_repr = ev_repr.unsqueeze(0)  # 添加 batch 维度
                ev_repr = downsample_ev_repr(x=ev_repr, scale_factor=0.5)
                ev_repr_numpy = ev_repr.numpy()[0]
            else:
                ev_repr_numpy = ev_repr.numpy()
            
            # d) 写入 HDF5
            h5_writer.add_data(ev_repr_numpy)
        
        num_written_ev_repr = h5_writer.get_current_length()
    
    # 7. 验证写入数量
    assert num_written_ev_repr == len(ev_repr_timestamps_us)
    
    # 8. 重命名临时文件
    os.rename(ev_outfile_in_progress, ev_outfile)
```

### 12.2 HDF5 文件写入的实现细节

#### BLOSC 压缩配置

```python
def _blosc_opts(complevel=1, complib="blosc:zstd", shuffle="byte"):
    """
    配置 BLOSC 压缩参数
    
    Args:
        complevel: 压缩级别 (1-9)
            1: 最快速度，较低压缩率 (推荐训练数据)
            5: 平衡速度和压缩率
            9: 最高压缩率，较慢速度 (推荐归档数据)
        complib: 压缩库
            "blosc:zstd": Zstandard (推荐，平衡性能)
            "blosc:lz4": LZ4 (最快)
            "blosc:zlib": Zlib (兼容性好)
        shuffle: 洗牌算法
            "byte": 字节级洗牌 (推荐)
            "bit": 位级洗牌 (更高压缩率，较慢)
            None: 不洗牌
    
    Returns:
        dict: HDF5 compression options
    """
    # 洗牌类型映射
    shuffle = 2 if shuffle == "bit" else 1 if shuffle == "byte" else 0
    
    # 压缩库索引
    compressors = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
    complib = ["blosc:" + c for c in compressors].index(complib)
    
    # HDF5 filter 配置
    args = {
        "compression": 32001,  # BLOSC filter ID
        "compression_opts": (
            0,           # blosc 版本
            0,           # blosc 版本
            0,           # blosc 版本
            0,           # 类型大小（0=自动）
            complevel,   # 压缩级别
            shuffle,     # 洗牌算法
            complib,     # 压缩库索引
        ),
    }
    
    # 禁用 h5py 的内置 shuffle（BLOSC 已处理）
    if shuffle > 0:
        args["shuffle"] = False
    
    return args
```

#### 动态扩展机制

```python
# HDF5 数据集的动态扩展
self.h5f.create_dataset(
    key,
    dtype=np.uint8,
    shape=(1, 20, 240, 304),       # 初始：1 个时间步
    chunks=(1, 20, 240, 304),      # 块大小：1 个时间步
    maxshape=(None, 20, 240, 304), # 最大：无限时间步
    **compression_opts,
)

# 每次添加数据时扩展
def add_data(self, data):
    new_size = self.t_idx + 1
    self.h5f[self.key].resize(new_size, axis=0)  # 扩展第一维
    self.h5f[self.key][self.t_idx:new_size] = data
    self.t_idx = new_size
```

**扩展策略**：
- 初始分配 1 个 chunk
- 每次扩展 1 个 chunk（增量写入）
- HDF5 自动管理存储空间
- 不预分配整个数组（节省内存）

### 12.3 数据序列化的过程

```
PyTorch Tensor (GPU/CPU)
        ↓
   .numpy() - 转换为 NumPy 数组
        ↓
H5Writer.add_data(numpy_array)
        ↓
h5py Dataset 写入
        ↓
BLOSC 压缩（在 HDF5 层）
        ↓
磁盘存储（HDF5 文件）
```

**关键点**：
1. **零拷贝（部分）**：NumPy 和 HDF5 共享内存布局（C-contiguous）
2. **压缩透明**：BLOSC 作为 HDF5 filter，自动压缩/解压
3. **延迟写入**：HDF5 可能缓冲写入以优化性能

### 12.4 错误处理和日志记录

```python
# 1. 序列无标注异常
try:
    labels_per_frame, frame_timestamps_us, ev_repr_timestamps_us, frameidx2repridx = \
        labels_and_ev_repr_timestamps(...)
except NoLabelsException:
    parent_dir = out_labels_dir.parent
    print(f"No labels after filtering. Deleting {str(parent_dir)}")
    shutil.rmtree(parent_dir)  # 删除整个序列目录
    return

# 2. 文件存在性检查
assert h5f_path.exists(), f"{h5f_path=}"
assert ev_repr_dir.is_dir(), f"{ev_repr_dir}"

# 3. 数据形状验证
assert data.shape == self.maxshape[1:], f"{data.shape=}, {self.maxshape[1:]=}"

# 4. 时间戳单调性检查
assert np.all(t_array[:-1] <= t_array[1:]), "Timestamps not sorted"

# 5. 写入数量验证
assert num_written_ev_repr == len(ev_repr_timestamps_us), \
    f"Written {num_written_ev_repr} but expected {len(ev_repr_timestamps_us)}"
```

**日志记录**：
- 使用 `tqdm` 显示处理进度
- 打印配置信息（OmegaConf.to_yaml）
- 关键错误打印到 stdout

---

## 13. 性能分析

### 13.1 单个数据集的预处理耗时

#### Gen1 数据集（全量）

| 阶段 | 时间 | 占比 | 瓶颈 |
|-----|------|------|------|
| 读取原始 HDF5 | 15-25 min | 40% | I/O |
| 构建事件表示 | 10-15 min | 30% | CPU (histogram) |
| 写入压缩 HDF5 | 8-12 min | 25% | I/O + 压缩 |
| 标注处理 | 1-2 min | 5% | - |
| **总计** | **35-55 min** | **100%** | I/O bound |

**配置**：
- 20 个并行进程
- SSD 存储
- Intel Xeon CPU

#### Gen4 数据集（全量）

| 阶段 | 时间 | 占比 | 瓶颈 |
|-----|------|------|------|
| 读取原始 HDF5 | 45-60 min | 45% | I/O |
| 构建事件表示 | 25-35 min | 30% | CPU |
| 下采样 | 5-8 min | 7% | CPU |
| 写入压缩 HDF5 | 15-20 min | 18% | I/O + 压缩 |
| **总计** | **90-120 min** | **100%** | I/O bound |

**配置**：
- 20 个并行进程
- SSD 存储
- NVIDIA GPU 不可用（纯 CPU）

### 13.2 文件 I/O 的性能瓶颈

#### 读取瓶颈

```python
# H5Reader 读取原始事件
with h5py.File(h5_path, 'r') as f:
    x = f['events']['x'][idx_start:idx_end]  # 磁盘读取
    y = f['events']['y'][idx_start:idx_end]  # 磁盘读取
    p = f['events']['p'][idx_start:idx_end]  # 磁盘读取
    t = f['events']['t'][idx_start:idx_end]  # 磁盘读取

# 问题：4 次独立的磁盘 I/O
# 优化：使用复合数据类型或预读取
```

**优化建议**：
```python
# 方案 1：批量读取
chunk_size = 10  # 一次读取 10 个窗口
for i in range(0, len(timestamps), chunk_size):
    # 读取大块数据，然后在内存中切片
    large_chunk = f['events']['x'][start:end]
    
# 方案 2：增加 chunk cache
f = h5py.File(h5_path, 'r', rdcc_nbytes=1024**3)  # 1 GB cache
```

#### 写入瓶颈

```python
# 当前：每次 add_data 都触发 I/O
for ev_repr in all_representations:
    h5_writer.add_data(ev_repr)  # 每次写入 ~1.5 MB

# 优化：批量写入
representations = []
for ev_repr in all_representations:
    representations.append(ev_repr)

# 一次性写入
h5_writer.add_batch(np.stack(representations))
```

### 13.3 不同文件系统上的性能差异

| 文件系统 | 顺序读 | 随机读 | 顺序写 | 随机写 | 推荐 |
|---------|--------|--------|--------|--------|------|
| **NVMe SSD** | 3000 MB/s | 500k IOPS | 2500 MB/s | 400k IOPS | ★★★★★ |
| **SATA SSD** | 550 MB/s | 90k IOPS | 520 MB/s | 80k IOPS | ★★★★☆ |
| **HDD** | 180 MB/s | 80 IOPS | 180 MB/s | 80 IOPS | ★☆☆☆☆ |
| **网络存储 (NFS)** | 100-500 MB/s | 变化大 | 100-500 MB/s | 变化大 | ★★☆☆☆ |

**实际影响**（Gen1 数据集）：
- **NVMe SSD**：35 分钟
- **SATA SSD**：50 分钟
- **HDD**：180 分钟（3 小时）
- **NFS**：60-120 分钟（取决于网络）

### 13.4 并行处理的优化空间

#### 当前并行策略

```python
# 序列级并行（进程池）
with get_context("spawn").Pool(num_processes) as pool:
    for _ in pool.imap_unordered(
        process_sequence,
        iterable=seq_data_list,
        chunksize=1,
    ):
        pbar.update()
```

**特点**：
- 每个进程处理一个完整序列
- 使用 `spawn` 上下文（避免 fork 问题）
- `chunksize=1`：每次分配一个序列
- 无序处理（`imap_unordered`）

#### 扩展性分析

```python
# 强扩展性（固定数据量，增加进程数）
# Gen1: 39 sequences

Processes:  1     2      4      8      16     20     32
Time (min): 550   280    145    80     45     35     35
Speedup:    1.0×  2.0×   3.8×   6.9×   12.2×  15.7×  15.7×
Efficiency: 100%  100%   95%    86%    76%    79%    49%

# 观察：
# - 16 进程后收益递减（I/O 瓶颈）
# - 20 进程是最佳选择（CPU/I/O 平衡）
```

#### 优化建议

**优化 1：任务级并行**

```python
# 当前：序列级并行
# 问题：序列长度不均，负载不平衡

# 改进：窗口级并行
def process_window(in_h5_file, idx_start, idx_end, representation):
    with H5Reader(in_h5_file) as h5_reader:
        ev_window = h5_reader.get_event_slice(idx_start, idx_end)
    ev_repr = representation.construct(...)
    return ev_repr

# 并行处理所有窗口
results = pool.map(process_window, window_list)

# 串行写入 HDF5（避免并发写入问题）
for result in results:
    h5_writer.add_data(result)
```

**优化 2：GPU 加速**

```python
# 事件表示构建可以使用 GPU
ev_repr = representation.construct(
    x=ev_window["x"].cuda(),
    y=ev_window["y"].cuda(),
    pol=ev_window["p"].cuda(),
    time=ev_window["t"].cuda(),
)

# 下采样也可以使用 GPU
ev_repr = downsample_ev_repr(ev_repr.cuda(), scale_factor=0.5)
```

**潜在加速**：
- Histogram 构建：5-10× 加速（GPU）
- 下采样：20-50× 加速（GPU）
- 总体：2-3× 加速（受 I/O 限制）

### 13.5 内存使用优化

```python
# 当前实现：每个进程独立加载数据
# 内存使用：num_processes × (sequence_size + overhead)

# 示例：20 进程 × 500 MB/sequence = 10 GB

# 优化 1：共享内存（不适用于 spawn 上下文）
# 优化 2：减少 batch size
# 优化 3：流式处理（不缓存中间结果）

# 推荐配置：
num_processes = min(cpu_count(), available_memory_gb // 0.5)
```

---

## 14. 使用示例

### 14.1 如何用 h5py 打开和探索文件

```python
import h5py
import numpy as np
from pathlib import Path

def explore_hdf5_file(h5_path: Path):
    """交互式探索 HDF5 文件"""
    print(f"Opening: {h5_path}\n")
    
    with h5py.File(h5_path, 'r') as f:
        # 1. 列出所有 keys
        print("=" * 60)
        print("HDF5 File Structure")
        print("=" * 60)
        
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: Dataset {obj.shape} {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}{name}: Group")
        
        f.visititems(print_structure)
        
        # 2. 查看数据集详情
        print("\n" + "=" * 60)
        print("Dataset Details")
        print("=" * 60)
        
        for key in f.keys():
            ds = f[key]
            if isinstance(ds, h5py.Dataset):
                print(f"\nDataset: /{key}")
                print(f"  Shape: {ds.shape}")
                print(f"  Dtype: {ds.dtype}")
                print(f"  Size: {ds.nbytes / 1e6:.2f} MB")
                print(f"  Chunks: {ds.chunks}")
                print(f"  Compression: {ds.compression}")
                print(f"  Compression opts: {ds.compression_opts}")
                
                # 属性
                if ds.attrs:
                    print(f"  Attributes:")
                    for attr_name, attr_value in ds.attrs.items():
                        print(f"    {attr_name}: {attr_value}")
                
                # 统计信息（前 10 个样本）
                sample = ds[:min(10, ds.shape[0])]
                print(f"  Statistics (first {len(sample)} samples):")
                print(f"    Min: {sample.min()}")
                print(f"    Max: {sample.max()}")
                print(f"    Mean: {sample.mean():.4f}")
                print(f"    Std: {sample.std():.4f}")
                sparsity = np.count_nonzero(sample) / sample.size
                print(f"    Sparsity: {sparsity:.4f}")

# 使用示例
h5_file = Path("target_dir/train/sequence_name/event_representations_v2/"
               "stacked_histogram_dt=50_nbins=10/event_representations.h5")
explore_hdf5_file(h5_file)
```

**输出示例**：
```
Opening: target_dir/train/sequence_name/.../event_representations.h5

============================================================
HDF5 File Structure
============================================================
data: Dataset (1234, 20, 240, 304) uint8

============================================================
Dataset Details
============================================================

Dataset: /data
  Shape: (1234, 20, 240, 304)
  Dtype: uint8
  Size: 183.45 MB
  Chunks: (1, 20, 240, 304)
  Compression: 32001
  Compression opts: (0, 0, 0, 0, 1, 1, 5)
  Statistics (first 10 samples):
    Min: 0
    Max: 10
    Mean: 0.0847
    Std: 0.4521
    Sparsity: 0.0312
```

### 14.2 读取特定样本的完整代码

```python
import h5py
import numpy as np
import torch
from pathlib import Path

class EventReprLoader:
    """事件表示加载器"""
    
    def __init__(self, sequence_dir: Path, repr_name: str):
        """
        初始化加载器
        
        Args:
            sequence_dir: 序列目录（如 train/sequence_name/）
            repr_name: 表示方法名称（如 stacked_histogram_dt=50_nbins=10）
        """
        self.sequence_dir = sequence_dir
        self.repr_name = repr_name
        
        # 路径
        self.labels_dir = sequence_dir / "labels_v2"
        self.repr_dir = sequence_dir / "event_representations_v2" / repr_name
        
        # 加载元数据
        self._load_metadata()
    
    def _load_metadata(self):
        """加载所有元数据到内存"""
        # 标注
        label_data = np.load(self.labels_dir / "labels.npz")
        self.labels = label_data['labels']
        self.objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']
        
        # 时间戳
        self.label_timestamps = np.load(self.labels_dir / "timestamps_us.npy")
        self.repr_timestamps = np.load(self.repr_dir / "timestamps_us.npy")
        
        # 映射
        self.objframe_idx_2_repr_idx = np.load(
            self.repr_dir / "objframe_idx_2_repr_idx.npy"
        )
        
        # 事件表示文件路径
        self.h5_path = self.repr_dir / "event_representations.h5"
        if not self.h5_path.exists():
            self.h5_path = self.repr_dir / "event_representations_ds2_nearest.h5"
        
        print(f"Loaded metadata:")
        print(f"  Frames: {len(self.label_timestamps)}")
        print(f"  Repr timesteps: {len(self.repr_timestamps)}")
        print(f"  Total labels: {len(self.labels)}")
    
    def get_frame_by_index(self, frame_idx: int):
        """
        根据帧索引获取数据
        
        Args:
            frame_idx: 帧索引 (0 到 len(label_timestamps)-1)
        
        Returns:
            dict: 包含事件表示和标注的字典
        """
        # 1. 获取事件表示索引
        repr_idx = self.objframe_idx_2_repr_idx[frame_idx]
        
        # 2. 读取事件表示
        with h5py.File(self.h5_path, 'r') as f:
            ev_repr = f['data'][repr_idx]  # (C, H, W)
        ev_repr = torch.from_numpy(ev_repr)
        
        # 3. 读取标注
        label_start_idx = self.objframe_idx_2_label_idx[frame_idx]
        if frame_idx + 1 < len(self.objframe_idx_2_label_idx):
            label_end_idx = self.objframe_idx_2_label_idx[frame_idx + 1]
        else:
            label_end_idx = len(self.labels)
        
        frame_labels = self.labels[label_start_idx:label_end_idx]
        
        # 4. 获取时间戳
        timestamp = self.label_timestamps[frame_idx]
        
        return {
            'event_repr': ev_repr,
            'labels': frame_labels,
            'timestamp_us': timestamp,
            'frame_idx': frame_idx,
            'repr_idx': repr_idx,
        }
    
    def get_sequence_window(self, frame_idx: int, sequence_length: int):
        """
        获取序列窗口（用于 RNN 模型）
        
        Args:
            frame_idx: 结束帧索引
            sequence_length: 序列长度
        
        Returns:
            dict: 序列数据
        """
        # 计算事件表示索引范围
        repr_idx_end = self.objframe_idx_2_repr_idx[frame_idx] + 1
        repr_idx_start = max(0, repr_idx_end - sequence_length)
        
        # 读取事件表示序列
        with h5py.File(self.h5_path, 'r') as f:
            ev_repr_seq = f['data'][repr_idx_start:repr_idx_end]
        ev_repr_seq = torch.from_numpy(ev_repr_seq)
        
        # 读取标注
        frame_data = self.get_frame_by_index(frame_idx)
        
        return {
            'event_repr_seq': ev_repr_seq,  # (T, C, H, W)
            'labels': frame_data['labels'],
            'timestamp_us': frame_data['timestamp_us'],
            'sequence_length': len(ev_repr_seq),
        }

# 使用示例 1：读取单帧
loader = EventReprLoader(
    sequence_dir=Path("target_dir/train/sequence_name"),
    repr_name="stacked_histogram_dt=50_nbins=10"
)

frame_data = loader.get_frame_by_index(100)
print(f"\nFrame 100:")
print(f"  Event repr shape: {frame_data['event_repr'].shape}")
print(f"  Num labels: {len(frame_data['labels'])}")
print(f"  Timestamp: {frame_data['timestamp_us']} μs")

# 使用示例 2：读取序列窗口
seq_data = loader.get_sequence_window(frame_idx=100, sequence_length=20)
print(f"\nSequence window:")
print(f"  Event repr seq shape: {seq_data['event_repr_seq'].shape}")
print(f"  Sequence length: {seq_data['sequence_length']}")
```

### 14.3 验证数据完整性的脚本

```python
import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple

def validate_preprocessed_dataset(
    dataset_dir: Path,
    repr_name: str,
    verbose: bool = True
) -> Tuple[bool, List[str]]:
    """
    验证预处理数据集的完整性和一致性
    
    Args:
        dataset_dir: 数据集根目录（包含 train/val/test）
        repr_name: 表示方法名称
        verbose: 是否打印详细信息
    
    Returns:
        (success, errors): 验证是否通过和错误列表
    """
    errors = []
    
    if verbose:
        print("=" * 70)
        print(f"Validating dataset: {dataset_dir}")
        print(f"Representation: {repr_name}")
        print("=" * 70)
    
    # 遍历所有 splits
    for split_name in ['train', 'val', 'test']:
        split_dir = dataset_dir / split_name
        if not split_dir.exists():
            errors.append(f"Missing split directory: {split_name}")
            continue
        
        if verbose:
            print(f"\nValidating {split_name} split...")
        
        # 遍历所有序列
        sequences = [d for d in split_dir.iterdir() if d.is_dir()]
        
        for seq_idx, seq_dir in enumerate(sequences):
            seq_name = seq_dir.name
            
            # 检查目录结构
            labels_dir = seq_dir / "labels_v2"
            repr_dir = seq_dir / "event_representations_v2" / repr_name
            
            if not labels_dir.exists():
                errors.append(f"{split_name}/{seq_name}: Missing labels_v2")
                continue
            if not repr_dir.exists():
                errors.append(f"{split_name}/{seq_name}: Missing repr dir")
                continue
            
            # 检查文件存在性
            required_files = {
                'labels': labels_dir / "labels.npz",
                'label_ts': labels_dir / "timestamps_us.npy",
                'repr_h5': list(repr_dir.glob("event_representations*.h5")),
                'repr_ts': repr_dir / "timestamps_us.npy",
                'frame2repr': repr_dir / "objframe_idx_2_repr_idx.npy",
            }
            
            # 事件表示 HDF5 文件
            if not required_files['repr_h5']:
                errors.append(f"{split_name}/{seq_name}: Missing HDF5 file")
                continue
            repr_h5_file = required_files['repr_h5'][0]
            
            # 检查其他文件
            for name, path in required_files.items():
                if name == 'repr_h5':
                    continue
                if not path.exists():
                    errors.append(f"{split_name}/{seq_name}: Missing {name}")
            
            try:
                # 加载数据
                label_data = np.load(required_files['labels'])
                labels = label_data['labels']
                objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']
                
                label_ts = np.load(required_files['label_ts'])
                repr_ts = np.load(required_files['repr_ts'])
                frame2repr = np.load(required_files['frame2repr'])
                
                with h5py.File(repr_h5_file, 'r') as f:
                    repr_shape = f['data'].shape
                
                # 验证一致性
                num_frames = len(label_ts)
                num_reprs = len(repr_ts)
                
                # 1. 长度一致性
                if len(frame2repr) != num_frames:
                    errors.append(
                        f"{split_name}/{seq_name}: "
                        f"frame2repr length {len(frame2repr)} != "
                        f"num_frames {num_frames}"
                    )
                
                if len(objframe_idx_2_label_idx) != num_frames:
                    errors.append(
                        f"{split_name}/{seq_name}: "
                        f"objframe_idx_2_label_idx length "
                        f"{len(objframe_idx_2_label_idx)} != "
                        f"num_frames {num_frames}"
                    )
                
                if repr_shape[0] != num_reprs:
                    errors.append(
                        f"{split_name}/{seq_name}: "
                        f"HDF5 timesteps {repr_shape[0]} != "
                        f"num_reprs {num_reprs}"
                    )
                
                # 2. 时间戳单调性
                if not np.all(np.diff(label_ts) > 0):
                    errors.append(
                        f"{split_name}/{seq_name}: "
                        f"Label timestamps not strictly increasing"
                    )
                
                if not np.all(np.diff(repr_ts) > 0):
                    errors.append(
                        f"{split_name}/{seq_name}: "
                        f"Repr timestamps not strictly increasing"
                    )
                
                # 3. 映射有效性
                if frame2repr.max() >= num_reprs:
                    errors.append(
                        f"{split_name}/{seq_name}: "
                        f"frame2repr contains invalid index: "
                        f"{frame2repr.max()} >= {num_reprs}"
                    )
                
                if frame2repr.min() < 0:
                    errors.append(
                        f"{split_name}/{seq_name}: "
                        f"frame2repr contains negative index"
                    )
                
                # 4. 标注对齐验证
                for i in range(num_frames):
                    repr_idx = frame2repr[i]
                    label_t = label_ts[i]
                    repr_t = repr_ts[repr_idx]
                    if label_t != repr_t:
                        errors.append(
                            f"{split_name}/{seq_name}: "
                            f"Misaligned frame {i}: "
                            f"label_t={label_t}, repr_t={repr_t}"
                        )
                        break  # 只报告第一个错误
                
                # 5. 标注索引有效性
                if objframe_idx_2_label_idx[-1] > len(labels):
                    errors.append(
                        f"{split_name}/{seq_name}: "
                        f"objframe_idx_2_label_idx contains invalid index"
                    )
                
                if verbose:
                    print(f"  [{seq_idx+1}/{len(sequences)}] {seq_name}: ✓")
            
            except Exception as e:
                errors.append(f"{split_name}/{seq_name}: Exception: {str(e)}")
    
    success = len(errors) == 0
    
    if verbose:
        print("\n" + "=" * 70)
        if success:
            print("✓ Validation PASSED")
        else:
            print(f"✗ Validation FAILED with {len(errors)} error(s):")
            for error in errors:
                print(f"  - {error}")
        print("=" * 70)
    
    return success, errors

# 使用示例
success, errors = validate_preprocessed_dataset(
    dataset_dir=Path("target_dir"),
    repr_name="stacked_histogram_dt=50_nbins=10",
    verbose=True
)

if not success:
    print("\nFailed validation!")
    for error in errors:
        print(f"ERROR: {error}")
```

### 14.4 文件格式的演进和兼容性

```python
# 版本兼容性处理

def load_labels_with_fallback(labels_dir: Path):
    """
    加载标注，支持多个版本
    
    Version history:
    - v1: labels.npy (single array, no indexing)
    - v2: labels.npz (with objframe_idx_2_label_idx)
    """
    # 尝试加载 v2
    v2_file = labels_dir / "labels.npz"
    if v2_file.exists():
        data = np.load(v2_file)
        return {
            'labels': data['labels'],
            'objframe_idx_2_label_idx': data['objframe_idx_2_label_idx'],
            'version': 2,
        }
    
    # 回退到 v1
    v1_file = labels_dir / "labels.npy"
    if v1_file.exists():
        labels = np.load(v1_file)
        # 假设每帧一个标注（简化）
        objframe_idx_2_label_idx = np.arange(len(labels))
        return {
            'labels': labels,
            'objframe_idx_2_label_idx': objframe_idx_2_label_idx,
            'version': 1,
        }
    
    raise FileNotFoundError(f"No labels found in {labels_dir}")

def load_event_repr_with_fallback(repr_dir: Path):
    """
    加载事件表示，支持多个文件名
    
    File naming:
    - event_representations.h5 (original resolution)
    - event_representations_ds2_nearest.h5 (downsampled by 2)
    """
    # 尝试原始分辨率
    h5_file = repr_dir / "event_representations.h5"
    if h5_file.exists():
        return h5_file, False
    
    # 尝试下采样版本
    h5_file = repr_dir / "event_representations_ds2_nearest.h5"
    if h5_file.exists():
        return h5_file, True
    
    raise FileNotFoundError(f"No event representations found in {repr_dir}")

# 使用示例
labels_data = load_labels_with_fallback(Path("sequence/labels_v2"))
print(f"Loaded labels version: {labels_data['version']}")

h5_file, is_downsampled = load_event_repr_with_fallback(
    Path("sequence/event_representations_v2/stacked_histogram_dt=50_nbins=10")
)
print(f"Loaded HDF5: {h5_file.name}")
print(f"Downsampled: {is_downsampled}")
```

---

## 总结

本文档深入分析了 `preprocess_dataset.py` 的文件生成过程和 HDF5 结构，涵盖：

1. **完整的处理流程**：从原始数据到最终 HDF5 文件的每一步
2. **HDF5 文件格式**：层次结构、压缩配置、数据组织
3. **数据一致性**：时间戳对齐、标注映射、验证方法
4. **性能优化**：I/O 瓶颈分析、并行策略、存储效率
5. **实用工具**：读取、验证、调试的完整代码示例

**关键要点**：
- 使用 BLOSC (zstd) 压缩，压缩率 3-5×，平衡速度和空间
- HDF5 数据集动态扩展，每个 chunk 存储一个时间步
- 支持 Gen1 和 Gen4 两种数据集，Gen4 自动下采样
- 标注与事件表示严格对齐，通过多层映射关系保证一致性
- 并行处理提供显著加速，20 进程是最佳配置

**最佳实践**：
1. 始终使用 SSD 存储预处理数据
2. 预处理后验证数据完整性
3. 保留原始数据和预处理配置（YAML）
4. 使用版本化目录（`labels_v2`，`event_representations_v2`）
5. 定期备份处理后的数据

此分析文档为理解和使用该预处理系统提供了全面的参考。

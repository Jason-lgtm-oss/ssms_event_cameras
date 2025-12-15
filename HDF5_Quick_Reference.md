# HDF5 预处理文件快速参考指南

## 目录结构速查

```
target_dir/
├── train/
│   └── {sequence_name}/
│       ├── labels_v2/
│       │   ├── labels.npz                        # 标注数据
│       │   └── timestamps_us.npy                 # 标注时间戳
│       └── event_representations_v2/
│           └── {representation_name}/
│               ├── event_representations.h5      # 事件表示（主文件）
│               ├── objframe_idx_2_repr_idx.npy   # 帧到表示的映射
│               └── timestamps_us.npy             # 表示时间戳
├── val/
└── test/
```

## 文件格式速查表

| 文件 | 格式 | 形状 | 数据类型 | 说明 |
|-----|------|------|---------|------|
| `event_representations.h5` | HDF5 | `(T, C, H, W)` | `uint8`/`int8` | T=时间步, C=通道数 |
| `labels.npz` | NPZ | `(N,)` | structured | N=总标注数 |
| `timestamps_us.npy` | NPY | `(F,)` | `int64` | F=帧数，单位微秒 |
| `objframe_idx_2_repr_idx.npy` | NPY | `(F,)` | `int64` | 帧索引→表示索引 |

## 常用代码片段

### 1. 快速读取单帧数据

```python
import h5py
import numpy as np

# 读取事件表示
with h5py.File('event_representations.h5', 'r') as f:
    ev_repr = f['data'][timestep_idx]  # 形状: (C, H, W)

# 读取对应标注
label_data = np.load('labels.npz')
labels = label_data['labels']
objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']

# 获取某帧的标注
start_idx = objframe_idx_2_label_idx[frame_idx]
end_idx = objframe_idx_2_label_idx[frame_idx + 1] if frame_idx + 1 < len(objframe_idx_2_label_idx) else len(labels)
frame_labels = labels[start_idx:end_idx]
```

### 2. 遍历所有帧

```python
import h5py
import numpy as np

# 加载元数据
label_data = np.load('labels.npz')
frame2repr = np.load('objframe_idx_2_repr_idx.npy')
num_frames = len(frame2repr)

# 遍历
with h5py.File('event_representations.h5', 'r') as f:
    for frame_idx in range(num_frames):
        repr_idx = frame2repr[frame_idx]
        ev_repr = f['data'][repr_idx]
        
        # 获取标注
        start_idx = label_data['objframe_idx_2_label_idx'][frame_idx]
        end_idx = (label_data['objframe_idx_2_label_idx'][frame_idx + 1] 
                   if frame_idx + 1 < num_frames else len(label_data['labels']))
        frame_labels = label_data['labels'][start_idx:end_idx]
        
        # 处理数据
        process(ev_repr, frame_labels)
```

### 3. 验证文件完整性

```python
import h5py
import numpy as np
from pathlib import Path

def quick_validate(sequence_dir, repr_name):
    labels_dir = sequence_dir / "labels_v2"
    repr_dir = sequence_dir / "event_representations_v2" / repr_name
    
    # 检查文件存在
    assert (labels_dir / "labels.npz").exists()
    assert (labels_dir / "timestamps_us.npy").exists()
    assert (repr_dir / "objframe_idx_2_repr_idx.npy").exists()
    assert (repr_dir / "timestamps_us.npy").exists()
    
    h5_files = list(repr_dir.glob("event_representations*.h5"))
    assert len(h5_files) > 0
    
    # 检查一致性
    label_data = np.load(labels_dir / "labels.npz")
    label_ts = np.load(labels_dir / "timestamps_us.npy")
    repr_ts = np.load(repr_dir / "timestamps_us.npy")
    frame2repr = np.load(repr_dir / "objframe_idx_2_repr_idx.npy")
    
    with h5py.File(h5_files[0], 'r') as f:
        num_reprs = f['data'].shape[0]
    
    num_frames = len(label_ts)
    
    assert len(frame2repr) == num_frames
    assert frame2repr.max() < num_reprs
    assert len(repr_ts) == num_reprs
    
    print(f"✓ Validation passed: {num_frames} frames, {num_reprs} repr timesteps")
```

### 4. 批量读取事件表示

```python
import h5py
import torch

def load_event_repr_sequence(h5_path, start_idx, end_idx):
    """读取事件表示序列（用于RNN）"""
    with h5py.File(h5_path, 'r') as f:
        ev_repr_seq = f['data'][start_idx:end_idx]  # (T, C, H, W)
    return torch.from_numpy(ev_repr_seq)

# 示例：读取 20 个时间步
seq = load_event_repr_sequence('event_representations.h5', 100, 120)
print(seq.shape)  # torch.Size([20, 20, 240, 304])
```

### 5. 提取事件表示统计信息

```python
import h5py
import numpy as np

def compute_statistics(h5_path, num_samples=100):
    """计算事件表示的统计信息"""
    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        total_timesteps = data.shape[0]
        
        # 随机采样
        indices = np.random.choice(total_timesteps, min(num_samples, total_timesteps), replace=False)
        samples = data[indices]
        
        stats = {
            'mean': samples.mean(),
            'std': samples.std(),
            'min': samples.min(),
            'max': samples.max(),
            'sparsity': np.count_nonzero(samples) / samples.size,
        }
        
        return stats

stats = compute_statistics('event_representations.h5')
print(f"Mean: {stats['mean']:.4f}, Sparsity: {stats['sparsity']:.4f}")
```

## 数据集规格

### Gen1

```
分辨率: 240 × 304
通道数: 20 (Stacked Histogram)
       10 (Mixed Density)
类别数: 3 (pedestrian, two_wheeler, car)
帧率: ~10 Hz
序列数: train=25, val=6, test=8
```

### Gen4

```
原始分辨率: 720 × 1280
下采样分辨率: 360 × 640
通道数: 20 (Stacked Histogram)
       10 (Mixed Density)
类别数: 3 (过滤后)
帧率: ~10 Hz
序列数: train=40, val=10, test=12
文件名后缀: event_representations_ds2_nearest.h5
```

## 命令速查

### 运行预处理

```bash
# Gen1
python RVT/scripts/genx/preprocess_dataset.py \
    /path/to/gen1_raw/ \
    /path/to/gen1_processed/ \
    RVT/scripts/genx/conf_preprocess/representation/stacked_hist.yaml \
    RVT/scripts/genx/conf_preprocess/extraction/const_duration.yaml \
    RVT/scripts/genx/conf_preprocess/filter_gen1.yaml \
    -ds gen1 -np 20

# Gen4
python RVT/scripts/genx/preprocess_dataset.py \
    /path/to/gen4_raw/ \
    /path/to/gen4_processed/ \
    RVT/scripts/genx/conf_preprocess/representation/stacked_hist.yaml \
    RVT/scripts/genx/conf_preprocess/extraction/const_duration.yaml \
    RVT/scripts/genx/conf_preprocess/filter_gen4.yaml \
    -ds gen4 -np 20
```

### 检查文件

```bash
# 使用提供的工具脚本
python HDF5_file_structure_examples.py inspect-sequence \
    /path/to/sequence/ \
    --repr-name stacked_histogram_dt=50_nbins=10

# 使用 h5ls（需要安装 HDF5 工具）
h5ls -r event_representations.h5

# 使用 Python
python -c "import h5py; f = h5py.File('event_representations.h5', 'r'); print(f['data'].shape)"
```

## 常见问题排查

### 问题 1: 文件无法打开

```python
# 检查文件是否损坏
try:
    with h5py.File('event_representations.h5', 'r') as f:
        print(f['data'].shape)
except OSError as e:
    print(f"File corrupted: {e}")
    # 解决方案：删除并重新生成
```

### 问题 2: 时间戳不对齐

```python
# 验证对齐
label_ts = np.load('labels_v2/timestamps_us.npy')
repr_ts = np.load('event_representations_v2/.../timestamps_us.npy')
frame2repr = np.load('event_representations_v2/.../objframe_idx_2_repr_idx.npy')

for i in range(len(label_ts)):
    assert label_ts[i] == repr_ts[frame2repr[i]], f"Misaligned at frame {i}"
```

### 问题 3: 内存不足

```python
# 使用分块读取
with h5py.File('event_representations.h5', 'r') as f:
    data = f['data']
    chunk_size = 100
    for i in range(0, data.shape[0], chunk_size):
        chunk = data[i:i+chunk_size]
        process_chunk(chunk)
```

### 问题 4: 压缩库缺失

```bash
# 安装必要的库
pip install h5py hdf5plugin

# 验证安装
python -c "import hdf5plugin; print(hdf5plugin.version)"
```

## 性能优化建议

### 1. 使用 SSD 存储
```
NVMe SSD: 推荐
SATA SSD: 可接受
HDD: 不推荐（速度慢 5-10×）
```

### 2. 合理设置并行进程数
```bash
# 推荐配置
-np 20  # 对于 16-32 核 CPU

# 计算公式
num_processes = min(cpu_count, available_memory_gb / 0.5)
```

### 3. 增加 HDF5 缓存
```python
# 读取时增大 chunk cache
with h5py.File(h5_path, 'r', rdcc_nbytes=1024**3) as f:  # 1 GB cache
    data = f['data'][:]
```

### 4. 批量操作
```python
# 避免
for i in range(1000):
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][i]

# 推荐
with h5py.File(h5_path, 'r') as f:
    for i in range(1000):
        data = f['data'][i]
```

## 标注字段说明

```python
labels.dtype = [
    ('t', '<u8'),              # 时间戳（微秒）
    ('x', '<f4'),              # 边界框左上角 x 坐标
    ('y', '<f4'),              # 边界框左上角 y 坐标
    ('w', '<f4'),              # 边界框宽度
    ('h', '<f4'),              # 边界框高度
    ('class_id', 'u1'),        # 类别 ID (0=pedestrian, 1=two_wheeler, 2=car)
    ('class_confidence', '<f4'), # 类别置信度
    ('track_id', '<u4'),       # 跟踪 ID
]
```

## 更多资源

- **主文档**: `HDF5文件生成与结构深度分析.md`
- **示例脚本**: `HDF5_file_structure_examples.py`
- **源代码**: `RVT/scripts/genx/preprocess_dataset.py`
- **配置文件**: `RVT/scripts/genx/conf_preprocess/`

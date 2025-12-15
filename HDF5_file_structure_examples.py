#!/usr/bin/env python3
"""
HDF5 文件结构示例和实用工具脚本

提供读取、验证和可视化预处理数据的工具函数
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


# ============================================================================
# 1. 文件探索工具
# ============================================================================

def inspect_event_representation_file(h5_path: Path, show_samples: int = 5):
    """
    详细检查事件表示 HDF5 文件
    
    Args:
        h5_path: HDF5 文件路径
        show_samples: 显示前几个样本的统计信息
    """
    print("=" * 80)
    print(f"Event Representation File Inspection")
    print(f"File: {h5_path}")
    print("=" * 80)
    
    with h5py.File(h5_path, 'r') as f:
        # 基本信息
        print("\n1. File Structure:")
        print(f"   Keys: {list(f.keys())}")
        
        # 数据集信息
        data = f['data']
        print("\n2. Dataset 'data':")
        print(f"   Shape: {data.shape}")
        print(f"   Dtype: {data.dtype}")
        print(f"   Chunks: {data.chunks}")
        print(f"   Compression: {data.compression}")
        if data.compression_opts:
            print(f"   Compression opts: {data.compression_opts}")
        
        # 大小信息
        print("\n3. Storage Information:")
        print(f"   Logical size (uncompressed): {data.size * data.dtype.itemsize / 1e6:.2f} MB")
        print(f"   File size (compressed): {Path(h5_path).stat().st_size / 1e6:.2f} MB")
        compression_ratio = (data.size * data.dtype.itemsize) / Path(h5_path).stat().st_size
        print(f"   Compression ratio: {compression_ratio:.2f}×")
        
        # 读取样本进行统计
        print(f"\n4. Data Statistics (first {show_samples} timesteps):")
        sample = data[:show_samples]
        
        print(f"   Min value: {sample.min()}")
        print(f"   Max value: {sample.max()}")
        print(f"   Mean: {sample.mean():.4f}")
        print(f"   Std: {sample.std():.4f}")
        
        # 稀疏性分析
        total_elements = sample.size
        non_zero = np.count_nonzero(sample)
        sparsity = non_zero / total_elements
        print(f"   Non-zero elements: {non_zero} / {total_elements} ({sparsity:.4%})")
        
        # 按通道分析（假设是 Stacked Histogram）
        if data.shape[1] == 20:  # 2 polarities × 10 bins
            print("\n5. Channel Statistics:")
            positive_channels = sample[:, :10]  # 正极性
            negative_channels = sample[:, 10:]  # 负极性
            print(f"   Positive polarity mean: {positive_channels.mean():.4f}")
            print(f"   Negative polarity mean: {negative_channels.mean():.4f}")
            print(f"   Positive/Negative ratio: {positive_channels.sum() / max(negative_channels.sum(), 1):.2f}")


def inspect_labels_file(labels_dir: Path):
    """
    检查标注文件
    
    Args:
        labels_dir: labels_v2 目录路径
    """
    print("=" * 80)
    print(f"Labels File Inspection")
    print(f"Directory: {labels_dir}")
    print("=" * 80)
    
    # 加载数据
    label_data = np.load(labels_dir / "labels.npz")
    labels = label_data['labels']
    objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']
    timestamps_us = np.load(labels_dir / "timestamps_us.npy")
    
    print("\n1. Basic Information:")
    print(f"   Total frames: {len(timestamps_us)}")
    print(f"   Total labels: {len(labels)}")
    print(f"   Labels per frame (avg): {len(labels) / len(timestamps_us):.2f}")
    
    print("\n2. Label Fields:")
    print(f"   Dtype: {labels.dtype}")
    print(f"   Fields: {labels.dtype.names}")
    
    print("\n3. Label Statistics:")
    # 类别分布
    class_ids = labels['class_id']
    unique_classes, counts = np.unique(class_ids, return_counts=True)
    print(f"   Class distribution:")
    class_names = {0: 'pedestrian', 1: 'two_wheeler', 2: 'car'}
    for class_id, count in zip(unique_classes, counts):
        class_name = class_names.get(class_id, 'unknown')
        print(f"     {class_name} (id={class_id}): {count} ({count/len(labels):.2%})")
    
    # 边界框尺寸
    widths = labels['w']
    heights = labels['h']
    print(f"\n   Bounding box sizes:")
    print(f"     Width:  min={widths.min():.1f}, max={widths.max():.1f}, mean={widths.mean():.1f}")
    print(f"     Height: min={heights.min():.1f}, max={heights.max():.1f}, mean={heights.mean():.1f}")
    
    # 时间信息
    print(f"\n4. Temporal Information:")
    print(f"   First timestamp: {timestamps_us[0]} μs ({timestamps_us[0]/1e6:.2f} s)")
    print(f"   Last timestamp:  {timestamps_us[-1]} μs ({timestamps_us[-1]/1e6:.2f} s)")
    print(f"   Duration: {(timestamps_us[-1] - timestamps_us[0])/1e6:.2f} s")
    
    # 时间间隔
    if len(timestamps_us) > 1:
        diffs = np.diff(timestamps_us)
        print(f"   Frame intervals (ms):")
        print(f"     Min: {diffs.min()/1000:.2f}")
        print(f"     Max: {diffs.max()/1000:.2f}")
        print(f"     Mean: {diffs.mean()/1000:.2f}")
        print(f"     Median: {np.median(diffs)/1000:.2f}")


def inspect_sequence_directory(sequence_dir: Path, repr_name: str):
    """
    检查完整的序列目录
    
    Args:
        sequence_dir: 序列目录路径
        repr_name: 表示方法名称
    """
    print("=" * 80)
    print(f"Sequence Directory Inspection")
    print(f"Path: {sequence_dir}")
    print(f"Representation: {repr_name}")
    print("=" * 80)
    
    labels_dir = sequence_dir / "labels_v2"
    repr_dir = sequence_dir / "event_representations_v2" / repr_name
    
    # 检查目录存在性
    print("\n1. Directory Structure:")
    print(f"   labels_v2: {'✓' if labels_dir.exists() else '✗ MISSING'}")
    print(f"   event_representations_v2/{repr_name}: {'✓' if repr_dir.exists() else '✗ MISSING'}")
    
    if not labels_dir.exists() or not repr_dir.exists():
        print("\n   ERROR: Required directories missing!")
        return
    
    # 检查文件
    print("\n2. Files:")
    label_files = {
        'labels.npz': labels_dir / "labels.npz",
        'timestamps_us.npy': labels_dir / "timestamps_us.npy",
    }
    repr_files = {
        'event_representations*.h5': list(repr_dir.glob("event_representations*.h5")),
        'objframe_idx_2_repr_idx.npy': repr_dir / "objframe_idx_2_repr_idx.npy",
        'timestamps_us.npy': repr_dir / "timestamps_us.npy",
    }
    
    print("\n   Labels:")
    for name, path in label_files.items():
        exists = path.exists()
        size = path.stat().st_size / 1e3 if exists else 0
        print(f"     {name}: {'✓' if exists else '✗'} ({size:.1f} KB)")
    
    print("\n   Event Representations:")
    for name, path_or_list in repr_files.items():
        if isinstance(path_or_list, list):
            if path_or_list:
                path = path_or_list[0]
                size = path.stat().st_size / 1e6
                print(f"     {path.name}: ✓ ({size:.1f} MB)")
            else:
                print(f"     {name}: ✗")
        else:
            exists = path_or_list.exists()
            size = path_or_list.stat().st_size / 1e3 if exists else 0
            print(f"     {name}: {'✓' if exists else '✗'} ({size:.1f} KB)")
    
    # 详细检查
    if label_files['labels.npz'].exists():
        print("\n" + "-" * 80)
        inspect_labels_file(labels_dir)
    
    h5_files = list(repr_dir.glob("event_representations*.h5"))
    if h5_files:
        print("\n" + "-" * 80)
        inspect_event_representation_file(h5_files[0])


# ============================================================================
# 2. 数据验证工具
# ============================================================================

def validate_sequence_consistency(sequence_dir: Path, repr_name: str) -> Tuple[bool, List[str]]:
    """
    验证序列数据的一致性
    
    Args:
        sequence_dir: 序列目录路径
        repr_name: 表示方法名称
    
    Returns:
        (success, errors): 是否通过验证和错误列表
    """
    errors = []
    
    labels_dir = sequence_dir / "labels_v2"
    repr_dir = sequence_dir / "event_representations_v2" / repr_name
    
    # 加载数据
    try:
        label_data = np.load(labels_dir / "labels.npz")
        labels = label_data['labels']
        objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']
        
        label_timestamps = np.load(labels_dir / "timestamps_us.npy")
        repr_timestamps = np.load(repr_dir / "timestamps_us.npy")
        frame2repr = np.load(repr_dir / "objframe_idx_2_repr_idx.npy")
        
        h5_files = list(repr_dir.glob("event_representations*.h5"))
        if not h5_files:
            errors.append("No HDF5 file found")
            return False, errors
        
        with h5py.File(h5_files[0], 'r') as f:
            repr_shape = f['data'].shape
    
    except Exception as e:
        errors.append(f"Failed to load files: {str(e)}")
        return False, errors
    
    # 验证
    num_frames = len(label_timestamps)
    num_reprs = len(repr_timestamps)
    
    # 1. 数组长度一致性
    if len(frame2repr) != num_frames:
        errors.append(
            f"frame2repr length ({len(frame2repr)}) != num_frames ({num_frames})"
        )
    
    if len(objframe_idx_2_label_idx) != num_frames:
        errors.append(
            f"objframe_idx_2_label_idx length ({len(objframe_idx_2_label_idx)}) "
            f"!= num_frames ({num_frames})"
        )
    
    if repr_shape[0] != num_reprs:
        errors.append(
            f"HDF5 timesteps ({repr_shape[0]}) != num_reprs ({num_reprs})"
        )
    
    # 2. 时间戳单调性
    if len(label_timestamps) > 1:
        label_diffs = np.diff(label_timestamps)
        if not np.all(label_diffs > 0):
            errors.append(f"Label timestamps not strictly increasing")
    
    if len(repr_timestamps) > 1:
        repr_diffs = np.diff(repr_timestamps)
        if not np.all(repr_diffs > 0):
            errors.append(f"Repr timestamps not strictly increasing")
    
    # 3. 索引有效性
    if frame2repr.max() >= num_reprs:
        errors.append(
            f"frame2repr contains invalid index: {frame2repr.max()} >= {num_reprs}"
        )
    
    if frame2repr.min() < 0:
        errors.append(f"frame2repr contains negative index: {frame2repr.min()}")
    
    if objframe_idx_2_label_idx.max() > len(labels):
        errors.append(
            f"objframe_idx_2_label_idx contains invalid index: "
            f"{objframe_idx_2_label_idx.max()} > {len(labels)}"
        )
    
    # 4. 标注-表示对齐
    misaligned_count = 0
    for i in range(num_frames):
        repr_idx = frame2repr[i]
        if label_timestamps[i] != repr_timestamps[repr_idx]:
            misaligned_count += 1
    
    if misaligned_count > 0:
        errors.append(f"{misaligned_count} / {num_frames} frames misaligned")
    
    success = len(errors) == 0
    return success, errors


# ============================================================================
# 3. 数据可视化工具
# ============================================================================

def visualize_event_representation(
    h5_path: Path,
    timestep_idx: int,
    save_path: Optional[Path] = None
):
    """
    可视化单个时间步的事件表示
    
    Args:
        h5_path: HDF5 文件路径
        timestep_idx: 时间步索引
        save_path: 保存图像路径（可选）
    """
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][timestep_idx]  # (C, H, W)
    
    num_channels = data.shape[0]
    
    # 假设是 Stacked Histogram (20 通道)
    if num_channels == 20:
        # 分离正负极性
        positive_data = data[:10]  # 前 10 个通道：正极性
        negative_data = data[10:]  # 后 10 个通道：负极性
        
        # 求和得到总事件密度
        positive_sum = positive_data.sum(axis=0)
        negative_sum = negative_data.sum(axis=0)
        
        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = axes[0].imshow(positive_sum, cmap='hot')
        axes[0].set_title(f'Positive Polarity (sum of 10 bins)')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(negative_sum, cmap='hot')
        axes[1].set_title(f'Negative Polarity (sum of 10 bins)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # 组合视图
        combined = positive_sum - negative_sum
        im3 = axes[2].imshow(combined, cmap='RdBu_r', vmin=-10, vmax=10)
        axes[2].set_title(f'Combined (Pos - Neg)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        plt.suptitle(f'Event Representation at Timestep {timestep_idx}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
    else:
        # 通用可视化：显示所有通道
        ncols = min(4, num_channels)
        nrows = (num_channels + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
        axes = axes.flatten() if num_channels > 1 else [axes]
        
        for i in range(num_channels):
            im = axes[i].imshow(data[i], cmap='hot')
            axes[i].set_title(f'Channel {i}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
        
        # 隐藏多余的子图
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Event Representation at Timestep {timestep_idx}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()


def plot_temporal_statistics(sequence_dir: Path, repr_name: str, save_path: Optional[Path] = None):
    """
    绘制序列的时间统计信息
    
    Args:
        sequence_dir: 序列目录路径
        repr_name: 表示方法名称
        save_path: 保存路径（可选）
    """
    labels_dir = sequence_dir / "labels_v2"
    repr_dir = sequence_dir / "event_representations_v2" / repr_name
    
    # 加载数据
    label_data = np.load(labels_dir / "labels.npz")
    labels = label_data['labels']
    label_timestamps = np.load(labels_dir / "timestamps_us.npy")
    repr_timestamps = np.load(repr_dir / "timestamps_us.npy")
    
    # 计算统计
    label_intervals = np.diff(label_timestamps) / 1000  # 转换为毫秒
    repr_intervals = np.diff(repr_timestamps) / 1000
    
    # 每帧的标注数量
    objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']
    labels_per_frame = np.diff(
        np.append(objframe_idx_2_label_idx, len(labels))
    )
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 标注时间间隔分布
    axes[0, 0].hist(label_intervals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Interval (ms)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Label Frame Intervals')
    axes[0, 0].axvline(label_intervals.mean(), color='r', linestyle='--', 
                       label=f'Mean: {label_intervals.mean():.2f} ms')
    axes[0, 0].legend()
    
    # 2. 事件表示时间间隔分布
    axes[0, 1].hist(repr_intervals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Interval (ms)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Event Repr Intervals')
    axes[0, 1].axvline(repr_intervals.mean(), color='r', linestyle='--', 
                       label=f'Mean: {repr_intervals.mean():.2f} ms')
    axes[0, 1].legend()
    
    # 3. 每帧标注数量
    axes[1, 0].plot(labels_per_frame, linewidth=0.5)
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('Number of Labels')
    axes[1, 0].set_title('Labels per Frame')
    axes[1, 0].axhline(labels_per_frame.mean(), color='r', linestyle='--', 
                       label=f'Mean: {labels_per_frame.mean():.2f}')
    axes[1, 0].legend()
    
    # 4. 时间轴对比
    axes[1, 1].plot(label_timestamps / 1e6, np.arange(len(label_timestamps)), 
                    'o-', markersize=2, label='Label Frames', alpha=0.7)
    axes[1, 1].plot(repr_timestamps / 1e6, np.arange(len(repr_timestamps)), 
                    'x-', markersize=1, label='Event Repr', alpha=0.7)
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Index')
    axes[1, 1].set_title('Temporal Alignment')
    axes[1, 1].legend()
    
    plt.suptitle(f'Temporal Statistics: {sequence_dir.name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ============================================================================
# 4. 命令行接口
# ============================================================================

def main():
    """命令行主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='HDF5 文件结构分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 检查单个 HDF5 文件
  python hdf5_structure_examples.py inspect-h5 path/to/event_representations.h5
  
  # 检查标注文件
  python hdf5_structure_examples.py inspect-labels path/to/labels_v2/
  
  # 检查完整序列
  python hdf5_structure_examples.py inspect-sequence path/to/sequence/ \\
      --repr-name stacked_histogram_dt=50_nbins=10
  
  # 验证序列一致性
  python hdf5_structure_examples.py validate path/to/sequence/ \\
      --repr-name stacked_histogram_dt=50_nbins=10
  
  # 可视化事件表示
  python hdf5_structure_examples.py visualize path/to/event_representations.h5 \\
      --timestep 100 --save output.png
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # inspect-h5 命令
    parser_h5 = subparsers.add_parser('inspect-h5', help='检查 HDF5 文件')
    parser_h5.add_argument('h5_path', type=Path, help='HDF5 文件路径')
    parser_h5.add_argument('--samples', type=int, default=5, help='显示样本数')
    
    # inspect-labels 命令
    parser_labels = subparsers.add_parser('inspect-labels', help='检查标注文件')
    parser_labels.add_argument('labels_dir', type=Path, help='labels_v2 目录路径')
    
    # inspect-sequence 命令
    parser_seq = subparsers.add_parser('inspect-sequence', help='检查完整序列')
    parser_seq.add_argument('sequence_dir', type=Path, help='序列目录路径')
    parser_seq.add_argument('--repr-name', required=True, help='表示方法名称')
    
    # validate 命令
    parser_val = subparsers.add_parser('validate', help='验证序列一致性')
    parser_val.add_argument('sequence_dir', type=Path, help='序列目录路径')
    parser_val.add_argument('--repr-name', required=True, help='表示方法名称')
    
    # visualize 命令
    parser_vis = subparsers.add_parser('visualize', help='可视化事件表示')
    parser_vis.add_argument('h5_path', type=Path, help='HDF5 文件路径')
    parser_vis.add_argument('--timestep', type=int, default=0, help='时间步索引')
    parser_vis.add_argument('--save', type=Path, help='保存路径')
    
    args = parser.parse_args()
    
    if args.command == 'inspect-h5':
        inspect_event_representation_file(args.h5_path, args.samples)
    
    elif args.command == 'inspect-labels':
        inspect_labels_file(args.labels_dir)
    
    elif args.command == 'inspect-sequence':
        inspect_sequence_directory(args.sequence_dir, args.repr_name)
    
    elif args.command == 'validate':
        success, errors = validate_sequence_consistency(args.sequence_dir, args.repr_name)
        print("\n" + "=" * 80)
        if success:
            print("✓ Validation PASSED")
        else:
            print(f"✗ Validation FAILED with {len(errors)} error(s):")
            for error in errors:
                print(f"  - {error}")
        print("=" * 80)
    
    elif args.command == 'visualize':
        visualize_event_representation(args.h5_path, args.timestep, args.save)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

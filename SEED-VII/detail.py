import os
import numpy as np
from scipy.io import loadmat

# 请根据实际情况修改为你的预处理后 EEG 数据所在目录
dataset_dir = r"D:\dachuang\EEG_preprocessed"

# 采样率（Hz）
fs = 200.0


def inspect_seed_vii(dataset_dir):
    """
    遍历 dataset_dir 下所有 .mat 文件，
    对每个受试者打印其试次数、通道数、采样点数和时长（秒）
    """
    for fname in sorted(os.listdir(dataset_dir)):
        if not fname.lower().endswith('.mat'):
            continue
        subj_id = os.path.splitext(fname)[0]
        mat_path = os.path.join(dataset_dir, fname)
        print(f"\n=== Subject {subj_id} ({fname}) ===")

        try:
            mat = loadmat(mat_path, squeeze_me=True)
        except Exception as e:
            print(f"  [!] 无法加载 .mat 文件: {e}")
            continue

        # 筛选所有顶层的数字字段，作为试次 ID
        trial_ids = [k for k in mat.keys() if k.isdigit()]
        if not trial_ids:
            print("  [!] 没有找到数字命名的 trial 字段。")
            continue

        print(f"  Trials found: {len(trial_ids)}")
        for vid in sorted(trial_ids, key=lambda x: int(x)):
            trial_data = mat[vid]
            if isinstance(trial_data, np.ndarray):
                # 期望 shape = (n_channels, n_samples)
                if trial_data.ndim == 2:
                    n_channels, n_samples = trial_data.shape
                    duration_sec = n_samples / fs
                    print(f"    Video {vid:>2}: channels={n_channels}, "
                          f"samples={n_samples}, duration≈{duration_sec:.1f}s")
                else:
                    print(f"    Video {vid:>2}: 非 2D 数组 (ndim={trial_data.ndim}, "
                          f"shape={trial_data.shape})")
            else:
                print(f"    Video {vid:>2}: 非 ndarray 类型 ({type(trial_data)})")


if __name__ == "__main__":
    inspect_seed_vii(dataset_dir)

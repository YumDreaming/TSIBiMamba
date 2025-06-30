import os
import pickle
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

# -----------------------------------------------------------------------------
# 1) 按被试无泄露地划分 train/valid/test（8:1:1）
# -----------------------------------------------------------------------------
ALL_SUBJECTS = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
random.seed(42)
random.shuffle(ALL_SUBJECTS)
n = len(ALL_SUBJECTS)            # 22
n_train = int(0.8 * n)           # 17
n_val   = int(0.1 * n)           #  2
SUBJECT_SPLITS = {
    "train": ALL_SUBJECTS[:n_train],
    "valid": ALL_SUBJECTS[n_train:n_train+n_val],
    "test":  ALL_SUBJECTS[n_train+n_train+n_val:]
}

# -----------------------------------------------------------------------------
# 2) 计算训练集上的全局统计量（Robust → Min–Max）
# -----------------------------------------------------------------------------
def compute_train_stats(raw_root: str, train_subjects: List[str]):
    """
    1) 针对 train_subjects 加载原始 .dat
    2) 对每 trial 提取 EEG(32) + log-GSR → feat (8064,33)
    3) 全部垂直拼接成 (N,33)，计算 median, IQR
    4) Robust 缩放后计算 min/max
    """
    all_feats = []
    for subj in train_subjects:
        dat_path = Path(raw_root) / f"{subj}.dat"
        with open(dat_path, "rb") as f:
            data_dict = pickle.load(f, encoding="latin1")
        raw = data_dict['data']  # (40, 40, 8064)
        for trial in raw:
            eeg = trial[:32, :].T               # (8064,32)
            gsr = np.log(trial[36, :].reshape(-1,1) + 1e-6)  # log-GSR → (8064,1)
            feat = np.concatenate([eeg, gsr], axis=1)       # (8064,33)
            all_feats.append(feat)
    concat = np.vstack(all_feats)                           # (40*17*8064,33)
    median = np.median(concat, axis=0)                      # (33,)
    q75, q25 = np.percentile(concat, [75,25], axis=0)
    iqr    = q75 - q25                                      # (33,)
    robust = (concat - median) / (iqr + 1e-6)               # robust scaled
    min_   = robust.min(axis=0)                             # (33,)
    max_   = robust.max(axis=0)                             # (33,)
    return median, iqr, min_, max_

# -----------------------------------------------------------------------------
# 3) 预处理 Transform：log→robust→minmax（Offline 用）
# -----------------------------------------------------------------------------
class PreprocessTransform:
    def __init__(self, median, iqr, min_, max_):
        self.median = median
        self.iqr    = iqr
        self.min    = min_
        self.max    = max_
    def __call__(self, feat: np.ndarray) -> np.ndarray:
        # feat: (T,33)
        # 1) log-transform GSR 列 (index 32)
        feat[:,32] = np.log(feat[:,32] + 1e-6)
        # 2) robust scaling
        feat = (feat - self.median) / (self.iqr + 1e-6)
        # 3) min–max to [0,1]
        feat = (feat - self.min) / (self.max - self.min + 1e-6)
        return feat.astype(np.float32)

# -----------------------------------------------------------------------------
# 4) 离线预处理：生成或读取预处理文件
# -----------------------------------------------------------------------------
def generate_or_load_preprocessed(
    raw_root: str,
    pre_root: str,
    subjects: List[str],
    transform: PreprocessTransform
):
    """
    在 pre_root 下为每个 subject 生成两个文件：
      - {subj}_features.npy: shape (40, 8064, 33)
      - {subj}_labels.npy:   shape (40,)
    如果已存在则跳过。最终文件结构：
    /root/autodl-tmp/deap_pre/
      s01_features.npy
      s01_labels.npy
      s02_features.npy
      ...
    """
    pre_root = Path(pre_root)
    pre_root.mkdir(parents=True, exist_ok=True)
    for subj in subjects:
        feat_path = pre_root / f"{subj}_features.npy"
        lbl_path  = pre_root / f"{subj}_labels.npy"
        if feat_path.exists() and lbl_path.exists():
            print(f"[SKIP] {subj} already preprocessed")
            continue
        print(f"[PREPROCESS] {subj}")
        dat_path = Path(raw_root) / f"{subj}.dat"
        with open(dat_path, "rb") as f:
            data_dict = pickle.load(f, encoding="latin1")
        raw = data_dict['data']     # (40,40,8064)
        labels = data_dict['labels'][:,1]  # arousal col, (40,)
        bin_lbl = (labels > 5).astype(int)
        feats = []
        for trial in raw:
            eeg = trial[:32, :].T               # (8064,32)
            gsr = trial[36, :].reshape(-1,1)    # (8064,1)
            feat = np.concatenate([eeg, gsr], axis=1)  # (8064,33)
            feat = transform(feat)                    # apply preprocess
            feats.append(feat)
        feats = np.stack(feats, axis=0)  # (40,8064,33)
        np.save(feat_path, feats)
        np.save(lbl_path, bin_lbl.astype(np.int64))
        print(f"  saved → {feat_path}, {lbl_path}")

# -----------------------------------------------------------------------------
# 5) Dataset 直接读取预处理好的 .npy，并做切片增强
# -----------------------------------------------------------------------------
class DEAPDataset(data.Dataset):
    def __init__(
        self,
        pre_root: Union[str, Path],
        subjects: List[str],
        fold: str = "train",
        aug: bool = False
    ):
        """
        pre_root: 预处理后数据根目录（.npy 文件所在）
        """
        super().__init__()
        self.pre_root = Path(pre_root)
        self.subjects = subjects
        self.fold = fold
        self.aug = aug

        self.features = []  # List of np.ndarray (T,33)
        self.labels   = []  # List of int

        for subj in subjects:
            feats = np.load(self.pre_root / f"{subj}_features.npy")  # (40,8064,33)
            lbls  = np.load(self.pre_root / f"{subj}_labels.npy")    # (40,)
            for i in range(feats.shape[0]):
                feat = feats[i]  # (8064,33)
                self.features.append(feat)
                self.labels.append(int(lbls[i]))
                if self.aug and self.fold=="train":
                    L = feat.shape[0]
                    for _ in range(5):
                        sl = int(random.random() * L)
                        if sl < 4000: continue
                        st = random.randint(0, L-sl)
                        self.features.append(feat[st:st+sl])
                        self.labels.append(int(lbls[i]))

        print(f"[{fold}] total samples={len(self.labels)}, +={sum(self.labels)}, -={len(self.labels)-sum(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# -----------------------------------------------------------------------------
# 6) collate_fn & DataLoader
# -----------------------------------------------------------------------------
def deap_collate_fn(batch):
    feats, lbls = zip(*batch)
    tensors = [torch.from_numpy(f) for f in feats]
    padded = pad_sequence(tensors, batch_first=True)  # (B,T_max,33)
    mask   = (padded.sum(dim=-1) != 0).long()          # (B,T_max)
    labels = torch.tensor(lbls, dtype=torch.long)      # (B,)
    return padded, labels, mask

def get_deap_dataloader(pre_root, fold="train", batch_size=8, aug=False):
    ds = DEAPDataset(pre_root, SUBJECT_SPLITS[fold], fold, aug)
    return data.DataLoader(
        ds, batch_size=batch_size,
        shuffle=(fold=="train"),
        collate_fn=deap_collate_fn
    )

# -----------------------------------------------------------------------------
# 7) 主流程：预处理检查/生成 → 构造 loader → 打印测试信息
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    RAW_ROOT = "/root/deap/data_preprocessed_python"
    PRE_ROOT = "/root/autodl-tmp/deap_pre"

    # 7.1 计算 train 统计量
    median, iqr, min_, max_ = compute_train_stats(RAW_ROOT, SUBJECT_SPLITS["train"])
    preprocess = PreprocessTransform(median, iqr, min_, max_)

    # 7.2 离线生成或加载预处理数据
    generate_or_load_preprocessed(RAW_ROOT, PRE_ROOT, ALL_SUBJECTS, preprocess)

    # 7.3 构造并测试各 split 的 DataLoader
    for split in ("train", "valid", "test"):
        loader = get_deap_dataloader(PRE_ROOT, split, batch_size=16, aug=(split=="train"))
        print(f"{split} dataset size: {len(loader.dataset)}")
        feats, labels, mask = next(iter(loader))
        print(f"A {split} batch → feats: {feats.shape}, labels: {labels.shape}, mask: {mask.shape}")

    # -----------------------------------------------------------------------------
    # 最终生成的文件结构预览 (/root/autodl-tmp/deap_pre):
    # ├── s01_features.npy
    # ├── s01_labels.npy
    # ├── s02_features.npy
    # ├── s02_labels.npy
    # └── ... 共 22*2 个文件
    # -----------------------------------------------------------------------------

import os
import pickle
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Union
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1) 定义被试列表并按 8:1:1 划分（避免数据泄露）
# -----------------------------------------------------------------------------
ALL_SUBJECTS = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
random.seed(42)
random.shuffle(ALL_SUBJECTS)
n = len(ALL_SUBJECTS)           # 22
n_train = int(0.8 * n)          # 17
n_val   = int(0.1 * n)          #  2
SUBJECT_SPLITS = {
    "train": ALL_SUBJECTS[:n_train],
    "valid": ALL_SUBJECTS[n_train:n_train + n_val],
    "test":  ALL_SUBJECTS[n_train + n_val:]
}

# -----------------------------------------------------------------------------
# 2) 统计训练集参数：signed-log1p → median/IQR → Min–Max
# -----------------------------------------------------------------------------
def compute_train_stats(raw_root: str, subjects: List[str]):
    all_feats = []
    for subj in tqdm(subjects, desc="Compute stats (subjects)"):
        dat = pickle.load(open(Path(raw_root)/f"{subj}.dat","rb"), encoding="latin1")
        for trial in tqdm(dat['data'], desc=f"{subj}", leave=False):
            eeg = trial[:32,:].T                       # (8064,32)
            gsr = trial[36,:].reshape(-1,1)           # (8064,1)
            feat = np.concatenate([eeg, gsr], axis=1)
            # signed-log1p
            feat = np.sign(feat) * np.log1p(np.abs(feat))
            all_feats.append(feat)
    concat = np.vstack(all_feats)                   # (N,33)
    median = np.median(concat, axis=0)
    q75, q25 = np.percentile(concat, 75, axis=0), np.percentile(concat, 25, axis=0)
    iqr = q75 - q25
    robust = (concat - median) / (iqr + 1e-6)
    min_, max_ = robust.min(axis=0), robust.max(axis=0)
    return median, iqr, min_, max_

# -----------------------------------------------------------------------------
# 3) Transform：signed-log1p → robust-scale → min–max
# -----------------------------------------------------------------------------
class PreprocessTransform:
    def __init__(self, median, iqr, min_, max_):
        self.median, self.iqr = median, iqr
        self.min,    self.max = min_,   max_
    def __call__(self, feat: np.ndarray) -> np.ndarray:
        feat = np.sign(feat) * np.log1p(np.abs(feat))
        feat = (feat - self.median) / (self.iqr + 1e-6)
        feat = (feat - self.min) / (self.max - self.min + 1e-6)
        return feat.astype(np.float32)

# -----------------------------------------------------------------------------
# 4) 预处理并保存；若已存在则跳过
# -----------------------------------------------------------------------------
def preprocess_and_save(raw_root: str, pre_root: str):
    raw_root, pre_root = Path(raw_root), Path(pre_root)
    if pre_root.exists():
        print(f"{pre_root} exists; skipping preprocessing.")
        return
    pre_root.mkdir(parents=True)

    # 4.1 计算训练统计量
    median, iqr, min_, max_ = compute_train_stats(raw_root, SUBJECT_SPLITS["train"])
    np.savez(pre_root/"stats.npz",
             median=median, iqr=iqr, min_=min_, max_=max_)
    transform = PreprocessTransform(median, iqr, min_, max_)

    # 4.2 对每个 split/subject/trial 执行预处理并保存
    for split, subs in SUBJECT_SPLITS.items():
        for subj in tqdm(subs, desc=f"Preprocessing {split}"):
            subj_dir = pre_root/split/subj
            subj_dir.mkdir(parents=True, exist_ok=True)
            dat = pickle.load(open(raw_root/f"{subj}.dat","rb"), encoding="latin1")
            raw, labels = dat['data'], dat['labels'][:,1]
            bin_labels = (labels > 5).astype(int)
            # 保存 labels.csv
            with open(subj_dir/"labels.csv","w") as lf:
                lf.write("trial,label\n")
                for t, lab in enumerate(bin_labels):
                    lf.write(f"{t},{lab}\n")
            # 保存特征 .npy
            for t, trial in enumerate(tqdm(raw, desc=f"{subj}", leave=False)):
                feat = np.concatenate([trial[:32,:].T,
                                       trial[36,:].reshape(-1,1)],
                                      axis=1)
                feat = transform(feat)
                np.save(subj_dir/f"{t:02d}.npy", feat)
        print(f"Finished split '{split}'")

# -----------------------------------------------------------------------------
# 5) Dataset 读取预处理数据；train 时做随机切片增强
# -----------------------------------------------------------------------------
class DEAPDataset(data.Dataset):
    def __init__(self, pre_root: str, split: str, aug: bool=False):
        super().__init__()
        base = Path(pre_root)/split
        self.features, self.labels, self.aug = [], [], aug
        for subj_dir in sorted(base.iterdir()):
            if not subj_dir.is_dir(): continue
            # 读取 labels.csv
            lines = (subj_dir/"labels.csv").read_text().splitlines()[1:]
            lbls = {int(l.split(",")[0]): int(l.split(",")[1]) for l in lines}
            for t, lab in lbls.items():
                feat = np.load(subj_dir/f"{t:02d}.npy")
                self.features.append(feat)
                self.labels.append(lab)
                if aug and split=="train":
                    L = feat.shape[0]
                    for _ in range(5):
                        sl = int(random.random()*L)
                        if sl < 4000: continue
                        st = random.randint(0, L-sl)
                        self.features.append(feat[st:st+sl])
                        self.labels.append(lab)
        print(f"[{split}] loaded {len(self.labels)} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]

# -----------------------------------------------------------------------------
# 6) collate_fn & DataLoader (CPU tensors, move to CUDA later)
# -----------------------------------------------------------------------------
def deap_collate_fn(batch):
    feats, lbls = zip(*batch)
    ts = [torch.from_numpy(f) for f in feats]
    padded = pad_sequence(ts, batch_first=True)
    mask   = (padded.sum(-1)!=0).long()
    labels = torch.tensor(lbls, dtype=torch.long)
    return padded, labels, mask

def get_deap_loader(pre_root, split, bs, aug):
    ds = DEAPDataset(pre_root, split, aug)
    return data.DataLoader(
        ds,
        batch_size=bs,
        shuffle=(split=="train"),
        pin_memory=True,
        num_workers=min(8, os.cpu_count()),
        collate_fn=deap_collate_fn
    )

# -----------------------------------------------------------------------------
# 7) 主流程：预处理→加载→打印→展示结构
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    RAW = "/root/deap/data_preprocessed_python"
    PRE = "/root/autodl-tmp/deap_signal"

    # 7.1 预处理
    preprocess_and_save(RAW, PRE)

    # 7.2 测试各 split loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for split in ("train","valid","test"):
        loader = get_deap_loader(PRE, split, bs=8, aug=(split=="train"))
        print(f"{split}: {len(loader.dataset)} samples")
        feats, labs, mask = next(iter(loader))
        feats, labs, mask = feats.to(device), labs.to(device), mask.to(device)
        print(f"  batch → feats: {feats.shape}, labels: {labs.shape}, mask: {mask.shape}")

    # 7.3 展示目录结构
    print("\nPreprocessed folder structure:")
    for root, dirs, files in os.walk(PRE):
        level = root.replace(PRE, "").count(os.sep)
        print("    "*level + os.path.basename(root) + "/")
        for fname in files:
            print("    "*(level+1) + fname)

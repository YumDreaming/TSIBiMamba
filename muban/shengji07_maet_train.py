#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-subject × single-session training of MAET on pre-computed
SEED-VII DE features.  Outputs metrics in the exact style requested.

执行:
    python shengji07_maet_train.py            # 使用默认目录
或
    python shengji07_maet_train.py --de_dir /path/DE --mat_dir /path/mat
"""

import os, math, argparse, warnings, sys
import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support)
from tqdm import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------  MAET backbone  ------------------
from maet import MAET          # 确保 maet.py 放在 PYTHONPATH

# ------------------  常量与标签  ------------------
FS          = 200
SEG_SIZE    = 100                         # 0.5 s
SESS_VIDS   = [range(0,20), range(20,40), range(40,60), range(60,80)]
SUBJ_NUM    = 20
METRIC_KEYS = ('Acc', 'Prec', 'Rec', 'F1')

emotion2idx = {'Disgust':0,'Fear':1,'Sad':2,'Neutral':3,
               'Happy':4,'Anger':5,'Surprise':6}
labels_text = [
    'Happy','Neutral','Disgust','Sad','Anger','Anger','Sad','Disgust','Neutral','Happy',
    'Happy','Neutral','Disgust','Sad','Anger','Anger','Sad','Disgust','Neutral','Happy',
    'Anger','Sad','Fear','Neutral','Surprise','Surprise','Neutral','Fear','Sad','Anger',
    'Anger','Sad','Fear','Neutral','Surprise','Surprise','Neutral','Fear','Sad','Anger',
    'Happy','Surprise','Disgust','Fear','Anger','Anger','Fear','Disgust','Surprise','Happy',
    'Happy','Surprise','Disgust','Fear','Anger','Anger','Fear','Disgust','Surprise','Happy',
    'Disgust','Sad','Fear','Surprise','Happy','Happy','Surprise','Fear','Sad','Disgust',
    'Disgust','Sad','Fear','Surprise','Happy','Happy','Surprise','Fear','Sad','Disgust',
]
video_labels = np.array([emotion2idx[e] for e in labels_text], dtype=np.int64)


# ================= 1. 还原 subject-session 索引 =================
def build_index(mat_dir):
    """
    返回 subject_indices[subject_id][session_id] = list(dataset_idx)
    dataset_idx 对应 X.npy/y.npy 的行号
    """
    subj_map = [[[] for _ in range(4)] for _ in range(SUBJ_NUM)]
    global_idx = 0

    mats = sorted([f for f in os.listdir(mat_dir) if f.endswith('.mat')])[:SUBJ_NUM]
    assert len(mats) == SUBJ_NUM, "未找到 20 个 .mat 文件"

    for sid, fn in enumerate(mats, 0):            # 0~19
        mat = sio.loadmat(os.path.join(mat_dir, fn))
        trials = sorted([k for k,v in mat.items()
                         if isinstance(v,np.ndarray) and v.ndim==2 and v.shape[0]==62])
        if len(trials)!=80:
            raise RuntimeError(f"{fn} trial 数 ≠ 80")
        for vid, tk in enumerate(trials):
            segs = mat[tk].shape[1] // SEG_SIZE
            idx_range = range(global_idx, global_idx+segs)
            global_idx += segs
            # 放入对应 session
            for sess_id, v_range in enumerate(SESS_VIDS):
                if vid in v_range:
                    subj_map[sid][sess_id].extend(idx_range)
                    break
    return subj_map


# ================= 2. 数据集封装 =================
class DEFeatureDataset(Dataset):
    def __init__(self, X, y, indices):
        self.x = torch.from_numpy(X[indices].reshape(-1, 62*5)).float()
        self.y = torch.from_numpy(y[indices]).long()
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return self.x[i], self.y[i]


# ================= 3. 评估函数 =================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); preds, trues = [], []
    for x,y in loader:
        logits = model(eeg=x.to(device), eye=None)
        preds.append(logits.argmax(1).cpu()); trues.append(y)
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(trues).numpy()

    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1


# ================= 4. 主流程 =================
def main(args):
    # ---------- 加载 X/y ----------
    X = np.load(os.path.join(args.de_dir, "X.npy"))
    y = np.load(os.path.join(args.de_dir, "y.npy"))
    assert X.ndim==3 and X.shape[1:]==(62,5), "X.npy 维度应为 (N,62,5)"

    subj_idx = build_index(args.mat_dir)

    # ---------- 结果容器 ----------
    subj_metrics = np.full((SUBJ_NUM, 4, 4), np.nan)   # subj × sess × metric
    sess_metrics = {k:([] for _ in range(4)) for k in METRIC_KEYS}  # later填

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # ---------- 训练循环 ----------
    subj_bar = tqdm(range(SUBJ_NUM), desc="被试")
    for sid in subj_bar:            # 0~19
        for sess in range(4):
            idx = subj_idx[sid][sess]
            if len(idx)<10:         # 过少则跳过
                continue
            X_sess, y_sess = X[idx], y[idx]

            # 9:1 split
            sss = StratifiedShuffleSplit(1, test_size=0.1,
                                         random_state=sid*4+sess)
            train_idx, test_idx = next(sss.split(X_sess.reshape(len(X_sess), -1),
                                                 y_sess))

            train_set = DEFeatureDataset(X_sess, y_sess, train_idx)
            test_set  = DEFeatureDataset(X_sess, y_sess, test_idx)
            train_loader = DataLoader(train_set, args.batch, shuffle=True)
            test_loader  = DataLoader(test_set,  args.batch, shuffle=False)

            model = MAET(eeg_dim=310, eye_dim=33, num_classes=7,
                         embed_dim=32, depth=3, eeg_seq_len=5,
                         num_heads=4, prompt=False).to(device)
            optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

            for ep in tqdm(range(args.epochs),
                           desc=f"S{sid+1:02d}-Sess{sess+1}",
                           leave=False):
                model.train()
                for xb,yb in DataLoader(train_set, args.batch, shuffle=True):
                    xb,yb = xb.to(device), yb.to(device)
                    optim.zero_grad()
                    loss = criterion(model(eeg=xb, eye=None), yb)
                    loss.backward(); optim.step()

            acc, prec, rec, f1 = evaluate(model, test_loader, device)
            subj_metrics[sid, sess] = (acc, prec, rec, f1)

        # ----- 打印当前被试结果 -----
        lines = [f"===== 被试 {sid+1:2d}/{SUBJ_NUM} ====="]
        for s in range(4):
            a,p,r,f = subj_metrics[sid,s]
            if not np.isnan(a):
                lines.append(f"  会话{s+1} —— "
                             f"Acc: {a*100:5.2f}%  Prec: {p*100:5.2f}%  "
                             f"Rec: {r*100:5.2f}%  F1: {f*100:5.2f}%")
        tqdm.write("\n".join(lines))

    # ---------- 统计 session 汇总 ----------
    print("\n=== 各会话汇总（均值 ± 标准差） ===\n")
    for m_i, m_name in enumerate(METRIC_KEYS):
        print(f"{m_name}:")
        for s in range(4):
            vals = subj_metrics[:,s,m_i]
            vals = vals[~np.isnan(vals)]
            print(f"  会话{s+1}: {np.mean(vals)*100:5.2f}% ± {np.std(vals)*100:5.2f}%")
        print()

    # ---------- 打印大表 ----------
    header = ["Subject"]
    for s in range(4):
        header += [f"S{s+1}_{k}" for k in METRIC_KEYS]
    print("\t".join(header))

    for sid in range(SUBJ_NUM):
        row = [f"{sid+1}"]
        for s in range(4):
            if np.isnan(subj_metrics[sid,s,0]):
                row += ["--"]*4
            else:
                for m in range(4):
                    row.append(f"{subj_metrics[sid,s,m]*100:5.2f}%")
        print("\t".join(row))


# ================= 5. 入口 =================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--de_dir",  default="/root/autodl-tmp/DE",
                        help="包含 X.npy / y.npy 的目录")
    parser.add_argument("--mat_dir", default="/root/autodl-tmp/EEG_preprocessed",
                        help="原始 .mat 文件目录，用于恢复索引")
    parser.add_argument("--epochs",  type=int, default=20)
    parser.add_argument("--batch",   type=int, default=128,
                        help="batch_size")
    parser.add_argument("--lr",      type=float, default=8e-5)
    parser.add_argument("--device",  default="cuda",
                        help="cuda / cuda:0 / cpu")
    args = parser.parse_args()
    main(args)

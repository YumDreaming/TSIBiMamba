#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEED-VII 单分支 PSD 数据训练脚本（单被试内 9:1 划分，含学习率调度与 per-class 准确率输出）
仅加载 PSD 分支，单分支 Mamba 处理（bidirectional=False）。
"""

import os
import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from mamba_ssm import Mamba
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

# ——— 全局占位（将在 main() 中被覆盖） ———
NUM_CLASSES = 7
IMG_ROWS    = 8
IMG_COLS    = 9
NUM_CHAN    = 4   # SEED-VII 总通道数 C=5，去掉第一通道后为 4
SEQ_LEN     = 6
EPOCHS      = 100
BATCH_SIZE  = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42

# -------------------- 基础模块 --------------------

class BaseCNN(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 64,  kernel_size=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc    = nn.Linear(64 * (IMG_ROWS//2) * (IMG_COLS//2), 512)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return torch.relu(self.fc(x))


class EnResBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.net   = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop)
        self.skip  = (nn.Conv1d(input_size, output_size, 1, bias=False)
                      if input_size != output_size else None)
        nn.init.xavier_uniform_(self.conv1.weight)

    def forward(self, x):
        out = self.net(x)
        res = self.skip(x) if self.skip is not None else x
        return out + res


class EnBiMambaLayer(nn.Module):
    def __init__(self, d_model, dropout, mamba_config):
        super().__init__()
        cfg = copy.deepcopy(mamba_config)
        cfg.pop('bidirectional', None)
        self.mamba = Mamba(d_model=d_model, **cfg)
        self.norm  = nn.LayerNorm(d_model, eps=1e-6)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, inference_params=None):
        out = self.mamba(x, inference_params)
        return x + self.norm(out)


class EnSSM(nn.Module):
    def __init__(self, input_size, output_sizes, dropout, mamba_config):
        super().__init__()
        self.res_layers   = nn.ModuleList()
        self.mamba_layers = nn.ModuleList()
        for i, out_sz in enumerate(output_sizes):
            in_sz = input_size if i == 0 else output_sizes[i-1]
            self.res_layers.append(EnResBlock(in_sz, out_sz, dropout))
            self.mamba_layers.append(
                EnBiMambaLayer(d_model=out_sz,
                               dropout=dropout,
                               mamba_config=mamba_config)
            )

    def forward(self, x, inference_params=None):
        out = x
        for res, mmb in zip(self.res_layers, self.mamba_layers):
            tmp = res(out.permute(0,2,1))
            out = tmp.permute(0,2,1)
            out = mmb(out, inference_params)
        return out


class MambaVision(nn.Module):
    """单分支 PSD → CNN → EnSSM(Mamba) → 分类"""
    def __init__(self, mm_input_size, mm_output_sizes, dropout, causal, mamba_config):
        super().__init__()
        self.base_cnn   = BaseCNN(in_chan=NUM_CHAN)
        self.encoder    = EnSSM(input_size=mm_input_size,
                                output_sizes=mm_output_sizes,
                                dropout=dropout,
                                mamba_config=mamba_config)
        self.classifier = nn.Linear(mm_output_sizes[-1], NUM_CLASSES)

    def forward(self, r, inference_params=None):
        # r: (B, T, C, H, W)
        B, T, C, H, W = r.shape
        x = r.view(B*T, C, H, W)
        feat = self.base_cnn(x)
        feat = feat.view(B, T, -1)
        enc  = self.encoder(feat, inference_params)
        last = enc[:, -1, :]
        return self.classifier(last)


# -------------------- 数据加载与预处理 --------------------

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_branch_npy(branch_dir: str):
    x_path = os.path.join(branch_dir, "t6x_89.npy")
    y_path = os.path.join(branch_dir, "t6y_89.npy")
    assert os.path.isfile(x_path), x_path
    assert os.path.isfile(y_path), y_path
    X = np.load(x_path)
    y = np.load(y_path)
    X = X.transpose(0,1,2,5,3,4)[..., 1:, :, :].copy()
    return X, y

def prepare_psd_data(PSD_dir: str, T: int):
    X_psd, y_psd = load_branch_npy(PSD_dir)
    num_subj, NS = X_psd.shape[0], X_psd.shape[1]
    flat = X_psd.reshape((-1, T, X_psd.shape[3], IMG_ROWS, IMG_COLS))
    groups = np.repeat(np.arange(num_subj), NS)
    return torch.from_numpy(flat).float(), torch.from_numpy(y_psd).long(), groups


# -------------------- 主训练流程 --------------------

def main():
    cfg = load_config("only_PSD.yaml")

    # 强制类型转换
    epochs        = int(cfg["epochs"])
    batch_size    = int(cfg["batch_size"])
    learning_rate = float(cfg["learning_rate"])
    lr_sched      = cfg["lr_scheduler"].lower()
    seed          = int(cfg.get("seed", 42))

    # 覆盖全局
    global SEQ_LEN, NUM_CHAN, IMG_ROWS, IMG_COLS, NUM_CLASSES
    SEQ_LEN     = int(cfg["input"]["T"])
    NUM_CHAN    = int(cfg["input"]["C"]) - 1
    IMG_ROWS    = int(cfg["input"]["H"])
    IMG_COLS    = int(cfg["input"]["W"])
    NUM_CLASSES = int(cfg["num_classes"])
    device      = torch.device(cfg["device"][0] if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Mamba 配置
    mm_cfg       = cfg["mmmamba"]
    mm_input     = int(mm_cfg["mm_input_size"])
    mm_outputs   = [int(x) for x in mm_cfg["mm_output_sizes"]]
    dropout      = float(mm_cfg["dropout"])
    causal       = bool(mm_cfg["causal"])
    mamba_config = copy.deepcopy(mm_cfg["mamba_config"])

    # 加载 PSD 数据
    PSD_dir = os.path.join(cfg["data_dir"], "PSD")
    print("加载 PSD 分支数据…")
    all_r, all_y, groups = prepare_psd_data(PSD_dir, SEQ_LEN)
    N_total  = all_y.shape[0]
    num_subj = int(groups.max()) + 1
    NS       = N_total // num_subj
    print(f"{num_subj} 位被试，每位 {NS} 样本，总 {N_total}")

    subj_accs, all_true, all_pred = [], [], []

    for subj in range(num_subj):
        print(f"\n=== 被试 {subj+1}/{num_subj} ===")
        idxs   = np.where(groups==subj)[0]
        R_sub  = all_r[idxs]
        y_sub  = all_y[idxs].numpy()

        tr_idx, te_idx = train_test_split(
            np.arange(NS), test_size=0.1,
            stratify=y_sub, random_state=seed
        )
        R_tr, R_te = R_sub[tr_idx], R_sub[te_idx]
        y_tr = torch.from_numpy(y_sub[tr_idx]).long()
        y_te = torch.from_numpy(y_sub[te_idx]).long()

        train_loader = DataLoader(
            TensorDataset(R_tr, y_tr),
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        test_loader  = DataLoader(
            TensorDataset(R_te, y_te),
            batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )

        model = MambaVision(
            mm_input_size   = mm_input,
            mm_output_sizes = mm_outputs,
            dropout         = dropout,
            causal          = causal,
            mamba_config    = mamba_config
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                     if lr_sched=="cos" else None)
        criterion = nn.CrossEntropyLoss()

        # 训练
        model.train()
        for epoch in range(1, epochs+1):
            loop = tqdm(train_loader, desc=f"E{epoch}/{epochs}", leave=False)
            for r_b, y_b in loop:
                r_b, y_b = r_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(r_b), y_b)
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())
            if scheduler: scheduler.step()

        # 测试 & 混淆矩阵
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for r_b, y_b in test_loader:
                r_b = r_b.to(device)
                preds = model(r_b).argmax(dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(y_b.numpy())

        acc = np.mean(np.array(y_pred)==np.array(y_true))*100
        print(f"被试{subj+1} 准确率: {acc:.2f}%")
        subj_accs.append(acc)
        all_true.extend(y_true)
        all_pred.extend(y_pred)

        # 保存
        sd = os.path.join(cfg["save_dir"], f"subject_{subj+1}")
        ensure_dir(sd)
        torch.save(model.state_dict(), os.path.join(sd, "model.pth"))

    # 汇总
    mean_acc, std_acc = np.mean(subj_accs), np.std(subj_accs)
    print(f"\n平均准确率: {mean_acc:.2f}%  Std: {std_acc:.2f}%")

    cm = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:,None] * 100

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm_norm, cmap=plt.cm.Blues)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Percentage (%)', rotation=270, labelpad=15)
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels([str(i) for i in range(NUM_CLASSES)], rotation=45, ha="right")
    ax.set_yticklabels([str(i) for i in range(NUM_CLASSES)])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Normalized Confusion Matrix (%)')
    thresh = cm_norm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            col = "white" if cm_norm[i,j] > thresh else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center", color=col)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["save_dir"], "confusion_matrix_psd.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
SEED-VII 数据集 7 分类训练脚本——BiTransformer 版本
将 CoBiMambaLayer 与 EnBiMambaLayer 均替换为双向 Transformer 层，
便于与 Mamba 模块性能对比。
"""

import os
import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from torch.nn import TransformerEncoderLayer

# —— Transformer 超参（可根据需要调整） ——
TRANS_NHEAD       = 8
TRANS_DIM_FEEDFWD = 2048
TRANS_DROPOUT     = 0.1

# —— 全局变量（main 中会动态覆盖） ——
NUM_CLASSES = 7
BATCH_SIZE  = 128
IMG_ROWS    = 8
IMG_COLS    = 9
NUM_CHAN    = 4
SEQ_LEN     = 6
EPOCHS      = 100
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42

# -------------------- 模型定义 --------------------

class BaseCNN(nn.Module):
    def __init__(self, in_chan=NUM_CHAN):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc    = nn.Linear(64 * (IMG_ROWS//2) * (IMG_COLS//2), 512)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return x

class ResBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2, dilation=1):
        super().__init__()
        self.e_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.e_bn   = nn.BatchNorm1d(output_size)
        self.g_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.g_bn   = nn.BatchNorm1d(output_size)
        self.relu   = nn.ReLU()
        self.e_drop = nn.Dropout(dropout)
        self.g_drop = nn.Dropout(dropout)
        self.e_net  = nn.Sequential(self.e_conv, self.e_bn, self.relu, self.e_drop)
        self.g_net  = nn.Sequential(self.g_conv, self.g_bn, self.relu, self.g_drop)

        if input_size != output_size:
            self.e_skipconv = nn.Conv1d(input_size, output_size, 1, bias=False)
            self.g_skipconv = nn.Conv1d(input_size, output_size, 1, bias=False)
        else:
            self.e_skipconv = None
            self.g_skipconv = None

        nn.init.xavier_uniform_(self.e_conv.weight)
        nn.init.xavier_uniform_(self.g_conv.weight)

    def forward(self, g, r):
        g_out = self.e_net(g)
        r_out = self.g_net(r)
        g_res = g if self.e_skipconv is None else self.e_skipconv(g)
        r_res = r if self.g_skipconv is None else self.g_skipconv(r)
        return g_out + g_res, r_out + r_res

class CoBiTransformerLayer(nn.Module):
    """替换 CoBiMambaLayer 的双分支 BiTransformer 层."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.trans_g = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.trans_r = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop  = nn.Dropout(dropout)

    def forward(self, g, r, *_):
        # 双向注意力：is_causal=False
        g2 = self.trans_g(g.transpose(0,1), is_causal=False).transpose(0,1)
        r2 = self.trans_r(r.transpose(0,1), is_causal=False).transpose(0,1)
        g_out = g + self.drop(self.norm1(g2))
        r_out = r + self.drop(self.norm2(r2))
        return g_out, r_out

class CoSSM(nn.Module):
    def __init__(self, input_size, output_sizes, dropout=0.2):
        super().__init__()
        res_list   = []
        tr_list    = []
        for i, out_sz in enumerate(output_sizes):
            in_sz = input_size if i==0 else output_sizes[i-1]
            res_list.append(ResBlock(in_sz, out_sz, dropout=dropout))
            tr_list.append(CoBiTransformerLayer(
                d_model=out_sz,
                nhead=TRANS_NHEAD,
                dim_feedforward=TRANS_DIM_FEEDFWD,
                dropout=dropout
            ))
        self.res_layers   = nn.ModuleList(res_list)
        self.trans_layers = nn.ModuleList(tr_list)

    def forward(self, g_x, r_x, *args):
        g_out, r_out = g_x, r_x
        for res, tr in zip(self.res_layers, self.trans_layers):
            g_tmp, r_tmp = res(g_out.permute(0,2,1), r_out.permute(0,2,1))
            g_out = g_tmp.permute(0,2,1)
            r_out = r_tmp.permute(0,2,1)
            g_out, r_out = tr(g_out, r_out)
        return g_out, r_out

class EnResBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.net   = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop)
        self.conv  = None if input_size==output_size else nn.Conv1d(input_size, output_size, 1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.conv is None else self.conv(x)
        return out + res

class EnBiTransformerLayer(nn.Module):
    """替换 EnBiMambaLayer 的单分支 BiTransformer 层."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.trans  = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.norm1  = nn.LayerNorm(d_model, eps=1e-6)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x, *_):
        # 双向注意力：is_causal=False
        x2 = self.trans(x.transpose(0,1), is_causal=False).transpose(0,1)
        return x + self.drop(self.norm1(x2))

class EnSSM(nn.Module):
    def __init__(self, input_size, output_sizes, dropout=0.2):
        super().__init__()
        res_list = []
        tr_list  = []
        for i, out_sz in enumerate(output_sizes):
            in_sz = input_size if i==0 else output_sizes[i-1]
            res_list.append(EnResBlock(in_sz, out_sz, dropout=dropout))
            tr_list.append(EnBiTransformerLayer(
                d_model=out_sz,
                nhead=TRANS_NHEAD,
                dim_feedforward=TRANS_DIM_FEEDFWD,
                dropout=dropout
            ))
        self.res_layers   = nn.ModuleList(res_list)
        self.trans_layers = nn.ModuleList(tr_list)

    def forward(self, x, *args):
        out = x
        for res, tr in zip(self.res_layers, self.trans_layers):
            tmp = res(out.permute(0,2,1))
            out = tmp.permute(0,2,1)
            out = tr(out)
        return out

class BiTransformerVision(nn.Module):
    def __init__(self, mm_input_size=512, mm_output_sizes=None, dropout=0.1):
        super().__init__()
        if mm_output_sizes is None:
            mm_output_sizes = [mm_input_size]
        self.base_cnn_g    = BaseCNN()
        self.base_cnn_r    = BaseCNN()
        self.cossm_encoder = CoSSM(mm_input_size, mm_output_sizes, dropout=dropout)
        self.enssm_encoder = EnSSM(mm_output_sizes[-1]*2, [mm_output_sizes[-1]*2], dropout=dropout)
        self.classifier    = nn.Linear(mm_output_sizes[-1]*2, NUM_CLASSES)

    def forward(self, g, r, *args):
        B, T, C, H, W = g.shape
        g_out = g.view(B*T, C, H, W)
        r_out = r.view(B*T, C, H, W)
        g_cnn = self.base_cnn_g(g_out)
        r_cnn = self.base_cnn_r(r_out)
        g_feat = g_cnn.view(B, T, -1)
        r_feat = r_cnn.view(B, T, -1)
        g_cossm, r_cossm = self.cossm_encoder(g_feat, r_feat)
        x = torch.cat([g_cossm, r_cossm], dim=-1)
        x = self.enssm_encoder(x)
        last = x[:, -1, :]
        return self.classifier(last)

# -------------------- 数据加载与预处理 --------------------

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_branch_npy(branch_dir: str):
    x_path = os.path.join(branch_dir, "t6x_89.npy")
    y_path = os.path.join(branch_dir, "t6y_89.npy")
    assert os.path.isfile(x_path), f"找不到 {x_path}"
    assert os.path.isfile(y_path), f"找不到 {y_path}"
    X = np.load(x_path)  # (20,4666,6,8,9,5)
    y = np.load(y_path)  # (93320,)
    assert X.ndim==6 and X.shape[2]==SEQ_LEN and X.shape[3]==IMG_ROWS and X.shape[4]==IMG_COLS and X.shape[5]==5
    assert y.shape[0]==X.shape[0]*X.shape[1]
    # (20,4666,6,8,9,5) -> (20,4666,6,5,8,9)
    X = X.transpose(0,1,2,5,3,4).copy()
    # 选后 4 通道 -> (20,4666,6,4,8,9)
    X = X[:,:,:,1:,:,:].copy()
    return X, y

def prepare_all_data(DE_dir: str, PSD_dir: str, T: int):
    X_de, y_de   = load_branch_npy(DE_dir)
    X_psd, y_psd = load_branch_npy(PSD_dir)
    assert np.array_equal(y_de, y_psd), "标签不一致"
    y_all = y_de
    num_subj, NS = X_de.shape[0], X_de.shape[1]
    de_flat  = X_de.reshape((-1, T, X_de.shape[3], IMG_ROWS, IMG_COLS))
    psd_flat = X_psd.reshape((-1, T, X_psd.shape[3], IMG_ROWS, IMG_COLS))
    groups   = np.repeat(np.arange(num_subj), NS)
    return (torch.from_numpy(de_flat).float(),
            torch.from_numpy(psd_flat).float(),
            torch.from_numpy(y_all).long(),
            groups)

# -------------------- 主训练流程 --------------------

def main():
    # 1. 载入配置
    cfg       = load_config("config_VII_7.yaml")
    data_dir  = cfg["data_dir"].rstrip("/\\")
    save_dir  = cfg["save_dir"].rstrip("/\\")
    ensure_dir(save_dir)
    EPOCHS     = int(cfg.get("epochs", 100))
    BATCH_SIZE = int(cfg.get("batch_size", 128))
    LR         = float(cfg.get("learning_rate", 8e-5))
    sched_t    = cfg.get("lr_scheduler", "none").lower()
    use_wandb  = bool(cfg.get("if_wandb", False))
    DEVICE     = torch.device(cfg["device"][0] if torch.cuda.is_available() else "cpu")
    SEED       = int(cfg.get("seed", 42))
    torch.manual_seed(SEED); np.random.seed(SEED)

    # 动态覆盖全局常量
    global SEQ_LEN, NUM_CHAN, IMG_ROWS, IMG_COLS, NUM_CLASSES
    SEQ_LEN     = int(cfg["input"]["T"])
    NUM_CHAN    = int(cfg["input"]["C"]) - 1
    IMG_ROWS    = int(cfg["input"]["H"])
    IMG_COLS    = int(cfg["input"]["W"])
    NUM_CLASSES = int(cfg.get("num_classes", 7))

    # 额外导入
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import itertools

    # 加载数据
    DE_dir, PSD_dir = os.path.join(data_dir, "DE"), os.path.join(data_dir, "PSD")
    all_g, all_r, all_y, groups = prepare_all_data(DE_dir, PSD_dir, SEQ_LEN)

    num_subj = int(groups.max()) + 1
    subj_accs = []

    for subj in range(num_subj):
        # 按被试划分
        idxs   = np.where(groups == subj)[0]
        Xg_sub = all_g[idxs]
        Xr_sub = all_r[idxs]
        y_sub  = all_y[idxs].numpy()

        # 9:1 划分训练/测试
        g_tr, g_te, r_tr, r_te, y_tr_np, y_te_np = train_test_split(
            Xg_sub, Xr_sub, y_sub,
            test_size=0.1,
            stratify=y_sub,
            random_state=SEED
        )
        y_tr = torch.from_numpy(y_tr_np).long()
        y_te = torch.from_numpy(y_te_np).long()

        train_loader = DataLoader(
            TensorDataset(g_tr, r_tr, y_tr),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            TensorDataset(g_te, r_te, y_te),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True
        )

        # 模型、优化器、损失
        model = BiTransformerVision(
            mm_input_size   = cfg["mmmamba"]["mm_input_size"],
            mm_output_sizes = cfg["mmmamba"]["mm_output_sizes"],
            dropout         = cfg["mmmamba"]["dropout"]
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
                     if sched_t == "cos" else None)
        criterion = nn.CrossEntropyLoss()

        # 训练
        model.train()
        for epoch in range(1, EPOCHS+1):
            loop = tqdm(train_loader,
                        desc=f"Subj{subj+1} Ep{epoch}/{EPOCHS}",
                        leave=False)
            for g_b, r_b, y_b in loop:
                g_b, r_b, y_b = g_b.to(DEVICE), r_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                logits = model(g_b, r_b)
                loss   = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())
            if scheduler:
                scheduler.step()

        # 测试并收集预测
        model.eval()
        all_preds = []
        all_true  = []
        with torch.no_grad():
            for g_b, r_b, y_b in test_loader:
                g_b, r_b, y_b = g_b.to(DEVICE), r_b.to(DEVICE), y_b.to(DEVICE)
                preds = model(g_b, r_b).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_true.extend(y_b.cpu().numpy().tolist())

        # 计算并保存混淆矩阵
        cm = confusion_matrix(all_true, all_preds)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Subject {subj+1} Confusion Matrix")
        plt.colorbar()
        ticks = np.arange(NUM_CLASSES)
        plt.xticks(ticks, [str(i) for i in ticks], rotation=45)
        plt.yticks(ticks, [str(i) for i in ticks])
        thresh = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()

        sub_dir = os.path.join(save_dir, f"subject_{subj+1}")
        ensure_dir(sub_dir)
        plt.savefig(os.path.join(sub_dir, "BiTransformer_confusion.png"))
        plt.close()

        # 计算并打印准确率
        overall_acc = np.mean(np.array(all_preds) == np.array(all_true)) * 100
        print(f"被试{subj+1} 整体准确率: {overall_acc:.2f}%")
        subj_accs.append(overall_acc)

        # 保存模型
        torch.save(model.state_dict(),
                   os.path.join(sub_dir, "BiTransformer.pth"))

    # 所有被试结果汇总
    overall_mean = np.mean(subj_accs)
    overall_std  = np.std(subj_accs)
    print(f"\n=== 所有被试 9:1 平均准确率: {overall_mean:.2f}%  Std: {overall_std:.2f}% ===")


if __name__ == '__main__':
    main()

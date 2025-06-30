# -*- coding: utf-8 -*-
"""
SEED-VII 数据集 7 分类训练脚本（单被试内 5 折交叉验证，含学习率调度与 per-class 准确率输出）
--------------------------------------------------------------------------------
此脚本基于 DEAP 二分类训练代码改写，用于 SEED-VII 数据集的七分类任务。
每个被试单独训练和测试，使用 StratifiedKFold(5) 进行被试内 5 折交叉验证。
在训练过程中可选 CosineAnnealingLR 学习率调度；测试时输出七个类别的单独准确率。
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

# —— 引入模型定义（与 DEAP 时相同）——
from models.selective_scan_interface import mamba_inner_fn_no_out_proj, selective_scan_fn
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm import Mamba
from mm_bimamba import Mamba as MMBiMamba
from bimamba import Mamba as BiMamba

# ——— 全局变量（会在 main() 中根据配置覆盖） ———
NUM_CLASSES = 7
BATCH_SIZE  = 128
IMG_ROWS    = 8
IMG_COLS    = 9
NUM_CHAN    = 4    # SEED-VII 选择后 4 个通道
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
        x = torch.relu(self.conv1(x))   # (B,64,8,9)
        x = torch.relu(self.conv2(x))   # (B,128,8,9)
        x = torch.relu(self.conv3(x))   # (B,256,8,9)
        x = torch.relu(self.conv4(x))   # (B,64,8,9)
        x = self.pool(x)                # (B,64,4,4)
        x = x.view(x.size(0), -1)       # (B,1024)
        x = torch.relu(self.fc(x))      # (B,512)
        return x

class ResBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2, dilation=1):
        super().__init__()
        self.e_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.e_bn   = nn.BatchNorm1d(output_size)
        self.g_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.g_bn   = nn.BatchNorm1d(output_size)
        self.relu   = nn.SiLU()
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

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.e_conv.weight.data)
        nn.init.xavier_uniform_(self.g_conv.weight.data)

    def forward(self, g, r):
        # g, r: (batch, feat_dim, seq_len)
        g_out = self.e_net(g)
        r_out = self.g_net(r)
        g_res = g
        r_res = r
        if self.e_skipconv is not None:
            g_res = self.e_skipconv(g_res)
        if self.g_skipconv is not None:
            r_res = self.g_skipconv(r_res)
        g_out = g_out + g_res
        r_out = r_out + r_res
        return g_out, r_out  # (batch, output_size, seq_len)

class CoBiMambaLayer(nn.Module):
    def __init__(self, d_model, dropout=0.0, activation='Swish', causal=False, mamba_config=None):
        super().__init__()
        if activation == 'Swish':
            activation = 'Swish'
        elif activation == "GELU":
            activation = torch.nn.GELU

        bidirectional = mamba_config.pop('bidirectional')
        self.bidirectional = bidirectional

        if causal or (not bidirectional):
            self.mamna_g = Mamba(d_model=d_model, **mamba_config)
            self.mamna_r = Mamba(d_model=d_model, **mamba_config)
        else:
            self.mamna = MMBiMamba(d_model=d_model, bimamba_type='v2', **mamba_config)
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop  = nn.Dropout(dropout)

    def forward(self, g, r, g_inference_params=None, r_inference_params=None):
        # g, r: (batch, seq_len, d_model)
        if self.bidirectional:
            g_out1, r_out1 = self.mamna(g, r, g_inference_params, r_inference_params)
        else:
            g_out1 = self.mamna_g(g)
            r_out1 = self.mamna_r(r)
        g_out = g + self.norm1(g_out1)
        r_out = r + self.norm2(r_out1)
        return g_out, r_out

class CoSSM(nn.Module):
    def __init__(self, input_size, output_sizes=[1024], dropout=0.2, activation='GELU', causal=False, mamba_config=None):
        super().__init__()
        res_list   = []
        mamba_list = []
        for i in range(len(output_sizes)):
            in_sz  = input_size if i < 1 else output_sizes[i - 1]
            out_sz = output_sizes[i]
            res_list.append(ResBlock(input_size=in_sz, output_size=out_sz, dropout=dropout))
            mamba_list.append(CoBiMambaLayer(d_model=out_sz, dropout=dropout,
                                             activation=activation, causal=causal, mamba_config=mamba_config))
        self.res_layers   = nn.ModuleList(res_list)
        self.mamba_layers = nn.ModuleList(mamba_list)

    def forward(self, g_x, r_x, g_inference_params=None, r_inference_params=None):
        # g_x, r_x: (batch, seq_len, feat_dim)
        g_out = g_x
        r_out = r_x
        for res_layer, mamba_layer in zip(self.res_layers, self.mamba_layers):
            g_tmp, r_tmp = res_layer(g_out.permute(0,2,1), r_out.permute(0,2,1))
            g_out = g_tmp.permute(0,2,1)
            r_out = r_tmp.permute(0,2,1)
            g_out, r_out = mamba_layer(g_out, r_out, g_inference_params, r_inference_params)
        return g_out, r_out  # (batch, seq_len, last_out_sz)

class EnResBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2, activation='GELU', dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.net   = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop)
        if input_size != output_size:
            self.conv = nn.Conv1d(input_size, output_size, 1, bias=False)
        else:
            self.conv = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data)

    def forward(self, x):
        # x: (batch, feat_dim, seq_len)
        out = self.net(x)
        if self.conv is not None:
            x = self.conv(x)
        out = out + x
        return out  # (batch, output_size, seq_len)

class EnBiMambaLayer(nn.Module):
    def __init__(self, d_model, dropout=0.0, activation='Swish', causal=False, mamba_config=None):
        super().__init__()
        if activation == 'Swish':
            activation = 'Swish'
        elif activation == "GELU":
            activation = torch.nn.GELU

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamna = Mamba(d_model=d_model, **mamba_config)
        else:
            self.mamna = BiMamba(d_model=d_model, bimamba_type='v2', **mamba_config)
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, inference_params=None):
        # x: (batch, seq_len, feat_dim)
        out = x + self.norm1(self.mamna(x, inference_params))
        return out  # (batch, seq_len, d_model)

class EnSSM(nn.Module):
    def __init__(self, input_size, output_sizes=[1024], activation='GELU', dropout=0.2, causal=False, mamba_config=None):
        super().__init__()
        res_list   = []
        mamba_list = []
        for i in range(len(output_sizes)):
            in_sz  = input_size if i < 1 else output_sizes[i - 1]
            out_sz = output_sizes[i]
            res_list.append(EnResBlock(input_size=in_sz, output_size=out_sz, dropout=dropout))
            mamba_list.append(EnBiMambaLayer(d_model=out_sz, dropout=dropout,
                                             activation=activation, causal=causal, mamba_config=mamba_config))
        self.res_layers   = nn.ModuleList(res_list)
        self.mamba_layers = nn.ModuleList(mamba_list)

    def forward(self, x, inference_params=None):
        # x: (batch, seq_len, feat_dim)
        out = x
        for res_layer, mamba_layer in zip(self.res_layers, self.mamba_layers):
            tmp = res_layer(out.permute(0,2,1))
            out = tmp.permute(0,2,1)  # (batch, seq_len, out_sz)
            out = mamba_layer(out, inference_params=inference_params)
        return out  # (batch, seq_len, last_out_sz)

class BiMambaVision(nn.Module):
    def __init__(self, mm_input_size=512, mm_output_sizes=[512],
                 dropout=0.1, activation='GELU', causal=False, mamba_config=None):
        super().__init__()
        self.base_cnn_g = BaseCNN()
        self.base_cnn_r = BaseCNN()
        self.cossm_encoder = CoSSM(mm_input_size, mm_output_sizes,
                                   dropout=dropout, activation=activation,
                                   causal=causal, mamba_config=mamba_config)
        self.enssm_encoder = EnSSM(mm_output_sizes[-1] * 2,
                                   [mm_output_sizes[-1] * 2],
                                   activation=activation, dropout=dropout,
                                   causal=causal, mamba_config=mamba_config)
        self.classifier = nn.Linear(mm_output_sizes[-1] * 2, NUM_CLASSES)

    def forward(self, g, r, g_inference_params=None, r_inference_params=None):
        # g, r: (B, T=6, C=4, H=8, W=9)
        B, T, C, H, W = g.shape
        g_out = g.view(B*T, C, H, W)
        r_out = r.view(B*T, C, H, W)
        g_cnn = self.base_cnn_g(g_out)  # (B*T, 512)
        r_cnn = self.base_cnn_r(r_out)  # (B*T, 512)
        g_feat = g_cnn.view(B, T, -1)   # (B, T, 512)
        r_feat = r_cnn.view(B, T, -1)   # (B, T, 512)
        g_cossm, r_cossm = self.cossm_encoder(g_feat, r_feat,
                                              g_inference_params, r_inference_params)
        x = torch.cat([g_cossm, r_cossm], dim=-1)  # (B, T, 1024)
        x = self.enssm_encoder(x)                  # (B, T, 1024)
        last = x[:, -1, :]                         # (B, 1024)
        out  = self.classifier(last)               # (B, NUM_CLASSES)
        return out

# -------------------- 数据加载与预处理 --------------------

def load_config(config_path: str):
    """从 YAML 文件载入配置字典。"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_branch_npy(branch_dir: str):
    """
    从 SEED-VII 分支目录加载 t6x_89.npy 与 t6y_89.npy：
    - t6x_89.npy: (20, 4666, 6, 8, 9, 5)
    - t6y_89.npy: (93320,)
    本函数：
      1. 检查文件存在性
      2. 加载 .npy
      3. 将通道维度移到第三位 → (20,4666,6,5,8,9)
      4. 选后 4 个通道 → (20,4666,6,4,8,9)
    返回：
      - X: np.ndarray 形状 (20,4666,6,4,8,9)
      - y: np.ndarray 形状 (93320,)
    """
    x_path = os.path.join(branch_dir, "t6x_89.npy")
    y_path = os.path.join(branch_dir, "t6y_89.npy")
    assert os.path.isfile(x_path), f"找不到 {x_path}"
    assert os.path.isfile(y_path), f"找不到 {y_path}"

    X = np.load(x_path)  # (20,4666,6,8,9,5)
    y = np.load(y_path)  # (93320,)
    assert X.ndim == 6 and X.shape[0] == 20 and X.shape[2] == SEQ_LEN \
           and X.shape[3] == IMG_ROWS and X.shape[4] == IMG_COLS and X.shape[5] == 5, \
           f"{x_path} 形状应为 (20,4666,{SEQ_LEN},{IMG_ROWS},{IMG_COLS},5)，当前为 {X.shape}"
    assert y.shape[0] == 20 * X.shape[1], f"标签长度 {y.shape[0]} ≠ 20×{X.shape[1]}"

    # 通道移位： (20,4666,6,8,9,5) → (20,4666,6,5,8,9)
    X = X.transpose(0,1,2,5,3,4).copy()
    # 选后 4 个通道： (20,4666,6,5,8,9) → (20,4666,6,4,8,9)
    X = X[:, :, :, 1:, :, :].copy()

    return X, y

def prepare_all_data(DE_dir: str, PSD_dir: str, T: int):
    """
    加载 SEED-VII DE/PSD 分支 3D 滑窗数据并展开：
    - all_g: torch.FloatTensor (N_total, T, C, H, W)
    - all_r: torch.FloatTensor (N_total, T, C, H, W)
    - all_y: torch.LongTensor  (N_total,)
    - groups: np.ndarray      (N_total,) 每个样本对应被试 ID（0–19）
    """
    X_de, y_de   = load_branch_npy(DE_dir)   # (20,4666,6,4,8,9), (93320,)
    X_psd, y_psd = load_branch_npy(PSD_dir)  # (20,4666,6,4,8,9), (93320,)
    assert np.array_equal(y_de, y_psd), "DE 分支与 PSD 分支的标签不一致！"
    y_all = y_de

    num_subj = X_de.shape[0]   # 20
    NS       = X_de.shape[1]   # 4666

    de_flat  = X_de.reshape((-1, T, X_de.shape[3], IMG_ROWS, IMG_COLS))    # (93320,6,4,8,9)
    psd_flat = X_psd.reshape((-1, T, X_psd.shape[3], IMG_ROWS, IMG_COLS))  # (93320,6,4,8,9)

    groups = np.repeat(np.arange(num_subj), NS)  # (93320,)

    all_g = torch.from_numpy(de_flat).float()
    all_r = torch.from_numpy(psd_flat).float()
    all_y = torch.from_numpy(y_all).long()

    return all_g, all_r, all_y, groups

# -------------------- 主训练流程 --------------------

def main():
    # —— 0. 额外依赖 ——（仅本函数内部用）
    import copy
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm

    # —— 1. 载入配置 ——
    cfg      = load_config("config_VII_7.yaml")
    data_dir = cfg["data_dir"].rstrip("/\\")
    save_dir = cfg["save_dir"].rstrip("/\\")
    ensure_dir(save_dir)

    # 超参
    EPOCHS     = int(cfg.get("epochs",        100))
    BATCH_SIZE = int(cfg.get("batch_size",    128))
    LR         = float(cfg.get("learning_rate", 8e-5))
    sched_t    = cfg.get("lr_scheduler", "none").lower()
    SEED       = int(cfg.get("seed", 42))

    # 设备 & 随机种子
    DEVICE = torch.device(cfg["device"][0] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # —— 2. 动态覆盖全局常量 ——
    global SEQ_LEN, NUM_CHAN, IMG_ROWS, IMG_COLS, NUM_CLASSES
    SEQ_LEN     = int(cfg["input"]["T"])
    NUM_CHAN    = int(cfg["input"]["C"]) - 1
    IMG_ROWS    = int(cfg["input"]["H"])
    IMG_COLS    = int(cfg["input"]["W"])
    NUM_CLASSES = int(cfg.get("num_classes", 7))
    globals()["DEVICE"] = DEVICE

    # —— 3. 加载数据 ——
    print(">> 正在加载 SEED-VII 3D 滑窗数据 …")
    all_g, all_r, all_y, groups = prepare_all_data(
        os.path.join(data_dir, "DE"),
        os.path.join(data_dir, "PSD"),
        SEQ_LEN
    )

    # 只取第 4 位被试 (index=3)
    subj = 3
    idxs   = np.where(groups == subj)[0]
    Xg_sub = all_g[idxs]   # (NS, T, C, H, W)
    Xr_sub = all_r[idxs]
    y_sub  = all_y[idxs].numpy()
    NS     = len(idxs)

    # 10 折交叉验证
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_accs, y_true_all, y_pred_all = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(NS), y_sub), start=1):
        print(f"\n--- Subject {subj+1} Fold {fold}/10 ---")
        Xg_tr, Xg_te = Xg_sub[tr_idx], Xg_sub[te_idx]
        Xr_tr, Xr_te = Xr_sub[tr_idx], Xr_sub[te_idx]
        y_tr = torch.from_numpy(y_sub[tr_idx]).long()
        y_te = torch.from_numpy(y_sub[te_idx]).long()

        train_loader = DataLoader(
            TensorDataset(Xg_tr, Xr_tr, y_tr),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            TensorDataset(Xg_te, Xr_te, y_te),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True
        )

        # 初始化模型
        model = BiMambaVision(
            mm_input_size   = cfg["mmmamba"]["mm_input_size"],
            mm_output_sizes = cfg["mmmamba"]["mm_output_sizes"],
            dropout         = cfg["mmmamba"]["dropout"],
            activation      = cfg["mmmamba"]["activation"],
            causal          = cfg["mmmamba"]["causal"],
            mamba_config    = copy.deepcopy(cfg["mmmamba"]["mamba_config"])
        ).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = (
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            if sched_t == "cos" else None
        )
        criterion = nn.CrossEntropyLoss()

        # —— 4.1 训练 ——
        model.train()
        for epoch in range(1, EPOCHS + 1):
            loop = tqdm(
                train_loader,
                desc=f"S{subj+1} F{fold} Ep{epoch}/{EPOCHS}",
                leave=False
            )
            for g_b, r_b, y_b in loop:
                g_b, r_b, y_b = g_b.to(DEVICE), r_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(g_b, r_b), y_b)
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())
            if scheduler:
                scheduler.step()

        # —— 4.2 测试 & 特征收集 ——
        model.eval()
        correct, total = 0, 0
        class_correct = np.zeros(NUM_CLASSES, int)
        class_total   = np.zeros(NUM_CLASSES, int)
        feat_list, label_list = [], []

        with torch.no_grad():
            for g_b, r_b, y_b in test_loader:
                g_b, r_b, y_b = g_b.to(DEVICE), r_b.to(DEVICE), y_b.to(DEVICE)
                # 手动抽取“倒数第二层”特征
                B, T, C, H, W = g_b.shape
                g_out = g_b.view(B*T, C, H, W)
                r_out = r_b.view(B*T, C, H, W)
                g_cnn = model.base_cnn_g(g_out)
                r_cnn = model.base_cnn_r(r_out)
                g_feat = g_cnn.view(B, T, -1)
                r_feat = r_cnn.view(B, T, -1)
                g_cossm, r_cossm = model.cossm_encoder(g_feat, r_feat, None, None)
                x = torch.cat([g_cossm, r_cossm], dim=-1)
                x = model.enssm_encoder(x)
                last = x[:, -1, :]

                preds = model.classifier(last).argmax(dim=1)
                correct += (preds == y_b).sum().item()
                total   += y_b.size(0)
                for i in range(y_b.size(0)):
                    lbl = y_b[i].item()
                    class_total[lbl]   += 1
                    class_correct[lbl] += int(preds[i] == y_b[i])

                y_true_all.extend(y_b.cpu().tolist())
                y_pred_all.extend(preds.cpu().tolist())
                feat_list.append(last.cpu().numpy())
                label_list.append(y_b.cpu().numpy())

        fold_acc = correct / total * 100
        fold_accs.append(fold_acc)
        print(f"  Fold {fold} Accuracy: {fold_acc:.2f}%")
        for cls in range(NUM_CLASSES):
            acc_c = (
                class_correct[cls] / class_total[cls] * 100
                if class_total[cls] else 0.0
            )
            print(f"    Class {cls}: {acc_c:.2f}% "
                  f"({class_correct[cls]}/{class_total[cls]})")

        # 保存模型
        model_dir = os.path.join(save_dir, f"subject_{subj+1}")
        ensure_dir(model_dir)
        torch.save(
            model.state_dict(),
            os.path.join(model_dir, f"fold{fold}.pth")
        )

        # —— 4.3 三维决策边界可视化 ——
        X_feat = np.vstack(feat_list)
        y_lab  = np.hstack(label_list)

        # PCA 降到 3 维
        pca = PCA(n_components=3, random_state=SEED)
        X3d = pca.fit_transform(X_feat)

        # One-vs-Rest 逻辑回归
        lr_base = LogisticRegression(max_iter=500, random_state=SEED)
        lr_clf  = OneVsRestClassifier(lr_base)
        lr_clf.fit(X3d, y_lab)

        # 构造三维网格（20×20×20）
        x_min, x_max = X3d[:,0].min() - 1, X3d[:,0].max() + 1
        y_min, y_max = X3d[:,1].min() - 1, X3d[:,1].max() + 1
        z_min, z_max = X3d[:,2].min() - 1, X3d[:,2].max() + 1

        grid_x = np.linspace(x_min, x_max, 20)
        grid_y = np.linspace(y_min, y_max, 20)
        grid_z = np.linspace(z_min, z_max, 20)
        xx, yy, zz = np.meshgrid(
            grid_x, grid_y, grid_z, indexing='xy'
        )
        grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

        # 预测并重塑
        Z = lr_clf.predict(grid_points).reshape(xx.shape)

        # 绘图
        fig = plt.figure(figsize=(8, 8))
        ax  = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('tab10', NUM_CLASSES)

        # 绘制网格点（低 alpha）
        for cls in range(NUM_CLASSES):
            mask = (Z == cls)
            ax.scatter(
                xx[mask], yy[mask], zz[mask],
                color=cmap(cls), alpha=0.05, s=5
            )

        # 绘制真实样本点
        ax.scatter(
            X3d[:,0], X3d[:,1], X3d[:,2],
            c=y_lab, cmap=cmap, edgecolor='k', s=30
        )

        ax.set_title(f"Subject {subj+1} Fold {fold} 3D Decision Boundary")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3", labelpad=5, rotation=-90)  # 使 PC3 竖直显示
        ax.zaxis.set_rotate_label(False)
        fig.tight_layout()

        db_path = os.path.join(
            model_dir, f"fold{fold}_decision_boundary_3d.png"
        )
        plt.savefig(db_path, dpi=300)
        plt.close()
        print(f"  3D decision boundary saved to: {db_path}")

    # —— 5. 汇总结果 ——
    print(f"\n>>> Subject {subj+1} 10-fold Avg Acc: "
          f"{np.mean(fold_accs):.2f}%  Std: {np.std(fold_accs):.2f}% <<<")

    # —— 6. 绘制混淆矩阵 ——
    cm  = confusion_matrix(y_true_all, y_pred_all, labels=list(range(NUM_CLASSES)))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(7,6))
    sns.heatmap(
        cmn, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=range(NUM_CLASSES),
        yticklabels=range(NUM_CLASSES)
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Subject {subj+1} Confusion Matrix (10-fold)")
    cm_path = os.path.join(save_dir,
                           f"subject_{subj+1}_confmat.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")



if __name__ == "__main__":
    main()

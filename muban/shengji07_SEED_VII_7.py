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
                                   [mm_output_sizes[-1] *4],
                                   activation=activation, dropout=dropout,
                                   causal=causal, mamba_config=mamba_config)
        self.classifier = nn.Linear(mm_output_sizes[-1] * 4, NUM_CLASSES)

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
    # —— 1. 载入配置 ——
    cfg = load_config("config_VII_7.yaml")
    data_dir = cfg["data_dir"].rstrip("/\\")
    save_dir = cfg["save_dir"].rstrip("/\\")
    ensure_dir(save_dir)

    # 训练开关 & 超参数
    do_train      = bool(cfg.get("train", True))
    EPOCHS        = int(cfg.get("epochs", 100))
    BATCH_SIZE    = int(cfg.get("batch_size", 128))
    LR            = float(cfg.get("learning_rate", 8e-5))
    sched_t       = cfg.get("lr_scheduler", "none").lower()  # 可选 "cos" 或 "none"
    use_wandb     = bool(cfg.get("if_wandb", False))

    # 设备 & 随机种子
    DEVICE = torch.device(cfg["device"][0] if torch.cuda.is_available() else "cpu")
    SEED   = int(cfg.get("seed", 42))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # —— 2. 动态覆盖全局常量，以便 BaseCNN 等模块读取正确值 ——
    global SEQ_LEN, NUM_CHAN, IMG_ROWS, IMG_COLS, NUM_CLASSES
    SEQ_LEN     = int(cfg["input"]["T"])    # 6
    # SEED-VII 原始有 5 个通道，但训练时只选后 4 个
    NUM_CHAN    = int(cfg["input"]["C"]) - 1 # 配置 C=5 → 用 4
    IMG_ROWS    = int(cfg["input"]["H"])    # 8
    IMG_COLS    = int(cfg["input"]["W"])    # 9
    NUM_CLASSES = int(cfg.get("num_classes", 7))  # 7 类

    globals()["DEVICE"] = DEVICE

    # Mamba 模块配置（DeepCopy 后弹出 bidirectional 标志）
    mcfg         = copy.deepcopy(cfg["mmmamba"])
    mamba_config = mcfg["mamba_config"]
    bidir_flag   = mamba_config["bidirectional"]

    # —— 3. 加载所有被试的 DE & PSD 数据 ——
    DE_dir  = os.path.join(data_dir, "DE")
    PSD_dir = os.path.join(data_dir, "PSD")
    print(">> 正在加载 SEED-VII 3D 滑窗数据 …")
    all_g, all_r, all_y, groups = prepare_all_data(DE_dir, PSD_dir, SEQ_LEN)
    N_total = all_y.shape[0]
    num_subj= int(groups.max()) + 1  # 20
    NS      = int(N_total / num_subj) # 4666
    print(f" 共计 {num_subj} 位被试，每位被试滑窗数 = {NS}，总样本数 = {N_total}")

    # —— 4. 单被试内 5 折交叉验证 ——
    subj_accs = []  # 存放每个被试的 5 折平均准确率

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    for subj in range(num_subj):
        print(f"\n===== 开始处理 被试 {subj+1}/{num_subj} =====")
        idxs   = np.where(groups == subj)[0]  # (NS,)
        Xg_sub = all_g[idxs]                  # (NS,6,4,8,9)
        Xr_sub = all_r[idxs]                  # (NS,6,4,8,9)
        y_sub  = all_y[idxs].numpy()          # (NS,)
        assert y_sub.min() >= 0 and y_sub.max() < NUM_CLASSES, \
               f"第{subj}被试标签范围应在 [0,{NUM_CLASSES-1}]，当前 [{y_sub.min()},{y_sub.max()}]"

        fold_accs = []  # 存放此被试每折的整体准确率

        for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(NS), y_sub), start=1):
            print(f"\n--- 被试 {subj+1} Fold {fold}/5 ---")
            Xg_tr = Xg_sub[tr_idx];    Xg_te = Xg_sub[te_idx]
            Xr_tr = Xr_sub[tr_idx];    Xr_te = Xr_sub[te_idx]
            y_tr  = torch.from_numpy(y_sub[tr_idx]).long()
            y_te  = torch.from_numpy(y_sub[te_idx]).long()

            train_ds = TensorDataset(Xg_tr, Xr_tr, y_tr)
            test_ds  = TensorDataset(Xg_te, Xr_te, y_te)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=4, pin_memory=True)
            test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=2, pin_memory=True)

            # 动态覆盖全局，以便 BaseCNN 等使用正确维度
            globals()["NUM_CHAN"  ] = NUM_CHAN
            globals()["IMG_ROWS"  ] = IMG_ROWS
            globals()["IMG_COLS"  ] = IMG_COLS
            globals()["SEQ_LEN"   ] = SEQ_LEN
            globals()["NUM_CLASSES"] = NUM_CLASSES

            model = BiMambaVision(
                mm_input_size   = cfg["mmmamba"]["mm_input_size"],
                mm_output_sizes = cfg["mmmamba"]["mm_output_sizes"],
                dropout         = cfg["mmmamba"]["dropout"],
                activation      = cfg["mmmamba"]["activation"],
                causal          = cfg["mmmamba"]["causal"],
                mamba_config    = copy.deepcopy(cfg["mmmamba"]["mamba_config"])
            ).to(DEVICE)

            optimizer = optim.Adam(model.parameters(), lr=LR)
            if sched_t == "cos":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            else:
                scheduler = None

            criterion = nn.CrossEntropyLoss()

            if use_wandb:
                import wandb
                wandb.init(
                    project="SEED-VII-7-Classes",
                    name=f"subj{subj+1}_fold{fold}",
                    config=cfg,
                    reinit=True
                )
                wandb.watch(model, log="all")

            # —— 5. 训练主循环 ——
            model.train()
            for epoch in range(1, EPOCHS + 1):
                loop = tqdm(train_loader,
                            desc=f"Subj{subj+1} Fold{fold} Ep{epoch}/{EPOCHS}",
                            leave=False)
                running_loss = 0.0
                for g_b, r_b, y_b in loop:
                    g_b = g_b.to(DEVICE, non_blocking=True)
                    r_b = r_b.to(DEVICE, non_blocking=True)
                    y_b = y_b.to(DEVICE, non_blocking=True)

                    optimizer.zero_grad()
                    logits = model(g_b, r_b)       # (batch_size, 7)
                    loss   = criterion(logits, y_b)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * g_b.size(0)
                    loop.set_postfix(loss=loss.item())

                if scheduler is not None:
                    scheduler.step()

                avg_loss = running_loss / len(train_loader.dataset)
                if use_wandb:
                    wandb.log({f"subj{subj+1}_fold{fold}/train_loss": avg_loss,
                               "epoch": epoch})

            # —— 6. 测试 & 计算整体及 per-class 准确率 ——
            model.eval()
            correct_total = 0
            total_samples = 0
            # 用于 per-class
            class_correct = np.zeros(NUM_CLASSES, dtype=int)
            class_total   = np.zeros(NUM_CLASSES, dtype=int)

            with torch.no_grad():
                for g_b, r_b, y_b in test_loader:
                    g_b = g_b.to(DEVICE, non_blocking=True)
                    r_b = r_b.to(DEVICE, non_blocking=True)
                    y_b = y_b.to(DEVICE, non_blocking=True)

                    logits = model(g_b, r_b)        # (batch_size, 7)
                    preds  = logits.argmax(dim=1)   # (batch_size,)

                    correct_total += (preds == y_b).sum().item()
                    total_samples += y_b.size(0)

                    for i in range(y_b.size(0)):
                        label = y_b[i].item()
                        class_total[label] += 1
                        if preds[i].item() == label:
                            class_correct[label] += 1

            overall_acc = correct_total / total_samples * 100
            print(f"被试{subj+1} Fold{fold} 整体准确率: {overall_acc:.2f}%")

            # 逐类输出准确率
            for cls in range(NUM_CLASSES):
                if class_total[cls] > 0:
                    acc_cls = class_correct[cls] / class_total[cls] * 100
                else:
                    acc_cls = 0.0
                print(f"  类别 {cls} 准确率: {acc_cls:.2f}% ({class_correct[cls]}/{class_total[cls]})")

            fold_accs.append(overall_acc)

            # —— 7. 保存该被试该折模型 ——
            model_sub_dir = os.path.join(save_dir, f"subject_{subj+1}")
            ensure_dir(model_sub_dir)
            model_path = os.path.join(model_sub_dir, f"fold{fold}.pth")
            # torch.save(model.state_dict(), model_path)
            print(f"[被试{subj+1} Fold{fold}] 模型已保存到: {model_path}")

            if use_wandb:
                wandb.log({f"subj{subj+1}_fold{fold}/test_acc": overall_acc})
                wandb.finish()

        # —— 4.3 每被试 5 折平均与标准差 ——
        mean_acc = np.mean(fold_accs)
        std_acc  = np.std(fold_accs)
        print(f"\n>>> 被试{subj+1} 10 折平均准确率: {mean_acc:.2f}%  Std: {std_acc:.2f}% <<<")
        subj_accs.append(mean_acc)

    # —— 8. 所有被试汇总结果 ——
    overall_mean = np.mean(subj_accs)
    overall_std  = np.std(subj_accs)
    print(f"\n=== 所有被试 5 折平均准确率: {overall_mean:.2f}%  Std: {overall_std:.2f}% ===")


if __name__ == '__main__':
    main()

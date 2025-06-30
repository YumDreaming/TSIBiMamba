import argparse
import copy
import json
import pickle
import random

import math
import os

import numpy as np
import yaml
from matplotlib import pyplot as plt
from pyts.image import RecurrencePlot
from sklearn.decomposition import PCA
from timm.models.vision_transformer import Mlp, PatchEmbed
from torch import nn
import torch
# from timm.models.layers import DropPath
from timm.layers import DropPath
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Subset
from sklearn.model_selection import GroupShuffleSplit, train_test_split, KFold, StratifiedKFold

# from .models.mamba.selective_scan_interface import mamba_inner_fn_no_out_proj
from models.selective_scan_interface import mamba_inner_fn_no_out_proj, selective_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from torchvision import transforms
import mne
import skimage
from skimage.transform import resize
from mamba_ssm import Mamba
from mm_bimamba import Mamba as MMBiMamba
from bimamba import Mamba as BiMamba
import scipy.io as sio

# —— 超参数 & 配置 ——
NUM_CLASSES = 2
BATCH_SIZE  = 64
IMG_ROWS, IMG_COLS, NUM_CHAN = 8, 9, 4
SEQ_LEN     = 6      # 每个样本由 6 帧组成
EPOCHS      = 100
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED        = 7

class BaseCNN(nn.Module):
    def __init__(self, in_chan=NUM_CHAN):
        super().__init__()
        # conv1: (B,4,8,9) → (B,64,8,9)
        self.conv1 = nn.Conv2d(in_chan, 64, kernel_size=5, padding=2)
        # conv2: (B,64,8,9) → (B,128,8,9)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # conv3: (B,128,8,9) → (B,256,8,9)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # conv4: (B,256,8,9) → (B,64,8,9)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1)
        # 池化： (B,64,8,9) → (B,64,4,4)
        self.pool = nn.MaxPool2d(2,2)
        # 全连接：4×4×64=1024 → 512
        self.fc   = nn.Linear(64*4*4, 512)

    def forward(self, x):
        # x: (B, 4, 8, 9)
        x = F.relu(self.conv1(x))   # → (B,64,8,9)
        x = F.relu(self.conv2(x))   # → (B,128,8,9)
        x = F.relu(self.conv3(x))   # → (B,256,8,9)
        x = F.relu(self.conv4(x))   # → (B,64,8,9)
        x = self.pool(x)            # → (B,64,4,4)
        x = x.view(x.size(0), -1)   # → (B,1024)
        x = F.relu(self.fc(x))      # → (B,512)
        return x                    # 不 reshape，因为 LSTM 接受 (B, T, 512)

class ResBlock(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 dropout=0.2,
                 causal=False,
                 dilation=1,):
        super().__init__()
        self.e_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.e_bn = nn.BatchNorm1d(output_size)

        self.g_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.g_bn = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()

        self.e_drop = nn.Dropout(dropout)
        self.g_drop = nn.Dropout(dropout)

        self.e_net = nn.Sequential(self.e_conv, self.e_bn, self.relu, self.e_drop)
        self.g_net = nn.Sequential(self.g_conv, self.g_bn, self.relu, self.g_drop)

        if input_size != output_size:
            self.e_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
            self.g_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.e_skipconv = None
            self.g_skipconv = None

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.e_conv.weight.data)
        nn.init.xavier_uniform_(self.g_conv.weight.data)
        # nn.init.xavier_uniform_(self.conv2.weight.data)

    def forward(self, g, r):
        # 主分支
        g_out = self.e_net(g)  # Res Block
        r_out = self.g_net(r)  # Res Block

        # 残差分支
        g_res = g
        r_res = r
        if self.e_skipconv is not None:
            g_res = self.e_skipconv(g_res)
        if self.g_skipconv is not None:
            r_res = self.g_skipconv(r_res)
        g_out = g_out + g_res
        r_out = r_out + r_res
        return g_out, r_out

class CoBiMambaLayer(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.0,
                 activation='Swish',
                 causal=False,
                 mamba_config=None):
        super().__init__()
        if activation == 'Swish':
            activation = 'Swish'
        elif activation == "GELU":
            activation = torch.nn.GELU


        bidirectional = mamba_config.pop('bidirectional')
        self.bidirectional = bidirectional
        if causal or (not bidirectional):
            self.mamna_g = Mamba(
                d_model=d_model,
                **mamba_config,
            )
            self.mamna_r = Mamba(
                d_model=d_model,
                **mamba_config,
            )
        else:
            self.mamna = MMBiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config,
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self,
                g, r,
                g_inference_params=None,
                r_inference_params=None,):
        if self.bidirectional:
            g_out1, r_out1 = self.mamna(g, r, g_inference_params, r_inference_params)
        else:
            g_out1 = self.mamna_g(g)
            r_out1 = self.mamna_r(r)

        g_out = g + self.norm1(g_out1)
        r_out = r + self.norm2(r_out1)

        return g_out, r_out

class CoSSM(nn.Module):
    def __init__(self,
                 input_size,
                 output_sizes=[1024],
                 dropout=0.2,
                 activation='GELU',
                 causal=False,
                 mamba_config=None
                 ):
        super().__init__()

        res_list = []
        mamba_list = []

        for i in range(len(output_sizes)):
            res_list.append(ResBlock(
                input_size=input_size if i < 1 else output_sizes[i - 1],
                output_size=output_sizes[i],
                dropout=dropout
            ))
            mamba_list.append(CoBiMambaLayer(
                d_model=output_sizes[i],
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config
            ))

        self.res_layers = torch.nn.ModuleList(res_list)
        self.mamba_layers = torch.nn.ModuleList(mamba_list)

    def forward(self, g_x, r_x, g_inference_params=None, r_inference_params=None):
        g_out = g_x
        r_out = r_x

        for res_layer, mamba_layer in zip(self.res_layers, self.mamba_layers):
            g_out, r_out = res_layer(g_out.permute(0, 2, 1), r_out.permute(0, 2, 1))
            g_out = g_out.permute(0, 2, 1)
            r_out = r_out.permute(0, 2, 1)
            g_out, r_out = mamba_layer(g_out, r_out, g_inference_params, r_inference_params)

        return g_out, r_out

class EnResBlock(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 dropout=0.2,
                 activation='GELU',
                 causal=False,
                 dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()

        self.drop = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop)

        if input_size != output_size:
            self.conv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.conv = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data)

    def forward(self, x):
        out = self.net(x)
        if self.conv is not None:
            x = self.conv(x)
        out = out + x
        return out

class EnBiMambaLayer(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.0,
                 activation='Swish',
                 causal=False,
                 mamba_config=None):
        super().__init__()

        if activation == 'Swish':
            activation = 'Swish'
        elif activation == "GELU":
            activation = torch.nn.GELU

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamna = Mamba(
                d_model=d_model,
                **mamba_config,
            )
        else:
            self.mamna = BiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config,
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, inference_params=None):
        out = x + self.norm1(self.mamna(x, inference_params))

        return out

class EnSSM(nn.Module):
    def __init__(self,
                 input_size,
                 output_sizes=[1024],
                 activation='GELU',
                 dropout=0.2,
                 causal=False,
                 mamba_config=None):
        super().__init__()

        res_list = []
        mamba_list = []

        for i in range(len(output_sizes)):
            res_list.append(EnResBlock(
                input_size=input_size if i < 1 else output_sizes[i - 1],
                output_size=output_sizes[i],
                dropout=dropout
            ))
            mamba_list.append(EnBiMambaLayer(
                d_model=output_sizes[i],
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config
            ))

        self.res_layers = torch.nn.ModuleList(res_list)
        self.mamba_layers = torch.nn.ModuleList(mamba_list)

    def forward(self, x, inference_params=None):
        out = x
        for res_layer, mamba_layer in zip(self.res_layers, self.mamba_layers):
            out = res_layer(out.permute(0, 2, 1))
            out = out.permute(0, 2, 1)
            out = mamba_layer(out, inference_params=inference_params)

        return out

class BiMambaVision(nn.Module):
    def __init__(self,
                 mm_input_size=512,
                 mm_output_sizes=[512],
                 dropout=0.1,
                 activation='GELU',
                 causal=False,
                 mamba_config=None
                 ):
        super().__init__()
        self.base_cnn_g = BaseCNN()
        self.base_cnn_r = BaseCNN()

        self.cossm_encoder = CoSSM(mm_input_size,
                                   mm_output_sizes,
                                   dropout=dropout,
                                   activation=activation,
                                   causal=causal,
                                   mamba_config=mamba_config)

        self.enssm_encoder = EnSSM(mm_output_sizes[-1] * 2,
                                   [512*8],
                                   activation=activation,
                                   dropout=dropout,
                                   causal=causal,
                                   mamba_config=mamba_config)

        self.classifier = nn.Linear(512*8,2)

    def forward(self, g, r, g_inference_params = None, r_inference_params = None):
        # x: (B, T=6, 4, 8, 9)
        B, T,  C, H, W = g.shape
        g_out = g
        r_out = r

        g_out = g_out.view(B*T, C, H, W) # → (B*T, 4, 8, 9)
        g_cnn = self.base_cnn_g(g_out) # → (B*T, 512)

        r_out = r_out.view(B*T, C, H, W) # → (B*T, 4, 8, 9)
        r_cnn = self.base_cnn_r(r_out) # → (B*T, 512)

        g_feat = g_cnn.view(B, T, 512)
        r_feat = r_cnn.view(B, T, 512)

        g_cossm, r_cossm = self.cossm_encoder.forward(g_feat, r_feat, g_inference_params, r_inference_params)

        x = torch.cat([g_cossm, r_cossm], dim=-1)

        x = self.enssm_encoder(x)

        last = x[:, -1, :]
        out = self.classifier(last)

        return out



# —— 1. 加载配置 ——
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# —— 2. 设备 & 随机种子 ——
DEVICE = torch.device(cfg['device'][0] if torch.cuda.is_available() else 'cpu')
torch.manual_seed(int(cfg.get('seed', 7)))
np.random.seed(int(cfg.get('seed', 7)))

# —— 3. 导入模型 ——
# from models.bimambavision import BiMambaVision

# —— 4. 辅助函数 ——
def load_branch(mat_path, label_key='valence_labels'):
    md     = sio.loadmat(mat_path)
    data   = md['data'].astype(np.float32)          # (4800,4,8,9)
    labels = md[label_key].flatten().astype(np.int64)
    return data, labels

def prepare_sequences(data, labels, T):
    n_seg = data.shape[0]
    n_seq = n_seg // T
    data = data[:n_seq * T]
    labels = labels[:n_seq * T]
    C, H, W = data.shape[1:]
    data_seq = data.reshape(n_seq, T, C, H, W)
    labels_seq = labels.reshape(n_seq, T)[:, 0]
    return data_seq, labels_seq

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# —— 5. 主函数 ——
def main():
    data_dir = cfg['data_dir']
    save_dir = cfg['save_dir']
    ensure_dir(save_dir)

    de_dir  = os.path.join(data_dir, 'DE')
    psd_dir = os.path.join(data_dir, 'PSD')

    # 强制类型转换
    T  = int(cfg['input']['T'])
    B  = int(cfg['batch_size'])
    E  = int(cfg['epochs'])
    LR = float(cfg['learning_rate'])
    scheduler_type = str(cfg.get('lr_scheduler', 'none')).lower()

    de_files = sorted([f for f in os.listdir(de_dir) if f.lower().endswith('.mat')])
    for de_fname in de_files:
        parts = de_fname.split('_', 1)
        if len(parts) != 2:
            print(f"跳过无法解析的文件名：{de_fname}")
            continue
        sid       = parts[1]                # e.g. "s01.mat"
        psd_fname = f"PSD_{sid}"            # e.g. "PSD_s01.mat"

        g_path = os.path.join(de_dir,  de_fname)
        r_path = os.path.join(psd_dir, psd_fname)
        if not os.path.isfile(r_path):
            print(f"找不到对应 PSD 文件，跳过：{r_path}")
            continue

        print(f"\n=== Subject {de_fname} / {psd_fname} ===")
        g_data, g_labels = load_branch(g_path, 'valence_labels')
        r_data, r_labels = load_branch(r_path, 'valence_labels')
        assert np.array_equal(g_labels, r_labels), "DE/PSD 标签不一致！"

        Xg, y = prepare_sequences(g_data, g_labels, T)
        Xr, _ = prepare_sequences(r_data, r_labels, T)

        Xg = torch.from_numpy(Xg).float().to(DEVICE)
        Xr = torch.from_numpy(Xr).float().to(DEVICE)
        y  = torch.from_numpy(y).long().to(DEVICE)

        skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(cfg.get('seed',7)))
        fold_acc = []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y.cpu().numpy()), 1):
            print(f"\n--- Fold {fold} ---")
            Xg_tr, Xr_tr, y_tr = Xg[tr_idx], Xr[tr_idx], y[tr_idx]
            Xg_te, Xr_te, y_te = Xg[te_idx], Xr[te_idx], y[te_idx]

            train_ds = TensorDataset(Xg_tr, Xr_tr, y_tr)
            test_ds  = TensorDataset(Xg_te, Xr_te, y_te)
            train_loader = DataLoader(train_ds, batch_size=B, shuffle=True)
            test_loader  = DataLoader(test_ds,  batch_size=B)

            mcfg = copy.deepcopy(cfg['mmmamba'])
            model = BiMambaVision(
                mm_input_size=mcfg["mm_input_size"],
                mm_output_sizes=mcfg["mm_output_sizes"],
                dropout=mcfg["dropout"],
                activation=mcfg["activation"],
                causal=mcfg["causal"],
                mamba_config=mcfg["mamba_config"]
            ).to(DEVICE)

            optimizer = optim.Adam(model.parameters(), lr=LR)
            if scheduler_type == 'cos':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=E)
            else:
                scheduler = None
            criterion = nn.CrossEntropyLoss()

            model.train()
            for epoch in range(1, E+1):
                loop = tqdm(train_loader,
                            desc=f"Fold{fold} Epoch{epoch}/{E}",
                            leave=False)
                for g_batch, r_batch, y_batch in loop:
                    optimizer.zero_grad()
                    logits = model(g_batch, r_batch)
                    loss   = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()
                    loop.set_postfix(loss=loss.item())
                if scheduler:
                    scheduler.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for g_batch, r_batch, y_batch in test_loader:
                    logits = model(g_batch, r_batch)
                    preds  = logits.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total   += y_batch.size(0)
            acc = correct / total * 100
            print(f"Fold {fold} Accuracy: {acc:.2f}%")
            fold_acc.append(acc)

            subj_id = os.path.splitext(de_fname)[0]
            save_p = os.path.join(save_dir, f"{subj_id}_fold{fold}.pth")
            torch.save(model.state_dict(), save_p)

        mean_a = np.mean(fold_acc)
        std_a  = np.std(fold_acc)
        print(f"\nSubject {de_fname} — Mean Acc: {mean_a:.2f}%  Std: {std_a:.2f}%")

if __name__ == '__main__':
    main()

# def main():
#     # 1. 在脚本里完整定义“配置”
#     config = {
#         # 数据相关
#         "data_dir": "/root/autodl-tmp/deap_map/3d/",
#         "save_dir": "/root/autodl-tmp/model",
#         # 训练开关 & 超参数
#         "train": True,
#         "epochs": 120,
#         "batch_size": 20,
#         "learning_rate": 8e-5,
#         "lr_scheduler": "cos",
#         "if_wandb": False,
#         # 设备
#         "device": ["cuda"] ,  # 优先用 cuda
#         # 模型 & 模块配置
#         "model": "DeapMamba",  # 只是个标记，真正用的类是 BiMambaVision
#         "dataset": "DEAP",
#         "mmmamba": {
#             "mm_input_size": 512,
#             "mm_output_sizes": [512],
#             "dropout": 0.1,
#             "activation": "Tanh",
#             "causal": False,
#             "mamba_config": {
#                 "d_state": 12,
#                 "expand": 2,
#                 "d_conv": 4,
#                 "bidirectional": True,
#             },
#         },
#         # 类别数
#         "num_classes": 2,
#         # 输入尺寸
#         "input": {
#             "T": 6,
#             "C": 4,
#             "H": 8,
#             "W": 9,
#         },
#     }
#
#     # 2. 设备准备
#     device_name = config["device"][0] if torch.cuda.is_available() else "cpu"
#     device = torch.device(device_name)
#     print(f"Using device: {device}")
#
#     # 3. 实例化模型
#     # 深拷贝 mamba_config，避免内部 .pop 破坏原 dict
#     mcfg = copy.deepcopy(config["mmmamba"])
#     model = BiMambaVision(
#         mm_input_size=mcfg["mm_input_size"],
#         mm_output_sizes=mcfg["mm_output_sizes"],
#         dropout=mcfg["dropout"],
#         activation=mcfg["activation"],
#         causal=mcfg["causal"],
#         mamba_config=mcfg["mamba_config"],
#     ).to(device)
#     model.eval()
#
#     # 4. 构造 dummy 数据： (B, T, C, H, W)
#     B = config["batch_size"]
#     T = config["input"]["T"]
#     C = config["input"]["C"]
#     H = config["input"]["H"]
#     W = config["input"]["W"]
#     g = torch.randn(B, T, C, H, W, device=device)
#     r = torch.randn(B, T, C, H, W, device=device)
#
#     # 5. 前向检验
#     with torch.no_grad():
#         out = model(g, r)
#     print(f"Forward 输出 shape = {tuple(out.shape)}，应为 ({B}, {config['num_classes']})")
#
# if __name__ == "__main__":
#     main()
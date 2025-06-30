import abc
import argparse
import copy
import json
import pickle
import random

import math
import os

import numpy as np
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
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Subset
from sklearn.model_selection import GroupShuffleSplit, train_test_split

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
import skimage
from skimage.transform import resize
# from speechbrain.nnet.activations import Swish
from mamba_ssm import Mamba
from mm_bimamba_fold import Mamba as MMBiMamba
from bimamba import Mamba as BiMamba

class BaseNet(nn.Module, abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def feature_extractor(self, x, mask=None):
        pass

    @abc.abstractmethod
    def classifier(self, x):
        pass

    def forward(self, x, mask=None):
        x = self.feature_extractor(x, mask)
        x = self.classifier(x)
        return x

class MMCNNEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
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

    def forward(self, xe, xg):
        # 主分支
        e_out = self.e_net(xe) # Res Block
        g_out = self.g_net(xg) # Res Block

        # 残差分支
        if self.e_skipconv is not None:
            xe = self.e_skipconv(xe)
        if self.g_skipconv is not None:
            xg = self.g_skipconv(xg)
        e_out = e_out+xe
        g_out = g_out+xg
        return e_out, g_out

class MMMambaEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 d_ffn,
                 activation='Swish',
                 dropout=0.0,
                 causal=False,
                 mamba_config=None):
        """
            mamba_config:
                d_state: 16
                expand: 4
                d_conv: 4
                bidirectional: true
        """
        super().__init__()
        if activation == 'Swish':
            activation = 'Swish'
        elif activation == "GELU":
            activation = torch.nn.GELU

        bidirectional = mamba_config.pop('bidirectional')
        self.bidirectional = bidirectional
        if causal or (not bidirectional):
            self.mamna_e = Mamba(
                d_model=d_model,
                **mamba_config,
            )
            self.mamna_g = Mamba(
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

        self.a_downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(d_model),
        )

    def forward(self,
                e_x, g_x,
                e_inference_params = None,
                g_inference_params = None,):
        if self.bidirectional:
            e_out1, g_out1 = self.mamna(e_x, g_x, e_inference_params, g_inference_params)
        else:
            e_out1 = self.mamna_e(e_x)
            g_out1 = self.mamna_g(g_x)
        e_out = e_x + self.norm1(e_out1)
        g_out = g_x + self.norm2(g_out1)

        return e_out, g_out


class CoSSM(nn.Module):
    def __init__(self,
                 num_layer,
                 input_size,
                 output_sizes=[128,256, 512],
                 d_ffn=1024,
                 activation="Swish",
                 dropout=0.1,
                 kernel_size=3,
                 causal=False,
                 mamba_config=None
                 ):
        super().__init__()

        cnn_list = []
        mamba_list = []

        for i in range(len(output_sizes)):
            cnn_list.append(MMCNNEncoderLayer(
                    input_size = input_size if i<1 else output_sizes[i-1],
                    output_size = output_sizes[i],
                    dropout=dropout
                ))
            mamba_list.append(MMMambaEncoderLayer(
                d_model=output_sizes[i],
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            ))

        self.cnn_layers = torch.nn.ModuleList(cnn_list)
        self.mamba_layers = torch.nn.ModuleList(mamba_list)

    def forward(self, e_x, g_x, e_inference_params=None, g_inference_params=None):
        e_out = e_x
        g_out = g_x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            e_out, g_out = cnn_layer(e_out.permute(0, 2, 1), g_out.permute(0, 2, 1))
            e_out = e_out.permute(0, 2, 1)
            g_out = g_out.permute(0, 2, 1)
            e_out, g_out = mamba_layer(e_out, g_out, e_inference_params, g_inference_params)

        return e_out, g_out

class CNNEncoderLayer(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 dropout=0.0,
                 causal=False,
                 dilation=1,):
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

class MambaEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 d_ffn,
                 activation="Swish",
                 dropout=0.0,
                 causal=False,
                 mamba_config=None,):
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
                 num_layer,
                 input_size,
                 output_sizes=[128,256, 512],
                 d_ffn=1024,
                 activation="Swish",
                 dropout=0.1,
                 kernel_size=3,
                 causal=False,
                 mamba_config=None):
        super().__init__()

        cnn_list = []
        mamba_list = []

        for i in range(len(output_sizes)):
            cnn_list.append(CNNEncoderLayer(
                input_size=input_size if i<1 else output_sizes[i-1],
                output_size=output_sizes[i],
                dropout=dropout,
            ))
            mamba_list.append(MambaEncoderLayer(
                d_model=output_sizes[i],
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            ))

        self.cnn_layers = torch.nn.ModuleList(cnn_list)
        self.mamba_layers = torch.nn.ModuleList(mamba_list)

    def forward(self, x, inference_params = None,):
        out = x
        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            out = cnn_layer(out.permute(0, 2, 1))
            out = out.permute(0, 2, 1)
            out = mamba_layer(out, inference_params=inference_params)

        return out

class DepMamba(BaseNet):
    def __init__(self,
                 eeg_input_size=32,
                 gsr_input_size=1,
                 mm_input_size=16,
                 mm_output_sizes=[256],
                 d_ffn=1024,
                 num_layers=8,
                 dropout=0.1,
                 activation='GELU',
                 causal=False,
                 mamba_config=None):
        super().__init__()
        self.conv_eeg = nn.Conv1d(eeg_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        self.conv_gsr = nn.Conv1d(gsr_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)

        self.cossm_encoder = CoSSM(num_layers,
                                   mm_input_size,
                                   mm_output_sizes,
                                   d_ffn,
                                   activation=activation,
                                   dropout=dropout,
                                   causal=causal,
                                   mamba_config=mamba_config)

        self.enssm_encoder = EnSSM(num_layers,
                                   mm_output_sizes[-1] * 2,
                                   [mm_output_sizes[-1] * 2],
                                   d_ffn,
                                   activation=activation,
                                   dropout=dropout,
                                   causal=causal,
                                   mamba_config=mamba_config)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.output = nn.Linear(mm_output_sizes[-1] * 2, 1)


    def feature_extractor(self, x, padding_mask=None, e_inference_params = None, g_inference_params = None):
        xe = x[:, :, :32]
        xg = x[:, :, 32:33]

        xe = self.conv_eeg(xe.permute(0,2,1)).permute(0,2,1)
        xg = self.conv_gsr(xg.permute(0,2,1)).permute(0,2,1)

        xe, xg = self.cossm_encoder(xe, xg, e_inference_params, g_inference_params)

        x = torch.cat([xe, xg], dim=-1)
        x = self.enssm_encoder(x)

        if padding_mask is not None:
            x = x * (padding_mask.unsqueeze(-1).float())
            x = x.sum(dim=1) / (padding_mask.unsqueeze(-1).float()
                                ).sum(dim=1, keepdim=False)  # Compute average
        else:
            x = self.pool(x.permute(0,2,1)).squeeze(-1)
        return x

    def classifier(self, x):
        return self.output(x)

def test_dep_mamba_forward():
    # 构造一个合理的 mamba_config
    base_mamba_cfg = {
        'd_state': 16,
        'expand': 4,
        'd_conv': 4,
        'bidirectional': True
    }

    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 实例化模型，并搬到 device
    model = DepMamba(
        eeg_input_size=32,
        gsr_input_size=1,
        mm_input_size=64,
        mm_output_sizes=[128, 256],
        num_layers=2,
        dropout=0.1,
        activation='GELU',
        causal=False,
        mamba_config=copy.deepcopy(base_mamba_cfg)
    ).to(device)
    model.eval()  # 只做前向，不跑梯度

    # 准备假数据：batch=4，序列长度=50，总通道=33 (32 EEG + 1 GSR)，并搬到 device
    x = torch.randn(4, 50, 33, device=device)
    mask = torch.ones(4, 50, device=device)
    mask[0, -5:] = 0

    # 前向测试
    try:
        out1 = model(x)  # 不带 mask
        assert out1.shape == (4, 1), f"Expected (4,1), got {out1.shape}"
        print("✅ Forward without mask 输出 shape:", out1.shape)

        out2 = model(x, mask)  # 带 mask
        assert out2.shape == (4, 1), f"Expected (4,1), got {out2.shape}"
        print("✅ Forward with mask    输出 shape:", out2.shape)

    except RuntimeError as e:
        if 'x.is_cuda()' in str(e):
            print("⚠️ 运行时检测到 CUDA-only 操作，但当前张量在 CPU 上。")
            print("   - 如果有 GPU，可在运行脚本时确保 `torch.cuda.is_available()` 返回 True；")
            print("   - 否则需要在 `selective_scan_interface` 中为 CPU 提供 fallback 实现。")
        raise  # 继续抛出，方便定位

if __name__ == "__main__":
    print("开始测试 DepMamba 前向逻辑...")
    test_dep_mamba_forward()
    print("全部测试通过！")
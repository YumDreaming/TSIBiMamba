import argparse
import random
from datetime import time

import math
import os

import numpy as np
from matplotlib import pyplot as plt
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
from models.selective_scan_interface import mamba_inner_fn_no_out_proj

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from torchvision import transforms

# 空间上的轻量增强，仅对 H×W 做操作
train_aug = transforms.Compose([
    transforms.RandomResizedCrop((128, 128), scale=(0.8, 1.0), antialias=True),  # 随机裁剪并缩放到 224×224
    transforms.RandomHorizontalFlip(),                            # 随机水平翻转
])

# 验证/测试时只做 resize 保证输入尺寸一致
val_aug = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),
])



# 输入嵌入模块
class PatchEmbed(nn.Module):
    def __init__(self,
                 in_chans=32,
                 in_dim=64,
                 dim=96
                 # img_size=224, patch_size=16, in_chans=3, embed_dim=768
                 ):
        super().__init__()
        self.proj = nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        # x(B, T, C, H, W)
        B, T, C, H, W = x.shape
        # 1) 时间维展平
        x = x.view(B * T, C, H, W)
        x = self.proj(x)
        x = self.conv(x)
        _, D, H2, W2 = x.shape
        x = x.view(B, T, D, H2, W2)
        return x


# 卷积块
class ConvBlock(nn.Module):
    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()
        # padding = tuple(k//2 for k in kernel_size)
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=padding)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=padding)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        # self.layer_scale = layer_scale

        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.use_layer_scale = True
        else:
            self.use_layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        shortcut = x

        # 1) 展平时间维到 batch
        x = x.view(B * T, C, H, W)

        # 2) 两层空间卷积
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)

        if self.use_layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)

        # 4) DropPath + 恢复形状
        x = self.drop_path(x)
        x = x.view(B, T, C, H, W)

        # 5) 残差连接
        return shortcut + x


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 通过双向依赖捕获的多模态融合
class CoSSM(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
            if_devide_out=True,  # False
            init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.if_devide_out = if_devide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.g_gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)
            self.r_gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.g_in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.r_in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.g_conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.r_conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.g_x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.g_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.r_x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.r_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.g_dt_proj.weight, dt_init_std)
            nn.init.constant_(self.r_dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.g_dt_proj.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.r_dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        g_dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        g_inv_dt = g_dt + torch.log(-torch.expm1(-g_dt))
        with torch.no_grad():
            self.g_dt_proj.bias.copy_(g_inv_dt)
        self.g_dt_proj.bias._no_reinit = True

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        r_dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        r_inv_dt = r_dt + torch.log(-torch.expm1(-r_dt))
        with torch.no_grad():
            self.r_dt_proj.bias.copy_(r_inv_dt)
        self.r_dt_proj.bias._no_reinit = True

        # S4D real initialization
        # Shared A
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.g_D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.g_D._no_weight_decay = True
        self.r_D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.r_D._no_weight_decay = True

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        self.g_conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.r_conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.g_x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.g_dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.r_x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.r_dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.g_D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # GASF分支反向跳跃连接, 保持fp32
        self.g_D_b._no_weight_decay = True  # 禁用权重衰减
        self.r_D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Recurrence分支反向跳跃连接, 保持fp32
        self.r_D_b._no_weight_decay = True

        self.g_out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.r_out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, g_hidden_states, r_hidden_states, g_inference_params=None, r_inference_params=None):
        # 确保两个分支输入形状一致
        assert g_hidden_states.shape == r_hidden_states.shape  # shape: (B, L, D)
        batch, seqlen, dim = g_hidden_states.shape

        # 初始化卷积和SSM状态（用于增量推理）
        g_conv_state, g_ssm_state = None, None
        r_conv_state, r_ssm_state = None, None

        # gasf branch，分支投影，将原始特征映射到更高维空间
        g_xz = rearrange(
            self.g_in_proj.weight @ rearrange(g_hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.g_in_proj.bias is not None:
            g_xz = g_xz + rearrange(self.g_in_proj.bias.to(dtype=g_xz.dtype), "d -> d 1")

        # rec branch
        r_xz = rearrange(
            self.r_in_proj.weight @ rearrange(r_hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.r_in_proj.bias is not None:
            r_xz = r_xz + rearrange(self.r_in_proj.bias.to(dtype=r_xz.dtype), "d -> d 1")

        # Compute ∆ A B C D, the state space parameters.
        #     A, D 是独立于输入的 (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C 是依赖于输入的 (这是Mamba模型和 linear time invariant S4 的主要区别,这也是为什么Mamba被称为selective state spaces

        # 状态空间参数计算
        A = -torch.exp(self.A_log.float())  # 状态转移矩阵, 离散化
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and g_inference_params is None and r_inference_params is None:  # Doesn't support outputting the states
            A_b = -torch.exp(self.A_b_log.float())  # 反向扫描用A

            # 正向扫描（GASF分支）
            g_out = mamba_inner_fn_no_out_proj(
                g_xz,
                self.g_conv1d.weight,  # 因果卷积核
                self.g_conv1d.bias,
                self.g_x_proj.weight,  # 生成 Δ,B,C
                self.g_dt_proj.weight,  # Δ 调整
                A,  # 状态矩阵
                None,  # input-dependent B
                None,  # input-dependent C
                D=self.g_D.float(),  # 跳跃连接
                delta_bias=self.g_dt_proj.bias.float(),
                delta_softplus=True,
            )

            # 反向扫描（翻转序列）
            g_out_b = mamba_inner_fn_no_out_proj(
                g_xz.flip([-1]),
                self.g_conv1d_b.weight,
                self.g_conv1d_b.bias,
                self.g_x_proj_b.weight,
                self.g_dt_proj_b.weight,
                A_b,
                None,  # input-dependent B
                None,  # input-dependent C
                self.g_D_b.float(),
                delta_bias=self.g_dt_proj_b.bias.float(),
                delta_softplus=True,
            )

            r_out = mamba_inner_fn_no_out_proj(
                r_xz,
                self.r_conv1d.weight,
                self.r_conv1d.bias,
                self.r_x_proj.weight,
                self.r_dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.r_D.float(),
                delta_bias=self.r_dt_proj.bias.float(),
                delta_softplus=True,
            )

            r_out_b = mamba_inner_fn_no_out_proj(
                r_xz.flip([-1]),
                self.r_conv1d_b.weight,
                self.r_conv1d_b.bias,
                self.r_x_proj_b.weight,
                self.r_dt_proj_b.weight,
                A_b,
                None,  # input-dependent B
                None,  # input-dependent C
                self.r_D_b.float(),
                delta_bias=self.r_dt_proj_b.bias.float(),
                delta_softplus=True,
            )

            # 合并双向结果
            if not self.if_devide_out:
                g_out = F.linear(rearrange(g_out + g_out_b.flip([-1]), "b d l -> b l d"), self.g_out_proj.weight,
                                 self.g_out_proj.bias)
                r_out = F.linear(rearrange(r_out + r_out_b.flip([-1]), "b d l -> b l d"), self.r_out_proj.weight,
                                 self.r_out_proj.bias)
            else:
                g_out = F.linear(rearrange(0.5 * g_out + 0.5 * g_out_b.flip([-1]), "b d l -> b l d"),
                                 self.g_out_proj.weight, self.g_out_proj.bias)
                r_out = F.linear(rearrange(0.5 * r_out + 0.5 * r_out_b.flip([-1]), "b d l -> b l d"),
                                 self.r_out_proj.weight, self.r_out_proj.bias)
        # 层缩放（可选）
        if self.init_layer_scale is not None:
            g_out = g_out * self.g_gamma
            r_out = r_out * self.r_gamma
        return g_out, r_out

class CoBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 transformer_blocks,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=True,
                 drop=0.1,
                 attn_drop=0.1,
                 drop_path=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 # CoSSM-specific params:
                 # cossm_d_state: int = 16,
                 # cossm_d_conv: int = 5,
                 # cossm_expand: int = 2,
                 ):
        super().__init__()
        self.norm1g = norm_layer(dim)
        self.norm1r = norm_layer(dim)
        # self.cossm_encoder = CoSSM()
        self.trans = counter in transformer_blocks
        if self.trans:
            # self.trans = True
            self.att1g = Attention(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   qk_norm=qk_scale,
                                   attn_drop=attn_drop,
                                   proj_drop=drop,
                                   norm_layer=norm_layer)
            self.att1r = Attention(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   qk_norm=qk_scale,
                                   attn_drop=attn_drop,
                                   proj_drop=drop,
                                   norm_layer=norm_layer)
        else:

            self.cossm_encoder = CoSSM(d_model=dim,
                                       d_state=8,
                                       d_conv=3,
                                       expand=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2g = norm_layer(dim)
        self.norm2r = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(torch.ones(dim) * (layer_scale or 1.0), requires_grad=bool(layer_scale))
        self.gamma_2 = nn.Parameter(torch.ones(dim) * (layer_scale or 1.0), requires_grad=bool(layer_scale))
        self.gamma_3 = nn.Parameter(torch.ones(dim) * (layer_scale or 1.0), requires_grad=bool(layer_scale))
        self.gamma_4 = nn.Parameter(torch.ones(dim) * (layer_scale or 1.0), requires_grad=bool(layer_scale))

    def forward(self, g, r):
        # g->(N, L, D) r->(N, L, D)
        ori_g = g
        ori_r = r

        g = self.norm1g(g)
        r = self.norm1r(r)

        if self.trans:
            g = self.att1g(g)
            r = self.att1r(r)
        else:
            g, r = self.cossm_encoder(g, r)

        # 残差 & DropPath
        g = ori_g + self.drop_path(self.gamma_1 * g)
        r = ori_r + self.drop_path(self.gamma_2 * r)

        # MLP 分支
        g = g + self.drop_path(self.gamma_3 * self.mlp(self.norm2g(g)))
        r = r + self.drop_path(self.gamma_4 * self.mlp(self.norm2r(r)))

        return g, r

    # CoSSM 分支：加大状态维度 & 卷积核
    # self.cossm_encoder = CoSSM(
    #     d_model=dim,
    #     d_state=cossm_d_state,
    #     d_conv=cossm_d_conv,
    #     expand=cossm_expand,
    #     dt_rank="auto",
    #     dt_min=0.001, dt_max=0.1,
    #     dt_init="random", dt_scale=1.0,
    #     init_layer_scale=layer_scale
    # )


class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 downsample_temporal=False,
                 activation=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        layers = [
            nn.Conv2d(dim, dim_out,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False)
        ]

        if activation:
            layers += [
                nn.BatchNorm2d(dim_out, eps=1e-5),
                nn.ReLU(inplace=True),
            ]

        self.reduction = nn.Sequential(*layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        x = self.reduction(x) # → (B*T, dim_out, H//2, W//2)

        C2, H2, W2 = x.shape[1:]
        x = x.view(B, T, C2, H2, W2)

        return x


class VisionLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 ws_t,
                 num_heads,
                 window_size,
                 conv=False,
                 conv3d=True,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks=[],
                 activation=False,
                 **kwargs
                 ):
        super().__init__()
        # (B, 9, 96, 32, 32)
        self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale_conv)
                                     for i in range(depth)])

        self.downsample = None if not downsample else Downsample(dim=dim, activation=activation)
        # self.downsample = None if not downsample else nn.Conv3d(dim, dim * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
        #                             bias=False) # out->(N, dim*2, D_out, H_out, W_out) 后三位折半
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        # _, _, H, W = x.shape
        # (B, 9, 96, 32, 32)
        for i, blk in enumerate(self.blocks):
            x = blk(x) # output->(B, 9, C, 32, 32)
        if self.downsample is None:
            return x
        return self.downsample(x)


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x

def st_window_partition(x, ws_t, ws):
    # x shape: (B, T, C, H, W)   <— 正确
    B, T, C, H, W = x.shape
    # permute to (B, C, T, H, W)
    x = x.permute(0, 2, 1, 3, 4)

    # --- 1) pad so T,H,W 都能整除窗口 ---
    pad_t = (ws_t - T % ws_t) % ws_t
    pad_h = (ws  - H % ws ) % ws
    pad_w = (ws  - W % ws ) % ws
    x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))  # pad 格式: (..., W前,W后, H前,H后, T前,T后)
    T_p, H_p, W_p = T + pad_t, H + pad_h, W + pad_w

    # --- 2) 时空切块 ---
    # reshape: (B,C,T_p//ws_t, ws_t, H_p//ws,ws, W_p//ws,ws)
    x = x.view(
        B, C,
        T_p // ws_t, ws_t,
        H_p // ws,  ws,
        W_p // ws,  ws
    )
    # permute 到 (B, nt, nh, nw, ws_t, ws, ws, C)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
    # flatten 小块： (B*nt*nh*nw, ws_t*ws*ws, C)
    windows = x.reshape(-1, ws_t * ws * ws, C)
    return windows, (T_p, H_p, W_p)

def st_window_reverse(windows, ws_t, ws, pad_sizes, B):
    """
    Args:
        windows: (num_windows*B, ws_t*ws*ws, C)
        ws_t, ws: same as partition
        pad_sizes: (T_p, H_p, W_p) from partition
        B: original batch size
    Returns:
        x: (B, C, T_p, H_p, W_p) padded reconstruction
    """
    T_p, H_p, W_p = pad_sizes
    # 先恢复为 (B, nt, nh, nw, ws_t, ws, ws, C)
    nt, nh, nw = T_p // ws_t, H_p // ws, W_p // ws
    C = windows.size(-1)
    x = windows.reshape(B, nt, nh, nw, ws_t, ws, ws, C)
    # permute 回 (B, C, T_p//ws_t, ws_t, H_p//ws,ws, W_p//ws,ws)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    x = x.reshape(B, C, T_p, H_p, W_p)
    # permute to (B,T_p,C,H_p,W_p)
    x = x.permute(0, 2, 1, 3, 4)
    return x

class CoMambaVisionLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 ws_t,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks=[],
                 ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.ws_t = ws_t
        self.ws = window_size
        self.blocks = nn.ModuleList([
            CoBlock(dim=dim,
                    counter=i,
                    transformer_blocks=transformer_blocks,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale)
            for i in range(depth)])

        self.downsample_g = None if not downsample else Downsample(dim=dim)
        self.downsample_r = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        # self.window_size = window_size

    def forward(self, g, r):
        # input->(B, T=9, D, 4, 4)
        B, T, C, H, W = g.shape

        # pad_r = (self.window_size - W % self.window_size) % self.window_size
        # pad_b = (self.window_size - H % self.window_size) % self.window_size

        # --- 1) partition ---
        g_wins, pad_sizes = st_window_partition(g, self.ws_t, self.ws)
        r_wins, _ = st_window_partition(r, self.ws_t, self.ws)

        # if pad_r > 0 or pad_b > 0:
        #     g = torch.nn.functional.pad(g, (0, pad_r, 0, pad_b))
        #     r = torch.nn.functional.pad(r, (0, pad_r, 0, pad_b))
        #     _, _, Hp, Wp = g.shape
        # else:
        #     Hp, Wp = H, W

        # 窗口划分
        # g = window_partition(g, self.window_size)  # 形状变为 (num_windows*B, window_size^2, C)
        # r = window_partition(r, self.window_size)

        # --- 2) CoBlock 序列处理 ---
        for blk in self.blocks:
            g_wins, r_wins = blk(g_wins, r_wins)

        # 窗口还原
        # g = window_reverse(g, self.window_size, Hp, Wp)  # 恢复形状为 (B, C, Hp, Wp)
        # r = window_reverse(r, self.window_size, Hp, Wp)

        # --- 3) reverse ---
        g = st_window_reverse(g_wins, self.ws_t, self.ws, pad_sizes, B)
        r = st_window_reverse(r_wins, self.ws_t, self.ws, pad_sizes, B)

        # if pad_r > 0 or pad_b > 0:  # 裁剪填充部分
        #     g = g[:, :, :H, :W].contiguous()
        #     r = r[:, :, :H, :W].contiguous()

        # --- 4) 裁剪回原始 T,H,W ---
        T_p, H_p, W_p = pad_sizes
        g = g[:, :T, :, :H, :W].contiguous()
        r = r[:, :T, :, :H, :W].contiguous()

        # --- 5) 可选下采样 ---
        if self.downsample_g is not None and self.downsample_r is not None:
            g = self.downsample_g(g)
            r = self.downsample_r(r)

        return g, r


class EnSSM(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            bimamba_type="v2",
            if_devide_out=True,  # False
            init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        assert bimamba_type == 'v2'

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        if bimamba_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        conv_state, ssm_state = None, None

        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # Compute ∆ A B C D, the state space parameters.
        #     A, D 是独立于输入的 (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C 是依赖于输入的 (这是Mamba模型和 linear time invariant S4 的主要区别,这也是为什么Mamba被称为selective state spaces

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        if self.use_fast_path and inference_params is None:
            A_b = -torch.exp(self.A_b_log.float())

            out = mamba_inner_fn_no_out_proj(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            out_b = mamba_inner_fn_no_out_proj(
                xz.flip([-1]),
                self.conv1d_b.weight,
                self.conv1d_b.bias,
                self.x_proj_b.weight,
                self.dt_proj_b.weight,
                A_b,
                None,
                None,
                self.D_b.float(),
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
            )

            if not self.if_devide_out:
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight,
                               self.out_proj.bias)
            else:
                out = F.linear(rearrange(0.5 * out + 0.5 * out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight,
                               self.out_proj.bias)
        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out


class EnBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 transformer_blocks,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=True,
                 drop=0.1,
                 attn_drop=0.1,
                 drop_path=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=1e-6,
                 # EnSSM 超参
                 cossm_d_state: int = 16,
                 cossm_d_conv: int = 5,
                 cossm_expand: int = 2,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.trans = counter in transformer_blocks
        if self.trans:
            # Transformer 分支
            self.att = Attention(dim,
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 qk_norm=qk_scale,
                                 attn_drop=attn_drop,
                                 proj_drop=drop,
                                 norm_layer=norm_layer)
        else:
            # EnSSM 分支：使用更大容量的状态空间
            self.cossm_encoder = EnSSM(d_model=dim,
                                       d_state=8,
                                       d_conv=3,
                                       expand=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        # --- 1) 主分支（Attention or EnSSM） + 残差 & DropPath ---
        ori_x = x
        x = self.norm1(x)
        if self.trans:
            x = self.att(x)
        else:
            x = self.cossm_encoder(x)
        x = self.drop_path(self.gamma_1 * x)
        x = ori_x + x

        # --- 2) MLP 分支 + 残差 & DropPath ---
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class CrossModalAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 bias=True,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.att_g2r = nn.MultiheadAttention(embed_dim=dim,
                                             bias=bias,
                                             num_heads=num_heads,
                                              dropout=attn_drop, kdim=dim, vdim=dim)
        self.att_r2g = nn.MultiheadAttention(embed_dim=dim,
                                             bias=bias,
                                             num_heads=num_heads,
                                              dropout=attn_drop, kdim=dim, vdim=dim)
        self.proj = nn.Linear(2*dim, dim, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = nn.GELU()
    def forward(self, g, r):
        # g, r: (N, L, C) → transpose to (L, N, C)
        g_t, r_t = g.transpose(0,1), r.transpose(0,1)
        # GASF queries RP
        g2r, _ = self.att_g2r(query=g_t, key=r_t, value=r_t)  # (L,N,C)
        # RP queries GASF
        r2g, _ = self.att_r2g(query=r_t, key=g_t, value=g_t)
        # restore (N, L, C)
        g2r = g2r.transpose(0,1)
        r2g = r2g.transpose(0,1)
        # fuse cross-modal outputs
        x = torch.cat([g2r, r2g], dim=-1)  # (N, L, 2C)
        x = self.proj(x)                   # (N, L, C)
        x = self.proj_drop(x)
        x = self.act(x)
        return x


class EnMambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 ws_t,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks=[],
                 activation=False
                 ):

        super().__init__()
        transformer_blocks = transformer_blocks or []

        # —— 1) 融合前 / 后 LayerNorm + 1×1Conv ——
        # self.norm_fuse = nn.LayerNorm(2 * dim)
        # self.fuse = nn.Sequential(
        #     nn.Conv1d(2 * dim, dim, kernel_size=1, bias=False),
        #     nn.BatchNorm1d(dim, eps=1e-5),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(drop)
        # )

        # —— 2) Blocks 序列 ——
        self.blocks = nn.ModuleList([EnBlock(dim=dim,
                                             counter=i,
                                             transformer_blocks=transformer_blocks,
                                             num_heads=num_heads,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             drop=drop,
                                             attn_drop=attn_drop,
                                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                             layer_scale=layer_scale)
                                     for i in range(depth)])

        self.cross_attn = CrossModalAttention(dim=dim,
                                              num_heads=num_heads,
                                              bias=qkv_bias,
                                              # qk_scale=qk_scale,
                                              attn_drop=attn_drop,
                                              proj_drop=drop)

        # —— 3) 下采样 ——
        # self.fuse = nn.Conv1d(in_channels=2 * dim, out_channels=dim, kernel_size=1)
        self.downsample = None if not downsample else Downsample(dim=dim, activation=activation)
        self.do_gt = False
        # self.window_size = window_size
        self.ws_t = ws_t
        self.ws = window_size

    def forward(self, g, r):
        B, T, C, H, W = g.shape

        # --- 1) partition ---
        g_wins, pad_sizes = st_window_partition(g, self.ws_t, self.ws)
        r_wins, _ = st_window_partition(r, self.ws_t, self.ws)

        # 2) 拼接 & Fuse
        # x = torch.cat([g_wins, r_wins], dim=-1)  # (nB, L, 2C)
        # x = self.norm_fuse(x)
        # x = x.permute(0, 2, 1)                 # (nB, 2C, L)
        # x = self.fuse(x)                       # (nB, C, L)
        # x = x.permute(0, 2, 1)                 # (nB, L, C)
        x = self.cross_attn(g_wins, r_wins)

        for _, blk in enumerate(self.blocks):
            x = blk(x)

        # 4) 时空窗口还原
        x = st_window_reverse(x, self.ws_t, self.ws, pad_sizes, B)
        # 裁剪掉 pad
        T_p, H_p, W_p = pad_sizes
        x = x[:, :T, :, :H, :W].contiguous()

        # 5) 可选下采样（仅空间）
        if self.downsample is None:
            return x
        return self.downsample(x)


class BiMambaVision(nn.Module):
    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 ws_t,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path=0.2,
                 in_chans=32,
                 num_classes=2,
                 qkv_bias=True,
                 qk_scale=True,
                 drop_rate=0., # att->prj_drop
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 drop_path_rate=0.2,
                 flat_drop=0.1,
                 activation='Swish',  # EnSSM 块的激活函数
                 dropout=0.1,  # EnSSM 块的 dropout
                 causal=False,  # EnSSM 块的 causal
                 mamba_config=None,  # EnSSM 块的相关配置
                 **kwargs
                 ):
        super().__init__()
        self.ws_t = ws_t
        num_features = int(dim * 2 ** (len(depths) - 1))
        #
        self.patch_embed_g = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        self.patch_embed_r = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 生成线性递增的 Drop Path 概率列表
        self.levels_g = nn.ModuleList()
        self.levels_r = nn.ModuleList()
        # 第一，第二层，提取单模态特征
        for i in range(len(depths) - 2):
            conv = True if (i == 0 or i == 1) else False
            # (B, 9, 96, 32, 32)
            level_g = VisionLayer(dim=int(dim * 2 ** i),
                                depth=depths[i],
                                num_heads=num_heads[i],
                                ws_t=ws_t[i],
                                window_size=window_size[i],
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                conv=conv,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],  # 为每个阶段分配对应的 Drop Path 概率区间
                                downsample=(i < 3),
                                layer_scale=layer_scale,
                                layer_scale_conv=layer_scale_conv,
                                transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0
                                else list(range(depths[i] // 2, depths[i])),
                                **kwargs
                                )
            level_r = VisionLayer(dim=int(dim * 2 ** i),
                                depth=depths[i],
                                num_heads=num_heads[i],
                                ws_t=ws_t[i],
                                window_size=window_size[i],
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                conv=conv,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],  # 为每个阶段分配对应的 Drop Path 概率区间
                                downsample=(i < 3),
                                layer_scale=layer_scale,
                                layer_scale_conv=layer_scale_conv,
                                transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0
                                else list(range(depths[i] // 2, depths[i])),
                                **kwargs
                                )
            self.levels_g.append(level_g)
            self.levels_r.append(level_r)
        # 第三层，多模态特征融合 / 单模态注意力机制
        self.steps = nn.ModuleList()
        i = 2
        self.step = CoMambaVisionLayer(dim=int(dim * 2 ** i),
                                       depth=depths[i],
                                       num_heads=num_heads[i],
                                       ws_t=ws_t[i],
                                       window_size=window_size[i],
                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias,
                                       qk_scale=qk_scale,
                                       conv=conv,
                                       drop=drop_rate,
                                       attn_drop=attn_drop_rate,
                                       drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                       downsample=(i < 3),
                                       layer_scale=layer_scale,
                                       layer_scale_conv=layer_scale_conv,
                                       transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[
                                                                                                            i] % 2 != 0
                                       else list(range(depths[i] // 2, depths[i])),
                                       )

        self.steps.append(self.step)
        # 第四层，对拼接特征的处理
        self.en_layers = nn.ModuleList()
        i = 3
        self.en_layer = EnMambaVisionLayer(dim=int(dim * 2 ** i),
                                       depth=depths[i],
                                       num_heads=num_heads[i],
                                       ws_t=ws_t[i],
                                       window_size=window_size[i],
                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias,
                                       qk_scale=qk_scale,
                                       conv=conv,
                                       drop=drop_rate,
                                       attn_drop=attn_drop_rate,
                                       drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                       downsample=(i < 3),
                                       layer_scale=layer_scale,
                                       layer_scale_conv=layer_scale_conv,
                                       transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[
                                                                                                            i] % 2 != 0
                                       else list(range(depths[i] // 2, depths[i])),
                                       )
        self.en_layers.append(self.en_layer)
        # (B, T_p, C, H_p, W_p)
        self.norm = nn.BatchNorm3d(num_features)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flat_drop = nn.Dropout(p=flat_drop)
        self.classifier = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, g, r):
        g = self.patch_embed_g(g)
        r = self.patch_embed_r(r)
        # 单模态特征提取
        for level_g, level_r in zip(self.levels_g, self.levels_r):
            g = level_g(g)
            r = level_r(r)

        # 多模态特征融合 (SSM) / 单模态注意力机制 (Attention)
        # g, r = self.steps(g, r)
        for step in self.steps:  # self.steps 存放唯一的 layer
            g, r = step(g, r)

        x = None
        for en_layer in self.en_layers:
            x = en_layer(g, r)
        # (B, T_p, C, H_p, W_p)

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.norm(x)
        x = self.avgpool(x)  # (B, C,1,1,1)
        x = torch.flatten(x, 1)  # (B, C)
        x = self.flat_drop(x)
        logits = self.classifier(x)  # (B, num_classes)

        return logits

device = torch.device("cuda")

# === 3) 随便给一套超参（够小，确保几 GB 显存就能跑） ===
model = BiMambaVision(
    dim=96,           # stage-0 embed dim
    in_dim=64,        # PatchEmbed 中间维度
    depths=[1, 1, 2, 1],          # 每个 stage 的 block 数
    ws_t=[3, 3, 3, 3],            # temporal window
    window_size=[7, 7, 7, 7],     # spatial window
    mlp_ratio=4,
    num_heads=[3, 3, 6, 12],      # 每 stage 多头数，需能整除各自 dim
    in_chans=32,
    num_classes=2,
    layer_scale=1e-6,
).to(device).eval()               # eval() 关闭 Dropout/BN 动测，更快

# === 4) 构造假输入：两模态 (g, r) ===
B, T, C, H, W = 2, 4, 32, 128, 128   # batch=2、4 帧、32 通道、128×128
dummy_g = torch.randn(B, T, C, H, W, device=device)
dummy_r = torch.randn(B, T, C, H, W, device=device)

# === 5) 前向 & 计时 ===
with torch.no_grad():
    torch.cuda.synchronize()
    # t0 = time.time()
    logits = model(dummy_g, dummy_r)   # forward
    torch.cuda.synchronize()
    # dt = time.time() - t0

print(f"Output shape: {logits.shape}")   # 预期 (B, num_classes)
# print(f"Forward time: {dt*1000:.1f} ms") # 每个 batch 所耗毫秒
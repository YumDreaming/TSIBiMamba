import argparse

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


# 输入嵌入模块
class PatchEmbed(nn.Module):
    def __init__(self,
                 in_chans=32,
                 in_dim=64,
                 dim=128
                 # img_size=224, patch_size=16, in_chans=3, embed_dim=768
                 ):
        super().__init__()
        self.proj = nn.Identity()
        # 这里的参数需要调优
        self.conv = nn.Sequential(
            nn.Conv3d(in_chans, in_dim, kernel_size=(3, 3, 3),
                      stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-5),
            nn.ReLU(),
            nn.Conv3d(in_dim, dim, kernel_size=(3, 3, 3),
                      stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(dim, eps=1e-5),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv(x)
        return x


# 卷积块
class ConvBlock(nn.Module):
    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=(3, 3, 3),):
        super().__init__()
        padding = tuple(k//2 for k in kernel_size)
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=padding)
        self.norm1 = nn.BatchNorm3d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=padding)
        self.norm2 = nn.BatchNorm3d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1, 1)
        x = input + self.drop_path(x)
        return x


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
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, g, r):
        ori_g = g
        ori_r = r

        g = self.norm1g(g)
        r = self.norm1r(r)

        if self.trans:
            g = self.att1g(g)
            r = self.att1r(r)
        else:
            g, r = self.cossm_encoder(g, r)

        # g = self.drop_path(self.gamma_1 * g)
        # r = self.drop_path(self.gamma_2 * r)
        #
        # g = ori_g + g
        # r = ori_r + r
        #
        # g = g + self.drop_path(self.gamma_2 * self.mlp(self.norm2g(g)))
        # r = r + self.drop_path(self.gamma_2 * self.mlp(self.norm2r(r)))

        # 残差 & DropPath
        g = ori_g + self.drop_path(self.gamma_1.view(1, -1) * g)
        r = ori_r + self.drop_path(self.gamma_2.view(1, -1) * r)

        # MLP 分支
        g = g + self.drop_path(self.gamma_2.view(1, -1) * self.mlp(self.norm2g(g)))
        r = r + self.drop_path(self.gamma_2.view(1, -1) * self.mlp(self.norm2r(r)))

        return g, r


class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 downsample_temporal=False,
                 activation=True,
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
        if downsample_temporal:
            kernel = (3, 3, 3)
            stride = (2, 2, 2)
            padding = (1, 1, 1)
        else: # 进入这里
            kernel = (1, 3, 3)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
        # self.reduction = nn.Sequential(
        #     nn.Conv3d(dim, dim_out, kernel_size=kernel, stride=stride, padding=pad, bias=False),
        #     nn.BatchNorm3d(dim_out, eps=1e-5),
        #     nn.ReLU(inplace=True),
        # )
        layers = [
            nn.Conv3d(dim, dim_out,
                      kernel_size=kernel,
                      stride=stride,
                      padding=padding,
                      bias=False)
        ]

        if activation:
            layers += [
                nn.BatchNorm3d(dim_out, eps=1e-5),
                nn.ReLU(inplace=True),
            ]

        self.reduction = nn.Sequential(*layers)

    def forward(self, x):
        x = self.reduction(x)
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
                 **kwargs
                 ):
        super().__init__()
        self.conv = conv
        self.conv3d = conv3d
        self.transformer_block = False

        self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale_conv)
                                     for i in range(depth)])
        self.transformer_block = False
        # self.downsample = None if not downsample else Downsample(dim=dim)
        self.downsample = None if not downsample else nn.Conv3d(dim, dim * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                                    bias=False) # out->(N, dim*2, D_out, H_out, W_out) 后三位折半
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        # _, _, H, W = x.shape
        for i, blk in enumerate(self.blocks):
            x = blk(x)
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
    """
    Args:
        x: (B, C, T, H, W)
        ws_t: temporal window size (e.g. 3 frames)
        ws: spatial window size (e.g. 7 pixels)
    Returns:
        windows: (num_windows*B, ws_t*ws*ws, C)
        (T_pad, H_pad, W_pad): padded sizes for later还原
    """
    B, C, T, H, W = x.shape

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
    x = windows.reshape(B, nt, nh, nw, ws_t, ws, ws, -1)
    # permute 回 (B, C, T_p//ws_t, ws_t, H_p//ws,ws, W_p//ws,ws)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    # 最后 reshape 回时空立方体 (B, C, T_p, H_p, W_p)
    x = x.reshape(B, -1, T_p, H_p, W_p)
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

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        # self.window_size = window_size

    def forward(self, g, r):
        B, C, T, H, W = g.shape

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
        g = g[..., :T, :H, :W].contiguous()
        r = r[..., :T, :H, :W].contiguous()

        # --- 5) 可选下采样 ---
        if self.downsample is not None:
            g = self.downsample(g)
            r = self.downsample(r)

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
        transformer_blocks = transformer_blocks or []

        # —— 1) 融合前 / 后 LayerNorm + 1×1Conv ——
        self.norm_fuse = nn.LayerNorm(2 * dim)
        self.fuse = nn.Sequential(
            nn.Conv1d(2 * dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim, eps=1e-5),
            nn.ReLU(inplace=True),
        )

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

        # —— 3) 下采样 ——
        # self.fuse = nn.Conv1d(in_channels=2 * dim, out_channels=dim, kernel_size=1)
        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        # self.window_size = window_size
        self.ws_t = ws_t
        self.ws = window_size

    def forward(self, g, r):
        B, C, T, H, W = g.shape

        # --- 1) partition ---
        g_wins, pad_sizes = st_window_partition(g, self.ws_t, self.ws)
        r_wins, _ = st_window_partition(r, self.ws_t, self.ws)

        # 2) 拼接 & Fuse
        x = torch.cat([g_wins, r_wins], dim=-1)  # (nB, L, 2C)
        x = self.norm_fuse(x)
        x = x.permute(0, 2, 1)                 # (nB, 2C, L)
        x = self.fuse(x)                       # (nB, C, L)
        x = x.permute(0, 2, 1)                 # (nB, L, C)

        for _, blk in enumerate(self.blocks):
            x = blk(x)

        # 4) 时空窗口还原
        x = st_window_reverse(x, self.ws_t, self.ws, pad_sizes, B)
        # 裁剪掉 pad
        T_p, H_p, W_p = pad_sizes
        x = x[..., :T, :H, :W].contiguous()         # back to (B, C, T, H, W)

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
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 生成线性递增的 Drop Path 概率列表
        self.levels = nn.ModuleList()
        # 第一，第二层，提取单模态特征
        for i in range(len(depths) - 2):
            conv = True if (i == 0 or i == 1) else False
            level = VisionLayer(dim=int(dim * 2 ** i),
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
            self.levels.append(level)
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
        self.lists = nn.ModuleList()
        i = 3
        self.list = EnMambaVisionLayer(dim=int(dim * 2 ** i),
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
        self.lists.append(self.list)
        self.norm = nn.BatchNorm3d(num_features)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flat_drop = nn.Dropout(p=flat_drop)
        self.classifier = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, g, r):
        # (28, 32, 224, 224)
        g = self.patch_embed(g)
        r = self.patch_embed(r)
        # 单模态特征提取
        for level in self.levels:
            g = level(g)
            r = level(r)

        # 多模态特征融合 (SSM) / 单模态注意力机制 (Attention)
        # g, r = self.steps(g, r)
        for step in self.steps:
            g, r = step(g, r)

        for list in self.lists:
            x = list(g, r)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.flat_drop(x)
        logits = self.classifier(x)

        return logits

class MVDataset(Dataset):
    """
    自定义多视图 EEG 数据集，加载 GASF 和 RP 图像及其标签
    :param root_dir:
    """
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.entries = []  # 存储 (gasf_path, rp_path, label)

        # GASF 路径
        gasf_root = os.path.join(root_dir, 'GASF')
        # 遍历每个受试者子目录
        for subj in os.listdir(gasf_root):
            subj_g_dir = os.path.join(gasf_root, subj)
            if not os.path.isdir(subj_g_dir):
                continue
            # 构造标签文件路径，如 s01_labels.txt
            label_file = os.path.join(subj_g_dir, f"{subj}_labels.txt")
            # 逐行读取标签和文件名
            with open(label_file, 'r') as f:
                for line in f:
                    fname, lbl = line.strip().split(',')  # 文件名, 标签
                    gasf_path = os.path.join(subj_g_dir, fname)
                    rp_path = gasf_path.replace(
                        os.path.join(root_dir, 'GASF'),
                        os.path.join(root_dir, 'RP')
                    ).replace('_GASF.npy', '_RP.npy')
                    self.entries.append((gasf_path, rp_path, int(lbl)))

    def __len__(self):
        """返回样本总数"""
        return len(self.entries)

    def __getitem__(self, idx):
        """
        根据索引 idx 返回一条数据：
          g: GASF 图像张量，形状 (C, T, H, W)
          r: RP   图像张量，形状 (C, T, H, W)
          label: 类别标签
        """
        gasf_path, rp_path, label = self.entries[idx]
        # 加载 NumPy 数据
        g = np.load(gasf_path)  # GASF 数据
        r = np.load(rp_path)    # RP 数据
        # 转为 PyTorch 张量
        g = torch.from_numpy(g).float()
        r = torch.from_numpy(r).float()
        y = torch.tensor(label, dtype=torch.long)
        return g, r, y

def train_epoch(model, loader, criterion, optimizer, device):
    """
    执行一个训练 epoch，返回平均 loss 和准确率
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 展示进度条
    # for g, r, labels in tqdm(loader, desc='训练', leave=False):
    #     g, r, labels = g.to(device), r.to(device), labels.to(device)
    #     optimizer.zero_grad()
    #     # 双分支模型输入
    #     outputs = model(g, r)
    #     loss = criterion(outputs, labels)
    #     loss.backward()
    #     optimizer.step()
    #
    #     total_loss += loss.item() * labels.size(0)
    #     _, preds = outputs.max(dim=1)
    #     correct += preds.eq(labels).sum().item()
    #     total += labels.size(0)

    # 使用 tqdm 展示进度条
    loop = tqdm(loader, desc='训练', leave=False)
    for g, r, labels in loop:
        g, r, labels = g.to(device), r.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(g, r)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        # 每个 batch 后更新进度条上的 loss 和 acc
        cur_loss = total_loss / total
        cur_acc = correct / total * 100
        loop.set_postfix(loss=f'{cur_loss:.4f}', acc=f'{cur_acc:.2f}%')

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def eval_epoch(model, loader, criterion, device):
    """
    执行一个验证 epoch，返回平均 loss 和准确率
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # with torch.no_grad():
    #     for g, r, labels in tqdm(loader, desc='验证', leave=False):
    #         g, r, labels = g.to(device), r.to(device), labels.to(device)
    #         outputs = model(g, r)
    #         loss = criterion(outputs, labels)
    #
    #         total_loss += loss.item() * labels.size(0)
    #         _, preds = outputs.max(dim=1)
    #         correct += preds.eq(labels).sum().item()
    #         total += labels.size(0)

    loop = tqdm(loader, desc='验证', leave=False)
    with torch.no_grad():
        for g, r, labels in loop:
            g, r, labels = g.to(device), r.to(device), labels.to(device)
            outputs = model(g, r)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            cur_loss = total_loss / total
            cur_acc = correct / total * 100
            loop.set_postfix(val_loss=f'{cur_loss:.4f}', val_acc=f'{cur_acc:.2f}%')

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def main(args):
    # 检查 CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据集
    dataset = MVDataset(args.data_root)
    # 分割训练/验证集，保持标签分布一致
    indices = list(range(len(dataset)))
    labels = [lbl for _, _, lbl in dataset.entries]
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 构建模型
    model = BiMambaVision(
        depths=args.depths,
        num_heads=args.num_heads,
        ws_t=args.ws_t,
        window_size=args.window_size,
        dim=args.dim,
        in_dim=args.in_dim,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate,
        num_classes=args.num_classes,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        layer_scale=args.layer_scale,
        layer_scale_conv=args.layer_scale_conv,
        qkv_bias=args.qkv_bias,
        qk_scale=args.qk_scale,
        in_chans=args.in_chans,
        flat_drop=args.flat_drop
    ).to(device)

    # 损失函数、优化器与学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # 训练历史记录
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = eval_epoch(model, val_loader, criterion, device)
        # 根据验证准确率调整学习率
        scheduler.step(val_acc)

        # 保存历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印本轮结果
        print(f"训练 Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"验证 Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

    # 训练结束后绘制并保存曲线
    epochs = range(1, args.epochs + 1)
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')

    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('acc_curve.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiMambaVision 模型训练脚本')
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/deap_pic',
                        help='预处理后 GASF/RP 数据的根目录')
    parser.add_argument('--batch_size', type=int, default=20, help='每个 batch 的样本数')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--in_chans', type=int, default=32)
    parser.add_argument('--depths', nargs='+', type=int, default=[1, 3, 8, 4])
    parser.add_argument('--num_heads', nargs='+', type=int, default=[2, 4, 8, 16])
    parser.add_argument('--ws_t', nargs='+', type=int, default=[3, 3, 3, 3])
    parser.add_argument('--window_size', nargs='+', type=int, default=[4, 4, 8, 4])
    parser.add_argument('--dim', type=int, default=96)
    parser.add_argument('--in_dim', type=int, default=64)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--drop_path_rate', type=float, default=0.6)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--attn_drop_rate', type=float, default=0.2)
    parser.add_argument('--layer_scale', type=float, default=1e-6)
    parser.add_argument('--layer_scale_conv', type=float, default=1e-6)
    parser.add_argument('--qkv_bias', action='store_true', default=True)
    parser.add_argument('--qk_scale', action='store_true', default=True)
    parser.add_argument('--flat_drop', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
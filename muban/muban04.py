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
from sklearn.model_selection import GroupShuffleSplit

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
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-5),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
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
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
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
            x = x * self.gamma.view(1, -1, 1, 1)
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

    # def step(self, g_hidden_states, g_conv_state, g_ssm_state, r_hidden_states, r_conv_state, r_ssm_state):
    #     dtype = g_hidden_states.dtype
    #     assert g_hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
    #     g_xz = self.g_in_proj(g_hidden_states.squeeze(1))  # (B 2D)
    #     g_x, g_z = g_xz.chunk(2, dim=-1)  # (B D)
    #
    #     r_xz = self.r_in_proj(r_hidden_states.squeeze(1))  # (B 2D)
    #     r_x, r_z = r_xz.chunk(2, dim=-1)  # (B D)
    #
    #     # Conv step
    #     if causal_conv1d_update is None:
    #         g_conv_state.copy_(torch.roll(g_conv_state, shifts=-1, dims=-1))  # Update state (B D W)
    #         g_conv_state[:, :, -1] = g_x
    #         g_x = torch.sum(g_conv_state * rearrange(self.g_conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
    #         if self.g_conv1d.bias is not None:
    #             g_x = g_x + self.conv1d.bias
    #         g_x = self.act(g_x).to(dtype=dtype)
    #
    #         r_conv_state.copy_(torch.roll(r_conv_state, shifts=-1, dims=-1))  # Update state (B D W)
    #         r_conv_state[:, :, -1] = r_x
    #         r_x = torch.sum(r_conv_state * rearrange(self.r_conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
    #         if self.r_conv1d.bias is not None:
    #             r_x = r_x + self.conv1d.bias
    #         r_x = self.act(r_x).to(dtype=dtype)
    #     other:
    #         g_x = causal_conv1d_update(
    #             g_x,
    #             g_conv_state,
    #             rearrange(self.a_conv1d.weight, "d 1 w -> d w"),
    #             self.g_conv1d.bias,
    #             self.activation,
    #         )
    #
    #         r_x = causal_conv1d_update(
    #             r_x,
    #             r_conv_state,
    #             rearrange(self.r_conv1d.weight, "d 1 w -> d w"),
    #             self.r_conv1d.bias,
    #             self.activation,
    #         )
    #
    #     g_x_db = self.g_x_proj(g_x)  # (B dt_rank+2*d_state)
    #     g_dt, g_B, g_C = torch.split(g_x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    #     # Don't add dt_bias here
    #     g_dt = F.linear(g_dt, self.g_dt_proj.weight)  # (B d_inner)
    #
    #     r_x_db = self.r_x_proj(r_x)  # (B dt_rank+2*d_state)
    #     r_dt, r_B, r_C = torch.split(r_x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    #     # Don't add dt_bias here
    #     r_dt = F.linear(r_dt, self.r_dt_proj.weight)  # (B d_inner)
    #
    #     A = -torch.exp(self.A_log.float())  # (d_inner, d_state)


class CoBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 transformer_blocks,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1g = norm_layer(dim)
        self.norm1r = norm_layer(dim)
        # self.cossm_encoder = CoSSM()
        self.trans = False
        if counter in transformer_blocks:
            self.trans = True
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

        g = self.drop_path(self.gamma_1 * g)
        r = self.drop_path(self.gamma_2 * r)

        g = ori_g + g
        r = ori_r + r

        g = g + self.drop_path(self.gamma_2 * self.mlp(self.norm2g(g)))
        r = r + self.drop_path(self.gamma_2 * self.mlp(self.norm2r(r)))
        return g, r


class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
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
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class VisionLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
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
        super().__init__()
        self.conv = conv
        self.transformer_block = False

        self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale_conv)
                                     for i in range(depth)])
        self.transformer_block = False
        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
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


class CoMambaVisionLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
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
        self.window_size = window_size

    def forward(self, g, r):
        _, _, H, W = g.shape

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        if pad_r > 0 or pad_b > 0:
            g = torch.nn.functional.pad(g, (0, pad_r, 0, pad_b))
            r = torch.nn.functional.pad(r, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = g.shape
        else:
            Hp, Wp = H, W

        # 窗口划分
        g = window_partition(g, self.window_size)  # 形状变为 (num_windows*B, window_size^2, C)
        r = window_partition(r, self.window_size)

        for blk in self.blocks:
            g, r = blk(g, r)

        # 窗口还原
        g = window_reverse(g, self.window_size, Hp, Wp)  # 恢复形状为 (B, C, Hp, Wp)
        r = window_reverse(r, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:  # 裁剪填充部分
            g = g[:, :, :H, :W].contiguous()
            r = r[:, :, :H, :W].contiguous()
        if self.downsample is None:
            return g, r
        return self.downsample(g), self.downsample(r)


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
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.trans = False
        if counter in transformer_blocks:
            self.trans = True
            self.att = Attention(dim,
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 qk_norm=qk_scale,
                                 attn_drop=attn_drop,
                                 proj_drop=drop,
                                 norm_layer=norm_layer)
        else:
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
        # x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        # x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        ori_x = x
        x = self.norm1(x)
        if self.trans:
            x = self.att(x)
        else:
            x = self.cossm_encoder(x)
        x = self.drop_path(self.gamma_1 * x)
        x = ori_x + x
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

        self.fuse = nn.Conv1d(in_channels=2 * dim, out_channels=dim, kernel_size=1)
        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, g, r):
        _, _, H, W = g.shape

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        if pad_r > 0 or pad_b > 0:
            g = torch.nn.functional.pad(g, (0, pad_r, 0, pad_b))
            r = torch.nn.functional.pad(r, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = g.shape
        else:
            Hp, Wp = H, W
        g = window_partition(g, self.window_size)
        r = window_partition(r, self.window_size)

        x = torch.cat([g, r], dim=-1)  # 拼接双路特征 (B, L, 2*C)
        x = x.permute(0, 2, 1)  # 调整为 (B, 2*C, L)
        x = self.fuse(x)  # 1x1卷积降维 (B, C, L)
        x = x.permute(0, 2, 1)  # 恢复 (B, L, C)

        for _, blk in enumerate(self.blocks):
            x = blk(x)

        x = window_reverse(x, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()
        if self.downsample is None:
            return x
        return self.downsample(x)


class BiMambaVision(nn.Module):
    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path=0.2,
                 in_chans=32,
                 num_classes=2,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 drop_path_rate=0.2,
                 activation='Swish',  # EnSSM 块的激活函数
                 dropout=0.1,  # EnSSM 块的 dropout
                 causal=False,  # EnSSM 块的 causal
                 mamba_config=None,  # EnSSM 块的相关配置
                 **kwargs
                 ):
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.patch_embed = PatchEmbed(in_chans=in_chans, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 生成线性递增的 Drop Path 概率列表
        self.levels = nn.ModuleList()
        # 第一，第二层，提取单模态特征
        for i in range(len(depths) - 2):
            conv = True if (i == 0 or i == 1) else False
            level = VisionLayer(dim=int(dim * 2 ** i),
                                depth=depths[i],
                                num_heads=num_heads[i],
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
                                )
            self.levels.append(level)
        # 第三层，多模态特征融合 / 单模态注意力机制
        self.steps = nn.ModuleList()
        i = 2
        self.step = CoMambaVisionLayer(dim=int(dim * 2 ** i),
                                       depth=depths[i],
                                       num_heads=num_heads[i],
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
        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
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
        logits = self.classifier(x)

        return logits

class DEAPWindowDataset(Dataset):
    def __init__(self, gasf_root, rp_root, transform=None):
        self.samples = []
        # 遍历各被试子文件夹
        for subj in sorted(os.listdir(gasf_root)):
            g_subdir = os.path.join(gasf_root, subj)
            r_subdir = os.path.join(rp_root, subj)
            labels_txt = os.path.join(g_subdir, f"{subj}_labels.txt")
            label_map = {}
            with open(labels_txt, 'r') as f:
                for line in f:
                    fname, lbl = line.strip().split(',')
                    label_map[fname] = int(lbl)
            # 对每个 trial 和 window 生成样本
            for fname, lbl in label_map.items():
                g_path = os.path.join(g_subdir, fname)
                r_path = os.path.join(r_subdir, fname.replace('_GASF.npy', '_RP.npy'))
                # 使用 mmap_mode 读取减少一次性内存开销
                arr = np.load(g_path, mmap_mode='r')
                n_windows = arr.shape[1]
                for w in range(n_windows):
                    # self.samples.append((g_path, r_path, lbl, w))
                    self.samples.append((g_path, r_path, lbl, w, subj))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # g_path, r_path, lbl, w = self.samples[idx]
        g_path, r_path, lbl, w, subj = self.samples[idx]
        # mmap_mode 下只加载必要数据片段
        g_arr = np.load(g_path, mmap_mode='r')
        r_arr = np.load(r_path, mmap_mode='r')
        g = g_arr[:, w].astype(np.float32) / 255.0
        r = r_arr[:, w].astype(np.float32) / 255.0
        # 转 tensor
        g_tensor = torch.from_numpy(g)
        r_tensor = torch.from_numpy(r)
        if self.transform:
            g_tensor = self.transform(g)
            r_tensor = self.transform(r)
        # return g_tensor, r_tensor, lbl
        return g_tensor, r_tensor, lbl, subj

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    # tqdm 进度条展示训练状态
    pbar = tqdm(loader, desc="Train", leave=False)
    for g, r, lbl, _ in pbar:
        g, r, lbl = g.to(device, non_blocking=True), r.to(device, non_blocking=True), lbl.to(device)
        optimizer.zero_grad()
        # 混合精度前向
        with autocast():
            outputs = model(g, r)
            loss = criterion(outputs, lbl)
        # 混合精度反向与更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # 统计
        batch_size = g.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == lbl).sum().item()
        total += batch_size
        pbar.set_postfix(loss=total_loss/total, acc=100*correct/total)
    return total_loss/total, 100*correct/total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Eval ", leave=False)
    with torch.no_grad():
        for g, r, lbl, _ in pbar:
            g, r, lbl = g.to(device, non_blocking=True), r.to(device, non_blocking=True), lbl.to(device)
            with autocast():
                outputs = model(g, r)
                loss = criterion(outputs, lbl)
            batch_size = g.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == lbl).sum().item()
            total += batch_size
            pbar.set_postfix(loss=total_loss/total, acc=100*correct/total)
    return total_loss/total, 100*correct/total

def main():
    # 数据路径: 使用数据盘 /root/autodl-tmp
    gasf_root = '/root/autodl-tmp/deap_pic/GASF'
    rp_root   = '/root/autodl-tmp/deap_pic/RP'

    # 超参数
    batch_size = 32
    num_epochs = 20
    lr = 1e-4
    in_chans = 32
    depths = [1, 3, 8, 4]
    num_heads = [2, 4, 8, 16]
    window_size = [8, 8, 14, 7]
    dim = 128
    in_dim = 64
    mlp_ratio = 4
    resolution = 224
    drop_path_rate = 0.4
    num_classes = 2

    # 设备检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 构建数据集与加载器，pin_memory 提升 GPU 数据加载效率，persistent_workers 持续 worker
    dataset = DEAPWindowDataset(gasf_root, rp_root)
    # n_train = int(len(dataset) * 0.9)
    # n_test  = len(dataset) - n_train
    # train_ds, test_ds = random_split(dataset, [n_train, n_test])
    groups = [s[-1] for s in dataset.samples]  # subj 列
    gss = GroupShuffleSplit(test_size=0.1, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(dataset.samples, groups=groups))
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)

    # 模型、损失、优化器、调度器与 AMP
    model = BiMambaVision(depths=depths,
                          num_heads=num_heads,
                          window_size=window_size,
                          dim=dim,
                          in_dim=in_dim,
                          mlp_ratio=mlp_ratio,
                          resolution=resolution,
                          drop_path_rate=drop_path_rate,
                          num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 根据验证 loss 自动调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler()  # AMP 缩放器

    # 训练与评估循环
    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        test_loss, test_acc   = evaluate(model, test_loader, criterion, device)
        # 调度器 step
        scheduler.step(test_loss)
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

if __name__ == '__main__':
    main()  # 入口



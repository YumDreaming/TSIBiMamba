import math

from timm.models.vision_transformer import Mlp, PatchEmbed
from torch import nn
import torch
from timm.models.layers import DropPath
import torch.nn.functional as F
from einops import rearrange, repeat
from pre_test.selective_scan_interface import mamba_inner_fn_no_out_proj
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

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

class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
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
            self.r_gammagamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

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

        self.g_D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.g_D_b._no_weight_decay = True
        self.r_D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.r_D_b._no_weight_decay = True

        self.g_out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.r_out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, g_hidden_states, r_hidden_states, g_inference_params=None, r_inference_params=None):
        assert g_hidden_states.shape == r_hidden_states.shape
        batch, seqlen, dim = g_hidden_states.shape
        g_conv_state, g_ssm_state = None, None
        r_conv_state, r_ssm_state = None, None

        # gasf branch
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

        A = -torch.exp(self.A_log.float())
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and g_inference_params is None and r_inference_params is None:  # Doesn't support outputting the states
            A_b = -torch.exp(self.A_b_log.float())
            g_out = mamba_inner_fn_no_out_proj(
                g_xz,
                self.g_conv1d.weight,
                self.g_conv1d.bias,
                self.g_x_proj.weight,
                self.g_dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.g_D.float(),
                delta_bias=self.g_dt_proj.bias.float(),
                delta_softplus=True,
            )

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

            if not self.if_devide_out:
                g_out = F.linear(rearrange(g_out + g_out_b.flip([-1]), "b d l -> b l d"), self.g_out_proj.weight, self.g_out_proj.bias)
                r_out = F.linear(rearrange(r_out + r_out_b.flip([-1]), "b d l -> b l d"), self.r_out_proj.weight, self.r_out_proj.bias)
            else:
                g_out = F.linear(rearrange(0.5*g_out + 0.5*g_out_b.flip([-1]), "b d l -> b l d"), self.g_out_proj.weight, self.g_out_proj.bias)
                r_out = F.linear(rearrange(0.5*r_out + 0.5*r_out_b.flip([-1]), "b d l -> b l d"), self.r_out_proj.weight, self.r_out_proj.bias)
        if self.init_layer_scale is not None:
            g_out = g_out * self.gamma
            r_out = r_out * self.gamma
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

    def forward(self, g, r):
        g = self.norm1g(g)
        r = self.norm1r(r)
        if self.trans:
            g = self.att1g(g)
            r = self.att1r(r)
        else:
            g, r = self.cossm_encoder(g, r)


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
                 transformer_blocks = [],
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
        self.transformer_block = False
        self.blocks = nn.ModuleList([CoBlock(dim=dim,
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
        self.transformer_block = True

class BiMambaVision(nn.module):
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
                 **kwargs
                 ):
        super.__init__()
        self.patch_embed = PatchEmbed(in_chans=in_chans, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)-2):
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
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < 3),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0
                                        else list(range(depths[i] // 2, depths[i])),
                                     )
            self.levels.append(level)
        self.steps = nn.ModuleList()
        i = 3
        step = CoMambaVisionLayer(   dim=int(dim * 2 ** i),
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
                                     transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0
                                        else list(range(depths[i] // 2, depths[i])),
                                  )
    def forward(self, g, r):
        # (28, 32, 224, 224)
        g = self.patch_embed(g)
        r = self.patch_embed(r)
        for level in self.levels:
            g = level(g)
            r = level(r)


in_chans=32
depths = [1, 3, 8, 4]
num_heads = [2, 4, 8, 16]
window_size = [8, 8, 14, 7]
dim = 128
in_dim=64
mlp_ratio = 4
resolution = 224
drop_path_rate = 0.2
# model = BiMambaVision(in_chans, dim, drop_path_rate, depths)
model = BiMambaVision(depths=depths,
                      num_heads=num_heads,
                      window_size=window_size,
                      dim=dim,
                      in_dim=in_dim,
                      mlp_ratio=mlp_ratio,
                      resolution=resolution,
                      drop_path_rate=drop_path_rate)
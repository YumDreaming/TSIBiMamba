#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import warnings
import math
import torch
import matplotlib.pyplot as plt
import torchaudio.models as tm

# 忽略第三方库过期警告，启用 cuDNN 自动调优
warnings.filterwarnings("ignore", "Importing from timm.models.layers is deprecated")
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 测试序列长度
SEQ_LENGTHS = [2000, 5000, 10000, 15000, 20000, 24000]
NUM_CLASSES = 7
# 为混合精度创建上下文
AMP = torch.cuda.amp.autocast

# MAET 参数
MAET_BASE_KWARGS = dict(
    eeg_dim=310, eye_dim=33, num_classes=NUM_CLASSES,
    embed_dim=32, depth=3, num_heads=4,
    mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0
)

# —— 导入已有模型 & BaseCNN ——
from shengji07_SEED_VII_7_4mat import BaseCNN, BiMambaVision
from shengji07_SEED_VII_7_Transformer_02 import BiTransformerVision
from maet import MAET

# —— ConformerVision 定义 ——
class ConformerVision(torch.nn.Module):
    def __init__(self, max_len, d_model, conformer_args):
        super().__init__()
        self.base_cnn_g = BaseCNN()
        self.base_cnn_r = BaseCNN()
        # Conformer 编码器
        self.conformer = tm.Conformer(
            input_dim=d_model,
            num_heads=conformer_args["num_heads"],
            ffn_dim=conformer_args["ffn_dim"],
            num_layers=conformer_args["num_layers"],
            depthwise_conv_kernel_size=conformer_args["kernel_size"],
            dropout=conformer_args["dropout"],
            use_group_norm=False,
            convolution_first=False
        )
        # 最大位置嵌入长度，用于 slice
        self.max_len = max_len
        # 分类头
        self.classifier = torch.nn.Linear(d_model, NUM_CLASSES)

    def forward(self, g, r):
        B, T, C, H, W = g.shape
        # CNN 特征
        feats_g = self.base_cnn_g(g.view(B*T, C, H, W)).view(B, T, -1)
        feats_r = self.base_cnn_r(r.view(B*T, C, H, W)).view(B, T, -1)
        x = torch.cat([feats_g, feats_r], dim=-1)  # (B,T,d_model)
        # 生成 lengths
        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        with AMP():
            x, _ = self.conformer(x, lengths)
        return self.classifier(x[:, -1, :])

# —— 一次性实例化四个模型 ——
# TSIMamba & Transformer
mamba = BiMambaVision(
    mm_input_size=512, mm_output_sizes=[1024],
    dropout=0.1, activation='GELU', causal=False,
    mamba_config={'d_state':12,'expand':2,'d_conv':4,'bidirectional':True}
).to(DEVICE).eval()

transformer = BiTransformerVision(
    mm_input_size=512, mm_output_sizes=[768,1024,768,512],
    dropout=0.1
).to(DEVICE).eval()

# MAET 最大长度实例化
max_len = SEQ_LENGTHS[-1]
maet = MAET(**{**MAET_BASE_KWARGS, "eeg_seq_len":max_len, "eye_seq_len":max_len}).to(DEVICE).eval()

# Conformer 最大长度实例化
d_model = 1024*1  # 1024 from mm_output_sizes * 2 / 2? Actually mm_output_sizes[-1]*2 = 2048
# But BaseCNN yields 512 features each => 512*2=1024
d_model = 512*2
conformer_args = {"num_heads":8, "ffn_dim":2048, "num_layers":4, "kernel_size":31, "dropout":0.1}
conformer = ConformerVision(max_len, d_model, conformer_args).to(DEVICE).eval()

# —— 测内存 & 速度通用函数 ——
def measure_mem(model, is_maet=False):
    results = []
    for L in SEQ_LENGTHS:
        torch.cuda.empty_cache()
        # 构造输入
        if is_maet:
            eeg = torch.randn(1, MAET_BASE_KWARGS["eeg_dim"], device=DEVICE)
            eye = torch.randn(1, MAET_BASE_KWARGS["eye_dim"], device=DEVICE)
            fwd = lambda: model(eeg=eeg, eye=eye)
        else:
            g = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            r = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            fwd = lambda: model(g, r)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        try:
            with AMP():
                _ = fwd()
            peak = torch.cuda.max_memory_allocated(DEVICE) / 1e9
        except RuntimeError:
            peak = float("nan")
        results.append(peak)
    return results

def measure_speed(model, is_maet=False, repeats=3):
    results = []
    for L in SEQ_LENGTHS:
        # 构造输入
        if is_maet:
            eeg = torch.randn(1, MAET_BASE_KWARGS["eeg_dim"], device=DEVICE)
            eye = torch.randn(1, MAET_BASE_KWARGS["eye_dim"], device=DEVICE)
            fwd = lambda: model(eeg=eeg, eye=eye)
        else:
            g = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            r = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            fwd = lambda: model(g, r)
        # 暖机
        for _ in range(2):
            try:
                with AMP(): _ = fwd()
            except RuntimeError:
                break
        torch.cuda.synchronize(DEVICE)
        t0 = time.perf_counter()
        for _ in range(repeats):
            try:
                with AMP(): _ = fwd()
            except RuntimeError:
                break
        torch.cuda.synchronize(DEVICE)
        results.append((time.perf_counter()-t0)/repeats)
    return results

# —— 绘图 & OOM 斜率延伸 ——
def plot_results(title, ys, labels, colors, save_name):
    fig, ax = plt.subplots(figsize=(8,5))
    x = SEQ_LENGTHS
    for y, lab, c in zip(ys, labels, colors):
        ax.plot(x, y, 'o-', color=c, label=lab)
    # 对 MAET (idx=2) 和 Conformer (idx=3) 做断线优雅延伸
    for idx in (2,3):
        y = ys[idx]
        for i,v in enumerate(y):
            if math.isnan(v) and i>=2:
                x0,y0 = x[i-2], y[i-2]
                x1,y1 = x[i-1], y[i-1]
                orig = (y1-y0)/(x1-x0)
                mul = 1.5
                y_top = ax.get_ylim()[1]
                while True:
                    slope = orig * mul
                    x_end = x1 + (y_top-y1)/slope
                    nxt = x[i] if i<len(x) else x1*1.1
                    if x1 < x_end < nxt: break
                    mul *= 1.1
                ax.plot([x1,x_end], [y1,y_top], '--', color=colors[idx], linewidth=2)
                break
    ax.set_title(title)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("GPU Memory (GB)" if "Memory" in title else "Inference Time (s)")
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join("/root/autodl-tmp/pic_model", save_name), dpi=300)
    plt.close(fig)

def main():
    os.makedirs("/root/autodl-tmp/pic_model", exist_ok=True)

    # 测 GPU 内存
    mem1 = measure_mem(mamba, False)
    mem2 = measure_mem(transformer, False)
    mem3 = measure_mem(maet, True)
    mem4 = measure_mem(conformer, False)
    plot_results(
        "GPU Memory Comparison",
        [mem1, mem2, mem3, mem4],
        ["TSIMamba","TSITransformer","MAET","Conformer"],
        ["#4C72B0","#55A868","#C44E52","#8172B2"],
        "gpu_memory_comparison.png"
    )

    # 测推理速度
    sp1 = measure_speed(mamba, False)
    sp2 = measure_speed(transformer, False)
    sp3 = measure_speed(maet, True)
    sp4 = measure_speed(conformer, False)
    plot_results(
        "Inference Speed Comparison",
        [sp1, sp2, sp3, sp4],
        ["TSIMamba","TSITransformer","MAET","Conformer"],
        ["#4C72B0","#55A868","#C44E52","#8172B2"],
        "inference_speed_comparison.png"
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import warnings
import math
import torch
import matplotlib.pyplot as plt
import torchaudio.models as tm

# 忽略 timm 过期警告，启用 cuDNN benchmark
warnings.filterwarnings("ignore", "Importing from timm.models.layers is deprecated")
torch.backends.cudnn.benchmark = True

# 设备与测试序列长度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LENGTHS = [2000, 5000, 10000, 15000, 20000, 24000]
NUM_CLASSES = 7

# MAET 基础配置
MAET_BASE_KWARGS = dict(
    eeg_dim=310, eye_dim=33, num_classes=NUM_CLASSES,
    embed_dim=32, depth=3, num_heads=4,
    mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0
)

# —— 导入已有模型与模块 ——
from shengji07_SEED_VII_7_4mat import BaseCNN, BiMambaVision
from shengji07_SEED_VII_7_Transformer_02 import BiTransformerVision
from maet import MAET

# —— ConformerVision 定义 ——
class ConformerVision(torch.nn.Module):
    def __init__(self, mm_output_sizes=[1024], conformer_args=None):
        super().__init__()
        # CNN 特征提取
        self.base_cnn_g = BaseCNN()
        self.base_cnn_r = BaseCNN()
        # 拼接后维度
        d_model = mm_output_sizes[-1] * 2
        args = conformer_args or {}
        # 构造 Conformer 编码器
        self.conformer = tm.Conformer(
            input_dim=d_model,
            num_heads=args.get("num_heads", 8),
            ffn_dim=args.get("ffn_dim", 2048),
            num_layers=args.get("num_layers", 4),
            depthwise_conv_kernel_size=args.get("kernel_size", 31),
            dropout=args.get("dropout", 0.1),
            use_group_norm=False,
            convolution_first=False
        )
        self.classifier = torch.nn.Linear(d_model, NUM_CLASSES)

    def forward(self, g, r):
        B, T, C, H, W = g.shape
        # CNN → flatten → time
        g_feat = self.base_cnn_g(g.view(B*T, C, H, W)).view(B, T, -1)
        r_feat = self.base_cnn_r(r.view(B*T, C, H, W)).view(B, T, -1)
        x = torch.cat([g_feat, r_feat], dim=-1)  # (B, T, d_model)
        # Conformer 需要传入 lengths
        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths)
        return self.classifier(x[:, -1, :])      # (B, NUM_CLASSES)

# —— 通用测内存（TSIMamba/TSITransformer/Conformer）——
def measure_mem_seq(model, seqs):
    mem = []
    m = model.to(DEVICE).eval()
    with torch.no_grad():
        for L in seqs:
            torch.cuda.empty_cache()
            g = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            r = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            torch.cuda.reset_peak_memory_stats(DEVICE)
            try:
                _ = m(g, r)
                peak = torch.cuda.max_memory_allocated(DEVICE) / 1e9
            except RuntimeError:
                peak = float("nan")
                torch.cuda.empty_cache()
            mem.append(peak)
    return mem

# —— 专用测 MAET 内存 ——
def measure_mem_maet(seqs):
    mem = []
    eeg_dim = MAET_BASE_KWARGS["eeg_dim"]
    eye_dim = MAET_BASE_KWARGS["eye_dim"]
    for L in seqs:
        cfg = {**MAET_BASE_KWARGS, "eeg_seq_len": L, "eye_seq_len": L}
        m = MAET(**cfg).to(DEVICE).eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            eeg = torch.randn(1, eeg_dim, device=DEVICE)
            eye = torch.randn(1, eye_dim, device=DEVICE)
            torch.cuda.reset_peak_memory_stats(DEVICE)
            try:
                _ = m(eeg=eeg, eye=eye)
                peak = torch.cuda.max_memory_allocated(DEVICE) / 1e9
            except RuntimeError:
                peak = float("nan")
                torch.cuda.empty_cache()
        mem.append(peak)
        del m, eeg, eye
        torch.cuda.empty_cache()
    return mem

# —— 通用测速度（TSIMamba/TSITransformer/Conformer）——
def measure_speed_seq(model, seqs, repeats=10):
    sp = []
    m = model.to(DEVICE).eval()
    with torch.no_grad():
        for L in seqs:
            g = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            r = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            # 暖机
            for _ in range(3):
                try: _ = m(g, r)
                except RuntimeError: break
            torch.cuda.synchronize(DEVICE)
            t0 = time.perf_counter()
            for _ in range(repeats):
                try: _ = m(g, r)
                except RuntimeError: break
            torch.cuda.synchronize(DEVICE)
            sp.append((time.perf_counter() - t0) / repeats)
    return sp

# —— 专用测 MAET 速度 ——
def measure_speed_maet(seqs, repeats=10):
    sp = []
    eeg_dim = MAET_BASE_KWARGS["eeg_dim"]
    eye_dim = MAET_BASE_KWARGS["eye_dim"]
    for L in seqs:
        cfg = {**MAET_BASE_KWARGS, "eeg_seq_len": L, "eye_seq_len": L}
        m = MAET(**cfg).to(DEVICE).eval()
        with torch.no_grad():
            eeg = torch.randn(1, eeg_dim, device=DEVICE)
            eye = torch.randn(1, eye_dim, device=DEVICE)
            for _ in range(3):
                try: _ = m(eeg=eeg, eye=eye)
                except RuntimeError: break
            torch.cuda.synchronize(DEVICE)
            t0 = time.perf_counter()
            for _ in range(repeats):
                try: _ = m(eeg=eeg, eye=eye)
                except RuntimeError: break
            torch.cuda.synchronize(DEVICE)
            sp.append((time.perf_counter() - t0) / repeats)
        del m, eeg, eye
        torch.cuda.empty_cache()
    return sp

# —— 绘图（4 曲线，idx=2,3 做 OOM 斜率延伸）——
def plot_four(x, ys, xlabel, ylabel, title, labels, colors, save_path):
    fig, ax = plt.subplots(figsize=(8,5))
    for idx, (y, lab, c) in enumerate(zip(ys, labels, colors)):
        ax.plot(x, y, 'o-', color=c, label=lab)
    # 对 MAET(idx=2) 与 Conformer(idx=3)分别处理 OOM 断线
    for idx in (2, 3):
        y = ys[idx]
        for i, v in enumerate(y):
            if math.isnan(v) and i >= 2:
                x0, y0 = x[i-2], y[i-2]
                x1, y1 = x[i-1], y[i-1]
                orig_slope = (y1 - y0) / (x1 - x0)
                mul = 1.5
                y_top = ax.get_ylim()[1]
                while True:
                    slope = orig_slope * mul
                    x_end = x1 + (y_top - y1) / slope
                    next_tick = x[i] if i < len(x) else x1 * 1.1
                    if x1 < x_end < next_tick:
                        break
                    mul *= 1.1
                ax.plot([x1, x_end], [y1, y_top], '--', color=colors[idx], linewidth=2)
                break
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def main():
    out_dir = "/root/autodl-tmp/pic_model"
    os.makedirs(out_dir, exist_ok=True)

    # 四模型实例化
    mamba       = BiMambaVision(
        mm_input_size=512, mm_output_sizes=[1024],
        dropout=0.1, activation='GELU', causal=False,
        mamba_config={'d_state':12,'expand':2,'d_conv':4,'bidirectional':True}
    )
    transformer = BiTransformerVision(
        mm_input_size=512, mm_output_sizes=[768,1024,768,512],
        dropout=0.1
    )
    # MAET 及 Conformer 于测函数内部按 seq_len 实例化
    conformer   = ConformerVision(
        mm_output_sizes=[1024],
        conformer_args={'num_heads':8,'ffn_dim':2048,'num_layers':4,'kernel_size':31,'dropout':0.1}
    )

    # 1) GPU Memory Comparison
    mem1 = measure_mem_seq(mamba, SEQ_LENGTHS)
    mem2 = measure_mem_seq(transformer, SEQ_LENGTHS)
    mem3 = measure_mem_maet(SEQ_LENGTHS)
    mem4 = measure_mem_seq(conformer, SEQ_LENGTHS)
    plot_four(
        SEQ_LENGTHS,
        [mem1, mem2, mem3, mem4],
        xlabel="Sequence Length",
        ylabel="GPU Memory (GB)",
        title="GPU Memory Comparison",
        labels=["TSIMamba","TSITransformer","MAET","Conformer"],
        colors=["#4C72B0","#55A868","#C44E52","#8172B2"],
        save_path=os.path.join(out_dir, "gpu_memory_comparison.png")
    )

    # 2) Inference Speed Comparison
    sp1 = measure_speed_seq(mamba, SEQ_LENGTHS)
    sp2 = measure_speed_seq(transformer, SEQ_LENGTHS)
    sp3 = measure_speed_maet(SEQ_LENGTHS)
    sp4 = measure_speed_seq(conformer, SEQ_LENGTHS)
    plot_four(
        SEQ_LENGTHS,
        [sp1, sp2, sp3, sp4],
        xlabel="Sequence Length",
        ylabel="Inference Time (s)",
        title="Inference Speed Comparison",
        labels=["TSIMamba","TSITransformer","MAET","Conformer"],
        colors=["#4C72B0","#55A868","#C44E52","#8172B2"],
        save_path=os.path.join(out_dir, "inference_speed_comparison.png")
    )

if __name__ == "__main__":
    main()

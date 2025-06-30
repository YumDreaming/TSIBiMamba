#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchaudio.models as tm
from tqdm import tqdm

# 忽略 timm 过期警告，启用 cuDNN benchmark
warnings.filterwarnings("ignore", "Importing from timm.models.layers is deprecated")
torch.backends.cudnn.benchmark = True

# 设备 & 序列长度 —— 唯一修改处
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LENGTHS = [500, 1000, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000,10000, 13000, 15000]
NUM_CLASSES = 7

# MAET 基础配置
MAET_BASE_KWARGS = dict(
    eeg_dim=310, eye_dim=33, num_classes=NUM_CLASSES,
    embed_dim=32, depth=6, num_heads=4,
    mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0
)

# 导入已有模型
from shengji07_SEED_VII_7_4mat import BaseCNN, BiMambaVision
from shengji07_SEED_VII_7_Transformer_02 import BiTransformerVision
from maet import MAET

# -------------------- 模型定义 --------------------

class ConformerVision(nn.Module):
    def __init__(self, conformer_args=None):
        super().__init__()
        self.base_cnn_g = BaseCNN()
        self.base_cnn_r = BaseCNN()
        d_model = 512 * 2
        args = conformer_args or {}
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
        self.classifier = nn.Linear(d_model, NUM_CLASSES)

    def forward(self, g, r):
        B, T, C, H, W = g.shape
        g_feat = self.base_cnn_g(g.view(B*T, C, H, W)).view(B, T, -1)
        r_feat = self.base_cnn_r(r.view(B*T, C, H, W)).view(B, T, -1)
        x = torch.cat([g_feat, r_feat], dim=-1)
        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths)
        return self.classifier(x[:, -1, :])

class BigTransformerVision(nn.Module):
    """
    更大参数的 TransformerEncoder 代替 ViTransformer
    """
    def __init__(self, nhead=16, num_layers=8, dim_feedforward=4096, dropout=0.1):
        super().__init__()
        self.base_cnn_g = BaseCNN()
        self.base_cnn_r = BaseCNN()
        self.d_model = 512 * 2
        self.pos_embed = nn.Parameter(torch.zeros(1, 6, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(self.d_model, NUM_CLASSES)

    def forward(self, g, r):
        B, T, C, H, W = g.shape
        g_feat = self.base_cnn_g(g.view(B*T, C, H, W)).view(B, T, -1)
        r_feat = self.base_cnn_r(r.view(B*T, C, H, W)).view(B, T, -1)
        x = torch.cat([g_feat, r_feat], dim=-1)
        pe = self.pos_embed
        if pe.shape[1] != T:
            pe = F.interpolate(
                pe.permute(0,2,1), size=T, mode='linear', align_corners=False
            ).permute(0,2,1)
        x = x + pe
        x = self.transformer(x)
        return self.classifier(x[:, -1, :])

# -------------------- 测试函数 --------------------

def measure_mem_seq(model, seqs):
    mem = []
    m = model.to(DEVICE).eval()
    for L in tqdm(seqs, desc=f"Mem {model.__class__.__name__}", unit="len"):
        torch.cuda.empty_cache()
        g = torch.randn(1, L, 4, 8, 9, device=DEVICE)
        r = torch.randn(1, L, 4, 8, 9, device=DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        try:
            _ = m(g, r)
            mem.append(torch.cuda.max_memory_allocated(DEVICE) / 1e9)
        except RuntimeError:
            print(f"[OOM] {model.__class__.__name__} at L={L}")
            mem.append(float("nan"))
            torch.cuda.empty_cache()
    return mem

def measure_mem_maet(seqs):
    mem = []
    eeg_dim, eye_dim = MAET_BASE_KWARGS["eeg_dim"], MAET_BASE_KWARGS["eye_dim"]
    for L in tqdm(seqs, desc="Mem MAET", unit="len"):
        cfg = {**MAET_BASE_KWARGS, "eeg_seq_len":L, "eye_seq_len":L}
        m = MAET(**cfg).to(DEVICE).eval()
        torch.cuda.empty_cache()
        eeg = torch.randn(1, eeg_dim, device=DEVICE)
        eye = torch.randn(1, eye_dim, device=DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        try:
            _ = m(eeg=eeg, eye=eye)
            mem.append(torch.cuda.max_memory_allocated(DEVICE) / 1e9)
        except RuntimeError:
            print(f"[OOM] MAET at L={L}")
            mem.append(float("nan"))
            torch.cuda.empty_cache()
        del m, eeg, eye
        torch.cuda.empty_cache()
    return mem

def measure_speed_seq(model, seqs, repeats=3):
    sp = []
    m = model.to(DEVICE).eval()
    for L in tqdm(seqs, desc=f"Speed {model.__class__.__name__}", unit="len"):
        try:
            g = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            r = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            for _ in range(2): _ = m(g, r)
            torch.cuda.synchronize(DEVICE)
            t0 = time.perf_counter()
            for _ in range(repeats): _ = m(g, r)
            torch.cuda.synchronize(DEVICE)
            sp.append((time.perf_counter() - t0) / repeats)
        except RuntimeError:
            print(f"[OOM] {model.__class__.__name__} speed at L={L}")
            sp.append(float("nan"))
            torch.cuda.empty_cache()
    return sp

def measure_speed_maet(seqs, repeats=3):
    sp = []
    eeg_dim, eye_dim = MAET_BASE_KWARGS["eeg_dim"], MAET_BASE_KWARGS["eye_dim"]
    for L in tqdm(seqs, desc="Speed MAET", unit="len"):
        try:
            cfg = {**MAET_BASE_KWARGS, "eeg_seq_len":L, "eye_seq_len":L}
            m = MAET(**cfg).to(DEVICE).eval()
            eeg = torch.randn(1, eeg_dim, device=DEVICE)
            eye = torch.randn(1, eye_dim, device=DEVICE)
            for _ in range(2): _ = m(eeg=eeg, eye=eye)
            torch.cuda.synchronize(DEVICE)
            t0 = time.perf_counter()
            for _ in range(repeats): _ = m(eeg=eeg, eye=eye)
            torch.cuda.synchronize(DEVICE)
            sp.append((time.perf_counter() - t0) / repeats)
        except RuntimeError:
            print(f"[OOM] MAET speed at L={L}")
            sp.append(float("nan"))
            torch.cuda.empty_cache()
        del m, eeg, eye
        torch.cuda.empty_cache()
    return sp

# -------------------- 绘图函数 --------------------

def plot_five(x, ys, xlabel, ylabel, title, labels, colors, save_path):
    fig, ax = plt.subplots(figsize=(8,5))
    # 实线
    for idx, (y, lab, c) in enumerate(zip(ys, labels, colors)):
        ax.plot(x, y, 'o-', color=c, label=lab)
    y_top = ax.get_ylim()[1]
    # 虚线延伸
    for idx in (2, 3, 4):
        y = ys[idx]
        for i, v in enumerate(y):
            if math.isnan(v) and i >= 2:
                x0, y0 = x[i-2], y[i-2]
                x1, y1 = x[i-1], y[i-1]
                orig_slope = max((y1 - y0) / (x1 - x0), 1e-6)
                slope_req = orig_slope * 1.5
                next_tick = x[i] if i < len(x) else x1 * 1.1
                x_end = x1 + (y_top - y1) / slope_req
                if not (x1 < x_end < next_tick):
                    x_end = x1 + (next_tick - x1) * 0.999
                ax.plot([x1, x_end], [y1, y_top], '--', color=colors[idx], linewidth=2)
                break
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

# -------------------- 主流程 --------------------

def main():
    out = "/root/autodl-tmp/pic_model_02"
    os.makedirs(out, exist_ok=True)

    # 初始化模型
    mamba = BiMambaVision(
        mm_input_size=512, mm_output_sizes=[1024],
        dropout=0.1, activation='GELU', causal=False,
        mamba_config={'d_state':12,'expand':2,'d_conv':4,'bidirectional':True}
    )
    transformer = BiTransformerVision(
        mm_input_size=512, mm_output_sizes=[768,1024,768,512],
        dropout=0.1
    )
    conformer = ConformerVision(
        conformer_args={'num_heads':8,'ffn_dim':2048,'num_layers':6,'kernel_size':31,'dropout':0.1}
    )
    big_transformer = BigTransformerVision(
        nhead=16, num_layers=10, dim_feedforward=4096, dropout=0.1
    )

    seq_models = [
        ("TSIMamba", mamba),
        ("TSITransformer", transformer),
        ("Conformer", conformer),
        ("BigTransformer", big_transformer)
    ]
    model_names = ["TSIBiMamba", "TSIBiTransformer", "MAET", "Conformer", "ViTransformer"]

    # GPU Memory 比较
    mem_lists = []
    for name, model in tqdm(seq_models, desc="Measure GPU Memory", unit="model"):
        mem_lists.append(measure_mem_seq(model, SEQ_LENGTHS))
    mem_lists.insert(2, measure_mem_maet(SEQ_LENGTHS))

    # 输出每个模型在各序列长度下的显存占用
    print("\nGPU Memory (GB) usage per sequence length:")
    for model_name, mem in zip(model_names, mem_lists):
        print(f"{model_name}:")
        for L, m in zip(SEQ_LENGTHS, mem):
            print(f"  Length {L}: {m:.3f} GB")

    plot_five(
        SEQ_LENGTHS, mem_lists,
        xlabel="Sequence Length", ylabel="GPU Memory (GB)",
        title="GPU Memory Comparison",
        labels=model_names,
        colors=["#4C72B0","#55A868","#C44E52","#8172B2","#8C8C8C"],
        save_path=os.path.join(out, "gpu_memory_comparison.png")
    )

    # 推理速度比较
    sp_lists = []
    for name, model in tqdm(seq_models, desc="Measure Inference Speed", unit="model"):
        sp_lists.append(measure_speed_seq(model, SEQ_LENGTHS))
    sp_lists.insert(2, measure_speed_maet(SEQ_LENGTHS))

    # 输出每个模型在各序列长度下的推理时间
    print("\nInference Time (s) per sequence length:")
    for model_name, sp in zip(model_names, sp_lists):
        print(f"{model_name}:")
        for L, t in zip(SEQ_LENGTHS, sp):
            print(f"  Length {L}: {t:.6f} s")

    plot_five(
        SEQ_LENGTHS, sp_lists,
        xlabel="Sequence Length", ylabel="Inference Time (s)",
        title="Inference Speed Comparison",
        labels=model_names,
        colors=["#4C72B0","#55A868","#C44E52","#8172B2","#8C8C8C"],
        save_path=os.path.join(out, "inference_speed_comparison.png")
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import time
import warnings

import torch
import matplotlib.pyplot as plt

# 忽略 timm 的过期警告
warnings.filterwarnings("ignore", "Importing from timm.models.layers is deprecated")
# 开启 cuDNN 自动调优
torch.backends.cudnn.benchmark = True

# 输出目录
OUT_DIR = "/root/autodl-tmp/pic_model_4"
os.makedirs(OUT_DIR, exist_ok=True)

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 横轴：若干序列长度
SEQ_LENGTHS = [2000, 5000, 10000, 15000, 20000, 24000]

# MAET 基本配置（会动态覆盖 seq_len）
MAET_BASE_KWARGS = dict(
    eeg_dim=310, eye_dim=33, num_classes=7,
    embed_dim=32, depth=3, num_heads=4,
    mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0
)

# 导入模型
from shengji07_SEED_VII_7_4mat import BiMambaVision as TSIMamba
from shengji07_SEED_VII_7_Transformer_02 import BiTransformerVision as TSITransformer
from maet import MAET
from conformer import ViT as EEGConformer

def measure_mem_seq_model(model, seqs):
    """测 TSIMamba/TSITransformer 在不同 seq_len 下的峰值显存 (GB)。"""
    mem = []
    m = model.to(DEVICE).eval()
    with torch.no_grad():
        for L in seqs:
            torch.cuda.empty_cache()
            # 随机输入
            g = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            r = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            torch.cuda.reset_peak_memory_stats(DEVICE)
            _ = m(g, r)
            mem.append(torch.cuda.max_memory_allocated(DEVICE) / (1024**3))
    return mem

def measure_mem_maet(seqs):
    """测 MAET 在不同 seq_len 下的峰值显存 (GB)，OOM 记 nan。"""
    mem = []
    for L in seqs:
        cfg = {**MAET_BASE_KWARGS, "eeg_seq_len": L, "eye_seq_len": L}
        m = MAET(**cfg).to(DEVICE).eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            eeg = torch.randn(1, cfg["eeg_dim"], device=DEVICE)
            eye = torch.randn(1, cfg["eye_dim"], device=DEVICE)
            torch.cuda.reset_peak_memory_stats(DEVICE)
            try:
                _ = m(eeg=eeg, eye=eye)
                mem.append(torch.cuda.max_memory_allocated(DEVICE) / (1024**3))
            except RuntimeError:
                mem.append(float("nan"))
                torch.cuda.empty_cache()
        del m, eeg, eye
        torch.cuda.empty_cache()
    return mem

def measure_mem_conformer_param(seqs):
    """基于参数量估算 EEG_Conformer 的显存占用 (GB)，返回常量列表。"""
    m = EEGConformer().to(DEVICE).eval()
    # 参数占用字节数
    param_bytes = sum(p.numel() * p.element_size() for p in m.parameters())
    gb = param_bytes / (1024**3)
    return [gb] * len(seqs)

def measure_speed_seq_model(model, seqs, reps=10):
    """测 TSIMamba/TSITransformer 推理平均耗时 (s)。"""
    sp = []
    m = model.to(DEVICE).eval()
    with torch.no_grad():
        for L in seqs:
            g = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            r = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            # 暖机
            for _ in range(3): _ = m(g, r)
            torch.cuda.synchronize(DEVICE)
            t0 = time.perf_counter()
            for _ in range(reps): _ = m(g, r)
            torch.cuda.synchronize(DEVICE)
            sp.append((time.perf_counter() - t0) / reps)
    return sp

def measure_speed_maet(seqs, reps=10):
    """测 MAET 推理平均耗时 (s)，OOM 记 nan。"""
    sp = []
    for L in seqs:
        cfg = {**MAET_BASE_KWARGS, "eeg_seq_len": L, "eye_seq_len": L}
        m = MAET(**cfg).to(DEVICE).eval()
        try:
            with torch.no_grad():
                eeg = torch.randn(1, cfg["eeg_dim"], device=DEVICE)
                eye = torch.randn(1, cfg["eye_dim"], device=DEVICE)
                for _ in range(3): _ = m(eeg=eeg, eye=eye)
                torch.cuda.synchronize(DEVICE)
                t0 = time.perf_counter()
                for _ in range(reps): _ = m(eeg=eeg, eye=eye)
                torch.cuda.synchronize(DEVICE)
            sp.append((time.perf_counter() - t0) / reps)
        except RuntimeError:
            sp.append(float("nan"))
            torch.cuda.empty_cache()
        del m, eeg, eye
        torch.cuda.empty_cache()
    return sp

def measure_speed_conformer(reps=10):
    """测 EEG_Conformer 推理平均耗时 (s)，返回常量列表。"""
    m = EEGConformer().to(DEVICE).eval()
    with torch.no_grad():
        inp = torch.randn(1, 1, 62, 200, device=DEVICE)
        for _ in range(3): _ = m(inp)
        torch.cuda.synchronize(DEVICE)
        t0 = time.perf_counter()
        for _ in range(reps): _ = m(inp)
        torch.cuda.synchronize(DEVICE)
        val = (time.perf_counter() - t0) / reps
    return [val] * len(SEQ_LENGTHS)

def plot_curves(xs, ys_list, labels, colors, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(8,5))
    for ys, lab, col in zip(ys_list, labels, colors):
        plt.plot(xs, ys, 'o-', color=col, label=lab)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    # 实例化
    mamba = TSIMamba(
        mm_input_size=512,
        mm_output_sizes=[1024],
        dropout=0.1,
        activation='GELU',
        causal=False,
        mamba_config={'d_state':12,'expand':2,'d_conv':4,'bidirectional':True}
    )
    trans = TSITransformer(
        mm_input_size=512,
        mm_output_sizes=[768,1024,768,512],
        dropout=0.1
    )

    # GPU Memory Comparison
    mem1 = measure_mem_seq_model(mamba, SEQ_LENGTHS)
    mem2 = measure_mem_seq_model(trans, SEQ_LENGTHS)
    mem3 = measure_mem_maet(SEQ_LENGTHS)
    mem4 = measure_mem_conformer_param(SEQ_LENGTHS)
    plot_curves(
        SEQ_LENGTHS,
        [mem1, mem2, mem3, mem4],
        labels=["TSIMamba","TSITransformer","MAET","EEGConformer"],
        colors=["#4C72B0","#55A868","#C44E52","#8172B2"],
        xlabel="Sequence Length",
        ylabel="GPU Memory (GB)",
        title="GPU Memory Comparison",
        save_path=os.path.join(OUT_DIR, "gpu_memory_comparison.png")
    )

    # Inference Speed Comparison
    sp1 = measure_speed_seq_model(mamba, SEQ_LENGTHS)
    sp2 = measure_speed_seq_model(trans, SEQ_LENGTHS)
    sp3 = measure_speed_maet(SEQ_LENGTHS)
    sp4 = measure_speed_conformer()
    plot_curves(
        SEQ_LENGTHS,
        [sp1, sp2, sp3, sp4],
        labels=["TSIMamba","TSITransformer","MAET","EEGConformer"],
        colors=["#4C72B0","#55A868","#C44E52","#8172B2"],
        xlabel="Sequence Length",
        ylabel="Inference Time (s)",
        title="Inference Speed Comparison",
        save_path=os.path.join(OUT_DIR, "inference_speed_comparison.png")
    )

if __name__ == "__main__":
    main()

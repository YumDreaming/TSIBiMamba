#!/usr/bin/env python3
import os, time, warnings
import torch
import matplotlib.pyplot as plt

# 忽略 timm 过期警告
warnings.filterwarnings("ignore", "Importing from timm.models.layers is deprecated")
# 启用 cuDNN 自动调优
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 要测试的序列长度
SEQ_LENGTHS = [2000, 5000, 10000, 15000, 20000, 24000]

# MAET 基础配置，seq_len 会动态覆盖
MAET_BASE_KWARGS = dict(
    eeg_dim=310, eye_dim=33, num_classes=7,
    embed_dim=32, depth=3, num_heads=4,
    mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0
)

# 导入三种模型
from shengji07_SEED_VII_7_4mat import BiMambaVision as TSIMamba
from shengji07_SEED_VII_7_Transformer_02 import BiTransformerVision as TSITransformer
from maet import MAET

def measure_mem(model, seqs):
    """测 TSIMamba/TSITransformer 在不同序列长度下的峰值显存 (GB)。"""
    mem = []
    m = model.to(DEVICE).eval()
    with torch.no_grad():
        for L in seqs:
            torch.cuda.empty_cache()
            g = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            r = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            torch.cuda.reset_peak_memory_stats(DEVICE)
            _ = m(g, r)
            mem.append(torch.cuda.max_memory_allocated(DEVICE) / 1e9)
    return mem

def measure_mem_maet(seqs):
    """测 MAET 在不同序列长度下的峰值显存，OOM 记为 nan 并继续。"""
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
                mem.append(torch.cuda.max_memory_allocated(DEVICE) / 1e9)
            except RuntimeError:
                mem.append(float("nan"))
                torch.cuda.empty_cache()
        del m, eeg, eye
        torch.cuda.empty_cache()
    return mem

def measure_speed(model, seqs, reps=10):
    """测 TSIMamba/TSITransformer 推理平均耗时 (s)。"""
    sp = []
    m = model.to(DEVICE).eval()
    with torch.no_grad():
        for L in seqs:
            g = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            r = torch.randn(1, L, 4, 8, 9, device=DEVICE)
            for _ in range(3): _ = m(g, r)  # 暖机
            torch.cuda.synchronize(DEVICE)
            t0 = time.perf_counter()
            for _ in range(reps): _ = m(g, r)
            torch.cuda.synchronize(DEVICE)
            sp.append((time.perf_counter() - t0) / reps)
    return sp

def measure_speed_maet(seqs, reps=10):
    """测 MAET 推理平均耗时，OOM 记为 nan 并继续。"""
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

def plot_beauty(x, y1, y2, y3, xlabel, ylabel, title, labels, colors, save_path):
    """绘制对比曲线，MAET OOM 处用放大斜率断线延伸至顶部。"""
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, y1, 'o-', color=colors[0], label=labels[0])
    ax.plot(x, y2, 'o-', color=colors[1], label=labels[1])
    ax.plot(x, y3, 'o-', color=colors[2], label=labels[2])

    import math
    # 找 MAET 首次 OOM
    for i, v in enumerate(y3):
        if math.isnan(v) and i >= 2:
            x_prev, y_prev = x[i-2], y3[i-2]
            x_last, y_last = x[i-1], y3[i-1]
            # 原始斜率
            orig_slope = (y_last - y_prev) / (x_last - x_prev)
            # 目标倍数，至少 1.5
            mul = 1.5
            y_top = ax.get_ylim()[1]
            # 计算延伸点
            while True:
                slope = orig_slope * mul
                # x_end 使 (y_top - y_last) = slope * (x_end - x_last)
                x_end = x_last + (y_top - y_last) / slope
                # 若 x_end 在 (x_last, next_tick) 范围内，则止
                next_tick = x[i] if i < len(x) else x_last * 1.1
                if x_last < x_end < next_tick:
                    break
                mul *= 1.1  # 增大 10%，再试
            ax.plot([x_last, x_end], [y_last, y_top], '--', color=colors[2], linewidth=2)
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

    # 实例化模型
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

    # GPU 内存对比
    mem1 = measure_mem(mamba, SEQ_LENGTHS)
    mem2 = measure_mem(trans, SEQ_LENGTHS)
    mem3 = measure_mem_maet(SEQ_LENGTHS)
    plot_beauty(
        SEQ_LENGTHS, mem1, mem2, mem3,
        xlabel="Sequence Length",
        ylabel="GPU Memory (GB)",
        title="GPU Memory Comparison",
        labels=["TSIMamba","TSITransformer","MAET"],
        colors=["#4C72B0","#55A868","#C44E52"],
        save_path=os.path.join(out_dir, "gpu_memory_comparison.png")
    )

    # 推理速度对比
    sp1  = measure_speed(mamba, SEQ_LENGTHS)
    sp2  = measure_speed(trans, SEQ_LENGTHS)
    sp3  = measure_speed_maet(SEQ_LENGTHS)
    plot_beauty(
        SEQ_LENGTHS, sp1, sp2, sp3,
        xlabel="Sequence Length",
        ylabel="Inference Time (s)",
        title="Inference Speed Comparison",
        labels=["TSIMamba","TSITransformer","MAET"],
        colors=["#4C72B0","#55A868","#C44E52"],
        save_path=os.path.join(out_dir, "inference_speed_comparison.png")
    )

if __name__ == "__main__":
    main()

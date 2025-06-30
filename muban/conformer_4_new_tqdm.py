#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import warnings
import math
import torch
import matplotlib.pyplot as plt
import torchaudio.models as tm
from tqdm import tqdm

# 忽略 timm 过期警告，启用 cuDNN 自动调优
warnings.filterwarnings("ignore", "Importing from timm.models.layers is deprecated")
torch.backends.cudnn.benchmark = True

# 设备与序列长度（更精细）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LENGTHS = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000, 15000]
NUM_CLASSES = 7

# MAET 配置基准
MAET_BASE_KWARGS = dict(
    eeg_dim=310, eye_dim=33, num_classes=NUM_CLASSES,
    embed_dim=32, depth=3, num_heads=4,
    mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0
)

# 导入已有模型
from shengji07_SEED_VII_7_4mat import BaseCNN, BiMambaVision
from shengji07_SEED_VII_7_Transformer_02 import BiTransformerVision
from maet import MAET

# 修正后的 ConformerVision（d_model = 512*2 = 1024）
class ConformerVision(torch.nn.Module):
    def __init__(self, conformer_args=None):
        super().__init__()
        # CNN 特征提取
        self.base_cnn_g = BaseCNN()
        self.base_cnn_r = BaseCNN()
        # 拼接后维度
        d_model = 512 * 2
        args = conformer_args or {}
        # 构造 Conformer，input_dim=1024
        self.conformer = tm.Conformer(
            input_dim                  = d_model,
            num_heads                  = args.get("num_heads", 8),
            ffn_dim                    = args.get("ffn_dim", 2048),
            num_layers                 = args.get("num_layers", 4),
            depthwise_conv_kernel_size = args.get("kernel_size", 31),
            dropout                    = args.get("dropout", 0.1),
            use_group_norm             = False,
            convolution_first          = False
        )
        self.classifier = torch.nn.Linear(d_model, NUM_CLASSES)

    def forward(self, g, r):
        B, T, C, H, W = g.shape
        # CNN 提取 → reshape
        g_feat = self.base_cnn_g(g.view(B*T, C, H, W)).view(B, T, -1)
        r_feat = self.base_cnn_r(r.view(B*T, C, H, W)).view(B, T, -1)
        x = torch.cat([g_feat, r_feat], dim=-1)  # (B, T, 1024)
        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths)        # (B, T, 1024)
        return self.classifier(x[:, -1, :])      # (B, 7)

# 通用测内存（Mamba/Transformer/Conformer）
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
            peak = torch.cuda.max_memory_allocated(DEVICE) / 1e9
        except RuntimeError:
            print(f"[OOM] {model.__class__.__name__} at L={L}")
            peak = float("nan")
            torch.cuda.empty_cache()
        mem.append(peak)
    return mem

# 专用测 MAET 内存
def measure_mem_maet(seqs):
    mem = []
    eeg_dim, eye_dim = MAET_BASE_KWARGS["eeg_dim"], MAET_BASE_KWARGS["eye_dim"]
    for L in tqdm(seqs, desc="Mem MAET", unit="len"):
        cfg = {**MAET_BASE_KWARGS, "eeg_seq_len": L, "eye_seq_len": L}
        m = MAET(**cfg).to(DEVICE).eval()
        torch.cuda.empty_cache()
        eeg = torch.randn(1, eeg_dim, device=DEVICE)
        eye = torch.randn(1, eye_dim, device=DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        try:
            _ = m(eeg=eeg, eye=eye)
            peak = torch.cuda.max_memory_allocated(DEVICE) / 1e9
        except RuntimeError:
            print(f"[OOM] MAET at L={L}")
            peak = float("nan")
            torch.cuda.empty_cache()
        mem.append(peak)
        del m, eeg, eye
        torch.cuda.empty_cache()
    return mem

# 通用测速度（Mamba/Transformer/Conformer）
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

# 专用测 MAET 速度
def measure_speed_maet(seqs, repeats=3):
    sp = []
    eeg_dim, eye_dim = MAET_BASE_KWARGS["eeg_dim"], MAET_BASE_KWARGS["eye_dim"]
    for L in tqdm(seqs, desc="Speed MAET", unit="len"):
        try:
            cfg = {**MAET_BASE_KWARGS, "eeg_seq_len": L, "eye_seq_len": L}
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

# 绘图（MAET idx=2、Conformer idx=3 OOM 延伸）
def plot_four(x, ys, xlabel, ylabel, title, labels, colors, save_path):
    fig, ax = plt.subplots(figsize=(8,5))
    for idx, (y, lab, c) in enumerate(zip(ys, labels, colors)):
        ax.plot(x, y, 'o-', color=c, label=lab)
    y_top = ax.get_ylim()[1]
    for idx in (2, 3):
        y = ys[idx]
        next_tick_global = x[-1] * 1.1
        for i, v in enumerate(y):
            if math.isnan(v) and i >= 2:
                x0, y0 = x[i-2], y[i-2]
                x1, y1 = x[i-1], y[i-1]
                orig_slope = (y1-y0)/(x1-x0) if x1!=x0 else 0
                next_tick = x[i] if i<len(x) else next_tick_global
                if orig_slope<=0:
                    x_end = x1+(next_tick-x1)*0.25
                else:
                    mul=1.5
                    for _ in range(20):
                        slope=orig_slope*mul
                        x_end=x1+(y_top-y1)/slope
                        if x1<x_end<next_tick: break
                        mul*=1.1
                    else:
                        x_end=x1+(next_tick-x1)*0.25
                ax.plot([x1,x_end],[y1,y_top],'--',color=colors[idx],linewidth=2)
                break
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc='upper left'); fig.tight_layout()
    fig.savefig(save_path, dpi=300); plt.close(fig)

def main():
    out = "/root/autodl-tmp/pic_model"
    os.makedirs(out, exist_ok=True)

    # 实例化四模型
    mamba       = BiMambaVision(512,[1024],0.1,'GELU',False,{'d_state':12,'expand':2,'d_conv':4,'bidirectional':True})
    transformer = BiTransformerVision(512,[768,1024,768,512],0.1)
    conformer   = ConformerVision({'num_heads':8,'ffn_dim':2048,'num_layers':4,'kernel_size':31,'dropout':0.1})

    # GPU Memory 比较
    seq_models = [("TSIMamba",mamba),("TSITransformer",transformer),("Conformer",conformer)]
    mems=[]
    for name,mdl in tqdm(seq_models,desc="Measure GPU Memory",unit="model"):
        mems.append(measure_mem_seq(mdl,SEQ_LENGTHS))
    maet_mem=measure_mem_maet(SEQ_LENGTHS)
    mems.insert(2,maet_mem)
    plot_four(SEQ_LENGTHS,mems,"Sequence Length","GPU Memory (GB)","GPU Memory Comparison",
              ["TSIMamba","TSITransformer","MAET","Conformer"],
              ["#4C72B0","#55A868","#C44E52","#8172B2"],
              os.path.join(out,"gpu_memory_comparison.png"))

    # 推理速度比较
    sps=[]
    for name,mdl in tqdm(seq_models,desc="Measure Inference Speed",unit="model"):
        sps.append(measure_speed_seq(mdl,SEQ_LENGTHS))
    maet_sp=measure_speed_maet(SEQ_LENGTHS)
    sps.insert(2,maet_sp)
    plot_four(SEQ_LENGTHS,sps,"Sequence Length","Inference Time (s)","Inference Speed Comparison",
              ["TSIMamba","TSITransformer","MAET","Conformer"],
              ["#4C72B0","#55A868","#C44E52","#8172B2"],
              os.path.join(out,"inference_speed_comparison.png"))

if __name__=="__main__":
    main()

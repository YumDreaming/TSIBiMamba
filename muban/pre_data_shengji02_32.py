#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEAP 预处理（CUDA 版）
──────────────────
输出:
deap_pre/
├── Heat/<subject>/sXX_tYY_Heat.npy   (5,9,128,128)
├── RP/<subject>/sXX_tYY_RP.npy       (5,9,128,128)
└── split.json   (train/val subjects + PCA 参数)

2025-05-22  •  tested with PyTorch 2.3 + CUDA 12.1
"""

# -------------------------------------------------- #
# 0. 依赖与运行环境
# -------------------------------------------------- #
import os, json, pickle
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")                     # 服务器无 DISPLAY
import matplotlib.pyplot as plt

import mne
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from pyts.image import RecurrencePlot
from skimage.transform import resize

# GPU / CPU 自动选择
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True   # 根据输入尺寸自动挑最快算法
print("🤖  running on", DEVICE)

# -------------------------------------------------- #
# 1. 全局常量
# -------------------------------------------------- #
RAW_ROOT   = "/root/deap/data_preprocessed_python"
PRE_ROOT   = "/root/autodl-tmp/deap_pre64"
SPLIT_JSON = os.path.join(PRE_ROOT, "split.json")

CH_NAMES = [
    'Fp1','AF3','F3','F7','FC5','FC1','C3','T7',
    'CP5','CP1','P3','P7','PO3','O1','Oz','Pz',
    'Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz',
    'C4','T8','CP6','CP2','P4','P8','PO4','O2'
]
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}

S_RATE       = 128
BASELINE_SEC = 3
WIN_SEC      = 12
OVERLAP_SEC  = 6
WIN_SAMPLES  = WIN_SEC * S_RATE          # 1536
STEP_SAMPLES = WIN_SAMPLES - OVERLAP_SEC * S_RATE   # 768
N_WINDOWS    = 9                         # 0-8 windows / trial

# -------------------------------------------------- #
# 2. 通用函数
# -------------------------------------------------- #
def compute_band_power_cuda(seg_np: np.ndarray,
                            freqs_t: torch.Tensor,
                            masks: torch.Tensor) -> np.ndarray:
    """
    利用 GPU 计算 5 频段功率。
    ----------
    seg_np : (32,1536) float32  – 单个滑窗
    freqs_t: (769,)   float32   – rfftfreq 结果
    masks  : (5,769) complex64  – 每个频段的掩码 (已在 GPU)
    返回值 : (5,32)  np.float32
    """
    # (1) numpy → torch (GPU)
    seg = torch.as_tensor(seg_np, device=DEVICE)            # (32,1536)
    # (2) FFT (GPU)
    seg_fft = torch.fft.rfft(seg, dim=1)                    # (32,769)
    # (3) 掩码相乘 + irfft 求功率
    #     (5,769) ⊗ (32,769) → broadcasting → (5,32,769)
    sig_band = torch.fft.irfft(seg_fft.unsqueeze(0) * masks.unsqueeze(1),
                               n=WIN_SAMPLES, dim=-1)       # (5,32,1536)
    power = sig_band.pow(2).mean(dim=-1)                    # (5,32)
    return power.cpu().numpy().astype(np.float32)           # 回 CPU

def collect_pca_corpus(train_subjects):
    """遍历训练 subjects 所有滑窗，累积 (N,32) 样本用于 PCA。"""
    out = []
    for subj in tqdm(train_subjects, desc="PCA corpus"):
        with open(os.path.join(RAW_ROOT, f"{subj}.dat"), "rb") as f:
            data = pickle.load(f, encoding="latin1")["data"][:, :32, :]
        for trial in range(40):
            sig = data[trial][:, BASELINE_SEC*S_RATE:]          # (32,7680)
            for w in range(N_WINDOWS):
                st = w * STEP_SAMPLES
                out.append(sig[:, st:st+WIN_SAMPLES].T)         # (1536,32)
    return np.vstack(out).astype(np.float32)                    # (N,32)

# -------------------------------------------------- #
# 3.  主流程
# -------------------------------------------------- #
def preprocess_once():
    # 3-0. 已做过则跳过
    if os.path.isdir(PRE_ROOT) and os.path.isfile(SPLIT_JSON):
        print("缓存已存在，跳过。")
        return

    os.makedirs(os.path.join(PRE_ROOT, "Heat"), exist_ok=True)
    os.makedirs(os.path.join(PRE_ROOT, "RP"),   exist_ok=True)

    # 3-1. 划分 subject
    subjects = sorted(f[:-4] for f in os.listdir(RAW_ROOT) if f.endswith(".dat"))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(subjects, groups=subjects))
    train_subj = [subjects[i] for i in train_idx]
    val_subj   = [subjects[i] for i in val_idx]
    split_info = {"train": train_subj, "val": val_subj}
    print(f"Train {len(train_subj)} subjects:", train_subj)
    print(f"Val   {len(val_subj)} subjects:", val_subj)

    # 3-2. 拟合 PCA（CPU - sklearn）
    pca_corpus = collect_pca_corpus(train_subj)
    print("Fit PCA on", pca_corpus.shape)
    pca = PCA(n_components=5, svd_solver="full", random_state=0).fit(pca_corpus)
    split_info["pca_mean"]       = pca.mean_.tolist()
    split_info["pca_components"] = pca.components_.tolist()

    # 3-3. 预先准备共用资源
    freqs_t = torch.fft.rfftfreq(WIN_SAMPLES, 1 / S_RATE).to(DEVICE)
    # 掩码一次性准备好 (5,769)
    masks = []
    for fmin, fmax in BANDS.values():
        m = ((freqs_t >= fmin) & (freqs_t < fmax)).to(torch.complex64)
        masks.append(m)
    masks = torch.stack(masks, dim=0)                         # (5,769)
    rp_gen = RecurrencePlot(threshold="point", percentage=20)
    # mne 用于绘拓扑图 (CPU)
    mne_info = mne.create_info(CH_NAMES, sfreq=S_RATE, ch_types="eeg")
    mne_info.set_montage(mne.channels.make_standard_montage("standard_1020"))

    # 3-4. 正式遍历所有 subject / trial
    for subj in tqdm(subjects, desc="Cache"):
        heat_dir = os.path.join(PRE_ROOT, "Heat", subj)
        rp_dir   = os.path.join(PRE_ROOT, "RP",   subj)
        os.makedirs(heat_dir, exist_ok=True)
        os.makedirs(rp_dir,   exist_ok=True)

        with open(os.path.join(RAW_ROOT, f"{subj}.dat"), "rb") as f:
            raw = pickle.load(f, encoding="latin1")
            data   = raw["data"][:, :32, :]          # (40,32,8064)
            labels = raw["labels"][:, 0]             # valence

        for trial in range(40):
            valence = 0 if labels[trial] <= 5 else 1
            sig = data[trial][:, BASELINE_SEC*S_RATE:]       # (32,7680)

            # 预分配 (5,9,128,128)
            heat_trial = np.zeros((5, N_WINDOWS, 128, 128), np.float32)
            rp_trial   = np.zeros_like(heat_trial)

            for w in range(N_WINDOWS):
                st = w * STEP_SAMPLES
                seg = sig[:, st:st+WIN_SAMPLES]              # (32,1536)

                # (a) Heat – CUDA 加速求功率
                band_powers = compute_band_power_cuda(seg, freqs_t, masks)  # (5,32)
                # 绘制 5 张 128×128 灰度图
                for bi, power in enumerate(band_powers):
                    # 128×128 像素：2 inch × 64 dpi
                    fig, ax = plt.subplots(figsize=(2,2), dpi=64)
                    mne.viz.plot_topomap(
                        power, mne_info, axes=ax, show=False,
                        cmap="RdBu_r", contours=0,
                        image_interp="linear", extrapolate="box",
                        outlines=None, sphere=0.10, res=128
                    )
                    ax.axis("off")
                    fig.canvas.draw()
                    img = np.asarray(fig.canvas.buffer_rgba())[...,:3].mean(-1) / 255.
                    plt.close(fig)
                    heat_trial[bi, w] = img.astype(np.float32)

                # (b) RP – CPU，仅占少量时间
                pcs = pca.transform(seg.T).T                # (5,1536)
                rps = rp_gen.fit_transform(pcs)             # (5,1536,1536)
                for bi in range(5):
                    rp_trial[bi, w] = resize(
                        rps[bi], (128,128), order=1, anti_aliasing=True
                    ).astype(np.float32)

            # (c) 保存
            np.save(os.path.join(heat_dir, f"{subj}_t{trial:02d}_Heat.npy"), heat_trial)
            np.save(os.path.join(rp_dir,   f"{subj}_t{trial:02d}_RP.npy"),   rp_trial)
            with open(os.path.join(heat_dir, f"{subj}_labels.txt"), "a") as f:
                f.write(f"{subj}_t{trial:02d}_Heat.npy,{valence}\n")

    # 3-5. 写 split.json
    with open(SPLIT_JSON, "w") as fp:
        json.dump(split_info, fp, indent=2, ensure_ascii=False)
    print("✅  预处理完成，缓存位置:", PRE_ROOT)


# -------------------------------------------------- #
#  ⟁ 入口
# -------------------------------------------------- #
if __name__ == "__main__":
    preprocess_once()

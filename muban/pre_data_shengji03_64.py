#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEAP 预处理（CUDA 版）
──────────────────
输出:
deap_pre/
├── Heat/<subject>/sXX_tYY_Heat.npy   (5,9,128,128)
├── RP/<subject>/sXX_tYY_RP.npy       (5,9,128,128)
└── split.json   (subjects 列表 + 各 subject PCA 参数)

2025-05-27  •  tested with PyTorch 2.3 + CUDA 12.1
"""

import os
import json
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
import torch
from sklearn.decomposition import PCA
from pyts.image import RecurrencePlot
from skimage.transform import resize

# 自动选择设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
print("🤖  running on", DEVICE)

# ------------------------------ #
# 1. 全局常数
# ------------------------------ #
RAW_ROOT   = "/root/deap/data_preprocessed_python"
PRE_ROOT   = "/root/autodl-tmp/deap_pre64_single"
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
N_WINDOWS    = 9                         # 每 trial 的滑窗数

# ------------------------------ #
# 2. GPU 加速计算频段功率函数
# ------------------------------ #
def compute_band_power_cuda(seg_np: np.ndarray,
                            freqs_t: torch.Tensor,
                            masks: torch.Tensor) -> np.ndarray:
    """
    seg_np : (32,1536) float32
    freqs_t : (769,) float32
    masks   : (5,769) complex64
    返回   : (5,32) float32
    """
    seg = torch.as_tensor(seg_np, device=DEVICE)                  # (32,1536)
    seg_fft = torch.fft.rfft(seg, dim=1)                          # (32,769)
    # (5,769) × (32,769) -> (5,32,769)
    sig_band = torch.fft.irfft(
        seg_fft.unsqueeze(0) * masks.unsqueeze(1),
        n=WIN_SAMPLES, dim=-1
    )  # (5,32,1536)
    power = sig_band.pow(2).mean(dim=-1)                          # (5,32)
    return power.cpu().numpy().astype(np.float32)

# ------------------------------ #
# 3. 主流程
# ------------------------------ #
def preprocess_once():
    # 已存在则跳过
    if os.path.isdir(PRE_ROOT) and os.path.isfile(SPLIT_JSON):
        print("缓存已存在，跳过。")
        return

    # 创建目录
    os.makedirs(os.path.join(PRE_ROOT, "Heat"), exist_ok=True)
    os.makedirs(os.path.join(PRE_ROOT, "RP"),   exist_ok=True)

    # 收集所有 subject
    subjects = sorted(f[:-4] for f in os.listdir(RAW_ROOT) if f.endswith(".dat"))
    split_info = {"subjects": subjects, "pca": {}}

    # 准备共享资源
    freqs_t = torch.fft.rfftfreq(WIN_SAMPLES, 1 / S_RATE).to(DEVICE)  # (769,)
    masks = []
    for fmin, fmax in BANDS.values():
        m = ((freqs_t >= fmin) & (freqs_t < fmax)).to(torch.complex64)
        masks.append(m)
    masks = torch.stack(masks, dim=0)  # (5,769)
    rp_gen = RecurrencePlot(threshold="point", percentage=20)
    mne_info = mne.create_info(CH_NAMES, sfreq=S_RATE, ch_types="eeg")
    mne_info.set_montage(mne.channels.make_standard_montage("standard_1020"))

    # 遍历每个 subject
    for subj in tqdm(subjects, desc="Subject"):
        # 读取原始数据
        with open(os.path.join(RAW_ROOT, f"{subj}.dat"), "rb") as f:
            raw = pickle.load(f, encoding="latin1")
        data   = raw["data"][:, :32, :]    # (40,32,8064)
        labels = raw["labels"][:, 0]       # (40,)

        # 收集该 subject 的所有滑窗段，用于 PCA
        corpus = []
        for trial in range(40):
            sig = data[trial][:, BASELINE_SEC*S_RATE:]  # (32,7680)
            for w in range(N_WINDOWS):
                st = w * STEP_SAMPLES
                seg = sig[:, st:st+WIN_SAMPLES]         # (32,1536)
                corpus.append(seg.T)                    # (1536,32)
        corpus = np.vstack(corpus).astype(np.float32)  # (40*9*1536,32)

        # 在被试内拟合 PCA —— 降到 5 维
        pca = PCA(n_components=5, svd_solver="full", random_state=0)
        pca.fit(corpus)
        split_info["pca"][subj] = {
            "mean":       pca.mean_.tolist(),
            "components": pca.components_.tolist()
        }

        # 创建输出子目录
        heat_dir = os.path.join(PRE_ROOT, "Heat", subj)
        rp_dir   = os.path.join(PRE_ROOT, "RP",   subj)
        os.makedirs(heat_dir, exist_ok=True)
        os.makedirs(rp_dir,   exist_ok=True)

        # 逐 trial / 滑窗生成 Heat & RP
        for trial in range(40):
            valence = 0 if labels[trial] <= 5 else 1
            sig = data[trial][:, BASELINE_SEC*S_RATE:]  # (32,7680)

            heat_trial = np.zeros((5, N_WINDOWS, 128, 128), np.float32)
            rp_trial   = np.zeros_like(heat_trial)

            for w in range(N_WINDOWS):
                st = w * STEP_SAMPLES
                seg = sig[:, st:st+WIN_SAMPLES]        # (32,1536)

                # —— (a) Heat: 5 频段 Topomap
                band_pows = compute_band_power_cuda(seg, freqs_t, masks)  # (5,32)
                for bi, power in enumerate(band_pows):
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

                # —— (b) RP: PCA 降维后生成 RecurrencePlot
                pcs = pca.transform(seg.T).T           # (5,1536)
                rps = rp_gen.fit_transform(pcs)        # (5,1536,1536)
                for bi in range(5):
                    rp_trial[bi, w] = resize(
                        rps[bi], (128,128),
                        order=1, anti_aliasing=True
                    ).astype(np.float32)

            # 保存 npy & labels
            np.save(os.path.join(heat_dir, f"{subj}_t{trial:02d}_Heat.npy"), heat_trial)
            np.save(os.path.join(rp_dir,   f"{subj}_t{trial:02d}_RP.npy"),   rp_trial)
            with open(os.path.join(heat_dir, f"{subj}_labels.txt"), "a") as f:
                f.write(f"{subj}_t{trial:02d}_Heat.npy,{valence}\n")

    # 写入 split.json
    with open(SPLIT_JSON, "w") as fp:
        json.dump(split_info, fp, indent=2, ensure_ascii=False)
    print("✅  预处理完成，缓存位置:", PRE_ROOT)


if __name__ == "__main__":
    preprocess_once()

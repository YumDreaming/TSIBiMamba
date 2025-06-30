#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEAP 预处理（CUDA+被试内5折 PCA，无泄露）
────────────────────────────────────────
输出:
deap_pre/
├── Heat/<subject>/sXX_tYY_Heat.npy       # (5,9,128,128), 所有折共用
├── RP/fold_k/<subject>/sXX_tYY_RP.npy    # (5,9,128,128)，k=1..5
└── split.json   (记录每 subject 每折的 train/val 窗口索引 + PCA 参数)

2025-05-27  •  PyTorch 2.3 + CUDA 12.1
"""

import os, json, pickle
from tqdm import tqdm

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from pyts.image import RecurrencePlot
from skimage.transform import resize

# 设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
print("🤖 running on", DEVICE)

# ----- 全局常数 ----- #
RAW_ROOT   = "/root/deap/data_preprocessed_python"
PRE_ROOT   = "/root/autodl-tmp/deap_pre64_single"
SPLIT_JSON = os.path.join(PRE_ROOT, "split.json")

CH_NAMES = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7',
            'CP5','CP1','P3','P7','PO3','O1','Oz','Pz',
            'Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz',
            'C4','T8','CP6','CP2','P4','P8','PO4','O2']
BANDS = {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,45)}

S_RATE       = 128
BASELINE_SEC = 3
WIN_SEC      = 12
OVERLAP_SEC  = 6
WIN_SAMPLES  = WIN_SEC * S_RATE            # 1536
STEP_SAMPLES = WIN_SAMPLES - OVERLAP_SEC*S_RATE   # 768
N_WINDOWS    = 9

# ----- GPU 计算 5 波段功率 ----- #
def compute_band_power_cuda(seg_np, freqs_t, masks):
    """
    seg_np  : (32,1536)
    freqs_t : (769,)
    masks   : (5,769)
    返回    : (5,32)
    """
    seg = torch.as_tensor(seg_np, device=DEVICE)            # (32,1536)
    seg_fft = torch.fft.rfft(seg, dim=1)                    # (32,769)
    # 广播相乘 -> (5,32,769) -> irfft -> (5,32,1536)
    sig = torch.fft.irfft(seg_fft.unsqueeze(0)*masks.unsqueeze(1),
                          n=WIN_SAMPLES, dim=-1)
    power = sig.pow(2).mean(dim=-1)                         # (5,32)
    return power.cpu().numpy().astype(np.float32)

# ----- 主流程 ----- #
def preprocess_once():
    if os.path.isdir(PRE_ROOT) and os.path.isfile(SPLIT_JSON):
        print("缓存已存在，跳过。")
        return
    os.makedirs(os.path.join(PRE_ROOT,"Heat"), exist_ok=True)
    # RP 分 5 折
    for k in range(1,6):
        os.makedirs(os.path.join(PRE_ROOT,f"RP/fold_{k}"), exist_ok=True)

    # 载入 subjects
    subjects = sorted(f[:-4] for f in os.listdir(RAW_ROOT) if f.endswith(".dat"))
    split_info = {"folds":{}, "subjects": subjects}

    # 共享资源
    freqs_t = torch.fft.rfftfreq(WIN_SAMPLES,1/S_RATE).to(DEVICE)  # (769,)
    masks = torch.stack([
        ((freqs_t>=fmin)&(freqs_t<fmax)).to(torch.complex64)
        for fmin,fmax in BANDS.values()
    ], dim=0)  # (5,769)
    rp_gen = RecurrencePlot(threshold="point", percentage=20)
    mne_info = mne.create_info(CH_NAMES, sfreq=S_RATE, ch_types="eeg")
    mne_info.set_montage(mne.channels.make_standard_montage("standard_1020"))

    # 对每个 subject
    for subj in tqdm(subjects, desc="Subject"):
        # 读原始信号
        with open(os.path.join(RAW_ROOT,f"{subj}.dat"),"rb") as f:
            raw = pickle.load(f,encoding="latin1")
        data   = raw["data"][:, :32, :]   # (40,32,8064)
        labels = raw["labels"][:,0]       # (40,)

        # 一次生成 Heat（不依赖 PCA，所有折共用）
        heat_dir = os.path.join(PRE_ROOT,"Heat",subj)
        os.makedirs(heat_dir,exist_ok=True)
        for trial in range(40):
            valence = 0 if labels[trial]<=5 else 1
            sig = data[trial][:, BASELINE_SEC*S_RATE:]  # (32,7680)
            heat_trial = np.zeros((5,N_WINDOWS,128,128),np.float32)
            for w in range(N_WINDOWS):
                seg = sig[:, w*STEP_SAMPLES : w*STEP_SAMPLES+WIN_SAMPLES]
                band_pows = compute_band_power_cuda(seg, freqs_t, masks)  # (5,32)
                for bi,power in enumerate(band_pows):
                    fig,ax = plt.subplots(figsize=(2,2),dpi=64)
                    mne.viz.plot_topomap(
                        power, mne_info, axes=ax, show=False,
                        cmap="RdBu_r", contours=0,
                        image_interp="linear", extrapolate="box",
                        outlines=None, sphere=0.10, res=128
                    )
                    ax.axis("off")
                    fig.canvas.draw()
                    img = np.asarray(fig.canvas.buffer_rgba())[...,:3].mean(-1)/255.
                    plt.close(fig)
                    heat_trial[bi,w] = img.astype(np.float32)
            np.save(os.path.join(heat_dir,f"{subj}_t{trial:02d}_Heat.npy"), heat_trial)
            with open(os.path.join(heat_dir,f"{subj}_labels.txt"),"a") as f:
                f.write(f"{subj}_t{trial:02d}_Heat.npy,{valence}\n")

        # 收集所有窗口段，用于后续每折 PCA
        window_list = []  # [(trial, w, seg_np)]
        for trial in range(40):
            sig = data[trial][:, BASELINE_SEC*S_RATE:]
            for w in range(N_WINDOWS):
                seg = sig[:, w*STEP_SAMPLES : w*STEP_SAMPLES+WIN_SAMPLES]  # (32,1536)
                window_list.append((trial,w,seg.copy()))
        n_win = len(window_list)  # 40*9=360

        # 5 折划分
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        split_info["folds"][subj] = {}
        for fold_id, (train_idx, val_idx) in enumerate(kf.split(range(n_win)), start=1):
            # 在该折的训练窗口上拟合 PCA
            corpus = np.vstack([window_list[i][2].T for i in train_idx]).astype(np.float32)
            pca = PCA(n_components=5,svd_solver="full",random_state=0).fit(corpus)
            # 记录 PCA 参数
            split_info["folds"][subj][f"fold_{fold_id}"] = {
                "train_windows": train_idx.tolist(),
                "val_windows":   val_idx.tolist(),
                "pca_mean":       pca.mean_.tolist(),
                "pca_components": pca.components_.tolist()
            }

            # 生成 RP：对训练 & 验证窗口都用同一个 PCA
            rp_fold_dir = os.path.join(PRE_ROOT,f"RP/fold_{fold_id}",subj)
            os.makedirs(rp_fold_dir, exist_ok=True)
            for i in np.concatenate([train_idx, val_idx]):
                trial, w, seg = window_list[i]
                pcs = pca.transform(seg.T).T            # (5,1536)
                rps = rp_gen.fit_transform(pcs)         # (5,1536,1536)
                rp_small = np.zeros((5,128,128),np.float32)
                for bi in range(5):
                    rp_small[bi] = resize(rps[bi], (128,128),
                                          order=1,anti_aliasing=True)
                np.save(os.path.join(
                    rp_fold_dir, f"{subj}_t{trial:02d}_w{w:02d}_RP.npy"
                ), rp_small)

    # 写 split.json
    with open(SPLIT_JSON,"w") as fp:
        json.dump(split_info, fp, indent=2, ensure_ascii=False)
    print("✅  预处理完成，缓存位置:", PRE_ROOT)


if __name__=="__main__":
    preprocess_once()

#!/usr/bin/env python3
import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
import pywt
from sklearn.decomposition import PCA
from tqdm import tqdm

def make_electrode_grid(grid_size, device):
    """生成 32 电极的单位圆坐标和插值网格点"""
    angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
    elec_pos = np.stack([np.cos(angles), np.sin(angles)], axis=1)    # (32,2)
    elec_pos_t = torch.tensor(elec_pos, dtype=torch.float32, device=device)

    gx = torch.linspace(-1, 1, grid_size, device=device, dtype=torch.float32)
    gy = torch.linspace(-1, 1, grid_size, device=device, dtype=torch.float32)
    gx, gy = torch.meshgrid(gx, gy, indexing='xy')
    grid_pts = torch.stack([gx.flatten(), gy.flatten()], dim=1)      # (G^2,2)
    return elec_pos_t, grid_pts

def main():
    # —— 参数设定 ——
    file_path     = '/root/deap/data_preprocessed_python/s01.dat'  # 修改为你本地路径
    trial_idx     = 0
    sample_rate   = 128
    window_length = 12  # 秒
    overlap       = 6   # 秒
    pic_size      = 128
    num_scales    = 5

    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 45)
    }

    # —— 设备 ——
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # —— 加载数据 & 去基线 ——
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')['data']  # (40,40,8064)
    eeg = torch.tensor(data[trial_idx, :32, :], dtype=torch.float32, device=device)  # (32,8064)
    eeg = eeg[:, 3 * sample_rate :]  # 去掉前 3s 基线 → (32,7680)

    # —— 计算窗口参数 ——
    win_samples = window_length * sample_rate
    step        = win_samples - overlap * sample_rate
    num_wins    = (eeg.shape[1] - win_samples) // step + 1
    print(f"Total windows per trial: {num_wins}")

    # —— 电极网格 ——
    elec_pos_t, grid_pts = make_electrode_grid(pic_size, device)

    # —— PCA 选通道 ——
    trial_np = eeg.cpu().numpy().T  # (time,32)
    pca      = PCA(n_components=1).fit(trial_np)
    loadings = pca.components_[0]
    idxs     = np.argsort(np.abs(loadings))[::-1][:num_scales]
    sel_chs  = idxs.tolist()
    print(f"Selected channels by PCA: {sel_chs}")

    # 这里只取第一个窗口 i=0 做示例
    i   = 0
    seg = eeg[:, i * step : i * step + win_samples]  # (32, win_samples)

    # —— 1) 多频段头皮拓扑 ——
    freqs    = torch.fft.rfftfreq(win_samples, 1/sample_rate).to(device=device, dtype=torch.float32)
    seg_fft  = torch.fft.rfft(seg, dim=1)
    topo_maps = []
    for band, (f1, f2) in bands.items():
        mask   = ((freqs >= f1) & (freqs < f2)).to(torch.complex64)
        sig_t  = torch.fft.irfft(seg_fft * mask.unsqueeze(0), n=win_samples, dim=1)
        power  = sig_t.pow(2).mean(dim=1)  # (32,)

        # IDW 插值
        dists  = torch.cdist(grid_pts, elec_pos_t)  # (G^2,32)
        weights= 1.0 / (dists + 1e-6)
        weights= weights / weights.sum(dim=1, keepdim=True)
        img    = (weights * power.unsqueeze(0)).sum(dim=1).reshape(pic_size, pic_size)
        # 单图归一化
        arr    = img.cpu().numpy()
        arr    = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        topo_maps.append((band, arr))

    # —— 2) CWT 多尺度 Scalogram ——
    scales   = np.arange(1, pic_size+1)
    cwt_maps = []
    for ch in sel_chs:
        sig     = seg[ch].cpu().numpy()
        coeffs, _ = pywt.cwt(sig, scales, 'morl', sampling_period=1/sample_rate)  # (128,win_samples)
        C       = torch.tensor(coeffs, dtype=torch.float32, device=device)
        C       = C.abs().unsqueeze(0).unsqueeze(0)  # (1,1,128,win_samples)
        C_res   = torch.nn.functional.interpolate(C, size=(pic_size, pic_size),
                                                  mode='bilinear', align_corners=False)[0,0]
        # 对数 + 单图归一化
        C_log   = torch.log1p(C_res)
        C_norm  = (C_log - C_log.min()) / (C_log.max() - C_log.min() + 1e-8)
        cwt_maps.append((ch, C_norm.cpu().numpy()))

    # —— 可视化 ——
    # 画 Topo
    for band, arr in topo_maps:
        plt.figure(figsize=(4,4))
        plt.imshow(arr, origin='lower', extent=(-1,1,-1,1))
        plt.title(f'Topo: {band}')
        plt.axis('off')

    # 画 CWT
    for ch, arr in cwt_maps:
        plt.figure(figsize=(4,4))
        plt.imshow(arr, origin='lower', aspect='auto')
        plt.title(f'CWT (log+norm): Ch {ch}')
        plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()

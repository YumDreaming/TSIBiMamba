#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter

# -------------------------- 过滤器 ---------------------------

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# ------------------------ 读入数据 ---------------------------

def read_mat(file_path):
    """读取 DEAP 预处理好的 matlab 数据，返回 data[trial, channel, sample]"""
    mat = sio.loadmat(file_path)
    return mat['data']

# ------------------------ 特征计算 ---------------------------

def compute_DE(signal):
    """计算单段信号的 Differential Entropy"""
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2

def compute_PSD(signal):
    """计算单段信号的 PSD 能量和 (sum of squares)"""
    return np.sum(signal ** 2)

# ------------------------ 分解函数 ---------------------------

def decompose_feature(file_path, feature='DE'):
    """
    对单个文件进行分解：
    - feature='DE' 计算 4 波段 DE，共 120 段
    - feature='PSD' 计算 4 波段 PSD，共 120 段
    返回：
      base_feat: shape (40, 4*32)
      trial_feat: shape (4800, 4*32)
    """
    data = read_mat(file_path)    # (40 trials, 32 ch, 8064 samp)
    n_trials, n_chan, n_samp = data.shape
    fs = 128
    # 用于存储
    all_base = []
    all_trials = []

    for tr in range(n_trials):
        base_feats = []    # 暂存一个 trial 的 32*4 个基线值
        trial_feats = []   # 暂存一个 trial 的 120*4 波段值，按 channel 堆叠

        for ch in range(n_chan):
            sig = data[tr, ch, :]
            base_sig = sig[:384]       # 3s 预试信号
            task_sig = sig[384:]       # 60s 任务信号

            # 各波段滤波
            bands = {
                'theta': butter_bandpass_filter,
                'alpha': butter_bandpass_filter,
                'beta':  butter_bandpass_filter,
                'gamma': butter_bandpass_filter
            }
            params = {
                'theta': (4, 8),
                'alpha': (8, 14),
                'beta':  (14, 31),
                'gamma': (31, 45)
            }

            # 先计算基线：6 段，各 64 点，取平均
            base_vals = []
            for band, filt in bands.items():
                low, high = params[band]
                filtered = filt(base_sig, low, high, fs, order=3)
                seg_vals = []
                for i in range(6):
                    seg = filtered[i*64:(i+1)*64]
                    val = compute_DE(seg) if feature=='DE' else compute_PSD(seg)
                    seg_vals.append(val)
                base_vals.append(np.mean(seg_vals))
            base_feats.extend(base_vals)

            # 再计算任务段：120 段，各 64 点，不取平均
            for band, filt in bands.items():
                low, high = params[band]
                filtered = filt(task_sig, low, high, fs, order=3)
                for i in range(120):
                    seg = filtered[i*64:(i+1)*64]
                    val = compute_DE(seg) if feature=='DE' else compute_PSD(seg)
                    trial_feats.append(val)

        all_base.append(base_feats)      # 32*4
        # trial_feats 列表长度 = 32*120*1 波段数4  → reshape 为 (120, 4, 32) 再展平为 (120*4*32,)
        trial_arr = np.array(trial_feats).reshape(32, 4, 120).transpose(2,1,0).reshape(-1)
        all_trials.append(trial_arr)

        print(f"[{feature}] processed trial {tr+1}/{n_trials}")

    base_feat = np.vstack(all_base)      # (40, 4*32)
    trial_feat = np.vstack(all_trials)   # (4800, 4*32)
    return base_feat, trial_feat

# ------------------------ 标签读取 ---------------------------

def get_labels(file_path):
    """
    从原始文件中读取 labels，生成 4800 个段的二分类标签 (valence, arousal)
    """
    mat = sio.loadmat(file_path)
    raw = mat['labels']    # (40, 4)
    val = raw[:,0] > 5
    aro = raw[:,1] > 5

    # 每 trial 重复 120 次
    val_labels = np.repeat(val, 120)
    aro_labels = np.repeat(aro, 120)
    return aro_labels.astype(int), val_labels.astype(int)

# ---------------------------- 主程序 ---------------------------

def main():
    input_dir  = '/root/deap/data_preprocessed_matlab'
    output_dir = '/root/autodl-tmp/deap_map'

    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.endswith('.mat'):
            continue
        path = os.path.join(input_dir, fname)
        print(f"\n=== Processing file: {fname} ===")

        # 分支 g：DE
        base_de, trial_de = decompose_feature(path, feature='DE')
        aro, val = get_labels(path)
        out_de = os.path.join(output_dir, f'DE_{fname}')
        sio.savemat(out_de, {
            'base_data':       base_de,
            'data':            trial_de,
            'valence_labels':  val,
            'arousal_labels':  aro
        })
        print(f"  → DE saved to {out_de}")

        # 分支 r：PSD
        base_psd, trial_psd = decompose_feature(path, feature='PSD')
        # 标签同上
        out_psd = os.path.join(output_dir, f'PSD_{fname}')
        sio.savemat(out_psd, {
            'base_data':       base_psd,
            'data':            trial_psd,
            'valence_labels':  val,
            'arousal_labels':  aro
        })
        print(f"  → PSD saved to {out_psd}")

if __name__ == '__main__':
    main()

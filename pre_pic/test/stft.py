#!/usr/bin/env python3
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

# ==== 参数配置 ====
file_path    = '/root/deap/data_preprocessed_python/s01.dat'
trial_idx    = 0    # 第 1 个 trial
sample_rate  = 128
baseline_sec = 3
n_fft        = 256
hop_length   = n_fft // 2

# 1. 加载并截取信号
with open(file_path, 'rb') as f:
    raw = pickle.load(f, encoding='latin1')
data = raw['data']  # (40, 40, 8064)
eeg = data[trial_idx, :32, :]  # (32, 8064)
# 去基线
baseline_samples = baseline_sec * sample_rate
seg = eeg[:, baseline_samples:]  # (32, 7680)
seg_t = torch.tensor(seg, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

# 2. 计算 STFT
stft_list = []
for ch in range(seg_t.shape[0]):
    sig = seg_t[ch]
    stft = torch.stft(sig, n_fft=n_fft, hop_length=hop_length,
                      return_complex=True, center=False)  # (freq_bins, time_frames)
    mag = stft.abs()
    mag = torch.log1p(mag)  # 对数放大
    stft_list.append(mag.cpu().numpy())

stft_arr = np.stack(stft_list, axis=0)  # (32, freq_bins, time_frames)

# 3. 绘图
n_channels, n_freq, n_time = stft_arr.shape
n_cols = 4
n_rows = n_channels // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
for idx in range(n_channels):
    r, c = divmod(idx, n_cols)
    ax = axes[r, c]
    ax.imshow(stft_arr[idx], origin='lower', aspect='auto')
    ax.set_title(f'Ch {idx+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()

print("STFT branch shape:", stft_arr.shape)

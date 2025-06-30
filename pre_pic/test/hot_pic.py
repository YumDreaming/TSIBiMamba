import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import mne

# ==== 参数配置 ====
file_path    = '/root/deap/data_preprocessed_python/s01.dat'
trial_idx    = 0    # 第 1 个 trial
window_idx   = 0    # 第 1 个 窗口
sample_rate  = 128
baseline_sec = 3
window_sec   = 12
overlap_sec  = 6

# DEAP 32 通道名称（10-20 系统）
ch_names = [
    'Fp1','AF3','F3','F7','FC5','FC1','C3','T7',
    'CP5','CP1','P3','P7','PO3','O1','Oz','Pz',
    'Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz',
    'C4','T8','CP6','CP2','P4','P8','PO4','O2'
]

# 频段定义
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}

# ==== 1. 加载并截取信号 ====
with open(file_path, 'rb') as f:
    raw = pickle.load(f, encoding='latin1')

data = raw['data']  # shape = (40, 40, 8064)
eeg = data[trial_idx, :32, :]  # (32, 8064)

# 去基线 & 切第 window_idx 个窗口
baseline_samples = baseline_sec * sample_rate
eeg = eeg[:, baseline_samples:]  # (32, 7680)
win_samples = window_sec * sample_rate
step = win_samples - overlap_sec * sample_rate
seg = eeg[:, window_idx * step : window_idx * step + win_samples]  # (32, 1536)

# 转成 torch Tensor（CPU 计算）
seg_t = torch.tensor(seg, dtype=torch.float32)

# ==== 2. 计算频段功率 ====
n = seg_t.shape[1]
freqs = torch.fft.rfftfreq(n, 1/sample_rate)
seg_fft = torch.fft.rfft(seg_t, dim=1)

band_powers = []
for name, (fmin, fmax) in bands.items():
    mask     = ((freqs >= fmin) & (freqs < fmax)).to(torch.complex64)
    sig_band = torch.fft.irfft(seg_fft * mask.unsqueeze(0), n=n, dim=1)
    power    = sig_band.pow(2).mean(dim=1).numpy()  # (32,)
    band_powers.append(power)
band_powers = np.stack(band_powers, axis=0)  # (5, 32)

# ==== 3. 构建 MNE Info 与 Montage ====
sfreq   = sample_rate
info    = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# ==== 4. 绘制 5 张独立 Topo 图（128×128，无大脑轮廓，颜色填满画布） ====
for idx, (band_name, power) in enumerate(zip(bands.keys(), band_powers)):
    fig, ax = plt.subplots(figsize=(4, 4))
    mne.viz.plot_topomap(
        power, info, axes=ax, show=False,
        cmap='RdBu_r', contours=0,
        image_interp='linear',
        extrapolate='box',
        outlines=None,  # 不画头轮廓
        mask=None,
        sphere=0.10,
        res=128
    )
    ax.set_title(f'{band_name} band')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1. 文件路径
data_dir = r'D:\dachuang\data_preprocessed_python'
fname = os.path.join(data_dir, 's02.dat')

# 2. 加载 .dat 文件
with open(fname, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# 3. 提取信号
signals = data['data']               # shape: (n_trials, n_channels, n_samples)
n_trials, n_channels, n_samples = signals.shape
print(f'Loaded data shape: {signals.shape}')

# 选择要看的 trial（这里以第 1 个 trial 为例）
trial_idx = 0
trial_data = signals[trial_idx]      # shape: (40, 8064)

# 分离 EEG（通道 0–31）和 GSR（通道 36）
eeg = trial_data[0:32, :]            # (32, 8064)
gsr = trial_data[36, :]              # (8064,)

# 4. 构造时间轴（DEAP 采样率 128 Hz）
fs = 128.0
times = np.arange(n_samples) / fs    # 单位：秒

# 5. 为每个 EEG 通道分别绘图
for ch in range(32):
    plt.figure(figsize=(8, 3))
    plt.plot(times, eeg[ch, :], linewidth=0.8)
    plt.title(f'Trial {trial_idx+1} — EEG 通道 {ch+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.tight_layout()   # 只有一个子图，这里不会报错
    # 如果需要保存文件，取消下一行注释并指定路径
    # plt.savefig(f'EEG_Trial{trial_idx+1}_Ch{ch+1:02d}.png', dpi=150)
    plt.show()
    plt.close()

# 6. 为 GSR 通道单独绘图
plt.figure(figsize=(8, 3))
plt.plot(times, gsr, linewidth=0.8)
plt.title(f'Trial {trial_idx+1} — GSR 通道 (Ch37)')
plt.xlabel('Time (s)')
plt.ylabel('GSR Signal')
plt.tight_layout()
# plt.savefig(f'GSR_Trial{trial_idx+1}_Ch37.png', dpi=150)
plt.show()
plt.close()

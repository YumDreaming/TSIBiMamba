import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# 1) 定义文件路径
dat_path = r"D:\dachuang\data_preprocessed_python\s04.dat"
if not os.path.exists(dat_path):
    raise FileNotFoundError(f"找不到文件：{dat_path}")

# 2) 加载数据
with open(dat_path, "rb") as f:
    data_dict = pickle.load(f, encoding="latin1")

# raw_data.shape == (40 trials, 40 channels, 8064 samples)
raw_data = data_dict['data']

# 3) 计算每个 EEG 通道的峰值
#    EEG 通道在前 32 个通道（0-based index 0–31）
peaks = []
for ch in range(1):
    # 取出该通道在所有 trial 上的所有采样点
    ch_data = raw_data[:, 36, :]             # shape (40, 8064)
    peak_val = np.max(np.abs(ch_data))       # max|value|
    peaks.append(peak_val)

# 4) 可视化
plt.figure(figsize=(12, 5))
plt.bar(np.arange(1, 33), peaks)
plt.xticks(np.arange(1, 33))
plt.xlabel("EEG Channel (1–32)")
plt.ylabel("Peak Amplitude (absolute value)")
plt.title("Subject s01: EEG 32 Channels Peak Amplitudes")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

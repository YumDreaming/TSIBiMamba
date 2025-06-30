import pickle
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from skimage.transform import resize

# ==== 配置 ====
file_path    = '/root/deap/data_preprocessed_python/s01.dat'
trial_idx    = 0    # 第 1 个 trial
window_idx   = 0    # 第 1 个 窗口
sample_rate  = 128
baseline_sec = 3
window_sec   = 12
overlap_sec  = 6

# 1. 加载并截取信号
with open(file_path, 'rb') as f:
    raw = pickle.load(f, encoding='latin1')
data = raw['data']           # (40,40,8064)
signal = data[trial_idx, :32, :]  # 32 通道

# 去基线 & 切第一个窗口
signal = signal[:, baseline_sec*sample_rate:]
win_len = window_sec * sample_rate
step    = win_len - overlap_sec * sample_rate
seg = signal[:, window_idx*step : window_idx*step + win_len]  # (32,1536)

# 2. 生成 Recurrence Plot（只计算一次）
rp_gen = RecurrencePlot(threshold='point', percentage=20)
# pyts 要求输入 shape (n_samples, n_timestamps), 这里每通道作为一个样本
rps = rp_gen.fit_transform(seg)  # (32,1536,1536)

# 3. 选前 5 通道并下采样到 128x128
rp5 = rps[:5]  # 5 通道
rp5_resized = np.array([
    resize(rp, (128, 128), order=1, anti_aliasing=True)
    for rp in rp5
])

# 4. 可视化
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(rp5_resized[i], cmap='gray', origin='lower')
    ax.set_title(f'Channel {i}')
    ax.axis('off')
plt.tight_layout()
plt.show()

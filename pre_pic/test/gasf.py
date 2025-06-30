#!/usr/bin/env python3
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# —— 配置 ——
DATA_PATH   = '/root/deap/data_preprocessed_python'
SUBJECT     = 's01.dat'
TRIAL_IDX   = 0        # 第 1 个 trial
WINDOW_SEC  = 12
OVERLAP_SEC = 6
FS          = 128
PIC_SIZE    = 128
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载并切分第一个 trial
with open(os.path.join(DATA_PATH, SUBJECT), 'rb') as f:
    raw = pickle.load(f, encoding='latin1')
data = raw['data']           # (40,40,8064)
labels = raw['labels']       # (40,4)
signal = data[TRIAL_IDX,:32,:]  # (32,8064)

# 去基线
signal = signal[:, 3*FS:]    # (32,7680)
win_len = WINDOW_SEC * FS
step    = win_len - OVERLAP_SEC * FS
num_wins = (signal.shape[1] - win_len)//step + 1

# 2. 生成所有窗口的 Recurrence Plot
rp_generator = RecurrencePlot(threshold='point', percentage=10)  # 10% 最近邻
rp_images = []   # 存 (num_wins, 32, PIC_SIZE, PIC_SIZE)
for i in tqdm(range(num_wins), desc='RP Windows'):
    seg = signal[:, i*step : i*step+win_len]  # (32,1536)
    # 对每通道批量生成 RP
    # pyts expects shape (n_channels, n_timestamps)
    rps = rp_generator.fit_transform(seg)      # (32,1536,1536)
    # 转为 torch 并 resize
    t = torch.tensor(rps, dtype=torch.float32, device=DEVICE).unsqueeze(1)  # (32,1,1536,1536)
    t = F.interpolate(t, size=(PIC_SIZE,PIC_SIZE), mode='bilinear', align_corners=False)
    rp_images.append(t.cpu().numpy())   # list of arrays

rp_images = np.stack(rp_images, axis=0)  # (num_wins,32,128,128)
print("RP images shape:", rp_images.shape)

# 3. 可视化示例：前三个窗口 5 个通道
fig, axes = plt.subplots(3,5, figsize=(12,8))
for wi in range(3):
    for ch in range(5):
        axes[wi,ch].imshow(rp_images[wi,ch], cmap='gray', origin='lower')
        axes[wi,ch].set_title(f'W{wi} Ch{ch}')
        axes[wi,ch].axis('off')
plt.suptitle('Recurrence Plots (first 3 windows, first 5 channels)')
plt.tight_layout()
plt.show()

# 4. 简单分类验证：用 RQA 特征 + SVM
# 4.1 计算 RQA 指标（重现率 RR）作为特征
#    RQA RR = matrix.mean()
X, y = [], []
for win in range(num_wins):
    for ch in range(32):
        rp = rp_images[win,ch]
        rr = rp.mean()
        X.append(rr)
        lbl = 1 if labels[TRIAL_IDX][0] > 5 else 0
        y.append(lbl)
X = np.array(X).reshape(num_wins,32)  # (num_wins,32)
y = np.array(y[:num_wins])            # (num_wins,)

# 4.2 train/test split & SVM
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("RP+RR+SVM Accuracy:", accuracy_score(y_test, y_pred))

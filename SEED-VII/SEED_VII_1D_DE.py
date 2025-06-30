# -*- coding: utf-8 -*-
"""
SEED-VII 数据集：提取每个通道 5 个频段的 DE 特征，
并将 62 个通道转化为 8*9*5 的三维输入，
其中 8*9 表示 62 个通道转化后的二维平面，5 表示 5 个频段
输入：D:\dachuang\EEG_preprocessed\*.mat（每个文件 80 个 trial, 每 trial shape=(62, N)）
输出：D:/dachuang/EEG_preprocessed/DE/X.npy, y.npy, X89.npy
"""

import os
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter

# —— SEED-VII 的情绪到数字映射 ——
emotion2idx = {
    'Disgust': 0,
    'Fear': 1,
    'Sad': 2,
    'Neutral': 3,
    'Happy': 4,
    'Anger': 5,
    'Surprise': 6
}
# 四个 Session、20 个视频共 80 个标签
labels_text = [
    # Session1: 1–20
    'Happy', 'Neutral', 'Disgust', 'Sad', 'Anger',
    'Anger', 'Sad', 'Disgust', 'Neutral', 'Happy',
    'Happy', 'Neutral', 'Disgust', 'Sad', 'Anger',
    'Anger', 'Sad', 'Disgust', 'Neutral', 'Happy',
    # Session2: 21–40
    'Anger', 'Sad', 'Fear', 'Neutral', 'Surprise',
    'Surprise', 'Neutral', 'Fear', 'Sad', 'Anger',
    'Anger', 'Sad', 'Fear', 'Neutral', 'Surprise',
    'Surprise', 'Neutral', 'Fear', 'Sad', 'Anger',
    # Session3: 41–60
    'Happy', 'Surprise', 'Disgust', 'Fear', 'Anger',
    'Anger', 'Fear', 'Disgust', 'Surprise', 'Happy',
    'Happy', 'Surprise', 'Disgust', 'Fear', 'Anger',
    'Anger', 'Fear', 'Disgust', 'Surprise', 'Happy',
    # Session4: 61–80
    'Disgust', 'Sad', 'Fear', 'Surprise', 'Happy',
    'Happy', 'Surprise', 'Fear', 'Sad', 'Disgust',
    'Disgust', 'Sad', 'Fear', 'Surprise', 'Happy',
    'Happy', 'Surprise', 'Fear', 'Sad', 'Disgust',
]
# 转成数字标签列表
video_labels = np.array([emotion2idx[e] for e in labels_text], dtype=int)
assert video_labels.shape[0] == 80


# —— 帮助函数 ——
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


def compute_DE(signal):
    var = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * var) / 2


def decompose(mat_file):
    """
    对单个 .mat 文件（一个受试，共 80 个 trial）做 DE 特征提取
    返回：
      X_sub: shape = (sum_i num_segments_i, 62, 5)
      y_sub: shape = (sum_i num_segments_i,)
    """
    mat = sio.loadmat(mat_file)
    # 找出所有 trial 变量：二维数组、第一维=62
    trial_keys = [k for k, v in mat.items()
                  if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == 62]
    trial_keys.sort()  # 保证 Video1…Video80 顺序
    assert len(trial_keys) == 80, f"{os.path.basename(mat_file)} 中找到 {len(trial_keys)} 个 trial，需为 80"

    fs = 200  # 采样率
    all_X = np.empty((0, 62, 5), dtype=float)
    all_y = np.empty((0,), dtype=int)

    for idx, key in enumerate(trial_keys):
        data = mat[key]  # shape = (62, N)
        n_samples = data.shape[1]
        n_seg = n_samples // 100  # 100 pts = 0.5s @200Hz

        # 为每个 segment 生成标签
        lbl = np.full((n_seg,), video_labels[idx], dtype=int)

        # 为每个 channel、每个 band 计算 DE
        de_feats = np.zeros((62, 5, n_seg), dtype=float)
        for ch in range(62):
            sig = data[ch]
            d1 = butter_bandpass_filter(sig, 1, 4, fs, order=3)
            d2 = butter_bandpass_filter(sig, 4, 8, fs, order=3)
            d3 = butter_bandpass_filter(sig, 8, 14, fs, order=3)
            d4 = butter_bandpass_filter(sig, 14, 31, fs, order=3)
            d5 = butter_bandpass_filter(sig, 31, 51, fs, order=3)
            for s in range(n_seg):
                start, end = s * 100, (s + 1) * 100
                de_feats[ch, 0, s] = compute_DE(d1[start:end])
                de_feats[ch, 1, s] = compute_DE(d2[start:end])
                de_feats[ch, 2, s] = compute_DE(d3[start:end])
                de_feats[ch, 3, s] = compute_DE(d4[start:end])
                de_feats[ch, 4, s] = compute_DE(d5[start:end])

        # 变形为 (n_seg,62,5) 并累加
        trial_X = de_feats.transpose(2, 0, 1)
        all_X = np.vstack([all_X, trial_X])
        all_y = np.concatenate([all_y, lbl])

    return all_X, all_y


if __name__ == "__main__":
    # ——— 主流程 ———
    input_dir = r"D:\dachuang\EEG_preprocessed"
    output_dir = r"D:/dachuang/SEED_VII/DE"
    os.makedirs(output_dir, exist_ok=True)

    X = np.empty((0, 62, 5), dtype=float)
    y = np.empty((0,), dtype=int)

    # 遍历所有 .mat 文件
    mats = [f for f in os.listdir(input_dir) if f.endswith(".mat")]
    mats.sort()
    for fn in mats:
        fp = os.path.join(input_dir, fn)
        print("Processing", fn)
        X_sub, y_sub = decompose(fp)
        X = np.vstack([X, X_sub])
        y = np.concatenate([y, y_sub])

    # 保存 1D 特征
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    print("Saved X.npy/y.npy:", X.shape, y.shape)

    # ——— 生成 8×9×5 的空间布局 X89 ———
    N = y.shape[0]
    X89 = np.zeros((N, 8, 9, 5), dtype=float)
    # 按论文/原代码的电极映射
    X89[:, 0, 2, :] = X[:, 3, :]
    X89[:, 0, 3:6, :] = X[:, 0:3, :]
    X89[:, 0, 6, :] = X[:, 4, :]
    for i in range(5):
        X89[:, i + 1, :, :] = X[:, 5 + i * 9: 5 + (i + 1) * 9, :]
    X89[:, 6, 1:8, :] = X[:, 50:57, :]
    X89[:, 7, 2:7, :] = X[:, 57:62, :]

    np.save(os.path.join(output_dir, "X89.npy"), X89)
    print("Saved X89.npy:", X89.shape)

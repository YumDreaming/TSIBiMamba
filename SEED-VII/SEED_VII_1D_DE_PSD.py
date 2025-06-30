# -*- coding: utf-8 -*-
"""
SEED-VII 数据集预处理：对每个通道 5 个频段同时提取 DE 或 PSD（Welch）特征，
并将 62 通道重排成 8×9×5 的空间结构。
输入目录：D:\dachuang\EEG_preprocessed\*.mat
输出目录：D:/dachuang/SEED_VII/DE/ 和 …/PSD/
每个分支下生成 X.npy, y.npy, X89.npy
"""

import os
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter, welch

# —— 1. 情绪到数字映射 ——
emotion2idx = {
    'Disgust' : 0,
    'Fear'    : 1,
    'Sad'     : 2,
    'Neutral' : 3,
    'Happy'   : 4,
    'Anger'   : 5,
    'Surprise': 6
}

# 四个 Session、20 视频，共 80 标签
labels_text = [
    # S1: 1–20
    'Happy','Neutral','Disgust','Sad','Anger',
    'Anger','Sad','Disgust','Neutral','Happy',
    'Happy','Neutral','Disgust','Sad','Anger',
    'Anger','Sad','Disgust','Neutral','Happy',
    # S2: 21–40
    'Anger','Sad','Fear','Neutral','Surprise',
    'Surprise','Neutral','Fear','Sad','Anger',
    'Anger','Sad','Fear','Neutral','Surprise',
    'Surprise','Neutral','Fear','Sad','Anger',
    # S3: 41–60
    'Happy','Surprise','Disgust','Fear','Anger',
    'Anger','Fear','Disgust','Surprise','Happy',
    'Happy','Surprise','Disgust','Fear','Anger',
    'Anger','Fear','Disgust','Surprise','Happy',
    # S4: 61–80
    'Disgust','Sad','Fear','Surprise','Happy',
    'Happy','Surprise','Fear','Sad','Disgust',
    'Disgust','Sad','Fear','Surprise','Happy',
    'Happy','Surprise','Fear','Sad','Disgust',
]
# 转为 numpy 数组，方便索引
video_labels = np.array([emotion2idx[e] for e in labels_text], dtype=int)
assert video_labels.shape[0] == 80, "标签长度应为80"

# —— 2. 带通滤波和 DE 计算 ——
def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    生成带通滤波器系数
    """
    nyq = fs * 0.5
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    """
    对一维信号 data 做带通滤波
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def compute_DE(segment):
    """
    计算单段信号的 Differential Entropy(DE)
    DE = 0.5 * ln(2πe * var)
    """
    var = np.var(segment, ddof=1)
    return 0.5 * math.log(2 * math.pi * math.e * var)

# —— 3. Welch PSD 计算 ——
def compute_PSD(segment, sfreq, band):
    """
    用 Welch 方法估计单段信号在指定频段的功率
      - segment: 1D numpy array
      - sfreq: 采样率 (Hz)
      - band: (low, high) 频带
    返回带内总功率
    """
    # 用 nperseg = segment 长度，保证频率分辨率最大
    freqs, psd = welch(segment,
                       fs=sfreq,
                       window='hann',
                       nperseg=len(segment),
                       noverlap=0,
                       scaling='density')
    # 选出带内索引
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    # 频率分辨率
    df = freqs[1] - freqs[0]
    # 数值积分（矩形近似）
    return np.sum(psd[idx]) * df

# —— 4. 正式分解函数 ——
def decompose(mat_file, feature='DE'):
    """
    对单个 .mat 文件做特征提取
    Parameters
    ----------
    mat_file : str
        .mat 文件路径，包含 80 个 trial，每 trial shape=(62, N)
    feature : {'DE','PSD'}
        选择特征类型

    Returns
    -------
    X_all : np.ndarray, shape=(total_segments, 62, 5)
    y_all : np.ndarray, shape=(total_segments,)
    """
    mat = sio.loadmat(mat_file)
    # 找到所有 trial（62通道、二维数组）
    trials = [k for k,v in mat.items()
              if isinstance(v, np.ndarray) and v.ndim==2 and v.shape[0]==62]
    trials.sort()
    assert len(trials)==80, f"{os.path.basename(mat_file)} 找到 {len(trials)} 个 trial，应为 80"

    fs = 200        # 采样率
    seg_len = 100   # 每段长度 = 100 samples = 0.5 s
    bands = [(1,4),(4,8),(8,14),(14,31),(31,51)]

    X_all = np.empty((0,62,5), dtype=float)
    y_all = np.empty((0,), dtype=int)

    for vid_idx, key in enumerate(trials):
        data = mat[key]              # shape=(62, N)
        n_seg = data.shape[1] // seg_len
        # 该视频所有段的标签
        y_seg = np.full((n_seg,), video_labels[vid_idx], dtype=int)

        # 预分配：62通道×5频段×n_seg
        feats = np.zeros((62,5,n_seg), dtype=float)

        for ch in range(62):
            sig = data[ch]
            if feature == 'DE':
                # 先滤波再分段
                filtered = [butter_bandpass_filter(sig, lo, hi, fs)
                            for lo,hi in bands]
                for s in range(n_seg):
                    st, ed = s*seg_len, (s+1)*seg_len
                    for b in range(5):
                        feats[ch,b,s] = compute_DE(filtered[b][st:ed])
            else:
                # PSD 直接按段计算
                for s in range(n_seg):
                    seg = sig[s*seg_len:(s+1)*seg_len]
                    for b,(lo,hi) in enumerate(bands):
                        feats[ch,b,s] = compute_PSD(seg, fs, (lo,hi))

        # (62,5,n_seg) → (n_seg,62,5)
        X_seg = feats.transpose(2,0,1)
        X_all = np.vstack([X_all, X_seg])
        y_all = np.concatenate([y_all, y_seg])

    return X_all, y_all

# —— 5. 主流程 ——
if __name__ == "__main__":
    input_dir   = r"D:\dachuang\EEG_preprocessed"
    output_base = r"D:/dachuang/SEED_VII"
    features    = ['DE','PSD']

    # 获取所有 .mat 文件并排序
    mat_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.mat'))

    for feat in features:
        out_dir = os.path.join(output_base, feat)
        os.makedirs(out_dir, exist_ok=True)

        X_total = np.empty((0,62,5), dtype=float)
        y_total = np.empty((0,), dtype=int)

        for fn in mat_files:
            fp = os.path.join(input_dir, fn)
            print(f"[{feat}] Processing {fn} ...")
            X_sub, y_sub = decompose(fp, feature=feat)
            X_total = np.vstack([X_total, X_sub])
            y_total = np.concatenate([y_total, y_sub])

        # 保存扁平特征
        np.save(os.path.join(out_dir, "X.npy"), X_total)
        np.save(os.path.join(out_dir, "y.npy"), y_total)
        print(f"[{feat}] Saved X.npy ({X_total.shape}), y.npy ({y_total.shape})")

        # 构建 8×9×5 的空间布局
        N = y_total.shape[0]
        X89 = np.zeros((N,8,9,5), dtype=float)
        X89[:,0,2,:]   = X_total[:,3,:]
        X89[:,0,3:6,:] = X_total[:,0:3,:]
        X89[:,0,6,:]   = X_total[:,4,:]
        for i in range(5):
            X89[:, i+1, :, :] = X_total[:, 5 + i*9 : 5 + (i+1)*9, :]
        X89[:,6,1:8,:] = X_total[:,50:57,:]
        X89[:,7,2:7,:] = X_total[:,57:62,:]

        np.save(os.path.join(out_dir, "X89.npy"), X89)
        print(f"[{feat}] Saved X89.npy ({X89.shape})")

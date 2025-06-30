# -*- coding: utf-8 -*-
"""
SEED-VII 数据集 GPU 加速预处理：
  对每个通道 5 个频段同时提取 DE 或 PSD（periodogram 方式）特征，并将 62 通道重排成 8×9×5 的空间结构。
  GPU 加速：使用 CuPy 替代 NumPy，在 GPU 上并行完成滤波、DE、PSD 计算。

输入目录：D:\dachuang\EEG_preprocessed\*.mat
输出目录：D:/dachuang/SEED_VII/DE/ 和 D:/dachuang/SEED_VII/PSD/
每个分支下生成 X.npy, y.npy, X89.npy

依赖：
  - Python 3.x
  - CuPy （例如：pip install cupy-cuda11x）
  - SciPy
  - NumPy
  - SciPy.io (用于加载 .mat)
"""

import os
import math
import numpy as np  # 仅用于 CPU 侧小量操作、数据持久化
import scipy.io as sio  # 用于读取 .mat 文件（CPU）
from scipy.signal import butter, lfilter  # 仅用于生成滤波器系数（CPU）
import cupy as cp  # GPU 端数组与计算
import cupyx.scipy.signal as cpsig  # GPU 端信号处理（仅部分功能）

# —— 1. 情绪到数字映射 ——
emotion2idx = {
    'Disgust': 0,
    'Fear': 1,
    'Sad': 2,
    'Neutral': 3,
    'Happy': 4,
    'Anger': 5,
    'Surprise': 6
}

# 四个 Session、20 视频，共 80 标签
labels_text = [
    # S1: 视频 1–20
    'Happy', 'Neutral', 'Disgust', 'Sad', 'Anger',
    'Anger', 'Sad', 'Disgust', 'Neutral', 'Happy',
    'Happy', 'Neutral', 'Disgust', 'Sad', 'Anger',
    'Anger', 'Sad', 'Disgust', 'Neutral', 'Happy',
    # S2: 视频 21–40
    'Anger', 'Sad', 'Fear', 'Neutral', 'Surprise',
    'Surprise', 'Neutral', 'Fear', 'Sad', 'Anger',
    'Anger', 'Sad', 'Fear', 'Neutral', 'Surprise',
    'Surprise', 'Neutral', 'Fear', 'Sad', 'Anger',
    # S3: 视频 41–60
    'Happy', 'Surprise', 'Disgust', 'Fear', 'Anger',
    'Anger', 'Fear', 'Disgust', 'Surprise', 'Happy',
    'Happy', 'Surprise', 'Disgust', 'Fear', 'Anger',
    'Anger', 'Fear', 'Disgust', 'Surprise', 'Happy',
    # S4: 视频 61–80
    'Disgust', 'Sad', 'Fear', 'Surprise', 'Happy',
    'Happy', 'Surprise', 'Fear', 'Sad', 'Disgust',
    'Disgust', 'Sad', 'Fear', 'Surprise', 'Happy',
    'Happy', 'Surprise', 'Fear', 'Sad', 'Disgust',
]
video_labels = np.array([emotion2idx[e] for e in labels_text], dtype=int)
assert video_labels.shape[0] == 80, "标签长度应为80"


# —— 2. 带通滤波 和 DE 计算 ——
def butter_bandpass_coeffs(lowcut, highcut, fs, order=5):
    """
    在 CPU 端生成带通滤波器系数 (b,a)，
    CuPy 端通过这些系数进行并行滤波。
    """
    nyq = fs * 0.5
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')


def gpu_bandpass_filter(sig_gpu, lowcut, highcut, fs, order=3):
    """
    在 GPU 端对 1D CuPy 数组进行带通滤波：
      - sig_gpu: 1D CuPy 数组 (Nc,)
      - lowcut, highcut: 频带边界
      - fs: 采样率
    返回滤波后 GPU 数组 (Nc,)
    """
    # 在 CPU 端获取滤波器系数
    b_cpu, a_cpu = butter_bandpass_coeffs(lowcut, highcut, fs, order=order)
    # 将系数复制到 GPU
    b_gpu = cp.asarray(b_cpu)
    a_gpu = cp.asarray(a_cpu)
    # 使用 CuPy 的 lfilter 进行滤波
    # 注意：cupyx.scipy.signal.lfilter 与 scipy.signal.lfilter 接口类似
    return cpsig.lfilter(b_gpu, a_gpu, sig_gpu)


def gpu_compute_DE(seg_gpu):
    """
    GPU 端计算 Differential Entropy(DE)：
      - seg_gpu: CuPy 数组，shape = (seg_len,)
    DE = 0.5 * ln(2πe * var)
    """
    # 无偏方差 ddof=1
    var = seg_gpu.var(ddof=1)
    # DE 公式
    return 0.5 * cp.log(2 * math.pi * math.e * var)


# —— 3. GPU 端 PSD 计算（Periodogram） ——
def gpu_compute_PSD(seg_gpu, sfreq, band):
    """
    GPU 端用简单 Periodogram 估计单段信号在指定频段的功率：
      - seg_gpu: 1D CuPy 数组，长度为 seg_len
      - sfreq: 采样率 (Hz)
      - band: (low, high) 频带
    返回标量：band 内功率
    说明：直接使用 FFT 计算功率谱密度
    """
    N = seg_gpu.shape[0]
    # 执行实值 FFT (rfft)，长度 = N/2 + 1
    fft_gpu = cp.fft.rfft(seg_gpu)
    # 计算功率谱 (periodogram)：|X(k)|^2 / (sfreq * N)
    # 这里省略窗口因子，近似使用矩形窗
    psd_gpu = (cp.abs(fft_gpu) ** 2) / (sfreq * N)
    # 频率向量 (rfft 频率)
    freqs = cp.fft.rfftfreq(N, d=1.0 / sfreq)  # GPU 端
    # 选出 band 区间索引
    idx = cp.logical_and(freqs >= band[0], freqs <= band[1])
    # 频率分辨率 = freqs[1]-freqs[0]
    df = freqs[1] - freqs[0]
    # 带内功率 = Σ psd[k] * df
    return cp.sum(psd_gpu[idx]) * df


# —— 4. GPU 版 每个 .mat 文件分解 & 特征提取 ——
def decompose_gpu(mat_file, feature='DE'):
    """
    对单个 .mat 文件做 GPU 加速特征提取
    Parameters
    ----------
    mat_file : str
        .mat 文件路径，包含 80 个 trial，每 trial shape=(62, N)
    feature : {'DE','PSD'}
        选择特征类型

    Returns
    -------
    X_all_cpu : np.ndarray, shape=(total_segments, 62, 5)  (CPU 侧 numpy)
    y_all_cpu : np.ndarray, shape=(total_segments,)        (CPU 侧 numpy)
    """
    # 1) 在 CPU 端加载 .mat 文件
    mat = sio.loadmat(mat_file)
    # 2) 找到所有 trial 变量名：二维数组，第一维=62
    trials = [k for k, v in mat.items()
              if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == 62]
    trials.sort()  # 保证 Video1…Video80 顺序
    assert len(trials) == 80, f"{os.path.basename(mat_file)} 找到 {len(trials)} 个 trial，应为 80"

    # 基本参数
    fs = 200  # 采样率
    seg_len = 100  # 每段长度 = 100 样本 = 0.5s
    bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 51)]  # 5 个频带

    # 初始化 GPU 端容器（占位，用于堆叠）
    # 这里只给一个空 shape；后续使用 vstack 时，需先将其换成具体第一次堆叠生成的数组
    X_all_gpu = None  # 待会在第一次循环时赋值
    y_all_cpu = []  # 标签直接保留在 CPU 端

    for vid_idx, key in enumerate(trials):
        # 3) 逐 video 提取数据
        data_cpu = mat[key]  # shape=(62, N)，NumPy 数组（CPU）
        N = data_cpu.shape[1]
        n_seg = N // seg_len  # 每 trial 切分成 n_seg 段
        # 4) 生成本视频的标签：shape=(n_seg,)
        lbl_seg = np.full((n_seg,), video_labels[vid_idx], dtype=int)
        y_all_cpu.append(lbl_seg)  # 以后统一 concatenate

        # 5) 将 data_cpu 转到 GPU：shape=(62, N)
        data_gpu = cp.asarray(data_cpu)

        # 6) 为本 trial 分配 GPU 端特征数组：shape=(62, 5, n_seg)
        feats_gpu = cp.zeros((62, 5, n_seg), dtype=cp.float32)

        # 7) 对每个通道并行计算
        for ch in range(62):
            sig_gpu = data_gpu[ch]  # GPU 上的一维数组
            if feature == 'DE':
                # a) 先滤波：得到 5 个频带信号，每个信号均为 GPU 数组
                filtered_list = []
                for (lo, hi) in bands:
                    filt_ch = gpu_bandpass_filter(sig_gpu, lo, hi, fs, order=3)  # GPU 滤波
                    filtered_list.append(filt_ch)
                # b) 分段并计算 DE（GPU 版本）
                for s in range(n_seg):
                    st, ed = s * seg_len, (s + 1) * seg_len
                    for b_idx in range(5):
                        seg_gpu = filtered_list[b_idx][st:ed]  # 长度 = seg_len
                        # DE 计算：GPU 上标量
                        feats_gpu[ch, b_idx, s] = gpu_compute_DE(seg_gpu)

            else:  # feature == 'PSD'
                # PSD：对每个段直接计算 periodogram
                for s in range(n_seg):
                    st, ed = s * seg_len, (s + 1) * seg_len
                    seg_gpu = sig_gpu[st:ed]  # 长度 = seg_len
                    for b_idx, (lo, hi) in enumerate(bands):
                        # GPU 上计算 PSD
                        feats_gpu[ch, b_idx, s] = gpu_compute_PSD(seg_gpu, fs, (lo, hi))

        # 8) (62, 5, n_seg) → (n_seg, 62, 5)
        X_seg_gpu = feats_gpu.transpose(2, 0, 1)  # shape = (n_seg, 62, 5)

        # 9) 将结果累加到 X_all_gpu
        if X_all_gpu is None:
            X_all_gpu = X_seg_gpu.copy()  # 第一次赋值
        else:
            X_all_gpu = cp.concatenate([X_all_gpu, X_seg_gpu], axis=0)

        # 10) 释放本次循环的 data_gpu 以节省显存
        del data_gpu, feats_gpu, X_seg_gpu
        cp._default_memory_pool.free_all_blocks()

    # 11) 合并所有 y 标签到一个 1D 数组
    y_all_cpu = np.concatenate(y_all_cpu, axis=0)  # shape = (总段数，)

    # 12) 最后将 GPU 上的 X_all_gpu 转回 CPU（numpy）以便保存
    X_all_cpu = cp.asnumpy(X_all_gpu)  # shape = (总段数, 62, 5)

    # 13) 返回 CPU 端结果
    return X_all_cpu, y_all_cpu


# —— 5. 主流程 ——
if __name__ == "__main__":
    input_dir = r"D:\dachuang\EEG_preprocessed"
    output_base = r"D:/dachuang/SEED_VII_GPU"
    features = ['DE', 'PSD']

    # 获取所有 .mat 文件并排序
    mat_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mat')])

    for feat in features:
        out_dir = os.path.join(output_base, feat)
        os.makedirs(out_dir, exist_ok=True)

        # 初始化累加容器 (CPU)
        X_total = None  # GPU 计算后再转回 CPU 时第一次赋值
        y_total = []  # CPU 端标签列表

        for fn in mat_files:
            fp = os.path.join(input_dir, fn)
            print(f"[{feat}] GPU Processing {fn} ...")
            X_sub, y_sub = decompose_gpu(fp, feature=feat)

            if X_total is None:
                X_total = X_sub.copy()  # 第一次赋值
            else:
                X_total = np.concatenate([X_total, X_sub], axis=0)

            y_total.append(y_sub)

        # 合并所有标签
        y_total = np.concatenate(y_total, axis=0)  # shape = (总段数，)

        # 保存扁平特征 到 .npy
        np.save(os.path.join(out_dir, "X.npy"), X_total)
        np.save(os.path.join(out_dir, "y.npy"), y_total)
        print(f"[{feat}] Saved X.npy ({X_total.shape}), y.npy ({y_total.shape})")

        # —— 6. 构建 8×9×5 的空间布局 X89 ——
        N = y_total.shape[0]
        X89 = np.zeros((N, 8, 9, 5), dtype=np.float32)

        # 按论文/原代码的电极映射规则
        # 第一行：电极索引 [3], [0:3], [4]
        X89[:, 0, 2, :] = X_total[:, 3, :]
        X89[:, 0, 3:6, :] = X_total[:, 0:3, :]
        X89[:, 0, 6, :] = X_total[:, 4, :]

        # 中间 5 行，每行 9 个电极
        for i in range(5):
            start = 5 + i * 9
            end = 5 + (i + 1) * 9
            X89[:, i + 1, :, :] = X_total[:, start:end, :]

        # 倒数第二行：电极 50:57 对应 X89[6,1:8]
        X89[:, 6, 1:8, :] = X_total[:, 50:57, :]

        # 最后一行：电极 57:62 对应 X89[7,2:7]
        X89[:, 7, 2:7, :] = X_total[:, 57:62, :]

        # 保存空间布局特征
        np.save(os.path.join(out_dir, "X89.npy"), X89)
        print(f"[{feat}] Saved X89.npy ({X89.shape})")

    print("All done! GPU 加速特征提取完成。")

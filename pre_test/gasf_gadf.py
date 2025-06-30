import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import os


# 读取 DEAP 数据集中的 .dat 文件
def load_deap_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data['data'], data['labels']


# 归一化函数，将时间序列数据归一化到 [0, 1]
def rescale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# 切除前三秒的基线
def remove_baseline(data, sample_rate=128, baseline_duration=3):
    baseline_samples = baseline_duration * sample_rate
    return data[baseline_samples:]


def plot_gasf_and_gadf(X, idx=0):
    # 使用 Gramian Angular Field 生成 GASF 和 GADF 图像
    gasf = GramianAngularField(method='summation')
    gadf = GramianAngularField(method='difference')

    X_gasf = gasf.transform(X)
    X_gadf = gadf.transform(X)

    # 创建一个图像，显示原始信号、极坐标转换和 GASF/GADF 图像
    plt.figure(figsize=(12, 6))
    plt.suptitle('Sample ' + str(idx))

    # 显示归一化后的时间序列
    ax1 = plt.subplot(121)
    ax1.plot(np.arange(len(X[idx])), rescale(X[idx]))
    ax1.set_title('Rescaled Time Series')

    # 显示极坐标图
    ax2 = plt.subplot(122, polar=True)
    r = np.array(range(1, len(X[idx]) + 1)) / 150
    theta = np.arccos(np.array(rescale(X[idx]))) * 2 * np.pi  # radian -> Angle
    ax2.plot(theta, r, color='r', linewidth=3)
    ax2.set_title('Polar Coordinate')

    # 显示 GASF 图像
    plt.figure()
    ax3 = plt.subplot(121)
    im_gasf = ax3.imshow(X_gasf[idx])  # 使用 imshow 显示图像
    ax3.set_title('GASF')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im_gasf, cax=cax)  # 通过 imshow 获取颜色条

    # 显示 GADF 图像
    ax4 = plt.subplot(122)
    im_gadf = ax4.imshow(X_gadf[idx])  # 使用 imshow 显示图像
    ax4.set_title('GADF')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im_gadf, cax=cax)  # 通过 imshow 获取颜色条

    plt.show()



# 主函数
def main():
    # 设置数据文件路径
    data_path = r'D:\jupyter\deap_data\data_preprocessed_python\s01.dat'  # 请根据实际路径调整文件路径

    # 加载数据
    data, labels = load_deap_data(data_path)

    # 假设我们选取第一个视频数据
    eeg_data = data[0]  # 选择第一个视频
    print(f"EEG Data shape: {eeg_data.shape}")  # 打印数据形状

    # 去除前三秒基线
    sample_rate = 128  # 数据采样率为 128Hz
    baseline_duration = 3  # 基线持续时间为 3秒
    eeg_data_no_baseline = np.array([remove_baseline(channel_data, sample_rate, baseline_duration)
                                     for channel_data in eeg_data])

    # 滑动窗口（3秒）的大小
    window_size = 3 * sample_rate  # 每个窗口包含 3秒的数据
    step_size = window_size  # 步长设置为3秒

    # 遍历所有通道并生成 GASF 和 GADF 图像
    for ch in range(eeg_data_no_baseline.shape[0]):  # 40 通道
        print(f"Processing Channel {ch + 1}...")
        eeg_signal = eeg_data_no_baseline[ch, :]  # 获取单通道数据

        # 滑动时间窗口生成图像
        for start in range(0, len(eeg_signal) - window_size, step_size):
            window_data = eeg_signal[start:start + window_size]  # 获取当前窗口的数据
            plot_gasf_and_gadf(np.expand_dims(window_data, axis=0))  # 扩展维度以匹配要求 (1, 时间点)


# 运行主函数
if __name__ == "__main__":
    main()

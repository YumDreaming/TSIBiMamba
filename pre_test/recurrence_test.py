import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
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

# 生成 Recurrence Plot 图像
def generate_recurrence_plot(X, idx=0, output_dir="output_images"):
    # 使用 Recurrence Plot 生成递归图，time_delay=10
    rp2 = RecurrencePlot(dimension=3, time_delay=10)
    X_rp2 = rp2.transform(X)

    # 创建保存图像的目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取 Recurrence Plot 图像
    rp_image = X_rp2[idx]  # 获取当前窗口的递归图

    # 将递归图的大小调整为224x224
    fig, ax = plt.subplots(figsize=(6, 6))
    im_rp2 = ax.imshow(rp_image, cmap='binary', interpolation='nearest')
    ax.set_title('Recurrence Plot, dimension=3, time_delay=10')

    # 添加颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im_rp2, cax=cax)

    # 保存图像为224x224
    img_path = os.path.join(output_dir, f"recurrence_plot_{idx+1}.png")
    plt.savefig(img_path, dpi=72, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

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

    # 滑动窗口（6秒）的大小
    window_size = 6 * sample_rate  # 每个窗口包含 6秒的数据 (6s * 128Hz = 768样本)
    step_size = window_size  # 步长设置为6秒

    # 遍历所有通道并生成 Recurrence Plot 图像
    for ch in range(eeg_data_no_baseline.shape[0]):  # 40 通道
        print(f"Processing Channel {ch + 1}...")
        eeg_signal = eeg_data_no_baseline[ch, :]  # 获取单通道数据

        # 滑动时间窗口生成 Recurrence Plot 图像
        for start in range(0, len(eeg_signal) - window_size, step_size):
            window_data = eeg_signal[start:start + window_size]  # 获取当前窗口的数据
            generate_recurrence_plot(np.expand_dims(window_data, axis=0), idx=0)  # 扩展维度以匹配要求 (1, 时间点)


# 运行主函数
if __name__ == "__main__":
    main()

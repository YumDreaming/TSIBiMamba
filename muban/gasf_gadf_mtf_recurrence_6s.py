import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from skimage.transform import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取 DEAP 数据集中的 .dat 文件
def load_deap_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data['data'][:, :32, :], data['labels']  # 32 通道的 EEG 数据和标签


# 归一化函数，将时间序列数据归一化到 [0, 1]
def rescale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# 切除前三秒的基线
def remove_baseline(data, sample_rate=128, baseline_duration=3):
    baseline_samples = baseline_duration * sample_rate
    return data[baseline_samples:]

# 生成并显示 GASF 和 Recurrence Plot 图像
def plot_gasf_and_rp(X, idx=0):
    # GASF 生成器 (直接输出224x224)`
    gasf = GramianAngularField(method='summation', image_size=224)
    X_gasf = gasf.transform(X)  # 输出形状 (n_samples, 224, 224)

    # Recurrence Plot 生成器（需调整尺寸）
    rp = RecurrencePlot(dimension=3, time_delay=3)
    X_rp = rp.transform(X)  # 输出形状 (n_samples, N, N), N为原信号长度
    X_rp_resized = np.array([resize(img, (224, 224), order=1) for img in X_rp])  # 双线性插值

    # 验证尺寸
    print(f"GASF图像尺寸: {X_gasf[idx].shape}")
    print(f"Recurrence Plot调整后尺寸: {X_rp_resized[idx].shape}")

    # 创建可视化图像
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'样本 {idx}')

    # 显示GASF
    ax1 = plt.subplot(121)
    im_gasf = ax1.imshow(X_gasf[idx])
    ax1.set_title('GASF (224x224)')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im_gasf, cax=cax)

    # 显示Recurrence Plot
    ax2 = plt.subplot(122)
    im_rp = ax2.imshow(X_rp_resized[idx], cmap='binary')
    ax2.set_title('Recurrence Plot (224x224)')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im_rp, cax=cax)

    plt.tight_layout()
    plt.show()

def process_deap_file(file_path, output_root, baseline_duration=3, sample_rate=128, window_length=6, overlap=3, pic_size=224):
    # 初始化图像转换器
    gasf = GramianAngularField(method='summation', image_size=pic_size)
    rp = RecurrencePlot(dimension=3, time_delay=3)

    # 数据、标签、实验名称
    data, labels = load_deap_data(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # 输出目录
    g_output_root = os.path.join(output_root, 'GASF', base_name)
    r_output_root = os.path.join(output_root, 'RP', base_name)
    os.makedirs(g_output_root, exist_ok=True)
    os.makedirs(r_output_root, exist_ok=True)

    # 图像标签名列表
    g_labels_list = []
    r_labels_list = []

    window_size = window_length * sample_rate  # 窗口长度
    overlap = overlap * sample_rate  # 重叠率
    step_size = window_size - overlap  # 步长

    # 遍历所有实验
    for trial_idx in range(data.shape[0]):
        # 去除基线
        eeg_data_no_baseline = np.array([remove_baseline(channel_data, sample_rate, baseline_duration)
                                         for channel_data in data[trial_idx]])
        # 遍历所有通道
        g_channels = []
        r_channels = []

        # 设定标签
        valence = labels[trial_idx][0]
        label_bin = 1 if valence > 5 else 0

        for ch_idx in range(data.shape[1]):
            eeg_signal = eeg_data_no_baseline[ch_idx, :]  # 获取单通道数据

            # 计算实际窗口数量
            num_windows = (eeg_signal.shape[0] - (window_size - overlap)) // (window_size - overlap)

            # 滑动时间窗口生成图像
            g_winds = []
            r_winds = []
            for i in range(num_windows):
                start = i * step_size
                end = start + window_size
                window_data = eeg_signal[start:end]

                # 特征计算，pic_size*pic_size (224*224)
                gasf_features = gasf.transform(np.expand_dims(window_data, axis=0))[0]
                rp_features = rp.transform(np.expand_dims(window_data, axis=0))
                # rp_resized = np.array([resize(img, (pic_size, pic_size), order=1) for img in rp_features])  # 双线性插值
                rp_resized = resize(rp_features[0], (pic_size, pic_size), order=1)  # 双线性插值

                # 保存单个通道的特征图像
                g_winds.append(gasf_features)
                r_winds.append(rp_resized)

            # 保存单个实验的特征图像
            g_channels.append(g_winds)
            r_channels.append(r_winds)

        # 保存 trial 数据
        g_file_name = f"{base_name}_trial_{trial_idx}_GASF.npy"
        r_file_name = f"{base_name}_trial_{trial_idx}_RP.npy"

        print(g_file_name)
        # print(np.array(g_channels).shape)
        print(r_file_name)
        # print(np.array(r_channels).shape)

        # # TODO: 保存数据成 .npy 文件，结构为 (channels, windows, H, W)
        np.save(os.path.join(g_output_root, g_file_name), np.array(g_channels))
        np.save(os.path.join(r_output_root, r_file_name), np.array(r_channels))

        # 保存标签
        g_labels_list.append(f"{g_file_name},{label_bin}\n")
        r_labels_list.append(f"{r_file_name},{label_bin}\n")

    # 保存标签文件
    with open(os.path.join(g_output_root, f"{base_name}_labels.txt"), "w") as f:
        f.writelines(g_labels_list)
    with open(os.path.join(r_output_root, f"{base_name}_labels.txt"), "w") as f:
        f.writelines(r_labels_list)


# 运行主函数
if __name__ == "__main__":
    # main()

    INPUT_PATH = r'D:\dachuang\data_preprocessed_python'
    OUTPUT_PATH = r'E:\pic'

    for file in os.listdir(INPUT_PATH):
        if file.endswith('.dat'):
            file_path = os.path.join(INPUT_PATH, file)
            process_deap_file(file_path, OUTPUT_PATH)

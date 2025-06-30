#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle


def main():
    # 1. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    # 2. 加载 s01 第 1 个实验
    with open('/root/deap/data_preprocessed_python/s01.dat', 'rb') as f:
        data = pickle.load(f, encoding='latin1')['data']  # (40,40,8064)
    eeg = data[2, :32, :]  # 取第一个 trial，前 32 通道
    eeg = torch.tensor(eeg, dtype=torch.float32, device=device)

    # 3. 去基线（3秒）
    baseline_samples = 3 * 128
    eeg = eeg[:, baseline_samples:]  # shape → (32, 7680)

    # 4. 频段定义
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    # 5. FFT 计算每通道每频段平均功率
    n = eeg.shape[1]
    freqs = torch.fft.rfftfreq(n, 1 / 128).to(device=device, dtype=torch.float32)
    eeg_fft = torch.fft.rfft(eeg, dim=1)
    band_powers = {}
    for name, (f1, f2) in bands.items():
        mask = ((freqs >= f1) & (freqs < f2)).to(dtype=torch.complex64)
        sig_band = eeg_fft * mask.unsqueeze(0)
        sig_time = torch.fft.irfft(sig_band, n=n, dim=1)
        # 每通道平均功率
        band_powers[name] = (sig_time.pow(2).mean(dim=1)).cpu().numpy()  # (32,)

    # 6. 电极在单位圆上的近似坐标
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    elec_pos = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (32,2)
    elec_pos_t = torch.tensor(elec_pos, dtype=torch.float32, device=device)

    # 7. 构造插值网格
    grid_size = 128
    grid_x = torch.linspace(-1, 1, grid_size, device=device, dtype=torch.float32)
    grid_y = torch.linspace(-1, 1, grid_size, device=device, dtype=torch.float32)
    gx, gy = torch.meshgrid(grid_x, grid_y, indexing='xy')
    grid_pts = torch.stack([gx.flatten(), gy.flatten()], dim=1)  # (4096,2), float32

    # 8. IDW 插值并可视化
    for name, pwr in band_powers.items():
        p_t = torch.tensor(pwr, dtype=torch.float32, device=device)  # (32,)
        # 计算网格点到电极的距离
        dists = torch.cdist(grid_pts, elec_pos_t)  # (4096,32)
        weights = 1.0 / (dists + 1e-6)  # 避免除零
        weights = weights / weights.sum(dim=1, keepdim=True)  # 归一化
        interp = (weights * p_t.unsqueeze(0)).sum(dim=1)  # (4096,)
        img = interp.reshape(grid_size, grid_size).cpu().numpy()

        plt.figure(figsize=(4, 4))
        plt.imshow(img, origin='lower', extent=(-1, 1, -1, 1))
        plt.title(f'{name} band topography')
        plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import scipy.io as sio
import numpy as np
import torch

def read_file(mat_file):
    """
    读取 .mat 文件，返回：
      - trial_data: ndarray (4800,128)
      - base_data:  ndarray (40,128)
      - arousal:    ndarray (4800,)
      - valence:    ndarray (4800,)
    """
    md = sio.loadmat(mat_file)
    return (md['data'], md['base_data'],
            md['arousal_labels'].flatten(),
            md['valence_labels'].flatten())

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

# 通道到 8×9 网格的位置映射（row, col）
IDX_MAP = {
    0:(0,3),  1:(0,2),  2:(1,2),  3:(1,0),  4:(2,1),  5:(2,3),  6:(3,2),  7:(3,0),
    8:(4,1),  9:(4,3), 10:(5,2), 11:(5,0), 12:(6,3), 13:(7,3), 14:(7,4), 15:(5,4),
   16:(0,5), 17:(0,6), 18:(1,4), 19:(1,6), 20:(1,8), 21:(2,7), 22:(2,5), 23:(3,4),
   24:(3,6), 25:(3,8), 26:(4,7), 27:(4,5), 28:(5,6), 29:(5,8), 30:(6,5), 31:(7,5),
}

def preprocess_cuda(mat_file, device, use_baseline=True):
    # 1) 读 numpy 数据
    trial_np, base_np, aro_np, val_np = read_file(mat_file)
    # 2) 转 torch.Tensor 到 GPU
    trial = torch.from_numpy(trial_np).to(device=device, dtype=torch.float)  # (4800,128)
    base  = torch.from_numpy(base_np).to(device=device, dtype=torch.float)   # (40,128)

    # 3) 基线去除
    if use_baseline:
        # 重复每个 trial 基线 120 次
        base_exp = base.repeat_interleave(120, dim=0)   # (4800,128)
        data = trial - base_exp
    else:
        data = trial

    # 4) 每行标准化
    mean = data.mean(dim=1, keepdim=True)
    std  = data.std(dim=1, keepdim=True, unbiased=False)
    data_norm = (data - mean) / std  # (4800,128)

    # 5) 重塑为 (4800,4,32)
    n_seg, dim = data_norm.shape
    data4 = data_norm.view(n_seg, 4, dim // 4)  # (4800,4,32)

    # 6) 分配到 3D 网格 (4800,4,8,9)
    out3d = torch.zeros((n_seg, 4, 8, 9), device=device, dtype=torch.float)
    # 利用通道映射向量化赋值
    for ch, (r, c) in IDX_MAP.items():
        # out3d[:, :, r, c] shape = (4800,4)
        # data4[:, :, ch]    shape = (4800,4)
        out3d[:, :, r, c] = data4[:, :, ch]

    # 7) 回 CPU 并转换为 numpy，准备保存
    return out3d.cpu().numpy(), aro_np.astype(int), val_np.astype(int)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("⚠️ CUDA 不可用，正在使用 CPU，性能会大幅下降。")

    input_root  = '/root/autodl-tmp/deap_map'
    output_root = '/root/autodl-tmp/deap_map/3d'
    use_baseline = True

    for branch in ['DE', 'PSD']:
        in_dir  = os.path.join(input_root, branch)
        out_dir = os.path.join(output_root, branch)
        ensure_dir(out_dir)

        for fname in os.listdir(in_dir):
            if not fname.lower().endswith('.mat'):
                continue
            inpath  = os.path.join(in_dir, fname)
            outpath = os.path.join(out_dir, fname)
            print(f'[{branch}] Processing {fname} → 3D on {device} ...')

            data3d, aro, val = preprocess_cuda(inpath, device, use_baseline)
            sio.savemat(outpath, {
                'data': data3d,
                'arousal_labels': aro,
                'valence_labels': val
            })
            print(f'  → Saved to {outpath}')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.io as sio
from sklearn import preprocessing

def read_file(mat_file):
    """
    读取单个 .mat 文件，返回：
      - trial_data: (4800,128) 1D 特征
      - base_data:  (40,128)  基线特征
      - arousal_labels, valence_labels: (4800,) each
    """
    md = sio.loadmat(mat_file)
    return (md['data'], md['base_data'],
            md['arousal_labels'].flatten(),
            md['valence_labels'].flatten())

def get_vector_deviation(v1, v2):
    return v1 - v2

def get_dataset_deviation(trial_data, base_data):
    """
    对所有 4800 段做基线去除：
      第 i 段对应 trial = i//120，如果超出索引则取最后一 trial。
    """
    n_seg, dim = trial_data.shape
    new_ds = np.empty((0, dim))
    for i in range(n_seg):
        bi = i // 120
        if bi >= base_data.shape[0]:
            bi = base_data.shape[0] - 1
        rec = get_vector_deviation(trial_data[i], base_data[bi]).reshape(1, -1)
        new_ds = np.vstack([new_ds, rec])
    return new_ds

def data_1Dto2D(vec):
    """
    将 32 通道的一段特征 (长度 32) 填入 8×9 网格
    """
    g = np.zeros((8,9))
    g[0] = (0, 0, vec[1], vec[0], 0, vec[16], vec[17], 0, 0)
    g[1] = (vec[3], 0, vec[2], 0, vec[18], 0, vec[19], 0, vec[20])
    g[2] = (0, vec[4], 0, vec[5], 0, vec[22], 0, vec[21], 0)
    g[3] = (vec[7], 0, vec[6], 0, vec[23], 0, vec[24], 0, vec[25])
    g[4] = (0, vec[8], 0, vec[9], 0, vec[27], 0, vec[26], 0)
    g[5] = (vec[11], 0, vec[10], 0, vec[15], 0, vec[28], 0, vec[29])
    g[6] = (0, 0, 0, vec[12], 0, vec[30], 0, 0, 0)
    g[7] = (0, 0, 0, vec[13], vec[14], vec[31], 0, 0, 0)
    return g

def pre_process(mat_file, use_baseline="yes"):
    """
    1) 读取 1D 特征和基线
    2) 如果 use_baseline=="yes"，做基线去除并标准化；否则直接标准化 trial_data
    3) 将每个 128 维向量拆成 4 段（band），每段 32 维映射到 8×9
    4) 输出 shape 为 (4800, 4, 8, 9)
    """
    trial_data, base_data, arousal, valence = read_file(mat_file)

    if use_baseline == "yes":
        data = get_dataset_deviation(trial_data, base_data)
        data = preprocessing.scale(data, axis=1)
    else:
        data = preprocessing.scale(trial_data, axis=1)

    n_seg, dim = data.shape   # 4800, 128
    sub_len = dim // 4        # 32
    data_3D = np.empty((0, 8, 9))

    for vec in data:
        # 每个 vec 拆成 4 个频段
        for b in range(4):
            seg = vec[b*sub_len:(b+1)*sub_len]
            g2d = data_1Dto2D(seg).reshape(1, 8, 9)
            data_3D = np.vstack([data_3D, g2d])

    # 最终 reshape 成 (4800,4,8,9)
    data_3D = data_3D.reshape(-1, 4, 8, 9)
    return data_3D, arousal, valence

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def main():
    input_root  = '/root/autodl-tmp/deap_map/'
    output_root = '/root/autodl-tmp/deap_map/3d/'
    use_baseline = "yes"

    for branch in ['DE', 'PSD']:
        in_dir  = os.path.join(input_root, branch)
        out_dir = os.path.join(output_root, branch)
        ensure_dir(out_dir)

        for fname in os.listdir(in_dir):
            if not fname.lower().endswith('.mat'):
                continue
            inpath = os.path.join(in_dir, fname)
            print(f'Processing {branch}/{fname} ...')

            data3d, aro, val = pre_process(inpath, use_baseline)
            savepath = os.path.join(out_dir, fname)
            sio.savemat(savepath, {
                'data': data3d,
                'arousal_labels': aro,
                'valence_labels': val
            })
            print(f'  → Saved 3D data to {savepath}')

if __name__ == '__main__':
    main()

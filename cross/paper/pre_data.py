import os
import numpy as np
import scipy.io as sio


def feature_wrap(feature_3d: np.ndarray) -> np.ndarray:
    """
    将 3D 特征 (62, T, 5) 展平为 2D (T, 310)，
    310 = 62 个通道 × 5 个频带。
    """
    channels, T, bands = feature_3d.shape  # e.g. (62, 235, 5)
    feature_2d = np.zeros((T, channels * bands), dtype=feature_3d.dtype)  # (T, 310)
    for i in range(T):
        # 第 i 个窗口: feature_3d[:, i, :] → (62,5) → reshape → (310,)
        feature_2d[i, :] = feature_3d[:, i, :].reshape(-1)
    return feature_2d


def data_prepare_seed(root_dir: str):
    """
    等价于 MATLAB 的 data_prepare_seed.m：
    1. 读取 root_dir/label.mat (15×1)
    2. 遍历 15 个受试者，每人 3 次会话：
       - 从 ExtractedFeatures 中加载 de_LDS1..15
       - 对每段调用 feature_wrap，拼接成 (sum_windows,310)
       - 生成对应的标签向量 (sum_windows,) 并 +1
       - 保存为 feature/sub_{i}_session_{k}.mat
    """
    # 1. 载入标签
    label_path = os.path.join(root_dir, 'label.mat')
    mat = sio.loadmat(label_path)
    # MATLAB 中 label 是列向量 (15,1) 或 (1,15)，reshape 为 (15,)
    labels = mat['label'].reshape(-1)  # e.g. [-1,0,1,...]

    # 2. 找到所有分段特征文件
    feat_in_dir = os.path.join(root_dir, 'ExtractedFeatures')
    all_files = sorted(f for f in os.listdir(feat_in_dir) if f.endswith('.mat'))
    # 应有 45 个文件，按受试者每 3 个一组

    feat_out_dir = os.path.join(root_dir, 'feature')
    os.makedirs(feat_out_dir, exist_ok=True)

    # 3. 对每个受试者
    for subj in range(15):
        # 取该受试者的三次会话
        files = all_files[subj * 3: subj * 3 + 3]

        for sess_idx, fname in enumerate(files, start=1):
            data = sio.loadmat(os.path.join(feat_in_dir, fname))

            # 收集 15 段视频的特征与标签
            feat_list = []
            lbl_list = []
            for seg in range(1, 16):
                key = f'de_LDS{seg}'

                # F3 的形状为 (62, W_k, 5)
                F3 = data[key]  # shape (62, windows_seg, 5)

                # X2 的形状为 (W_k, 310)
                X2 = feature_wrap(F3)  # shape (windows_seg, 310)
                feat_list.append(X2)

                # 标签全段相同，长度 windows_seg
                seg_label = np.full(F3.shape[1], labels[seg - 1], dtype=np.int32)
                lbl_list.append(seg_label)
            # 循环结束后：
            # feat_list: [ (W_1, 310), (W_2, 310), …, (W_15, 310) ]
            # lbl_list : [ (W_1,),     (W_2,),     …, (W_15,)     ]

            # 拼接整个会话的所有窗口
            # feat_sess 的形状为 (W_1 + W_2 + … + W_15, 310) ※※※※※
            feat_sess = np.vstack(feat_list)  # (sum_windows, 310)

            # lbl_sess 的形状为 (W_1 + W_2 + … + W_15,)
            lbl_sess = np.concatenate(lbl_list)  # (sum_windows,)

            # 将原来的 {-1,0,1} 平移为 {0,1,2}
            lbl_sess += 1  # MATLAB 中 +1 (变成 1,2,3)

            # 准备保存的 dict
            # 如果要存成 (N,1) 结构，可以再 reshape：
            # lbl_sess 的形状为 (W_1 + W_2 + … + W_15, 1) ※※※※※
            ds_name = f'dataset_session{sess_idx}'
            save_dict = {
                f'{ds_name}': {
                    'feature': feat_sess,
                    'label': lbl_sess.reshape(-1, 1)  # 保持 (N,1) 结构
                }
            }

            out_fname = os.path.join(
                feat_out_dir,
                f'sub_{subj + 1}_session_{sess_idx}.mat'
            )
            sio.savemat(out_fname, save_dict)
            print(f'Wrote {out_fname}: feature={feat_sess.shape}, label={lbl_sess.shape}')


if __name__ == '__main__':
    # 将这里改成你的 SEED 根目录
    SEED_ROOT = r'F:\zhourushuang\SEED'
    data_prepare_seed(SEED_ROOT)

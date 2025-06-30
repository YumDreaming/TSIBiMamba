# -*- coding: utf-8 -*-
"""
第二阶段：SEED-VII 数据集「DE/PSD」 两分支的 3D 滑窗重组
--------------------------------------------------------------------------------
修复了“不一致的滑窗数”问题：NS 应当等于 ∑_j floor(min_seg[j]/t)，
而不是 floor(sum(min_seg)/t)。
"""

import os
import numpy as np
import scipy.io as sio

# —————————————— 1. 配置区 ——————————————
# SEED-VII 原始 .mat 所在目录（用于统计每个 trial 的原始段数）
MAT_DIR = r"D:\dachuang\EEG_preprocessed"

# 第一阶段输出的“DE”和“PSD”分支目录
BRANCH_INPUT_BASE = r"D:\dachuang\SEED_VII"

# 第二阶段输出的目录基路径
BRANCH_OUTPUT_BASE = r"D:\dachuang\SEED_VII_3D"

# 滑动窗口长度（单位：段数，1 段 = 0.5 秒 @200Hz = 100 个采样点）
t = 6

# —————————————— 2. 辅助函数 ——————————————

def count_segments_per_trial(mat_file, seg_len=100):
    """
    读取一个 SEED-VII .mat 文件，统计其中每个 trial 的段数 = floor(N_samples/seg_len)。
    每个 trial 对应一个 shape=(62, N_samples) 的数组。
    返回：长度为 80 的列表，每个元素为该 trial 的段数（int）。
    """
    mat = sio.loadmat(mat_file)
    # 找到所有“二维数组、第一维=62”的 key，当做 trial
    trials = [k for k, v in mat.items()
              if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == 62]
    trials.sort()
    if len(trials) != 80:
        raise ValueError(f"{os.path.basename(mat_file)}: 找到 {len(trials)} 个 trial，应为 80。")

    segments = []
    for key in trials:
        data = mat[key]               # shape = (62, N_samples)
        n_samples = data.shape[1]
        segments.append(n_samples // seg_len)
    return segments  # e.g. [122, 444, 378, ...]


def collect_all_subjects_segment_info(mat_dir, seg_len=100):
    """
    遍历目录下所有 .mat 文件，调用 count_segments_per_trial，返回：
      - mat_files:      排序后的 .mat 文件名列表
      - trial_segs:     二维列表，trial_segs[i] = 第 i 个受试者对应的 80 个 trial 的段数
      - total_segments: 一维列表，total_segments[i] = sum(trial_segs[i])
    """
    mat_files = sorted([f for f in os.listdir(mat_dir) if f.endswith(".mat")])
    trial_segs = []
    total_segments = []
    for fn in mat_files:
        fp = os.path.join(mat_dir, fn)
        segs = count_segments_per_trial(fp, seg_len=seg_len)
        trial_segs.append(segs)
        total_segments.append(sum(segs))
    return mat_files, trial_segs, total_segments


# —————————————— 3. 主流程 ——————————————
if __name__ == "__main__":
    # 3.1 统计每个受试者的每个 trial 段数，并计算每个 trial 的最小段数
    seg_len = 100  # 每段 100 个采样点 = 0.5 秒 @200Hz
    mat_files, subject_trial_segs_list, subject_total_segs = \
        collect_all_subjects_segment_info(MAT_DIR, seg_len=seg_len)

    num_subjects = len(mat_files)
    print(f"共找到 {num_subjects} 个受试者的 .mat 文件：")
    for idx, fn in enumerate(mat_files):
        print(f"  [{idx}] {fn}，总段数 = {subject_total_segs[idx]}")

    # 3.1.1 计算每个 trial 索引 j (0..79) 的最小段数 min_seg[j]
    num_trials = 80
    min_seg = []
    for j in range(num_trials):
        all_j_segs = [subject_trial_segs_list[i][j] for i in range(num_subjects)]
        min_seg.append(min(all_j_segs))
    total_trunc_segs_per_subject = sum(min_seg)
    print(f"\n每个 trial 的最小段数 min_seg = {min_seg}")
    print(f"裁剪后，每个受试者的总段数 = {total_trunc_segs_per_subject}")

    # 3.1.2 由于不重叠滑窗要在“每个 trial 内”分别做 floor(seg_j/t)，
    #       所以滑窗总数 NS 应当 := ∑_{j=0..79} floor(min_seg[j]/t)
    NS = sum(seg_j // t for seg_j in min_seg)
    print(f"滑动窗口长度 t = {t}，按 trial 计算后的滑窗总数 NS = {NS}")

    # 3.2 定义两个分支，要对它们执行相同的“裁剪 + 滑窗”重组
    branches = ["DE", "PSD"]

    # 3.3 对于每个分支，加载第一阶段生成的 X89.npy，然后做“裁剪 + 滑窗”
    for feat in branches:
        print(f"\n========== 开始处理分支：{feat} ==========")
        branch_in_dir = os.path.join(BRANCH_INPUT_BASE, feat)
        branch_out_dir = os.path.join(BRANCH_OUTPUT_BASE, feat)
        os.makedirs(branch_out_dir, exist_ok=True)

        # 3.3.1 加载该分支的 X89.npy
        x89_path = os.path.join(branch_in_dir, "X89.npy")
        if not os.path.isfile(x89_path):
            raise FileNotFoundError(f"找不到 {feat} 分支的 X89.npy: {x89_path}")
        X89_all = np.load(x89_path)
        expected_rows = sum(subject_total_segs)
        if X89_all.shape[0] != expected_rows:
            raise ValueError(f"{feat} 分支 X89.npy 的行数 ({X89_all.shape[0]}) "
                             f"与所有受试者合计段数 ({expected_rows}) 不符。")

        # 3.3.2 “裁剪”出 X89_all_trunc ——
        #       依次对每个受试者、每个 trial 取前 min_seg[j] 段
        truncated_subs = []
        base_idx = 0
        for subj_idx in range(num_subjects):
            tot_seg = subject_total_segs[subj_idx]
            X_sub_full = X89_all[base_idx : base_idx + tot_seg]  # shape=(tot_seg,8,9,5)

            trial_segs = subject_trial_segs_list[subj_idx]  # 长度=80，每个 trial 的原始段数
            truncated_trials = []
            start = 0
            for j in range(num_trials):
                orig_count = trial_segs[j]
                keep_count = min_seg[j]
                truncated_part = X_sub_full[start : start + keep_count]
                truncated_trials.append(truncated_part)
                start += orig_count

            X_sub_trunc = np.vstack(truncated_trials)  # shape=(sum(min_seg),8,9,5)
            if X_sub_trunc.shape[0] != total_trunc_segs_per_subject:
                raise ValueError(f"受试者 {subj_idx} 裁剪后段数 {X_sub_trunc.shape[0]} ≠ {total_trunc_segs_per_subject}。")
            truncated_subs.append(X_sub_trunc)
            base_idx += tot_seg
            print(f"  [裁剪完成] 受试者 {subj_idx+1}/{num_subjects}，原段数 {tot_seg} → 裁剪后 {total_trunc_segs_per_subject}")

        X89_all_trunc = np.vstack(truncated_subs)
        print(f"  构造完成 X89_all_trunc，形状 = {X89_all_trunc.shape} (应 = {num_subjects}×{total_trunc_segs_per_subject} = {num_subjects * total_trunc_segs_per_subject})")

        # 3.3.3 在 X89_all_trunc 上做“不重叠滑窗”重组
        #       new_x shape = (num_subjects, NS, t, 8,9,5)
        #       new_y shape = (num_subjects * NS,)
        new_x = np.zeros((num_subjects, NS, t, 8, 9, 5), dtype=float)
        new_y = np.zeros((num_subjects * NS,), dtype=int)

        # 重新构造 video_labels（80 个 trial 的情绪标签）
        emotion2idx = {
            'Disgust': 0, 'Fear': 1, 'Sad': 2,
            'Neutral': 3, 'Happy': 4, 'Anger': 5, 'Surprise': 6
        }
        labels_text = [
            # S1: Videos 1–20
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
        video_labels = np.array([emotion2idx[e] for e in labels_text], dtype=int)

        base_trunc_idx = 0
        for subj_idx in range(num_subjects):
            X_sub = X89_all_trunc[base_trunc_idx : base_trunc_idx + total_trunc_segs_per_subject]
            trial_start = 0
            win_count_acc = 0

            for j in range(num_trials):
                this_trial_seg = min_seg[j]
                this_trial_win = this_trial_seg // t
                this_label = video_labels[j]
                for w in range(this_trial_win):
                    st = trial_start + w * t
                    ed = st + t
                    new_x[subj_idx, win_count_acc] = X_sub[st:ed]  # shape=(t,8,9,5)
                    new_y[subj_idx * NS + win_count_acc] = this_label
                    win_count_acc += 1
                trial_start += this_trial_seg

            if win_count_acc != NS:
                raise ValueError(f"第 {subj_idx} 个受试者滑窗写入 {win_count_acc} ≠ NS({NS})。")
            base_trunc_idx += total_trunc_segs_per_subject
            print(f"  [滑窗完成] 受试者 {subj_idx+1}/{num_subjects} → {win_count_acc} 个滑窗")

        # 3.3.4 保存结果
        out_x_path = os.path.join(branch_out_dir, f"t{t}x_89.npy")
        out_y_path = os.path.join(branch_out_dir, f"t{t}y_89.npy")
        np.save(out_x_path, new_x)
        np.save(out_y_path, new_y)
        print(f"[{feat}] 保存完成：\n  {out_x_path}  → {new_x.shape}\n  {out_y_path}  → {new_y.shape}")

    print("\n====== 所有分支处理完毕 ======")

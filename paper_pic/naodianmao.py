import mne
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------
# 1. 定义 SEED-62 中常见的 62 个通道名
#    如果你的通道名略有不同，请相应修改这里。
# ------------------------------------------
seed62_ch_names = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz"
]

# ------------------------------------------
# 2. 加载 MNE 标准 10–20 布局，并只保留这 62 个通道
# ------------------------------------------
full_montage = mne.channels.make_standard_montage('standard_1020')

missing = [ch for ch in seed62_ch_names if ch not in full_montage.ch_names]
if missing:
    raise RuntimeError(f"以下通道在 standard_1020 中缺失，请检查名称拼写：{missing}")

info = mne.create_info(ch_names=seed62_ch_names, sfreq=1000.0, ch_types='eeg')
info.set_montage(full_montage)

# ------------------------------------------
# 3. 用 find_layout 得到 62 个通道投影到二维的坐标 pos_2d
# ------------------------------------------
layout = mne.channels.find_layout(info)
# 如果想保险一点，确保两个列表顺序一致：
assert layout.names == seed62_ch_names, "layout.names 顺序和 seed62_ch_names 不一致，请确认列表顺序"

pos_2d = layout.pos  # (62, 2) 数组，范围大致在 [-0.5, +0.5]

# ------------------------------------------
# 4. 准备一个零数组，只为了“占位”给 plot_topomap
# ------------------------------------------
zeros = np.zeros(len(seed62_ch_names))

# ------------------------------------------
# 5. 绘制头型轮廓和点（不带标签），然后手动在点旁写标签
# ------------------------------------------
plt.rcParams['font.size'] = 8  # 全局字体小一些，减少文字重叠概率

fig, ax = plt.subplots(figsize=(6, 6))

# 5.1 先让 plot_topomap 画出头型轮廓+点
#    - contours=0：不要等高线，只需要头型轮廓
#    - sphere=0.5：半径 0.5，与 pos_2d 的坐标范围匹配
#    - show=False：不立即 plt.show()，后面我们还要手动添加文字
mne.viz.plot_topomap(
    zeros,
    pos_2d,
    axes=ax,
    show=False,
    contours=0,
    sphere=0.5
)

# 5.2 在每个通道对应的 (x, y) 坐标位置，用 ax.text 写上通道名
for ch_name, (x, y) in zip(seed62_ch_names, pos_2d):
    # 这里用稍微偏右上方一点的offset=(0.01, 0.01) 来放置文字，可自行微调：
    ax.text(x + 0.01, y + 0.01, ch_name,
            fontsize=6,       # 字体调得更小，减少重叠
            horizontalalignment='left',
            verticalalignment='bottom')

ax.set_title("SEED 62-Channel 2D Layout", pad=20)

# 5.3 最后再显示整个图
plt.tight_layout()
plt.show()

# 如果需要把图片保存到本地，请取消下面一行注释：
# fig.savefig("seed62_2d_layout_labeled.png", dpi=300, bbox_inches='tight')

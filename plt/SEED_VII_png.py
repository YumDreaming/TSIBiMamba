import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义混淆矩阵数据
cm = np.array([
    [79.35, 3.25, 4.22, 2.92, 4.94, 3.31, 2.01],
    [ 3.50,71.75, 3.67, 5.33, 3.67, 5.67, 6.42],
    [ 5.00, 4.92,74.84, 1.53, 2.98, 7.18, 3.55],
    [ 6.44, 1.44, 3.27,77.31, 2.12, 4.52, 4.90],
    [ 6.82, 3.41, 2.05, 0.83,77.12, 3.18, 6.59],
    [ 3.75, 3.06, 5.88, 1.94, 1.50,81.25, 2.62],
    [ 3.50, 3.36, 3.50, 2.21, 5.36, 3.21,78.86],
])

# 2. 定义标签顺序
labels = ['Disgust', 'Fear', 'Sad', 'Neutral', 'Happy', 'Anger', 'Surprise']

# 3. 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, f"{cm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10)

plt.tight_layout()

# 4. 保存为位图（PNG）
out_dir = r'/root/autodl-tmp'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'confusion_matrix_SEED_VII.png')
plt.savefig(out_path, format='png', bbox_inches='tight')

# （可选）同时显示
plt.show()


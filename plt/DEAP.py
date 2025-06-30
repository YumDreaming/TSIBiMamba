# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix_percent(y_true, y_pred,
                                  classes=None,
                                  cmap=plt.cm.Blues,
                                  save_path=None):
    """
    计算并绘制按行归一化后的百分比混淆矩阵。

    参数：
    - y_true: 真实标签，形如 [0,1,0,1,...]
    - y_pred: 预测标签，形如 [0,1,1,0,...]
    - classes: 类别名称列表，长度需等于类别数
    - cmap: 配色方案
    - save_path: 若不为 None，则保存图片到该路径
    """
    # 1. 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 2. 按行归一化并转为百分比
    cm_percent = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True) * 100

    # 3. 绘图
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # 4. 坐标轴刻度和标签
    n_classes = cm.shape[0]
    if classes is None:
        classes = [str(i) for i in range(n_classes)]
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix (Normalized %)'
    )
    plt.setp(ax.get_xticklabels(), rotation=0, ha='right')

    # 5. 在每个单元格内写入百分比数值
    thresh = cm_percent.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i,
                    f"{cm_percent[i, j]:.2f}%",
                    ha="center", va="center",
                    color="white" if cm_percent[i, j] > thresh else "black")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    # —— 在这里替换为你的真实标签和预测标签 ——
    # 下面演示用你给的绝对值生成 y_true, y_pred
    # 第一行：1032 正确预测为 0，112 错误预测为 1
    # 第二行：115 错误预测为 0，1301 正确预测为 1
    y_true = np.array([0]*1144 + [1]*1416)
    y_pred = np.array([0]*1032 + [1]*112 + [0]*115 + [1]*1301)

    # 若是多分类，只需把 classes 换成相应名称列表即可
    plot_confusion_matrix_percent(
        y_true, y_pred,
        classes=['Class 0','Class 1'],
        save_path='cm_percent.png'
    )

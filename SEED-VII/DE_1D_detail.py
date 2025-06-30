import os
import numpy as np

# ——— 配置部分 ———
# 请根据实际路径修改
psd_dir = r"D:/dachuang/SEED_VII_02/DE"

# 对应的文件名
file_X    = os.path.join(psd_dir, "X.npy")
file_y    = os.path.join(psd_dir, "y.npy")
file_X89  = os.path.join(psd_dir, "X89.npy")

# ——— 加载并打印维度 ———
def load_and_print_shapes():
    # 检查文件是否存在
    for fpath in [file_X, file_y, file_X89]:
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"找不到文件: {fpath}")

    # 加载 X.npy
    X = np.load(file_X)
    print("Loaded X.npy")
    print("  shape of X:", X.shape)
    # 例如： (总段数, 62, 5)

    # 加载 y.npy
    y = np.load(file_y)
    print("\nLoaded y.npy")
    print("  shape of y:", y.shape)
    # 例如： (总段数,)

    # 加载 X89.npy
    X89 = np.load(file_X89)
    print("\nLoaded X89.npy")
    print("  shape of X89:", X89.shape)
    # 例如： (总段数, 8, 9, 5)

    # 可选：打印前三个样本的标签与对应通道-频段维度
    print("\n示例展示：")
    print("  y[0:3] =", y[:3])
    print("  X[0].shape  =", X[0].shape)   # 应为 (62,5)
    print("  X89[0].shape =", X89[0].shape) # 应为 (8,9,5)

if __name__ == "__main__":
    load_and_print_shapes()

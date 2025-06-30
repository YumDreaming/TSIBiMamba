import os
import numpy as np

# 目录路径
data_dir = '/root/autodl-tmp/SEED_VII_3D/DE'

# 方法一：动态列出所有 .npy 文件，并打印形状
print("== All .npy files in directory ==")
for fname in sorted(os.listdir(data_dir)):
    if fname.endswith('.npy'):
        path = os.path.join(data_dir, fname)
        try:
            arr = np.load(path)
            print(f"{fname:15s} → shape = {arr.shape}")
        except Exception as e:
            print(f"{fname:15s} → failed to load ({e})")

# 方法二：针对指定文件逐一检查
print("\n== Specific files ==")
for fname in ['t6x_89.npy', 't6y_89.npy']:
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        print(f"{fname:15s} → NOT FOUND")
        continue
    try:
        arr = np.load(path)
        print(f"{fname:15s} → shape = {arr.shape}")
    except Exception as e:
        print(f"{fname:15s} → ERROR loading ({e})")

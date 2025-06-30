import os
import numpy as np

def check_npy(path):
    arr = np.load(path)
    print(f"文件: {os.path.basename(path)}")
    print(f"  shape : {arr.shape}")
    print(f"  dtype : {arr.dtype}")
    # 如果是数值数组，统计一下 NaN、min、max、mean
    if np.issubdtype(arr.dtype, np.number):
        nan_count = np.isnan(arr).sum()
        print(f"  NaN 数量 : {nan_count}")
        # 对 y（标签）这种一维整型数组，可以跳过下面的统计
        if arr.ndim > 1 or np.issubdtype(arr.dtype, np.floating):
            print(f"  最小值 : {np.nanmin(arr):.4f}")
            print(f"  最大值 : {np.nanmax(arr):.4f}")
            print(f"  平均值 : {np.nanmean(arr):.4f}")
    print()

if __name__ == "__main__":
    # 根据你保存的路径修改这两行
    output_de   = r"D:\dachuang\SEED_preprocess\DE"
    output_psd  = r"D:\dachuang\SEED_preprocess\PSD"

    files_to_check = [
        os.path.join(output_de,  "X_1D_DE.npy"),
        os.path.join(output_de,  "y.npy"),
        os.path.join(output_psd, "X_1D_PSD.npy"),
        os.path.join(output_psd, "y.npy"),
        os.path.join(output_de,  "X89_DE.npy"),
        os.path.join(output_psd, "X89_PSD.npy"),
    ]

    for fp in files_to_check:
        if os.path.isfile(fp):
            check_npy(fp)
        else:
            print(f"找不到文件: {fp}\n")

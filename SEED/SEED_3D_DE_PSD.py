import os
import numpy as np
from tqdm import tqdm

# --- Configuration ---
img_rows, img_cols, num_chan = 8, 9, 5
t = 6  # window length

# trial segment lengths (points) and labels per trial
trials = [470, 466, 412, 476, 370, 390, 474, 432, 530, 474, 470, 466, 470, 476, 412]
all_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

# compute boundaries and number of segments per session
boundaries = np.concatenate(([0], np.cumsum(trials)))
ns = sum((length // t) for length in trials)
print(f"Segments per session (ns): {ns}")

# input/output paths
input_dirs = {
    'DE': r"D:\dachuang\SEED_preprocess\DE",
    'PSD': r"D:\dachuang\SEED_preprocess\PSD"
}
output_base = r"D:\dachuang\SEED_3D"

def segment_and_save(feature_type):
    # load 8x9 features
    x89_path = os.path.join(input_dirs[feature_type], f"X89_{feature_type}.npy")
    print(f"Loading {feature_type} features from: {x89_path}")
    X89 = np.load(x89_path)  # shape: (45*6788, 8, 9, 5)

    # reshape to sessions
    X_sess = X89.reshape(45, -1, img_rows, img_cols, num_chan)  # (45, 6788, 8,9,5)

    # prepare output arrays
    new_x = np.zeros((45, ns, t, img_rows, img_cols, num_chan), dtype=X89.dtype)
    new_y = []

    print(f"Segmenting {feature_type} into windows of length {t}...")
    for sess in tqdm(range(45), desc=f"{feature_type} sessions"):
        seg_idx = 0
        for trial_idx in range(len(trials)):
            start, end = boundaries[trial_idx], boundaries[trial_idx+1]
            label = all_labels[trial_idx]
            # sliding windows with step=t
            for i in range(start, end - t + 1, t):
                new_x[sess, seg_idx] = X_sess[sess, i:i+t]
                new_y.append(label)
                seg_idx += 1
    new_y = np.array(new_y)

    # create output directory
    out_dir = os.path.join(output_base, feature_type)
    os.makedirs(out_dir, exist_ok=True)

    # print dimensions
    print(f"{feature_type} segmentation complete.")
    print(f" new_x shape: {new_x.shape}")
    print(f" new_y shape: {new_y.shape}")

    # save
    x_out = os.path.join(out_dir, f"t{t}x_89.npy")
    y_out = os.path.join(out_dir, f"t{t}y_89.npy")
    np.save(x_out, new_x)
    np.save(y_out, new_y)
    print(f"Saved: {x_out}\n       {y_out}\n")

if __name__ == '__main__':
    for ft in ['DE', 'PSD']:
        segment_and_save(ft)

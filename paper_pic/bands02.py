import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ---------------------------------------------
# 1. Configuration and Parameters
# ---------------------------------------------
mat_file = r"D:\dachuang\EEG_preprocessed\1.mat"
fs = 200

# Duration to extract (in seconds)
duration_secs = 5
n_samples = fs * duration_secs  # 1,000 samples

# Frequency bands
bands = [
    ("Theta", 4, 8),
    ("Alpha", 8, 14),
    ("Beta", 14, 31),
    ("Gamma", 31, 51),
]

segment_secs = 0.5
n_samples_segment = int(fs * segment_secs)  # 100 samples

# ---------------------------------------------
# 2. Load .mat and Select Experiment
# ---------------------------------------------
mat = loadmat(mat_file)
exp_keys = [k for k in mat.keys() if not k.startswith("__")]
if not exp_keys:
    raise KeyError(f"No valid experiment keys found. Keys: {list(mat.keys())}")

exp_key = exp_keys[0]
exp_data_obj = mat[exp_key]

# ---------------------------------------------
# 3. Extract 62-Channel EEG Array
# ---------------------------------------------
def extract_eeg_array(obj):
    if isinstance(obj, np.ndarray) and obj.dtype != np.object_:
        if obj.ndim == 2 and 62 in obj.shape:
            return obj.copy()
        else:
            raise ValueError(f"Array shape {obj.shape} does not contain 62 channels.")
    if isinstance(obj, np.ndarray) and obj.dtype.names is not None:
        for field_name in obj.dtype.names:
            field_data = obj[field_name]
            if isinstance(field_data, np.ndarray) and field_data.dtype != np.object_:
                if field_data.ndim == 2 and 62 in field_data.shape:
                    return field_data.copy()
            if isinstance(field_data, np.ndarray) and field_data.dtype == np.object_:
                try:
                    nested0 = field_data[0, 0]
                except Exception:
                    nested0 = None
                if isinstance(nested0, np.ndarray) and nested0.dtype != np.object_:
                    if nested0.ndim == 2 and 62 in nested0.shape:
                        return nested0.copy()
                if isinstance(nested0, np.ndarray) and nested0.dtype.names is not None:
                    return extract_eeg_array(nested0)
        raise KeyError(f"No (62, N) field in struct for experiment '{exp_key}'. Fields: {obj.dtype.names}")
    raise TypeError(f"Cannot extract EEG from type {type(obj)} for experiment '{exp_key}'.")

raw = extract_eeg_array(exp_data_obj)
if raw.ndim != 2:
    raise ValueError(f"Extracted data has {raw.ndim} dims; expected 2.")

if raw.shape[0] == 62:
    eeg_data = raw.copy()
elif raw.shape[1] == 62:
    eeg_data = raw.T.copy()
else:
    raise ValueError(f"Extracted array shape {raw.shape} does not contain 62 channels.")

total_samples = eeg_data.shape[1]
if total_samples < n_samples:
    raise ValueError(f"Experiment '{exp_key}' has only {total_samples} samples (< {n_samples}).")

# ---------------------------------------------
# 4. Extract First `duration_secs` Seconds
# ---------------------------------------------
data_short = eeg_data[:, :n_samples]
t_axis = np.arange(n_samples) / fs  # [0, 5) seconds

# ---------------------------------------------
# 5. Plot Channels 1 & 2 in One Figure
#    Black border, no ticks/labels, signal touches edges, zero-line dashed
# ---------------------------------------------
fig1, axes1 = plt.subplots(nrows=2, ncols=1, figsize=(12, 4))

for idx in [0, 1]:
    ax = axes1[idx]
    # Plot the signal (blue)
    ax.plot(t_axis, data_short[idx, :], color="#1F77B4", linewidth=0.5)
    # Plot dashed zero line
    ax.axhline(0, linestyle="--", linewidth=0.5, color="gray")
    # Set x‐limits exactly to data range, remove margins
    ax.set_xlim(t_axis[0], t_axis[-1])
    ax.margins(x=0)
    # Black border (all spines)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)
    # Remove ticks/labels
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

# ---------------------------------------------
# 6. Plot Channel 62 in Separate Figure
#    Black border, no ticks/labels, signal touches edges, zero-line dashed
# ---------------------------------------------
fig2, ax2 = plt.subplots(figsize=(12, 2))

# Plot the last channel (index 61)
ax2.plot(t_axis, data_short[61, :], color="#1F77B4", linewidth=0.5)
# Dashed zero line
ax2.axhline(0, linestyle="--", linewidth=0.5, color="gray")
# Force x‐axis to data range, remove margins
ax2.set_xlim(t_axis[0], t_axis[-1])
ax2.margins(x=0)
# Black border
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(1)
# Remove ticks/labels
ax2.set_xticks([])
ax2.set_yticks([])

plt.show()

# ---------------------------------------------
# 7. Bandpass Filter Functions
# ---------------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, signal)

# ---------------------------------------------
# 8. Extract 0.5‐Second Segment from Fp1
# ---------------------------------------------
channel_fp1 = eeg_data[0, :]
segment_fp1 = channel_fp1[:n_samples_segment]
t_seg = np.arange(n_samples_segment) / fs  # [0, 0.5) seconds

# ---------------------------------------------
# 9. Filter Segment into Four Bands
# ---------------------------------------------
filtered = {}
for band_name, low_f, high_f in bands:
    filtered[band_name] = bandpass_filter(segment_fp1, low_f, high_f, fs, order=4)

# ---------------------------------------------
# 10. Plot Filtered Bands, Stacked, Black Border,
#     Blue Signal, Dashed Zero Line, Signal Touches Edges
# ---------------------------------------------
fig3, axes3 = plt.subplots(nrows=len(bands), ncols=1, figsize=(10, 6))

for idx, (band_name, low_f, high_f) in enumerate(bands):
    ax = axes3[idx]
    # Plot blue signal
    ax.plot(t_seg, filtered[band_name], color="#1F76B3", linewidth=1.0)
    # Dashed zero line
    ax.axhline(0, linestyle="--", linewidth=0.5, color="gray")
    # Force x‐axis to data range, remove margins
    ax.set_xlim(t_seg[0], t_seg[-1])
    ax.margins(x=0)
    # Black border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)
    # Remove ticks/labels
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

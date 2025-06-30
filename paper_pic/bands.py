import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ---------------------------------------------
# 1. Configuration and Parameters
# ---------------------------------------------
# Path to your .mat file containing 80 experiments
mat_file = r"D:\dachuang\EEG_preprocessed\1.mat"

# Sampling rate (Hz)
fs = 200

# Duration to extract for the full 62‐channel plot (in seconds)
duration_secs = 5
n_samples_100s = fs * duration_secs  # 200 * 100 = 20,000 samples

# Frequency bands for filtering
bands = [
    ("Theta", 4, 8),
    ("Alpha", 8, 14),
    ("Beta", 14, 31),
    ("Gamma", 31, 51),
]

# Length of the short segment for channel Fp1 (in seconds)
segment_secs = 0.5
n_samples_segment = int(fs * segment_secs)  # 200 * 0.5 = 100 samples

# ---------------------------------------------
# 2. Load the .mat File and Inspect Experiment Keys
# ---------------------------------------------
mat = loadmat(mat_file)

# Filter out MATLAB‐internal keys beginning with "__"
exp_keys = [k for k in mat.keys() if not k.startswith("__")]
if len(exp_keys) == 0:
    raise KeyError(f"No valid experiment keys found in .mat file. Available keys: {list(mat.keys())}")

# For this example, we'll process the first experiment (key '1').
# You can loop over exp_keys or select a different index if needed.
exp_key = exp_keys[0]  # e.g., '1'
print(f"Using experiment key: '{exp_key}' from the .mat file.")

# Extract the raw data object for experiment '1'
exp_data_obj = mat[exp_key]

# ---------------------------------------------
# 3. Dig into exp_data_obj to Find the 62‐Channel Array
#    (It may be a plain array or a MATLAB struct)
# ---------------------------------------------
def extract_eeg_array(obj):
    """
    Given a loaded MATLAB object 'obj' for one experiment, attempt to find a (62, N) NumPy array.
    Handles cases where 'obj' is:
      1) A plain 2D NumPy array of shape (62, N) or (N, 62),
      2) A structured array (MATLAB struct) with fields containing the EEG matrix.
    """
    # Case A: Plain NumPy ndarray
    if isinstance(obj, np.ndarray) and obj.dtype != np.object_:
        # If it's 2D and one dimension is 62, assume that's channels
        if obj.ndim == 2 and 62 in obj.shape:
            return obj.copy()
        else:
            raise ValueError(f"Experiment array has shape {obj.shape}, but neither dimension is 62.")

    # Case B: MATLAB struct → NumPy structured ndarray with dtype.names
    if isinstance(obj, np.ndarray) and obj.dtype.names is not None:
        # MATLAB structs load as an ndarray of dtype=object or structured with fields.
        # Often the actual EEG channels live under a field named 'data', 'EEG', or similar.
        # We'll search all fields for a sub‐array whose first dimension is 62.
        for field_name in obj.dtype.names:
            field_data = obj[field_name]
            # If this field is itself a 2D array of shape (62, N) or (N, 62), return that
            if isinstance(field_data, np.ndarray) and field_data.dtype != np.object_:
                if field_data.ndim == 2 and 62 in field_data.shape:
                    return field_data.copy()
            # If this field is another nested struct (object dtype), dive one more level
            if isinstance(field_data, np.ndarray) and field_data.dtype == np.object_:
                nested = field_data
                # Often nested is of shape (1,1), so index [0,0]
                try:
                    nested0 = nested[0, 0]
                except Exception:
                    nested0 = None
                if isinstance(nested0, np.ndarray) and nested0.dtype != np.object_:
                    if nested0.ndim == 2 and 62 in nested0.shape:
                        return nested0.copy()
                # If nested0 is itself a struct, recursively search its fields
                if isinstance(nested0, np.ndarray) and nested0.dtype.names is not None:
                    return extract_eeg_array(nested0)

        # If we reach here, no suitable field found
        raise KeyError(
            f"Could not find a (62, N) array in any field of the struct for experiment '{exp_key}'. "
            f"Struct fields: {obj.dtype.names}"
        )

    # Otherwise, we don't know how to extract
    raise TypeError(f"Unsupported data type for experiment '{exp_key}': {type(obj)}")


# Attempt to extract a raw EEG array of shape (62, total_samples)
raw = extract_eeg_array(exp_data_obj)

# At this point, `raw` should be a NumPy array with shape (62, total_samples) or (total_samples, 62)
if raw.ndim != 2:
    raise ValueError(f"Extracted EEG data has unexpected number of dimensions: {raw.ndim}")

# If shape is (N, 62), transpose to (62, N)
if raw.shape[1] == 62 and raw.shape[0] != 62:
    eeg_data = raw.T.copy()
elif raw.shape[0] == 62:
    eeg_data = raw.copy()
else:
    raise ValueError(f"Extracted EEG array has shape {raw.shape}, but neither dimension is 62.")

total_samples = eeg_data.shape[1]
if total_samples < n_samples_100s:
    raise ValueError(
    f"Experiment '{exp_key}' only contains {total_samples} samples (< {n_samples_100s} needed for 100 seconds)."
)

print(f"Extracted EEG data for experiment '{exp_key}' has shape {eeg_data.shape} (62 channels, {total_samples} samples).")

# ---------------------------------------------
# 4. Extract the First 100 Seconds of All 62 Channels
# ---------------------------------------------
data_100s = eeg_data[:, :n_samples_100s]  # shape = (62, 20,000)

# ---------------------------------------------
# 5. Plot All 62 Channels (First 100 Seconds)
# ---------------------------------------------
t_full = np.arange(n_samples_100s) / fs  # time axis (0 to 100 s)

fig1, axes1 = plt.subplots(
    nrows=62,
    ncols=1,
    sharex=True,
    figsize=(12, 2 * 62),
    constrained_layout=True,
)
fig1.suptitle(f"Experiment {exp_key}: 62 Channels (First {duration_secs} s)", fontsize=16, y=0.92)

for ch_idx in range(62):
    ax = axes1[ch_idx]
    ax.plot(t_full, data_100s[ch_idx, :], linewidth=0.5)
    ax.set_ylabel(f"Ch {ch_idx + 1}", fontsize=8)
    ax.set_xlim(0, duration_secs)
    ax.tick_params(labelsize=6)
    if ch_idx < 61:
        ax.set_xticklabels([])

axes1[-1].set_xlabel("Time (s)", fontsize=10)
plt.show()

# ---------------------------------------------
# 6. Bandpass Filter Functions
# ---------------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Design a Butterworth bandpass filter.
    Returns filter coefficients (b, a).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Apply zero‐phase Butterworth bandpass filter to 1D signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = filtfilt(b, a, signal)
    return filtered

# ---------------------------------------------
# 7. Extract a 0.5‐Second Segment from Channel Fp1
# ---------------------------------------------
# Assumption: channel index 0 corresponds to Fp1. Adjust if your channel order differs.
channel_fp1 = eeg_data[0, :]  # shape = (total_samples,)
segment_fp1 = channel_fp1[:n_samples_segment]  # first 0.5 seconds
t_segment = np.arange(n_samples_segment) / fs  # time axis (0 to 0.5 s)

# ---------------------------------------------
# 8. Filter the 0.5‐Second Segment into Theta, Alpha, Beta, Gamma
# ---------------------------------------------
filtered_segments = {}
for band_name, low_f, high_f in bands:
    filtered = bandpass_filter(segment_fp1, low_f, high_f, fs, order=4)
    filtered_segments[band_name] = filtered

# ---------------------------------------------
# 9. Plot the Filtered Bands for the 0.5‐Second Segment (Channel Fp1)
# ---------------------------------------------
fig2, axes2 = plt.subplots(
    nrows=len(bands),
    ncols=1,
    sharex=True,
    figsize=(10, 2 * len(bands)),
    constrained_layout=True,
)
fig2.suptitle(f"Experiment {exp_key}, Channel Fp1: {segment_secs}-Second Segment Filtered into Bands", fontsize=16, y=0.9)

for idx, (band_name, low_f, high_f) in enumerate(bands):
    ax = axes2[idx]
    ax.plot(t_segment, filtered_segments[band_name], linewidth=1.0)
    ax.set_ylabel(f"{band_name}\n({low_f}-{high_f} Hz)", fontsize=8)
    ax.set_xlim(0, segment_secs)
    ax.tick_params(labelsize=6)
    if idx < len(bands) - 1:
        ax.set_xticklabels([])

axes2[-1].set_xlabel("Time (s)", fontsize=10)
plt.show()

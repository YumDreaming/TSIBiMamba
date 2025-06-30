import os
import math
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter

# GPU and progress bar support
try:
    import cupy as cp
    from cupyx.scipy.signal import spectrogram
    use_gpu = True
except ImportError:
    use_gpu = False

from tqdm import tqdm

# ----- Feature computation functions -----
def compute_DE(signal):
    """
    Compute Differential Entropy for a signal segment.
    """
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


def compute_PSD(segment, sfreq, band):
    """
    Estimate band power of a signal segment in the given frequency band.
    Uses GPU via CuPy spectrogram if available, else CPU via SciPy welch.
    """
    if use_gpu:
        x = cp.asarray(segment)
        f, _, Sxx = spectrogram(
            x,
            fs=sfreq,
            window='hann',
            nperseg=len(segment),
            noverlap=0,
            mode='psd'
        )
        mask = (f >= band[0]) & (f <= band[1])
        df = f[1] - f[0]
        power = cp.sum(Sxx[mask]) * df
        return float(power.get())
    else:
        from scipy.signal import welch
        freqs, psd = welch(
            segment,
            fs=sfreq,
            window='hann',
            nperseg=len(segment),
            noverlap=0,
            scaling='density'
        )
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        df = freqs[1] - freqs[0]
        return np.sum(psd[idx]) * df


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


def decompose(file_path, name):
    """
    Extract DE and PSD features for all trials in one session file.
    Returns DE, PSD arrays and labels.
    """
    data = loadmat(file_path)
    sfreq = 200  # sampling rate

    decomposed_de = []
    decomposed_psd = []
    labels = []
    all_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 51)]

    for trial in tqdm(range(15), desc=f"{name} Trials", leave=False):
        key = f"{name}_eeg{trial+1}"
        trial_data = data[key]
        n_points = trial_data.shape[1]
        n_seg = n_points // 100
        labels.extend([all_labels[trial]] * n_seg)

        de_trial = np.zeros((n_seg, 62, 5))
        psd_trial = np.zeros((n_seg, 62, 5))

        for ch in tqdm(range(62), desc="Channels", leave=False):
            signal = trial_data[ch]
            # filter each band once
            filtered = [butter_bandpass_filter(signal, low, high, sfreq, order=3) for (low, high) in bands]

            for b_idx, band_signal in enumerate(filtered):
                for seg_idx in range(n_seg):
                    seg = band_signal[seg_idx*100:(seg_idx+1)*100]
                    de_trial[seg_idx, ch, b_idx] = compute_DE(seg)
                    psd_trial[seg_idx, ch, b_idx] = compute_PSD(seg, sfreq, bands[b_idx])

        decomposed_de.append(de_trial)
        decomposed_psd.append(psd_trial)

    decomposed_de = np.vstack(decomposed_de)
    decomposed_psd = np.vstack(decomposed_psd)
    labels = np.array(labels)
    return decomposed_de, decomposed_psd, labels


if __name__ == "__main__":
    data_root = r"D:\dachuang\SEED\Preprocessed_EEG"
    output_de = r"D:\dachuang\SEED_preprocess_GPU\DE"
    output_psd = r"D:\dachuang\SEED_preprocess_GPU\PSD"
    os.makedirs(output_de, exist_ok=True)
    os.makedirs(output_psd, exist_ok=True)

    people_name = [
        '1_20131027','1_20131030','1_20131107', '6_20130712','6_20131016','6_20131113',
        '7_20131027','7_20131030','7_20131106','15_20130709','15_20131016','15_20131105',
        '12_20131127','12_20131201','12_20131207','10_20131130','10_20131204','10_20131211',
        '2_20140404','2_20140413','2_20140419','5_20140411','5_20140418','5_20140506',
        '8_20140511','8_20140514','8_20140521','13_20140527','13_20140603','13_20140610',
        '3_20140603','3_20140611','3_20140629','14_20140601','14_20140615','14_20140627',
        '11_20140618','11_20140625','11_20140630','9_20140620','9_20140627','9_20140704',
        '4_20140621','4_20140702','4_20140705'
    ]
    short_name = ['djc','djc','djc','mhw','mhw','mhw','phl','phl','phl',
                  'zjy','zjy','zjy','wyw','wyw','wyw','ww','ww','ww',
                  'jl','jl','jl','ly','ly','ly','sxy','sxy','sxy',
                  'xyl','xyl','xyl','jj','jj','jj','ys','ys','ys',
                  'wsf','wsf','wsf','wk','wk','wk','lqj','lqj','lqj']

    X_de = np.empty((0, 62, 5))
    X_psd = np.empty((0, 62, 5))
    y = np.array([])

    for p, sn in tqdm(zip(people_name, short_name), desc="Sessions"):
        fpath = os.path.join(data_root, p + ('.mat' if p.endswith('.mat') else ''))
        de_feat, psd_feat, labels = decompose(fpath, sn)
        X_de = np.vstack([X_de, de_feat])
        X_psd = np.vstack([X_psd, psd_feat])
        y = np.append(y, labels)

    # save 1D features
    np.save(os.path.join(output_de, "X_1D_DE.npy"), X_de)
    np.save(os.path.join(output_de, "y.npy"), y)
    np.save(os.path.join(output_psd, "X_1D_PSD.npy"), X_psd)
    np.save(os.path.join(output_psd, "y.npy"), y)

    # map to 8x9 grid
    X89_de = np.zeros((len(y), 8, 9, 5))
    X89_psd = np.zeros((len(y), 8, 9, 5))
    X89_de[:, 0, 2, :] = X_de[:, 3, :];   X89_psd[:, 0, 2, :] = X_psd[:, 3, :]
    X89_de[:, 0, 3:6, :] = X_de[:, 0:3, :]; X89_psd[:, 0, 3:6, :] = X_psd[:, 0:3, :]
    X89_de[:, 0, 6, :] = X_de[:, 4, :];   X89_psd[:, 0, 6, :] = X_psd[:, 4, :]
    for r in range(5):
        X89_de[:, r+1, :, :] = X_de[:, 5+r*9:5+(r+1)*9, :]
        X89_psd[:, r+1, :, :] = X_psd[:, 5+r*9:5+(r+1)*9, :]
    X89_de[:, 6, 1:8, :] = X_de[:, 50:57, :]; X89_psd[:, 6, 1:8, :] = X_psd[:, 50:57, :]
    X89_de[:, 7, 2:7, :] = X_de[:, 57:62, :]; X89_psd[:, 7, 2:7, :] = X_psd[:, 57:62, :]

    np.save(os.path.join(output_de, "X89_DE.npy"), X89_de)
    np.save(os.path.join(output_psd, "X89_PSD.npy"), X89_psd)

    print("All features extracted and saved.")

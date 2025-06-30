#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def load_deap_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    # EEG channels (32) and labels
    return data['data'][:, :32, :], data['labels']

def remove_baseline(signal, sample_rate, baseline_duration):
    start = int(baseline_duration * sample_rate)
    return signal[start:]

def compute_gaf(x):
    # x: 1D torch tensor on device
    min_, max_ = x.min(), x.max()
    # scale to [-1,1]
    X = (x - min_) / (max_ - min_) * 2 - 1
    phi = torch.acos(torch.clamp(X, -1, 1))
    G = torch.cos(phi.unsqueeze(1) + phi.unsqueeze(0))
    return G

def compute_rp(x, embed_dim=3, time_delay=3):
    # embed the signal
    L = x.shape[0]
    M = L - (embed_dim - 1) * time_delay
    idx = torch.arange(embed_dim, device=x.device) * time_delay
    emb = torch.stack([x[i:i+M] for i in idx], dim=1)
    # pairwise distances
    D = torch.cdist(emb, emb, p=2)
    thresh = torch.median(D)
    R = (D <= thresh).float()
    return R

def process_file(file_path, output_root, args, device):
    data, labels = load_deap_data(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]
    # create output dirs
    g_out = os.path.join(output_root, 'GASF', base)
    r_out = os.path.join(output_root, 'RP',   base)
    os.makedirs(g_out, exist_ok=True)
    os.makedirs(r_out, exist_ok=True)

    # compute sizes
    window_samples = int(args.window_length * args.sample_rate)
    step         = int(window_samples - args.overlap * args.sample_rate)

    for trial_idx in tqdm(range(data.shape[0]), desc=base, unit="trial"):
        # move trial to GPU tensor
        trial = torch.tensor(data[trial_idx], dtype=torch.float32, device=device)
        # remove baseline
        trial = torch.stack([remove_baseline(ch, args.sample_rate, args.baseline_duration) for ch in trial])

        label_bin = 1 if labels[trial_idx][0] > 5 else 0
        gaf_channels, rp_channels = [], []

        for ch in range(trial.size(0)):
            sig = trial[ch]
            num_w = (sig.shape[0] - window_samples) // step + 1
            g_winds, r_winds = [], []

            for i in range(num_w):
                seg = sig[i * step : i * step + window_samples]

                # GASF
                G = compute_gaf(seg)
                G = G.unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
                G_resized = F.interpolate(G, (args.pic_size, args.pic_size), mode='bilinear', align_corners=False)[0,0]
                g_img = ((G_resized + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
                g_winds.append(g_img)

                # Recurrence Plot
                R = compute_rp(seg)
                R = R.unsqueeze(0).unsqueeze(0)
                R_resized = F.interpolate(R, (args.pic_size, args.pic_size), mode='bilinear', align_corners=False)[0,0]
                r_img = (R_resized * 255).clamp(0, 255).byte().cpu().numpy()
                r_winds.append(r_img)

            gaf_channels.append(np.stack(g_winds, axis=0))
            rp_channels.append(np.stack(r_winds, axis=0))

        g_arr = np.stack(gaf_channels, axis=0)
        r_arr = np.stack(rp_channels, axis=0)

        # save arrays
        np.save(os.path.join(g_out, f"{base}_trial_{trial_idx}_GASF.npy"), g_arr)
        np.save(os.path.join(r_out, f"{base}_trial_{trial_idx}_RP.npy"),   r_arr)

        # record labels
        with open(os.path.join(g_out, f"{base}_labels.txt"), 'a') as fg:
            fg.write(f"{base}_trial_{trial_idx}_GASF.npy,{label_bin}\n")
        with open(os.path.join(r_out, f"{base}_labels.txt"), 'a') as fr:
            fr.write(f"{base}_trial_{trial_idx}_RP.npy,{label_bin}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEAP preprocessing on GPU")
    parser.add_argument('-i', '--input',  default='/root/deap/data_preprocessed_python')
    parser.add_argument('-o', '--output', default='/root/autodl-tmp/deap_pic')
    parser.add_argument('--baseline_duration', type=int, default=3)
    parser.add_argument('--sample_rate',      type=int, default=128)
    parser.add_argument('--window_length',   type=int, default=12)
    parser.add_argument('--overlap',         type=int, default=6)
    parser.add_argument('--pic_size',        type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    for fname in os.listdir(args.input):
        if fname.endswith('.dat'):
            process_file(os.path.join(args.input, fname), args.output, args, device)

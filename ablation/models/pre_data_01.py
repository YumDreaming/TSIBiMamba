#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
from pyts.image import RecurrencePlot
import mne

# ─── 电极与频段配置 ────────────────────────────────────────
CH_NAMES = [
    'Fp1','AF3','F3','F7','FC5','FC1','C3','T7',
    'CP5','CP1','P3','P7','PO3','O1','Oz','Pz',
    'Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz',
    'C4','T8','CP6','CP2','P4','P8','PO4','O2'
]
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}

# ─── 工具函数 ─────────────────────────────────────────────
def load_deap(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data['data'][:, :32, :], data['labels']

def remove_baseline(sig, sr, baseline_sec):
    return sig[:, baseline_sec * sr:]

def fft_band_power(seg_t, fmin, fmax, sr):
    n = seg_t.shape[-1]
    freqs  = torch.fft.rfftfreq(n, d=1/sr).to(seg_t.device)
    fftseg = torch.fft.rfft(seg_t, dim=-1)
    mask   = ((freqs >= fmin) & (freqs < fmax)).to(torch.complex64)
    band   = torch.fft.irfft(fftseg * mask.unsqueeze(0), n=n, dim=-1)
    return band.pow(2).mean(dim=-1)  # (32,)

def topo_heatmap(power, elec_pos, pic_size):
    # elec_pos: (32,2) tensor on device
    # power: CPU numpy or torch tensor on device
    if not isinstance(power, torch.Tensor):
        power = torch.tensor(power, dtype=torch.float32, device=elec_pos.device)
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-1,1,pic_size,device=elec_pos.device),
        torch.linspace(-1,1,pic_size,device=elec_pos.device),
        indexing='xy'
    ), dim=-1).reshape(-1,2)  # (pic_size**2,2)
    d = torch.cdist(grid, elec_pos)               # (P,32)
    w = 1/(d+1e-6)
    w = w / w.sum(dim=1, keepdim=True)
    img = (w * power.unsqueeze(0)).sum(dim=1)
    img = img.reshape(pic_size, pic_size)
    img = (img - img.min())/(img.max()-img.min()+1e-6)
    return (img * 255).byte().cpu().numpy()

def compute_rp_torch(sig, embed_dim=3, tau=3):
    L = sig.shape[0]
    M = L - (embed_dim-1)*tau
    idx = torch.arange(embed_dim, device=sig.device)*tau
    emb = torch.stack([sig[i:i+M] for i in idx], dim=1)
    D = torch.cdist(emb, emb)
    return (D <= torch.median(D)).float()

# ─── 主流程 ───────────────────────────────────────────────
def process_subject(path_dat, out_root, args,
                    device, pca_model, info, elec_pos):
    data, labels = load_deap(path_dat)
    base = os.path.splitext(os.path.basename(path_dat))[0]

    # 保持旧结构目录名
    g_out = os.path.join(out_root, 'GASF', base)
    r_out = os.path.join(out_root, 'RP',   base)
    os.makedirs(g_out, exist_ok=True)
    os.makedirs(r_out, exist_ok=True)

    # 窗口参数
    W = args.window_length * args.sample_rate
    step = W - args.overlap * args.sample_rate
    pic = args.pic_size
    K = args.rp_k

    # 试次循环
    for trial_idx in tqdm(range(data.shape[0]),
                          desc=f"{base} trials", leave=False):
        trial = torch.tensor(data[trial_idx], dtype=torch.float32, device=device)
        trial = remove_baseline(trial, args.sample_rate, args.baseline_duration)

        label_bin = int(labels[trial_idx][0] > 5)
        topo_wins, rp_wins = [], []

        n_wins = (trial.shape[1] - W)//step + 1
        # 窗口循环
        for win in range(n_wins):
            seg = trial[:, win*step:win*step+W]  # (32,W)

            # —— Topo 分支 ——
            band_imgs = []
            for fmin,fmax in BANDS.values():
                pwr = fft_band_power(seg, fmin, fmax, args.sample_rate)
                img = topo_heatmap(pwr, elec_pos, pic)
                band_imgs.append(img)
            topo_wins.append(np.stack(band_imgs,0))  # (5,128,128)

            # —— RP 分支 ——
            # PCA 投影
            scores = pca_model.transform(seg.T.cpu().numpy()).T  # (K,W)
            comp_imgs = []
            for k in range(K):
                sig = torch.tensor(scores[k], dtype=torch.float32, device=device)
                R = compute_rp_torch(sig)
                R = R.unsqueeze(0).unsqueeze(0)
                R = F.interpolate(R, size=(pic,pic),
                                  mode='bilinear', align_corners=False)[0,0]
                comp_imgs.append((R*255).byte().cpu().numpy())
            rp_wins.append(np.stack(comp_imgs,0))       # (K,128,128)

        # 堆叠 → (channels, windows, H, W)
        g_arr = np.stack(topo_wins, 1)
        r_arr = np.stack(rp_wins,   1)

        # 保存
        np.save(os.path.join(g_out, f"{base}_trial_{trial_idx}_GASF.npy"), g_arr)
        np.save(os.path.join(r_out, f"{base}_trial_{trial_idx}_RP.npy"),   r_arr)

        # 记录标签
        with open(os.path.join(g_out, f"{base}_labels.txt"), 'a') as fg:
            fg.write(f"{base}_trial_{trial_idx}_GASF.npy,{label_bin}\n")
        with open(os.path.join(r_out, f"{base}_labels.txt"), 'a') as fr:
            fr.write(f"{base}_trial_{trial_idx}_RP.npy,{label_bin}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEAP Topo+RP preprocessing on GPU")
    parser.add_argument('-i','--input', default='/root/deap/data_preprocessed_python')
    parser.add_argument('-o','--output',default='/root/autodl-tmp/deap_pic01')
    parser.add_argument('--baseline_duration',type=int,default=3)
    parser.add_argument('--sample_rate',type=int,default=128)
    parser.add_argument('--window_length',type=int,default=12)
    parser.add_argument('--overlap',type=int,default=6)
    parser.add_argument('--pic_size',type=int,default=128)
    parser.add_argument('--rp_k',type=int,default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # —— 全局 PCA 拟合 ——
    all_mat = []
    for fn in os.listdir(args.input):
        if not fn.endswith('.dat'): continue
        d,_ = load_deap(os.path.join(args.input,fn))
        d2 = d[:,:,args.baseline_duration*args.sample_rate:]
        # 收集所有窗口原始时序
        W = args.window_length*args.sample_rate
        step = W - args.overlap*args.sample_rate
        for trial in d2:
            for i in range((trial.shape[1]-W)//step+1):
                all_mat.append(trial[:,i*step:i*step+W].T)  # (W,32)
    X_all = np.vstack(all_mat)  # (n_wins*W,32)
    pca_model = PCA(n_components=args.rp_k).fit(X_all)
    # 用于插值的电极位置
    info = mne.create_info(ch_names=CH_NAMES, sfreq=args.sample_rate, ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'))
    elec_pos = torch.tensor(
        np.stack([info['chs'][i]['loc'][:2] for i in range(32)]),
        dtype=torch.float32, device=device
    )

    # 对每个被试，显示其处理进度
    for fn in tqdm(os.listdir(args.input), desc="Subjects"):
        if fn.endswith('.dat'):
            process_subject(os.path.join(args.input,fn),
                            args.output, args, device,
                            pca_model, info, elec_pos)

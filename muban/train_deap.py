import os
import yaml
import copy
import scipy.io as sio
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from shengji06 import BiMambaVision

# 假设模型类定义在同一文件或可 import
# from your_model_module import BiMambaVision

class DEAPDataset(Dataset):
    def __init__(self, g_data, r_data, labels, seq_len=6):
        """
        g_data, r_data: numpy arrays of shape (4800,4,8,9)
        labels: numpy array of shape (4800,)
        seq_len: 时间窗口长度
        """
        self.g = g_data
        self.r = r_data
        self.labels = labels
        self.seq_len = seq_len
        self.num_windows = self.g.shape[0] // seq_len

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        s = idx * self.seq_len
        e = s + self.seq_len
        g_seq = self.g[s:e]                  # (T,4,8,9)
        r_seq = self.r[s:e]
        lbl  = self.labels[e - 1]            # 用最后一帧的标签
        # 二分类：阈值5分
        lbl = 1 if lbl > 5 else 0
        # 转为 Tensor
        g_seq = torch.from_numpy(g_seq).float()
        r_seq = torch.from_numpy(r_seq).float()
        lbl   = torch.tensor(lbl, dtype=torch.long)
        return g_seq, r_seq, lbl

def train_on_subject(subject_file, config, device):
    subj_id = os.path.splitext(subject_file)[0]
    print(f"=== Subject {subj_id} ===")

    # 加载预处理好的 data/labels
    de_mat  = sio.loadmat(os.path.join(config['data_dir'], 'DE',  subject_file))
    psd_mat = sio.loadmat(os.path.join(config['data_dir'], 'PSD', subject_file))
    g_data  = de_mat['data']              # (4800,4,8,9)
    r_data  = psd_mat['data']
    labels  = de_mat['valence_labels'].flatten()

    # 构造 Dataset
    full_ds = DEAPDataset(g_data, r_data, labels, seq_len=config['input']['T'])

    # 五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_ds), start=1):
        print(f"--- Fold {fold}/5 ---")
        train_ds = torch.utils.data.Subset(full_ds, train_idx)
        val_ds   = torch.utils.data.Subset(full_ds, val_idx)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                                  shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                                  shuffle=False, num_workers=4, pin_memory=True)

        # 每折重新实例化模型
        mcfg = copy.deepcopy(config['mmmamba'])
        model = BiMambaVision(
            mm_input_size=mcfg['mm_input_size'],
            mm_output_sizes=mcfg['mm_output_sizes'],
            dropout=mcfg['dropout'],
            activation=mcfg['activation'],
            causal=mcfg['causal'],
            mamba_config=mcfg['mamba_config']
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        if config['lr_scheduler'] == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
        else:
            scheduler = None

        # 训练循环
        for epoch in range(1, config['epochs'] + 1):
            model.train()
            running_loss = 0.0
            loop = tqdm(train_loader, desc=f"[S{subj_id} F{fold}] Epoch {epoch}/{config['epochs']}")
            for g_batch, r_batch, y_batch in loop:
                g_batch = g_batch.to(device)       # (B,T,4,8,9)
                r_batch = r_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(g_batch, r_batch)  # (B,2)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * g_batch.size(0)
                loop.set_postfix(loss=running_loss / ((loop.n+1)*config['batch_size']))

            if scheduler:
                scheduler.step()

            avg_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")

            # 验证
            model.eval()
            correct = 0
            with torch.no_grad():
                for g_batch, r_batch, y_batch in val_loader:
                    g_batch = g_batch.to(device)
                    r_batch = r_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds = model(g_batch, r_batch).argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
            acc = correct / len(val_loader.dataset)
            print(f"Epoch {epoch} Val Acc: {acc:.4f}")

        # 保存本折模型
        os.makedirs(config['save_dir'], exist_ok=True)
        ckpt_path = os.path.join(config['save_dir'], f"{subj_id}_fold{fold}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}\n")

def main():
    # 1) 加载配置
    config = yaml.safe_load(open('config.yaml', 'r'))

    # 2) 设备
    device_name = config['device'][0] if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f"Using device: {device}\n")

    # 3) 针对单被试训练（可修改为循环多个 subject）
    de_dir = os.path.join(config['data_dir'], 'DE')
    subject_files = sorted(os.listdir(de_dir))
    # 这里只训练第一个被试，如需遍历所有，把 [0] 去掉
    train_on_subject(subject_files[0], config, device)

if __name__ == '__main__':
    main()

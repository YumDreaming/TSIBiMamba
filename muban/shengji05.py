import os
import time
import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# —— 超参数 & 配置 ——
NUM_CLASSES = 2
BATCH_SIZE  = 64
IMG_ROWS, IMG_COLS, NUM_CHAN = 8, 9, 4
SEQ_LEN     = 6      # 每个样本由 6 帧组成
EPOCHS      = 100
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED        = 7

torch.manual_seed(SEED)
np.random.seed(SEED)

# —— 1. 定义 PyTorch 版 “BaseCNN” 子网 ——
class BaseCNN(nn.Module):
    def __init__(self, in_chan=NUM_CHAN):
        super().__init__()
        # conv1: (B,4,8,9) → (B,64,8,9)
        self.conv1 = nn.Conv2d(in_chan, 64, kernel_size=5, padding=2)
        # conv2: (B,64,8,9) → (B,128,8,9)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        # conv3: (B,128,8,9) → (B,256,8,9)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1)
        # conv4: (B,256,8,9) → (B,64,8,9)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1)
        # 池化： (B,64,8,9) → (B,64,4,4)
        self.pool = nn.MaxPool2d(2,2)
        # 全连接：4×4×64=1024 → 512
        self.fc   = nn.Linear(64*4*4, 512)

    def forward(self, x):
        # x: (B, 4, 8, 9)
        x = F.relu(self.conv1(x))   # → (B,64,8,9)
        x = F.relu(self.conv2(x))   # → (B,128,8,9)
        x = F.relu(self.conv3(x))   # → (B,256,8,9)
        x = F.relu(self.conv4(x))   # → (B,64,8,9)
        x = self.pool(x)            # → (B,64,4,4)
        x = x.view(x.size(0), -1)   # → (B,1024)
        x = F.relu(self.fc(x))      # → (B,512)
        return x                    # 不 reshape，因为 LSTM 接受 (B, T, 512)

# —— 2. 定义“4D-CRNN”主模型 ——
class FourDCRNN(nn.Module):
    def __init__(self, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.base_cnn = BaseCNN()
        # LSTM 接受的输入是每帧 512 维，输出隐藏层 128
        self.lstm = nn.LSTM(input_size=512, hidden_size=128,
                            batch_first=True)
        # 分类器
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        # x: (B, T=6, 4, 8, 9)
        B, T, C, H, W = x.shape

        # ——（1）把时序帧并到 batch 维度上，逐帧过 CNN ——
        x = x.view(B*T, C, H, W)          # → (B*T, 4, 8, 9)
        feat = self.base_cnn(x)          # → (B*T, 512)

        # ——（2）恢复时序维度，送入 LSTM ——
        feat = feat.view(B, T, 512)      # → (B, 6, 512)
        out_seq, _ = self.lstm(feat)     # → (B, 6, 128)

        # ——（3）取最后一帧的输出，做分类 ——
        last = out_seq[:, -1, :]         # → (B, 128)
        out  = self.classifier(last)     # → (B, 2)
        return out

# —— 3. 读数据 & 预处理函数 ——
def load_subject_data(mat_path, flag='a'):
    mat = sio.loadmat(mat_path)
    data = mat['data']               # (4800,4,8,9)
    y_v  = mat['valence_labels'][0]  # (4800,)
    y_a  = mat['arousal_labels'][0]  # (4800,)
    # 独热编码
    y_v = F.one_hot(torch.from_numpy(y_v).long(), NUM_CLASSES)
    y_a = F.one_hot(torch.from_numpy(y_a).long(), NUM_CLASSES)

    # 转成 PyTorch 张量 & 重排通道顺序
    # Keras: data.transpose([0,2,3,1]) → (4800,8,9,4)
    one_falx = torch.from_numpy(data).float().permute(0,2,3,1)
    # 分组 ⇒ (4800//6=800, 6, 8,9,4)
    one_falx = one_falx.reshape(-1, SEQ_LEN, IMG_ROWS, IMG_COLS, NUM_CHAN)

    # 构造每组的标签：每隔 6 帧取一次
    N = one_falx.size(0)  # 800
    if flag == 'v':
        y_full = y_v
    else:
        y_full = y_a
    one_y = y_full.reshape(N, SEQ_LEN, NUM_CLASSES)[:,0,:]  # 第 0 帧标签即可 ⇒ (800,2)

    # (800, 6, 8, 9, 4)
    # (800, NUM_CLASSES)
    return one_falx, one_y

# —— 4. 主循环：遍历每个被试，做 5 折 CV ——
dataset_dir = "…/with_base_0.5/"
acc_list, std_list = [], []

for subj in sorted(os.listdir(dataset_dir)):
    if not subj.endswith('.mat'): continue
    # 每个被试的数据
    # (800, 6, 8, 9, 4)
    # (800, NUM_CLASSES)
    one_falx, one_y = load_subject_data(os.path.join(dataset_dir, subj), flag='a')
    N = one_falx.size(0)  # 800

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_acc = []

    for train_idx, test_idx in kf.split(np.zeros(N), one_y.argmax(dim=1).numpy()):
        # 分割训练/测试集
        x_train = one_falx[train_idx].to(DEVICE)  # (640,6,8,9,4)
        y_train = one_y[train_idx].to(DEVICE)     # (640,2)
        x_test  = one_falx[test_idx].to(DEVICE)
        y_test  = one_y[test_idx].to(DEVICE)

        # DataLoader（可选）
        train_ds = TensorDataset(x_train, y_train)
        test_ds  = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

        # 新模型 & 优化器 & 损失
        model = FourDCRNN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # ——— 训练 ———
        model.train()
        for epoch in range(EPOCHS):
            for xb, yb in train_loader:
                # Forward
                logits = model(xb)           # → (batch,2)
                loss   = criterion(logits, yb.argmax(dim=1))
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ——— 测试 ———
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb)
                preds  = logits.argmax(dim=1)
                correct += (preds == yb.argmax(dim=1)).sum().item()
                total   += xb.size(0)
        acc = correct / total * 100
        all_acc.append(acc)
        print(f"Fold accuracy: {acc:.2f}%")

    mean_acc = np.mean(all_acc)
    std_acc  = np.std(all_acc)
    acc_list.append(mean_acc)
    std_list.append(std_acc)
    print(f"Subject {subj} — mean acc: {mean_acc:.2f}%, std: {std_acc:.2f}%\n")

print("Overall subjects — Acc:", acc_list)
print("Overall subjects — Std:", std_list)
print("Average Acc:", np.mean(acc_list), "Average Std:", np.mean(std_list))

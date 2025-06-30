print("hello world")


# -------- 自定义 Dataset --------
class DEAPWindowDataset(Dataset):
    def __init__(self, gasf_root, rp_root, transform=None):
        self.samples = []  # (g_path, r_path, label, window_idx)
        for subj in sorted(os.listdir(gasf_root)):
            g_subdir = os.path.join(gasf_root, subj)
            r_subdir = os.path.join(rp_root, subj)
            # 读取该被试的标签文件
            labels_txt = os.path.join(g_subdir, f"{subj}_labels.txt")
            label_map = {}
            with open(labels_txt, 'r') as f:
                for line in f:
                    fname, lbl = line.strip().split(',')
                    label_map[fname] = int(lbl)
            # 遍历所有 trial
            for g_fname, lbl in label_map.items():
                g_path = os.path.join(g_subdir, g_fname)
                r_fname = g_fname.replace('_GASF.npy', '_RP.npy')
                r_path = os.path.join(r_subdir, r_fname)
                # 再读一次获取 window 数
                arr = np.load(g_path)  # shape (32,14,224,224)
                n_windows = arr.shape[1]
                for w in range(n_windows):
                    self.samples.append((g_path, r_path, lbl, w))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        g_path, r_path, lbl, w = self.samples[idx]
        g_arr = np.load(g_path)      # (32,14,224,224)
        r_arr = np.load(r_path)
        g = g_arr[:, w, :, :].astype(np.float32) / 255.0  # (32,224,224)
        r = r_arr[:, w, :, :].astype(np.float32) / 255.0
        g = torch.from_numpy(g)
        r = torch.from_numpy(r)
        if self.transform:
            g = self.transform(g)
            r = self.transform(r)
        return g, r, lbl

if __name__ == '__main__':
    # TODO: 测试代码
    # 参数
    in_chans = 32
    depths = [1, 3, 8, 4]
    num_heads = [2, 4, 8, 16]
    window_size = [8, 8, 14, 7]
    dim = 128
    in_dim = 64
    mlp_ratio = 4
    resolution = 224
    drop_path_rate = 0.2
    num_classes = 2

    OUTPUT_PATH = r'D:\dachuang\pic05'
    gasf_root = os.path.join(OUTPUT_PATH, 'GASF')
    rp_root   = os.path.join(OUTPUT_PATH, 'RP')

    # 1. 构建 Dataset & DataLoader
    dataset = DEAPWindowDataset(gasf_root, rp_root)
    total = len(dataset)
    n_train = int(0.9 * total)
    n_test  = total - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test])

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. 模型、损失、优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BiMambaVision(depths=depths,
                          num_heads=num_heads,
                          window_size=window_size,
                          dim=dim,
                          in_dim=in_dim,
                          mlp_ratio=mlp_ratio,
                          resolution=resolution,
                          drop_path_rate=drop_path_rate,
                          num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 3. 准备实时可视化
    plt.ion()
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4))
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        # —— 训练 ——
        model.train()
        running_loss = 0.0
        correct = 0
        total_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False)
        for g, r, lbl in pbar:
            g, r, lbl = g.to(device), r.to(device), lbl.to(device)
            optimizer.zero_grad()
            outputs = model(g, r)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * g.size(0)
            _, preds = outputs.max(1)
            correct += (preds == lbl).sum().item()
            total_samples += g.size(0)
            pbar.set_postfix(loss=running_loss / total_samples, acc=100 * correct / total_samples)
        train_losses.append(running_loss / total_samples)
        train_accs.append(100 * correct / total_samples)

        # —— 测试 ——
        model.eval()
        running_loss = 0.0
        correct = 0
        total_samples = 0
        pbar = tqdm(test_loader, desc=f"Epoch {epoch}/{num_epochs} [Test ]", leave=False)
        with torch.no_grad():
            for g, r, lbl in pbar:
                g, r, lbl = g.to(device), r.to(device), lbl.to(device)
                outputs = model(g, r)
                loss = criterion(outputs, lbl)
                running_loss += loss.item() * g.size(0)
                _, preds = outputs.max(1)
                correct += (preds == lbl).sum().item()
                total_samples += g.size(0)
                pbar.set_postfix(loss=running_loss / total_samples, acc=100 * correct / total_samples)
        test_losses.append(running_loss / total_samples)
        test_accs.append(100 * correct / total_samples)

        # —— 实时绘图 ——
        ax_l.clear()
        ax_a.clear()
        ax_l.plot(train_losses, label='Train Loss')
        ax_l.plot(test_losses, label='Test Loss')
        ax_l.set_title('Loss Curve');
        ax_l.legend()
        ax_a.plot(train_accs, label='Train Acc')
        ax_a.plot(test_accs, label='Test Acc')
        ax_a.set_title('Accuracy Curve');
        ax_a.legend()
        plt.pause(0.01)

    plt.ioff()
    plt.show()
    print("训练完成！")

# 5) VisionLayer3D to replace VisionLayer for early/mid spatio-temporal stages
class VisionLayer3D(nn.Module):
    def __init__(self, dim, depth, ws_t, window_size, conv3d=False, downsample=True, **kwargs):
        super().__init__()
        self.conv3d = conv3d
        self.blocks = nn.ModuleList()
        dp = kwargs.get('drop_path', 0.)
        ls = kwargs.get('layer_scale_conv', None)
        for i in range(depth):
            _dp = dp[i] if isinstance(dp, (list, tuple)) else dp
            if self.conv3d:
                self.blocks.append(ConvBlock3D(dim=dim, drop_path=_dp, layer_scale=ls))
            else:
                # fallback to 2D ConvBlock per frame
                self.blocks.append(ConvBlock(dim=dim, drop_path=_dp, layer_scale=ls))
        # downsample: 3D conv for temporal-aware, else 2D downsample
        if downsample:
            if self.conv3d:
                self.downsample = nn.Conv3d(dim, dim * 2, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False)
            else:
                self.downsample = Downsample(dim)
        else:
            self.downsample = None
        self.ws_t = ws_t
        self.window_size = window_size

    def forward(self, x):  # x: (B, C, T, H, W) if conv3d else treated per-frame
        if self.conv3d:
            for blk in self.blocks:
                x = blk(x)
            if self.downsample is not None:
                x = self.downsample(x)
            return x
        else:
            # reshape to process spatial conv per frame
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            for blk in self.blocks:
                x = blk(x)
            if self.downsample is not None:
                x = self.downsample(x)
            C2, H2, W2 = x.shape[1:]
            x = x.reshape(B, T, C2, H2, W2).permute(0, 2, 1, 3, 4)
            return x

class CoMambaVisionLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 ws_t,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks=[],
                 ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.ws_t = ws_t
        self.ws = window_size
        self.blocks = nn.ModuleList([
            CoBlock(dim=dim,
                    counter=i,
                    transformer_blocks=transformer_blocks,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale)
            for i in range(depth)])

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        # self.window_size = window_size

    def forward(self, g, r):
        B, C, T, H, W = g.shape

        # pad_r = (self.window_size - W % self.window_size) % self.window_size
        # pad_b = (self.window_size - H % self.window_size) % self.window_size

        # --- 1) partition ---
        g_wins, pad_sizes = st_window_partition(g, self.ws_t, self.ws)
        r_wins, _ = st_window_partition(r, self.ws_t, self.ws)

        # if pad_r > 0 or pad_b > 0:
        #     g = torch.nn.functional.pad(g, (0, pad_r, 0, pad_b))
        #     r = torch.nn.functional.pad(r, (0, pad_r, 0, pad_b))
        #     _, _, Hp, Wp = g.shape
        # else:
        #     Hp, Wp = H, W

        # 窗口划分
        # g = window_partition(g, self.window_size)  # 形状变为 (num_windows*B, window_size^2, C)
        # r = window_partition(r, self.window_size)

        # --- 2) CoBlock 序列处理 ---
        for blk in self.blocks:
            g_wins, r_wins = blk(g_wins, r_wins)

        # 窗口还原
        # g = window_reverse(g, self.window_size, Hp, Wp)  # 恢复形状为 (B, C, Hp, Wp)
        # r = window_reverse(r, self.window_size, Hp, Wp)

        # --- 3) reverse ---
        g = st_window_reverse(g_wins, self.ws_t, self.ws, pad_sizes, B)
        r = st_window_reverse(r_wins, self.ws_t, self.ws, pad_sizes, B)

        # if pad_r > 0 or pad_b > 0:  # 裁剪填充部分
        #     g = g[:, :, :H, :W].contiguous()
        #     r = r[:, :, :H, :W].contiguous()

        # --- 4) 裁剪回原始 T,H,W ---
        T_p, H_p, W_p = pad_sizes
        g = g[..., :T, :H, :W]
        r = r[..., :T, :H, :W]

        # --- 5) 可选下采样 ---
        if self.downsample is not None:
            g = self.downsample(g)
            r = self.downsample(r)

        return g, r

class CoMambaVisionLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 ws_t,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks=[],
                 ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.ws_t = ws_t
        self.ws = window_size
        self.blocks = nn.ModuleList([
            CoBlock(dim=dim,
                    counter=i,
                    transformer_blocks=transformer_blocks,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale)
            for i in range(depth)])

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        # self.window_size = window_size

    def forward(self, g, r):
        # input->(B, T=9, D, 4, 4)
        B, T, C, H, W = g.shape

        # pad_r = (self.window_size - W % self.window_size) % self.window_size
        # pad_b = (self.window_size - H % self.window_size) % self.window_size

        # --- 1) partition ---
        g_wins, pad_sizes = st_window_partition(g, self.ws_t, self.ws)
        r_wins, _ = st_window_partition(r, self.ws_t, self.ws)

        # if pad_r > 0 or pad_b > 0:
        #     g = torch.nn.functional.pad(g, (0, pad_r, 0, pad_b))
        #     r = torch.nn.functional.pad(r, (0, pad_r, 0, pad_b))
        #     _, _, Hp, Wp = g.shape
        # else:
        #     Hp, Wp = H, W

        # 窗口划分
        # g = window_partition(g, self.window_size)  # 形状变为 (num_windows*B, window_size^2, C)
        # r = window_partition(r, self.window_size)

        # --- 2) CoBlock 序列处理 ---
        for blk in self.blocks:
            g_wins, r_wins = blk(g_wins, r_wins)

        # 窗口还原
        # g = window_reverse(g, self.window_size, Hp, Wp)  # 恢复形状为 (B, C, Hp, Wp)
        # r = window_reverse(r, self.window_size, Hp, Wp)

        # --- 3) reverse ---
        g = st_window_reverse(g_wins, self.ws_t, self.ws, pad_sizes, B)
        r = st_window_reverse(r_wins, self.ws_t, self.ws, pad_sizes, B)

        # if pad_r > 0 or pad_b > 0:  # 裁剪填充部分
        #     g = g[:, :, :H, :W].contiguous()
        #     r = r[:, :, :H, :W].contiguous()

        # --- 4) 裁剪回原始 T,H,W ---
        T_p, H_p, W_p = pad_sizes
        g = g[:, :T, :, :H, :W].contiguous()
        r = r[:, :T, :, :H, :W].contiguous()

        # --- 5) 可选下采样 ---
        if self.downsample is not None:
            g = self.downsample(g)
            r = self.downsample(r)

        return g, r
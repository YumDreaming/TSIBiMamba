#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import warnings
import torchaudio.models as tm
from shengji07_SEED_VII_7_4mat import BaseCNN
from tqdm import tqdm

warnings.filterwarnings("ignore", "Importing from timm.models.layers is deprecated")
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 极小的序列长度序列
TEST_LENGTHS = [10, 20, 50, 100, 200, 500]

# 复用之前的 ConformerVision 定义
class ConformerVision(torch.nn.Module):
    def __init__(self, mm_output_sizes=[1024], conformer_args=None):
        super().__init__()
        self.base_cnn_g = BaseCNN()
        self.base_cnn_r = BaseCNN()
        d_model = mm_output_sizes[-1] * 2
        args = conformer_args or {}
        self.conformer = tm.Conformer(
            input_dim                  = d_model,
            num_heads                  = args.get("num_heads", 8),
            ffn_dim                    = args.get("ffn_dim", 2048),
            num_layers                 = args.get("num_layers", 4),
            depthwise_conv_kernel_size = args.get("kernel_size", 31),
            dropout                    = args.get("dropout", 0.1),
            use_group_norm             = False,
            convolution_first          = False
        )
        self.classifier = torch.nn.Linear(d_model, 7)

    def forward(self, g, r):
        B, T, C, H, W = g.shape
        g_feat = self.base_cnn_g(g.view(B*T, C, H, W)).view(B, T, -1)
        r_feat = self.base_cnn_r(r.view(B*T, C, H, W)).view(B, T, -1)
        x = torch.cat([g_feat, r_feat], dim=-1)
        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths)
        return self.classifier(x[:, -1, :])

def debug_conformer():
    model = ConformerVision(mm_output_sizes=[1024],
                            conformer_args={'num_heads':8,'ffn_dim':2048,'num_layers':2,'kernel_size':15,'dropout':0.1})
    model = model.to(DEVICE).eval()
    print("=== ConformerSmokeTest ===")
    for L in TEST_LENGTHS:
        torch.cuda.empty_cache()
        g = torch.randn(1, L, 4, 8, 9, device=DEVICE)
        r = torch.randn(1, L, 4, 8, 9, device=DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        try:
            _ = model(g, r)
            peak = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
            print(f"Length={L:4d}  Peak Mem = {peak:.2f} GB")
        except RuntimeError as e:
            print(f"Length={L:4d}  **OOM**")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    debug_conformer()

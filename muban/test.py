import torch
from mamba_ssm import Mamba

print("PyTorch版本:", torch.__version__)  # 应输出 2.1.0
print("CUDA可用:", torch.cuda.is_available())  # True
print("CUDA版本:", torch.version.cuda)  # 11.8
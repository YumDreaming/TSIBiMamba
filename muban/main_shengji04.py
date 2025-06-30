#!/usr/bin/env python3
"""
main_shengji04.py

基于 DepMamba 仓库的 main.py 模板，改写用于训练和测试 DEAP 数据集上的 shengji04.DepMamba 模型。

将该脚本放在 `/root/deap/bimambavision/muban` 目录下，配置文件放在 `/root/deap/bimambavision/config/config.yaml`。
"""
import os
import sys
import argparse
import yaml
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# 确保当前目录在模块搜索路径中（可根据工程结构调整）
sys.path.append(os.path.dirname(__file__))  # 把 muban/ 加入 path

# 导入本地模块
# from deap_signal import preprocess_and_save, get_deap_loader
from shengji04 import DepMamba
from datasets.deap_signal import preprocess_and_save, get_deap_loader


def parse_args():
    """
    解析命令行参数并加载 YAML 配置。
    支持命令行对部分字段的覆盖，并进行必要的类型转换。
    返回一个配置字典 config。
    """
    parser = argparse.ArgumentParser(description="Train and test DepMamba on DEAP dataset")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml'),
                        help="Path to config file (YAML)")
    # 可选重写 config 中的字段
    parser.add_argument("--data_dir", type=str, help="预处理后的数据根目录")
    parser.add_argument("--save_dir", type=str, help="模型与日志保存目录")
    parser.add_argument("--train", type=lambda x: x.lower()=='true', help="是否执行训练阶段 (true/false)")
    parser.add_argument("--if_wandb", type=lambda x: x.lower()=='true', help="是否启用 Weights & Biases 日志 (true/false)")
    parser.add_argument("--device", type=str, nargs='+', help="CUDA 设备列表，例如 --device cuda 0")
    parser.add_argument("--epochs", type=int, help="训练总轮数")
    parser.add_argument("--batch_size", type=int, help="批量大小")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--lr_scheduler", type=str, help="学习率调度策略，例如 'cos'")
    args = parser.parse_args()

    # 加载 YAML 配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 覆盖 YAML
    for key in ['data_dir', 'save_dir', 'train', 'if_wandb', 'device',
                'epochs', 'batch_size', 'learning_rate', 'lr_scheduler']:
        val = getattr(args, key)
        if val is not None:
            config[key] = val

    # 类型转换，确保数值型
    try:
        config['epochs'] = int(config['epochs'])
        config['batch_size'] = int(config['batch_size'])
        config['learning_rate'] = float(config['learning_rate'])
    except Exception:
        raise ValueError(f"Config 中的 epochs/batch_size/learning_rate 必须是数值: {config['epochs']}, {config['batch_size']}, {config['learning_rate']}")
    # train 与 if_wandb 转为 bool
    config['train'] = bool(config.get('train', False))
    config['if_wandb'] = bool(config.get('if_wandb', False))

    return config


def train_epoch(model, loader, loss_fn, optimizer, device, scaler=None):
    """
    执行一个训练 epoch，支持 AMP 半精度训练。
    返回 (平均损失, 准确率)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y, mask in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device).unsqueeze(1).float()
        mask = mask.to(device)

        optimizer.zero_grad()
        with autocast(enabled=(scaler is not None)):
            logits = model(x, mask)
            loss = loss_fn(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 统计
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        preds = (torch.sigmoid(logits) > 0.5).int()
        total_correct += (preds == y.int()).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def evaluate(model, loader, loss_fn, device):
    """
    在验证/测试集上评估模型，计算损失、准确率、精确率、召回率和 F1。
    返回 (loss, accuracy, precision, recall, f1)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    TP = FP = TN = FN = 0

    with torch.no_grad():
        for x, y, mask in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device).unsqueeze(1).float()
            mask = mask.to(device)

            logits = model(x, mask)
            loss = loss_fn(logits, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = (torch.sigmoid(logits) > 0.5).int()
            y_int = y.int()
            TP += ((preds == 1) & (y_int == 1)).sum().item()
            FP += ((preds == 1) & (y_int == 0)).sum().item()
            TN += ((preds == 0) & (y_int == 0)).sum().item()
            FN += ((preds == 0) & (y_int == 1)).sum().item()

    loss_avg = total_loss / total_samples
    accuracy = (TP + TN) / total_samples
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return loss_avg, accuracy, precision, recall, f1


def main():
    # 解析配置
    config = parse_args()

    # 固定随机种子以复现结果
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置设备
    device_list = config['device']
    device = torch.device(device_list[0] if torch.cuda.is_available() else 'cpu')

    # 数据加载
    data_dir = config['data_dir']
    train_loader = get_deap_loader(data_dir, 'train', config['batch_size'], aug=True)
    val_loader   = get_deap_loader(data_dir, 'valid', config['batch_size'], aug=False)
    test_loader  = get_deap_loader(data_dir, 'test',  config['batch_size'], aug=False)

    # 模型实例化
    model = DepMamba(**config['mmmamba']).to(device)
    if len(device_list) > 1:
        model = nn.DataParallel(model, device_ids=list(range(len(device_list))))

    # 损失函数、优化器、AMP
    loss_fn = nn.BCEWithLogitsLoss()
    # 出错修复：确保 lr 是 float 类型
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scaler = GradScaler()

    # 学习率调度
    scheduler = None
    if config['lr_scheduler'].lower() == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=1e-6
        )

    # W&B 日志（可选）
    if config['if_wandb']:
        import wandb
        wandb.init(project="deap_mamba", config=config)

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    best_model_path = os.path.join(config['save_dir'], 'best_model.pt')
    best_f1 = -1.0

    # 训练 & 验证
    if config['train']:
        for epoch in range(1, config['epochs'] + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, loss_fn, optimizer, device, scaler)
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
                model, val_loader, loss_fn, device)

            # 以验证集 F1 选优
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)

            # 日志输出
            print(f"[Epoch {epoch:03d}/{config['epochs']}] "
                  f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
                  f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}, f1: {val_f1:.4f}")
            if config['if_wandb']:
                wandb.log({
                    'train/loss': train_loss, 'train/acc': train_acc,
                    'val/loss': val_loss, 'val/acc': val_acc,
                    'val/precision': val_prec, 'val/recall': val_rec, 'val/f1': val_f1
                })

            if scheduler is not None:
                scheduler.step()

    # 测试最优模型
    print("Loading best model from", best_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, test_loader, loss_fn, device)
    print(f"[Test] loss: {test_loss:.4f}, acc: {test_acc:.4f}, "
          f"precision: {test_prec:.4f}, recall: {test_rec:.4f}, f1: {test_f1:.4f}")

    # 保存结果
    result_file = os.path.join(config['save_dir'], 'test_results.txt')
    with open(result_file, 'w') as f:
        f.write(
            f"loss:{test_loss:.4f}, acc:{test_acc:.4f}, "
            f"precision:{test_prec:.4f}, recall:{test_rec:.4f}, f1:{test_f1:.4f}\n"
        )
    if config['if_wandb']:
        wandb.log({
            'test/loss': test_loss, 'test/acc': test_acc,
            'test/precision': test_prec, 'test/recall': test_rec, 'test/f1': test_f1
        })
        wandb.finish()


if __name__ == '__main__':
    main()

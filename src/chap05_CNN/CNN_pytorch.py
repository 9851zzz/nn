#!/usr/bin/env python
# coding: utf-8
"""
基于 PyTorch 的 CNN 实现 —— MNIST 手写数字识别(精简版)

特性:
  - 自动选择训练设备(CUDA / MPS / CPU)
  - 训练集数据增强 + MNIST 标准归一化
  - Conv-BN-ReLU 三阶段 CNN + Dropout / Dropout2d
  - AdamW + CosineAnnealingLR
  - Label Smoothing + 梯度裁剪
  - 验证集划分 + 早停 + 最佳模型保存
  - CUDA 自动混合精度(AMP,通过 enabled 标志统一代码路径)
  - 单张图片推理接口
"""

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


# =============================================================================
# 配置
# =============================================================================
@dataclass
class Config:
    # 路径
    data_dir: str = "./mnist"
    save_dir: str = "./checkpoints"
    ckpt_name: str = "best_cnn_mnist.pth"

    # 模型 / 训练
    num_classes: int = 10
    epochs: int = 15
    batch_size: int = 128
    test_batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 5e-4
    dropout: float = 0.3
    label_smoothing: float = 0.1
    grad_clip: Optional[float] = 5.0

    # 验证 / 早停
    val_ratio: float = 0.1
    early_stop_patience: Optional[int] = 5

    # DataLoader(Windows 下建议 0;Linux / macOS 可改为 2 或 4)
    num_workers: int = 0

    # 其他
    seed: int = 42
    log_interval: int = 100
    test_best_after_train: bool = True

    # MNIST 官方训练集统计量
    mean: Tuple[float, ...] = (0.1307,)
    std: Tuple[float, ...] = (0.3081,)


cfg = Config()


# =============================================================================
# 工具
# =============================================================================
def get_device() -> torch.device:
    """自动选择训练设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def set_seed(seed: int) -> None:
    """固定随机种子,尽可能提高实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ckpt_path() -> str:
    """获取最佳模型保存路径(顺便确保目录存在)。"""
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(cfg.save_dir, cfg.ckpt_name)


# =============================================================================
# 数据
# =============================================================================
def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """构建 MNIST 训练 / 验证 / 测试 DataLoader。"""
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])

    # 同一份 MNIST,两套 transform 分别用于训练子集和验证子集
    train_full = torchvision.datasets.MNIST(cfg.data_dir, train=True,  download=True, transform=train_tf)
    val_full   = torchvision.datasets.MNIST(cfg.data_dir, train=True,  download=True, transform=eval_tf)
    test_set   = torchvision.datasets.MNIST(cfg.data_dir, train=False, download=True, transform=eval_tf)

    # 按 val_ratio 切分训练 / 验证
    n_total = len(train_full)
    n_val = int(n_total * cfg.val_ratio)
    rng = torch.Generator().manual_seed(cfg.seed)
    idx = torch.randperm(n_total, generator=rng).tolist()
    train_set = Subset(train_full, idx[n_val:])
    val_set   = Subset(val_full,   idx[:n_val])

    common = dict(
        num_workers=cfg.num_workers,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size,      shuffle=True,  **common)
    val_loader   = DataLoader(val_set,   batch_size=cfg.test_batch_size, shuffle=False, **common)
    test_loader  = DataLoader(test_set,  batch_size=cfg.test_batch_size, shuffle=False, **common)

    print(f"训练 / 验证 / 测试 样本数: {len(train_set)} / {len(val_set)} / {len(test_set)}")
    return train_loader, val_loader, test_loader


# =============================================================================
# 模型
# =============================================================================
def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d + BatchNorm2d + ReLU 基础模块。"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CNN(nn.Module):
    """MNIST CNN 分类模型。输入 [B, 1, 28, 28] -> 输出 [B, 10]。"""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        p2d = dropout * 0.5

        self.features = nn.Sequential(
            conv_block(1, 32),    conv_block(32, 32),
            nn.MaxPool2d(2),      nn.Dropout2d(p2d),     #  32 x 14 x 14
            conv_block(32, 64),   conv_block(64, 64),
            nn.MaxPool2d(2),      nn.Dropout2d(p2d),     #  64 x  7 x  7
            conv_block(64, 128),  conv_block(128, 128),
            nn.MaxPool2d(2),      nn.Dropout2d(p2d),     # 128 x  3 x  3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# =============================================================================
# 训练 / 评估
# =============================================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    """在验证或测试集上评估,返回 (avg_loss, accuracy)。"""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        bs = images.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == labels).sum().item()
        total         += bs
    return total_loss / total, total_correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> Tuple[float, float]:
    """训练一个 epoch。AMP 通过 scaler / autocast 的 enabled 参数统一控制。"""
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    use_amp = DEVICE.type == "cuda"

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        # scaler.enabled=False 时,以下调用全部退化为 no-op + 原生 optimizer.step()
        scaler.scale(loss).backward()
        if cfg.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = images.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == labels).sum().item()
        total         += bs

        if step % cfg.log_interval == 0 or step == len(loader):
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:02d} [{step:04d}/{len(loader)}] "
                  f"loss={total_loss / total:.4f} "
                  f"acc={total_correct / total * 100:.2f}% "
                  f"lr={lr:.2e}")

    return total_loss / total, total_correct / total


def save_checkpoint(model: nn.Module, epoch: int, val_acc: float, path: str) -> None:
    """保存最佳模型(精简版:只保留模型权重和必要元信息)。"""
    torch.save({
        "epoch": epoch,
        "val_acc": val_acc,
        "model_state_dict": model.state_dict(),
    }, path)


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> float:
    """完整训练流程。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * len(train_loader)
    )
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    eval_loss_fn  = nn.CrossEntropyLoss()

    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    path = ckpt_path()

    print("=" * 60)
    print(f"设备: {DEVICE}  |  Batch: {cfg.batch_size}  |  Epochs: {cfg.epochs}  |  AMP: {use_amp}")
    print(f"LR: {cfg.lr}  |  WD: {cfg.weight_decay}  |  Dropout: {cfg.dropout}")
    print(f"Checkpoint: {path}")
    print("=" * 60)

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, train_loss_fn, optimizer, scheduler, scaler, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, eval_loss_fn)
        elapsed = time.time() - t0

        msg = (f"[Epoch {epoch:02d}] {elapsed:.1f}s | "
               f"train: loss={train_loss:.4f} acc={train_acc * 100:.2f}% | "
               f"val: loss={val_loss:.4f} acc={val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            save_checkpoint(model, epoch, val_acc, path)
            msg += "  [BEST → saved]"
        else:
            no_improve += 1
            msg += f"  [no-improve: {no_improve}]"
        print(msg)

        if cfg.early_stop_patience and no_improve >= cfg.early_stop_patience:
            print(f"早停: 验证集连续 {cfg.early_stop_patience} 轮未提升。")
            break

    print("=" * 60)
    print(f"训练完成,最佳验证准确率: {best_val_acc * 100:.2f}%")
    print("=" * 60)
    return best_val_acc


# =============================================================================
# 推理
# =============================================================================
@torch.no_grad()
def predict_single_image(model: nn.Module, image: torch.Tensor) -> Tuple[int, float]:
    """
    对单张 MNIST 图片做预测。

    参数:
        model: 训练好的模型
        image: shape 为 [1, 28, 28] 或 [1, 1, 28, 28] 的 Tensor,
               应已完成 ToTensor + Normalize 预处理。

    返回:
        (pred, confidence) —— 类别 (0-9) 与置信度 (0-1)。
    """
    model.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"输入维度应为 [1,28,28] 或 [1,1,28,28], 当前: {tuple(image.shape)}")

    image = image.to(DEVICE, non_blocking=True)
    probs = torch.softmax(model(image), dim=1)
    conf, pred = probs.max(dim=1)
    return pred.item(), conf.item()


# =============================================================================
# 主流程
# =============================================================================
def main() -> None:
    set_seed(cfg.seed)
    train_loader, val_loader, test_loader = build_dataloaders()

    model = CNN(num_classes=cfg.num_classes, dropout=cfg.dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.3f}M")

    train(model, train_loader, val_loader)

    if cfg.test_best_after_train:
        path = ckpt_path()
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss())
            print("=" * 60)
            print(f"测试集 (基于 Epoch {ckpt['epoch']} 的最佳模型):")
            print(f"  loss = {test_loss:.4f}   acc = {test_acc * 100:.2f}%")
            print("=" * 60)
        else:
            print("未找到最佳模型文件,跳过最终测试。")


if __name__ == "__main__":
    # Windows 下使用 DataLoader 多进程时,main 入口必须包在 if __name__ == "__main__" 里
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练已被用户手动中断。")
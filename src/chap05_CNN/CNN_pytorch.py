#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# 基于 PyTorch 的 CNN 实现 —— MNIST 手写数字识别（完整优化版）
#
# 主要改进：
#   1. 自动选择训练设备：CUDA / Apple MPS / CPU
#   2. 使用 MNIST 标准均值和标准差进行归一化
#   3. 训练集加入轻量数据增强
#   4. 使用 Conv-BN-ReLU 三阶段 CNN 结构
#   5. 使用 Dropout / Dropout2d 降低过拟合风险
#   6. 使用 AdamW 优化器和 CosineAnnealingLR 学习率调度
#   7. 训练损失支持 Label Smoothing
#   8. 支持验证集划分、早停和最佳模型保存
#   9. 支持从 checkpoint 恢复训练
#  10. 支持单张 MNIST 图片推理并返回置信度
# =============================================================================

import os
import time
import random
from pathlib import Path
from contextlib import nullcontext
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


# =============================================================================
# 配置区
# =============================================================================
class Config:
    # 路径配置
    DATA_DIR = "./mnist"
    SAVE_DIR = "./checkpoints"
    CKPT_NAME = "best_cnn_mnist.pth"

    # 数据与模型配置
    NUM_CLASSES = 10

    # 训练超参数
    EPOCHS = 15
    BATCH_SIZE = 128
    TEST_BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4
    DROPOUT = 0.3
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP_NORM = 5.0

    # 验证集比例
    VAL_RATIO = 0.1

    # 早停配置：None 表示关闭早停
    EARLY_STOPPING_PATIENCE = 5

    # 是否从已有 checkpoint 继续训练
    RESUME_FROM_CKPT = False

    # DataLoader 配置
    # Windows 下 NUM_WORKERS=0 最稳定；Linux/macOS 可改为 2 或 4
    NUM_WORKERS = 0

    # 随机种子
    SEED = 42

    # 日志打印间隔
    LOG_INTERVAL = 100

    # 是否在训练结束后加载最佳模型并重新测试
    TEST_BEST_AFTER_TRAIN = True

    # MNIST 官方训练集统计均值和标准差
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)


cfg = Config()


# =============================================================================
# 工具函数
# =============================================================================
def get_device() -> torch.device:
    """自动选择训练设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


def set_seed(seed: int = 42) -> None:
    """固定随机种子，尽可能提高实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 为了更稳定的复现，关闭 TF32。
        # 如果只追求速度，可以改为 True。
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id: int) -> None:
    """为 DataLoader worker 固定随机种子。"""
    worker_seed = cfg.SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_dir(path: str) -> None:
    """确保目录存在。"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_checkpoint_path() -> str:
    """获取最佳模型保存路径。"""
    ensure_dir(cfg.SAVE_DIR)
    return os.path.join(cfg.SAVE_DIR, cfg.CKPT_NAME)


def create_grad_scaler(device: torch.device) -> Optional[Any]:
    """创建 AMP GradScaler，仅在 CUDA 上启用。"""
    if device.type != "cuda":
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)

    return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device: torch.device):
    """创建 AMP autocast 上下文，仅在 CUDA 上启用。"""
    if device.type != "cuda":
        return nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)

    return torch.cuda.amp.autocast(enabled=True)


# =============================================================================
# 数据加载
# =============================================================================
def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """构建训练集和验证/测试集的数据预处理流程。"""

    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.95, 1.05),
        ),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MNIST_MEAN, cfg.MNIST_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.MNIST_MEAN, cfg.MNIST_STD),
    ])

    return train_transform, eval_transform


def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建 MNIST 训练集、验证集和测试集 DataLoader。

    说明：
        train_loader 使用数据增强；
        val_loader 不使用数据增强；
        test_loader 不使用数据增强。
    """

    train_transform, eval_transform = build_transforms()

    full_train_aug = torchvision.datasets.MNIST(
        root=cfg.DATA_DIR,
        train=True,
        transform=train_transform,
        download=True,
    )

    full_train_eval = torchvision.datasets.MNIST(
        root=cfg.DATA_DIR,
        train=True,
        transform=eval_transform,
        download=True,
    )

    test_set = torchvision.datasets.MNIST(
        root=cfg.DATA_DIR,
        train=False,
        transform=eval_transform,
        download=True,
    )

    total_size = len(full_train_aug)
    val_size = int(total_size * cfg.VAL_RATIO)
    train_size = total_size - val_size

    split_generator = torch.Generator().manual_seed(cfg.SEED)
    indices = torch.randperm(total_size, generator=split_generator).tolist()

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_set = Subset(full_train_aug, train_indices)
    val_set = Subset(full_train_eval, val_indices)

    loader_generator = torch.Generator().manual_seed(cfg.SEED)

    pin_memory = DEVICE.type == "cuda"
    persistent_workers = cfg.NUM_WORKERS > 0

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if cfg.NUM_WORKERS > 0 else None,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if cfg.NUM_WORKERS > 0 else None,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if cfg.NUM_WORKERS > 0 else None,
    )

    print(f"训练集样本数: {train_size}")
    print(f"验证集样本数: {val_size}")
    print(f"测试集样本数: {len(test_set)}")

    return train_loader, val_loader, test_loader


# =============================================================================
# 模型定义
# =============================================================================
class ConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + ReLU 基础模块。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN(nn.Module):
    """
    MNIST CNN 分类模型。

    输入:
        [batch_size, 1, 28, 28]

    输出:
        [batch_size, 10]
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # 1 x 28 x 28 -> 32 x 14 x 14
            ConvBlock(1, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout * 0.5),

            # 32 x 14 x 14 -> 64 x 7 x 7
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout * 0.5),

            # 64 x 7 x 7 -> 128 x 3 x 3
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout * 0.5),
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
        """初始化模型参数。"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# 训练与评估
# =============================================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    """在验证集或测试集上评估模型。"""

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any],
    epoch: int,
) -> Tuple[float, float]:
    """训练一个 epoch。"""

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    use_amp = DEVICE.type == "cuda"

    for step, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast_context(DEVICE):
                logits = model(images)
                loss = loss_fn(logits, labels)

            assert scaler is not None
            scaler.scale(loss).backward()

            if cfg.GRAD_CLIP_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=cfg.GRAD_CLIP_NORM,
                )

            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(images)
            loss = loss_fn(logits, labels)

            loss.backward()

            if cfg.GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=cfg.GRAD_CLIP_NORM,
                )

            optimizer.step()

        scheduler.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        if step % cfg.LOG_INTERVAL == 0 or step == len(train_loader):
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples

            print(
                f"Epoch [{epoch:02d}/{cfg.EPOCHS}] "
                f"Step [{step:04d}/{len(train_loader)}] "
                f"Loss: {train_loss:.4f} "
                f"Acc: {train_acc * 100:.2f}% "
                f"LR: {current_lr:.2e}"
            )

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    val_acc: float,
    path: str,
) -> None:
    """保存模型检查点。"""

    checkpoint = {
        "epoch": epoch,
        "val_acc": val_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": {
            "num_classes": cfg.NUM_CLASSES,
            "epochs": cfg.EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "test_batch_size": cfg.TEST_BATCH_SIZE,
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay": cfg.WEIGHT_DECAY,
            "dropout": cfg.DROPOUT,
            "label_smoothing": cfg.LABEL_SMOOTHING,
            "grad_clip_norm": cfg.GRAD_CLIP_NORM,
            "val_ratio": cfg.VAL_RATIO,
            "seed": cfg.SEED,
        },
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    map_location: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """加载模型检查点，可选恢复 optimizer 和 scheduler。"""

    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=map_location,
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> float:
    """完整训练流程。"""

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.EPOCHS * len(train_loader),
    )

    train_loss_fn = nn.CrossEntropyLoss(
        label_smoothing=cfg.LABEL_SMOOTHING,
    )

    eval_loss_fn = nn.CrossEntropyLoss()

    scaler = create_grad_scaler(DEVICE)
    ckpt_path = get_checkpoint_path()

    start_epoch = 1
    best_val_acc = 0.0

    if cfg.RESUME_FROM_CKPT and os.path.exists(ckpt_path):
        checkpoint = load_checkpoint(
            model=model,
            path=ckpt_path,
            map_location=DEVICE,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("val_acc", 0.0)

        print(f"已从 checkpoint 继续训练: {ckpt_path}")
        print(f"起始 Epoch: {start_epoch}")
        print(f"当前最佳验证准确率: {best_val_acc * 100:.2f}%")

    print("=" * 80)
    print("训练配置")
    print("=" * 80)
    print(f"设备              : {DEVICE}")
    print(f"训练样本数        : {len(train_loader.dataset)}")
    print(f"验证样本数        : {len(val_loader.dataset)}")
    print(f"Epochs            : {cfg.EPOCHS}")
    print(f"Batch Size        : {cfg.BATCH_SIZE}")
    print(f"Learning Rate     : {cfg.LEARNING_RATE}")
    print(f"Weight Decay      : {cfg.WEIGHT_DECAY}")
    print(f"Dropout           : {cfg.DROPOUT}")
    print(f"Label Smoothing   : {cfg.LABEL_SMOOTHING}")
    print(f"Grad Clip Norm    : {cfg.GRAD_CLIP_NORM}")
    print(f"Early Stopping    : {cfg.EARLY_STOPPING_PATIENCE}")
    print(f"AMP               : {DEVICE.type == 'cuda'}")
    print(f"Checkpoint Path   : {ckpt_path}")
    print("=" * 80)

    no_improve_epochs = 0

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=train_loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
        )

        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            loss_fn=eval_loss_fn,
        )

        elapsed = time.time() - start_time

        print("-" * 80)
        print(
            f"Epoch [{epoch:02d}/{cfg.EPOCHS}] 完成 "
            f"| 用时: {elapsed:.1f}s "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc * 100:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_acc=val_acc,
                path=ckpt_path,
            )

            print(f"发现新的最佳模型，验证集准确率: {best_val_acc * 100:.2f}%")
            print(f"模型已保存到: {ckpt_path}")
        else:
            no_improve_epochs += 1
            print(f"验证集准确率未提升，连续未提升轮数: {no_improve_epochs}")

        print("-" * 80)

        if (
            cfg.EARLY_STOPPING_PATIENCE is not None
            and no_improve_epochs >= cfg.EARLY_STOPPING_PATIENCE
        ):
            print(f"触发早停：验证集连续 {cfg.EARLY_STOPPING_PATIENCE} 轮未提升。")
            break

    print("=" * 80)
    print(f"训练完成，最佳验证集准确率: {best_val_acc * 100:.2f}%")
    print("=" * 80)

    return best_val_acc


# =============================================================================
# 单张图片推理函数
# =============================================================================
@torch.no_grad()
def predict_single_image(
    model: nn.Module,
    image: torch.Tensor,
) -> Tuple[int, float]:
    """
    对单张 MNIST 图片进行预测。

    参数:
        model: 训练好的模型
        image: shape 为 [1, 28, 28] 或 [1, 1, 28, 28] 的 Tensor。
               注意：输入应已完成 ToTensor 和 Normalize 预处理。

    返回:
        pred: 预测类别，范围 0-9
        confidence: 预测置信度，范围 0-1
    """

    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    if image.dim() != 4:
        raise ValueError(
            f"输入图片维度应为 [1, 28, 28] 或 [1, 1, 28, 28]，当前为 {tuple(image.shape)}"
        )

    image = image.to(DEVICE, non_blocking=True)

    logits = model(image)
    probs = torch.softmax(logits, dim=1)
    confidence, pred = probs.max(dim=1)

    return pred.item(), confidence.item()


# =============================================================================
# 主函数
# =============================================================================
def main() -> None:
    set_seed(cfg.SEED)

    train_loader, val_loader, test_loader = build_dataloaders()

    model = CNN(
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.DROPOUT,
    ).to(DEVICE)

    print("模型结构：")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量      : {total_params / 1e6:.3f} M")
    print(f"可训练参数量  : {trainable_params / 1e6:.3f} M")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    if cfg.TEST_BEST_AFTER_TRAIN:
        ckpt_path = get_checkpoint_path()

        if os.path.exists(ckpt_path):
            print("正在加载最佳模型进行最终测试...")

            checkpoint = load_checkpoint(
                model=model,
                path=ckpt_path,
                map_location=DEVICE,
            )

            test_loss_fn = nn.CrossEntropyLoss()

            final_loss, final_acc = evaluate(
                model=model,
                data_loader=test_loader,
                loss_fn=test_loss_fn,
            )

            print("=" * 80)
            print(f"最佳模型来自 Epoch: {checkpoint['epoch']}")
            print(f"保存时验证集准确率: {checkpoint['val_acc'] * 100:.2f}%")
            print(f"最终测试 Loss     : {final_loss:.4f}")
            print(f"最终测试 Acc      : {final_acc * 100:.2f}%")
            print("=" * 80)
        else:
            print("未找到最佳模型文件，跳过最终测试。")


if __name__ == "__main__":
    # Windows 下使用 DataLoader 多进程时，必须放在 main 入口中执行
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练已被用户手动中断。")
    except Exception as exc:
        print(f"\n程序运行出错: {exc}")
        raise
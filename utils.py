"""
utils.py — Metrics, logging, seeding, and checkpoint helpers.
"""

import logging
import random
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MetricTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.metrics: Dict[str, Dict[str, float]] = {}

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.metrics:
            self.metrics[name] = {"val": 0.0, "sum": 0.0, "count": 0, "avg": 0.0}
        m = self.metrics[name]
        m["val"] = value
        m["sum"] += value * n
        m["count"] += n
        m["avg"] = m["sum"] / m["count"]

    def avg(self, name: str) -> float:
        return self.metrics[name]["avg"] if name in self.metrics else 0.0

    def result(self) -> Dict[str, float]:
        return {k: v["avg"] for k, v in self.metrics.items()}


def _prepare_targets(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Convert single-channel integer masks to one-hot when needed."""
    if preds.shape[1] > 1 and target.shape[1] == 1:
        target_indices = target.squeeze(1).long().clamp(min=0)
        target = F.one_hot(target_indices, num_classes=preds.shape[1])
        target = target.permute(0, 4, 1, 2, 3).float()
    return target


def _predictions(preds: torch.Tensor) -> torch.Tensor:
    if preds.shape[1] > 1:
        return F.softmax(preds, dim=1)
    return torch.sigmoid(preds)


def dice_score(preds: torch.Tensor, target: torch.Tensor) -> float:
    target = _prepare_targets(preds, target)
    p = _predictions(preds)
    b, c = p.shape[:2]
    p_flat = p.reshape(b, c, -1)
    t_flat = target.reshape(b, c, -1)
    inter = (p_flat * t_flat).sum(dim=2)
    union = p_flat.sum(dim=2) + t_flat.sum(dim=2)
    return ((2.0 * inter + 1e-6) / (union + 1e-6)).mean().item()


def iou_score(preds: torch.Tensor, target: torch.Tensor) -> float:
    target = _prepare_targets(preds, target)
    if preds.shape[1] > 1:
        p = (F.softmax(preds, dim=1) > 0.5).float()
    else:
        p = (torch.sigmoid(preds) > 0.5).float()
    b, c = p.shape[:2]
    p_f = p.reshape(b, c, -1)
    t_f = target.reshape(b, c, -1)
    inter = (p_f * t_f).sum(dim=2)
    union = p_f.sum(dim=2) + t_f.sum(dim=2) - inter
    return (inter / (union + 1e-6)).mean().item()


def precision_score(preds: torch.Tensor, target: torch.Tensor) -> float:
    target = _prepare_targets(preds, target)
    if preds.shape[1] > 1:
        p = (F.softmax(preds, dim=1) > 0.5).float()
    else:
        p = (torch.sigmoid(preds) > 0.5).float()
    tp = (p * target).sum()
    fp = (p * (1 - target)).sum()
    return (tp / (tp + fp + 1e-6)).item()


def recall_score(preds: torch.Tensor, target: torch.Tensor) -> float:
    target = _prepare_targets(preds, target)
    if preds.shape[1] > 1:
        p = (F.softmax(preds, dim=1) > 0.5).float()
    else:
        p = (torch.sigmoid(preds) > 0.5).float()
    tp = (p * target).sum()
    fn = ((1 - p) * target).sum()
    return (tp / (tp + fn + 1e-6)).item()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: tuple = (1, 1, 128, 128, 128)) -> None:
    print(f"Model Parameters: {count_parameters(model):,}")


def setup_logger(log_path: str, level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger so all modules share file + console output."""
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    return root


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Global seed set to %d.", seed)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)
    logger.info("Checkpoint saved -> %s", path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    map_location = device if device is not None else "cpu"
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info("Checkpoint loaded <- %s (epoch %d)", path, checkpoint.get("epoch", -1))
    return checkpoint

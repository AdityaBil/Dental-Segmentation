"""
train.py — Full training pipeline for 3D Dental CBCT Segmentation.

Pipeline overview
─────────────────
  1. Logging & seed initialisation
  2. DataLoader construction (train / val / test)
  3. Model, loss, optimizer, and LR scheduler creation
  4. Training loop (100 epochs):
       a. Forward pass
       b. Loss backward + gradient clipping
       c. Optimizer step
       d. Epoch validation: Dice + IoU
       e. Best-model checkpoint saving
  5. Final test evaluation

Run
───
  python train.py

  To resume from a checkpoint:
  python train.py --resume checkpoints/best_model.pth
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

import torch
import torch.nn as nn
from torch.cuda import amp

# ── Add project root to path so all local modules resolve ───────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from augmentations import build_train_transform
from dataset import _local_dataset_ready
from drive_client import find_drive_mount
from losses import build_loss_fn
from models.unet3d import UNet3D
from preprocessing import build_dataloaders, build_preprocessing_pipeline
from utils import (
    MetricTracker,
    count_parameters,
    dice_score,
    iou_score,
    load_checkpoint,
    model_summary,
    precision_score,
    recall_score,
    save_checkpoint,
    set_seed,
    setup_logger,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3D U-Net for dental CBCT segmentation."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS,
        help=f"Number of training epochs (default: {config.EPOCHS}).",
    )
    parser.add_argument(
        "--lr", type=float, default=config.LEARNING_RATE,
        help=f"Learning rate (default: {config.LEARNING_RATE}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.BATCH_SIZE,
        help=f"Batch size (default: {config.BATCH_SIZE}).",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable in-memory caching of dataset volumes.",
    )
    parser.add_argument(
        "--dataset-root", type=str, default=None,
        help="Override dataset root directory (imagesTr/labelsTr parent).",
    )
    parser.add_argument(
        "--patience", type=int, default=config.EARLY_STOPPING_PATIENCE,
        help=f"Early stopping patience in epochs (default: {config.EARLY_STOPPING_PATIENCE}).",
    )
    parser.add_argument(
        "--drive", action="store_true",
        help="Load dataset lazily from Google Drive (one file at a time, cached locally).",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Force local dataset even if incomplete (disables auto Drive fallback).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# One training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
    grad_clip:  float,
    tracker:    MetricTracker,
    scaler:     amp.GradScaler,
) -> dict:
    """
    Run one full pass over the training set.

    Parameters
    ----------
    model     : nn.Module         U-Net model in training mode.
    loader    : DataLoader        Training DataLoader.
    optimizer : Optimizer         AdamW.
    criterion : nn.Module         Loss function (Dice, BCE+Dice, …).
    device    : torch.device      CPU or CUDA.
    grad_clip : float             Max gradient norm (0.0 = disabled).
    tracker   : MetricTracker     Accumulates batch metrics.

    Returns
    -------
    dict
        Epoch-level averages: {"loss", "dice", "iou"}.
    """
    model.train()
    tracker.reset()

    for batch_idx, (volumes, masks) in enumerate(loader):
        volumes = volumes.to(device, non_blocking=True)   # (B, 1, D, H, W)
        masks   = masks.to(device,   non_blocking=True)   # (B, 1, D, H, W)

        optimizer.zero_grad()
        # Use mixed precision when running on CUDA
        autocast_enabled = device.type == "cuda"
        with amp.autocast(enabled=autocast_enabled):
            preds = model(volumes)              # (B, C, D, H, W)
            loss = criterion(preds, masks)

        # Backward with GradScaler for mixed precision safety
        scaler.scale(loss).backward()

        # Gradient clipping (prevents exploding gradients in deep 3D networks)
        if grad_clip > 0.0:
            # Unscale before clipping
            if autocast_enabled:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # ── Metrics (detached, no grad) ───────────────────────────────────────
        with torch.no_grad():
            batch_n   = volumes.shape[0]
            batch_dice = dice_score(preds, masks)
            batch_iou  = iou_score(preds,  masks)

        tracker.update("loss", loss.item(), n=batch_n)
        tracker.update("dice", batch_dice,  n=batch_n)
        tracker.update("iou",  batch_iou,   n=batch_n)

        if (batch_idx + 1) % 10 == 0:
            logger.debug(
                "  [Batch %03d/%03d]  loss=%.4f  dice=%.4f  iou=%.4f",
                batch_idx + 1, len(loader),
                loss.item(), batch_dice, batch_iou,
            )

    return tracker.result()


# ─────────────────────────────────────────────────────────────────────────────
# Validation / evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    tracker:   MetricTracker,
    split:     str = "val",
) -> dict:
    """
    Evaluate the model on a dataloader (validation or test).

    Parameters
    ----------
    split : str   Label for log messages ("val" or "test").

    Returns
    -------
    dict
        Epoch-level averages: {"loss", "dice", "iou", "precision", "recall"}.
    """
    model.eval()
    tracker.reset()

    for volumes, masks in loader:
        volumes = volumes.to(device, non_blocking=True)
        masks   = masks.to(device,   non_blocking=True)

        preds = model(volumes)
        loss  = criterion(preds, masks)

        batch_n = volumes.shape[0]
        tracker.update("loss",      loss.item(),               n=batch_n)
        tracker.update("dice",      dice_score(preds, masks),  n=batch_n)
        tracker.update("iou",       iou_score(preds,  masks),  n=batch_n)
        tracker.update("precision", precision_score(preds, masks), n=batch_n)
        tracker.update("recall",    recall_score(preds, masks),    n=batch_n)

    avgs = tracker.result()
    logger.debug(
        "[%s] loss=%.4f  dice=%.4f  iou=%.4f  prec=%.4f  rec=%.4f",
        split.upper(),
        avgs["loss"], avgs["dice"], avgs["iou"],
        avgs.get("precision", 0), avgs.get("recall", 0),
    )
    return avgs


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.dataset_root:
        config.DATASET_ROOT = os.path.abspath(args.dataset_root)
        config.IMAGE_DIR = os.path.join(config.DATASET_ROOT, "imagesTr")
        config.MASK_DIR = os.path.join(config.DATASET_ROOT, "labelsTr")

    use_drive = args.drive or config.USE_DRIVE_DATASET
    if not use_drive and not args.local:
        mount = find_drive_mount()
        if mount:
            config.DATASET_ROOT = mount
            config.IMAGE_DIR = os.path.join(mount, "imagesTr")
            config.MASK_DIR = os.path.join(mount, "labelsTr")
            logger.info("Using Google Drive Desktop sync at %s", mount)
        elif not _local_dataset_ready():
            use_drive = True
            logger.info("Local dataset incomplete — switching to Google Drive lazy-load mode.")

    # ── Logging + seed ───────────────────────────────────────────────────────
    setup_logger(log_path=config.LOG_PATH)
    set_seed(config.RANDOM_SEED)

    logger.info("=" * 60)
    logger.info("3D Dental CBCT Segmentation - Training")
    logger.info("  Device      : %s", config.DEVICE)
    logger.info("  Data source : %s", "Google Drive (lazy cache)" if use_drive else "local")
    logger.info("  Epochs      : %d", args.epochs)
    logger.info("  Batch size  : %d", args.batch_size)
    logger.info("  LR          : %g", args.lr)
    logger.info("  Loss        : %s", config.LOSS_FN)
    logger.info("=" * 60)

    # ── Create checkpoint directory ───────────────────────────────────────────
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    logger.info("Building DataLoaders…")
    preprocess     = build_preprocessing_pipeline()
    train_transform = build_train_transform(preprocessing_pipeline=preprocess)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_transform=train_transform,
        val_transform=preprocess,
        test_transform=preprocess,
        batch_size=args.batch_size,
        cache=not args.no_cache,
        use_drive=use_drive,
    )
    logger.info(
        "DataLoaders ready - train: %d batches | val: %d | test: %d",
        len(train_loader), len(val_loader), len(test_loader),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Instantiating UNet3D…")
    model = UNet3D(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        base_filters=config.BASE_FILTERS,
        dropout_p=config.DROPOUT_P,
    ).to(config.DEVICE)

    model_summary(model, input_size=(1, 1, *config.IMAGE_SIZE))
    logger.info("Trainable parameters: %s", f"{count_parameters(model):,}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = build_loss_fn(name=config.LOSS_FN)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Cosine Annealing: smoothly decays LR from max to near-zero over T_max epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=getattr(config, "SCHEDULER_T0", 10),
        T_mult=2,
        eta_min=args.lr * 0.01,
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=args.lr * 0.001,
    )

    # ── Optional resume from checkpoint ──────────────────────────────────────
    start_epoch   = 0
    best_val_dice = 0.0

    if args.resume is not None:
        ckpt = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.DEVICE,
        )
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_dice = ckpt.get("metrics", {}).get("val_dice", 0.0)
        logger.info("Resuming from epoch %d (best val Dice: %.4f).", start_epoch, best_val_dice)

    # ── Metric trackers ───────────────────────────────────────────────────────
    train_tracker = MetricTracker()
    val_tracker   = MetricTracker()
    history = []
    epochs_without_improvement = 0

    # Mixed precision scaler (enabled only when using CUDA)
    scaler = amp.GradScaler(enabled=(config.DEVICE.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("Starting training…\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()

        # ── Train ────────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=config.DEVICE,
            grad_clip=config.GRAD_CLIP_NORM,
            tracker=train_tracker,
            scaler=scaler,
        )

        # ── Validate ─────────────────────────────────────────────────────────
        val_metrics = {}
        if (epoch + 1) % config.VALIDATE_EVERY == 0:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=config.DEVICE,
                tracker=val_tracker,
                split="val",
            )

        # ── LR step ───────────────────────────────────────────────────────────
        scheduler.step(epoch + 1)
        if val_metrics:
            plateau_scheduler.step(val_metrics.get("dice", 0.0))
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.perf_counter() - epoch_start

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics.get("loss", 0.0),
            "val_dice": val_metrics.get("dice", 0.0),
            "val_iou": val_metrics.get("iou", 0.0),
            "lr": current_lr,
            "time_s": epoch_time,
        })

        # ── Console / log output ──────────────────────────────────────────────
        print(
            f"\nEpoch [{epoch + 1:03d}/{args.epochs}]  "
            f"({epoch_time:.1f}s)  LR: {current_lr:.2e}"
        )
        print(
            f"  Train  - Loss: {train_metrics['loss']:.4f}  "
            f"Dice: {train_metrics['dice']:.4f}  "
            f"IoU:  {train_metrics['iou']:.4f}"
        )
        if val_metrics:
            print(
                f"  Val    - Loss: {val_metrics['loss']:.4f}  "
                f"Dice: {val_metrics['dice']:.4f}  "
                f"IoU:  {val_metrics['iou']:.4f}  "
                f"Prec: {val_metrics.get('precision', 0):.4f}  "
                f"Rec:  {val_metrics.get('recall', 0):.4f}"
            )

        logger.info(
            "Epoch %03d | train_loss=%.4f train_dice=%.4f | "
            "val_loss=%.4f val_dice=%.4f val_iou=%.4f | lr=%.2e",
            epoch + 1,
            train_metrics["loss"], train_metrics["dice"],
            val_metrics.get("loss", 0), val_metrics.get("dice", 0),
            val_metrics.get("iou", 0), current_lr,
        )

        # ── Save best model (highest validation Dice) ─────────────────────────
        val_dice = val_metrics.get("dice", 0.0)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "val_dice": val_dice,
                    "val_iou":  val_metrics.get("iou", 0.0),
                    "val_loss": val_metrics.get("loss", 0.0),
                },
                path=config.BEST_MODEL_PATH,
                scheduler=scheduler,
            )
            print(f"  * New best model saved (val_dice={best_val_dice:.4f})")
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch + 1} "
                    f"(no val Dice improvement for {args.patience} epochs)."
                )
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    args.patience,
                )
                break

        # ── Periodic checkpoint saves ─────────────────────────────────────────
        if (epoch + 1) % config.SAVE_EVERY == 0:
            periodic_path = os.path.join(
                config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1:04d}.pth"
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train_dice": train_metrics["dice"], **val_metrics},
                path=periodic_path,
                scheduler=scheduler,
            )

    # ── Save training history ───────────────────────────────────────────────
    history_path = os.path.join(config.CHECKPOINT_DIR, "training_history.json")
    csv_path = os.path.join(config.CHECKPOINT_DIR, "training_history.csv")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if history:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    _save_training_plot(history, os.path.join(config.CHECKPOINT_DIR, "training_curves.png"))

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("Training complete.  Evaluating on test set…")
    print("\n" + "=" * 60)

    if os.path.isfile(config.BEST_MODEL_PATH):
        print("Loading best model for final test evaluation…")
        load_checkpoint(
            path=config.BEST_MODEL_PATH,
            model=model,
            device=config.DEVICE,
        )
    else:
        print("No best-model checkpoint found; evaluating current weights.")

    test_tracker  = MetricTracker()
    test_metrics  = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=config.DEVICE,
        tracker=test_tracker,
        split="test",
    )

    print("\n-- Test Set Results ---------------------------------------")
    for metric, value in test_metrics.items():
        print(f"  {metric:<12s}: {value:.4f}")
    print("=" * 60)

    logger.info(
        "Test results: %s",
        {k: f"{v:.4f}" for k, v in test_metrics.items()},
    )

    results_path = os.path.join(config.CHECKPOINT_DIR, "test_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")


def _save_training_plot(history: list, path: str) -> None:
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping training plot.")
        return

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [row["train_dice"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_dice"] for row in history], label="val")
    axes[1].set_title("Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Training curves saved -> %s", path)


if __name__ == "__main__":
    main()

"""
verify_setup.py — Quick sanity check before training.

Usage:
    python verify_setup.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import config
from dataset import _discover_pairs, _local_dataset_ready
from models.unet3d import UNet3D
from utils import count_parameters


def main() -> None:
    print("=" * 60)
    print("ToothFairy project verification")
    print("=" * 60)

    print(f"\nProject root : {config.ROOT_DIR}")
    print(f"Dataset root : {config.DATASET_ROOT}")
    print(f"Images dir   : {config.IMAGE_DIR}")
    print(f"Masks dir    : {config.MASK_DIR}")
    print(f"Device       : {config.DEVICE}")

    images_ok = os.path.isdir(config.IMAGE_DIR)
    masks_ok = os.path.isdir(config.MASK_DIR)
    n_images = n_masks = n_pairs = 0

    if images_ok:
        n_images = sum(
            1 for f in os.scandir(config.IMAGE_DIR)
            if f.name.endswith((".nii", ".nii.gz"))
        )
    if masks_ok:
        n_masks = sum(
            1 for f in os.scandir(config.MASK_DIR)
            if f.name.endswith((".nii", ".nii.gz"))
        )

    if images_ok and masks_ok:
        try:
            n_pairs = len(_discover_pairs(config.IMAGE_DIR, config.MASK_DIR))
        except RuntimeError:
            n_pairs = 0
        print(f"\nDataset      : {n_pairs} matched pairs ({n_images} images, {n_masks} masks)")
    elif images_ok:
        print(f"\nDataset      : PARTIAL ({n_images} images, labelsTr/ missing)")
        print("  -> Use Google Drive mode: python train.py --drive")
    else:
        print("\nDataset      : MISSING locally")
        print("  -> Use Google Drive mode: python train.py --drive")

    drive_pairs = 0
    if not _local_dataset_ready():
        print("\nGoogle Drive check…")
        try:
            from drive_client import build_drive_manifest, discover_drive_pairs
            manifest = build_drive_manifest()
            drive_pairs = len(discover_drive_pairs(manifest))
            print(f"  Drive pairs: {drive_pairs} (lazy download on first use)")
            print(f"  Cache dir  : {config.DRIVE_CACHE_DIR}")
        except Exception as exc:
            print(f"  Drive      : FAILED ({exc})")

    print("\nModel check…")
    model = UNet3D(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        base_filters=config.BASE_FILTERS,
        dropout_p=config.DROPOUT_P,
    )
    print(f"  Parameters : {count_parameters(model):,}")

    dummy = torch.randn(1, config.IN_CHANNELS, *config.IMAGE_SIZE)
    out = model(dummy)
    assert out.shape == (1, config.NUM_CLASSES, *config.IMAGE_SIZE), out.shape
    print(f"  Forward    : OK -> output shape {tuple(out.shape)}")

    from preprocessing import build_preprocessing_pipeline, build_dataloaders
    from augmentations import build_train_transform
    from losses import build_loss_fn

    build_preprocessing_pipeline()
    build_train_transform()
    build_loss_fn(config.LOSS_FN)
    print("  Pipeline   : OK")

    if n_pairs > 0:
        print("\nDataLoader smoke test (local)…")
        preprocess = build_preprocessing_pipeline()
        train_transform = build_train_transform(preprocessing_pipeline=preprocess)
        train_loader, _, _ = build_dataloaders(
            train_transform=train_transform,
            val_transform=preprocess,
            test_transform=preprocess,
            batch_size=1,
            cache=False,
            use_drive=False,
        )
        volumes, masks = next(iter(train_loader))
        print(f"  Batch shape: image={tuple(volumes.shape)} mask={tuple(masks.shape)}")
        print("  DataLoader : OK")
    elif drive_pairs > 0:
        print("\nDrive DataLoader smoke test (downloads 1 sample)…")
        preprocess = build_preprocessing_pipeline()
        train_loader, _, _ = build_dataloaders(
            train_transform=preprocess,
            val_transform=preprocess,
            test_transform=preprocess,
            batch_size=1,
            cache=True,
            use_drive=True,
        )
        print(f"  Drive dataset size: {len(train_loader.dataset)} pairs")
        print("  Downloading first sample from Drive (may take a minute)…")
        volumes, masks = next(iter(train_loader))
        print(f"  Batch shape: image={tuple(volumes.shape)} mask={tuple(masks.shape)}")
        print("  Drive mode : OK")

    print("\n" + "=" * 60)
    if n_pairs > 0:
        print("Ready to train:  python train.py")
    elif drive_pairs > 0:
        print("Ready to train from Google Drive:")
        print("  python train.py --drive")
    else:
        print("Code is ready. Connect to Google Drive or download locally:")
        print("  python train.py --drive")
        print("  python download_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
preprocessing.py — Preprocessing transforms and DataLoader factory.

Implements:
  • MinMaxNormalizer      — (x - min) / (max - min)
  • ZScoreNormalizer      — (x - mean) / std
  • VolumeResizer         — resize volume with linear interpolation
  • MaskResizer           — resize mask with nearest-neighbour (preserves labels)
  • OneHotEncoder         — convert integer mask → one-hot channels
  • PreprocessingPipeline — composes all of the above into a single callable
  • build_dataloaders()   — splits dataset, wraps in DataLoaders

All transforms accept and return a dict {"image": np.ndarray, "mask": np.ndarray}
so they compose cleanly with augmentation transforms.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset

import config
from dataset import CBCTDataset, DriveCBCTDataset, _local_dataset_ready

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Individual transform classes
# ─────────────────────────────────────────────────────────────────────────────

class MinMaxNormalizer:
    """
    Linearly rescale voxel intensities to [0, 1].

        x_norm = (x - x_min) / (x_max - x_min + ε)

    Parameters
    ----------
    eps : float
        Small constant that prevents division by zero for constant volumes.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image = sample["image"].astype(np.float32)
        v_min, v_max = image.min(), image.max()
        sample["image"] = (image - v_min) / (v_max - v_min + self.eps)
        return sample


class ZScoreNormalizer:
    """
    Standardise voxel intensities to zero mean and unit variance.

        x_norm = (x - μ) / (σ + ε)

    Parameters
    ----------
    eps : float
        Small constant that prevents division by zero for constant volumes.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image = sample["image"].astype(np.float32)
        mu    = image.mean()
        sigma = image.std()
        sample["image"] = (image - mu) / (sigma + self.eps)
        return sample


class VolumeResizer:
    def __init__(self, target_size: tuple, order: int = 1):
        self.target_size = target_size
        self.order = order

    def __call__(self, sample: dict) -> dict:
        import numpy as np
        from scipy.ndimage import zoom
        image = sample['image']
        spatial_shape = image.shape[1:] if len(image.shape) == 4 else image.shape
        target = np.array(self.target_size, dtype=float)
        factors = target / np.array(spatial_shape, dtype=float)
        full_factors = [1.0] + list(factors) if len(image.shape) == 4 else factors
        resized = zoom(image, full_factors, order=self.order, prefilter=False)
        sample['image'] = resized
        return sample


class MaskResizer:
    def __init__(self, target_size: tuple, order: int = 0):
        self.target_size = target_size
        self.order = order

    def __call__(self, sample: dict) -> dict:
        import numpy as np
        from scipy.ndimage import zoom
        mask = sample['mask']
        spatial_shape = mask.shape[1:] if len(mask.shape) == 4 else mask.shape
        target = np.array(self.target_size, dtype=float)
        factors = target / np.array(spatial_shape, dtype=float)
        full_factors = [1.0] + list(factors) if len(mask.shape) == 4 else factors
        resized = zoom(mask, full_factors, order=self.order, prefilter=False)
        sample['mask'] = resized
        return sample


class OneHotEncoder:
    """
    Convert an integer-valued mask of shape (D, H, W) into a one-hot
    representation of shape (C, D, H, W), where C = num_classes.

    This is applied *after* resizing and stored back into sample["mask"]
    so that the final mask tensor has shape (C, D, H, W).

    Note: CBCTDataset.__getitem__ later adds a channel dim for the *raw*
    case (1, D, H, W).  When one-hot is enabled the mask is already
    multi-channel; the Dataset skips re-adding a channel dim if ndim == 4.

    Parameters
    ----------
    num_classes : int
        Total number of segmentation classes including background (class 0).
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES) -> None:
        self.num_classes = num_classes

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        mask = sample["mask"]
        if mask.ndim == 4:
            mask = mask.squeeze(0)
        mask = mask.astype(np.int64)
        D, H, W = mask.shape
        one_hot = np.zeros((self.num_classes, D, H, W), dtype=np.float32)
        for c in range(self.num_classes):
            one_hot[c] = (mask == c).astype(np.float32)
        sample["mask"] = one_hot   # (C, D, H, W)
        return sample


class Compose:
    """
    Sequentially apply a list of callables to a sample dict.

    Compatible with both preprocessing and augmentation transforms.
    """

    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        lines = ["Compose(["]
        for t in self.transforms:
            lines.append(f"  {t.__class__.__name__},")
        lines.append("])")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline factory
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    normalization: str = config.NORMALIZATION,
    target_size: Tuple[int, int, int] = config.IMAGE_SIZE,
    one_hot: bool = config.ONE_HOT_ENCODE,
) -> Compose:
    """
    Construct the base preprocessing pipeline (no augmentation).

    Applied identically to train, validation, and test splits.

    Parameters
    ----------
    normalization : str
        "minmax"  → MinMaxNormalizer
        "zscore"  → ZScoreNormalizer
    target_size : tuple of int
        Target spatial dimensions for volume and mask resize.
    one_hot : bool
        Whether to one-hot encode multi-class masks.

    Returns
    -------
    Compose
        A callable that accepts and returns a sample dict.
    """
    transforms = []

    # ── 1. Normalise intensity ───────────────────────────────────────────────
    if normalization == "minmax":
        transforms.append(MinMaxNormalizer())
    elif normalization == "zscore":
        transforms.append(ZScoreNormalizer())
    else:
        raise ValueError(
            f"Unknown normalization '{normalization}'. "
            "Choose 'minmax' or 'zscore'."
        )

    # ── 2. Resize volume (trilinear) ─────────────────────────────────────────
    transforms.append(VolumeResizer(target_size=target_size, order=1))

    # ── 3. Resize mask (nearest-neighbour) ───────────────────────────────────
    transforms.append(MaskResizer(target_size=target_size))

    # ── 4. Optional one-hot encoding ─────────────────────────────────────────
    if one_hot:
        transforms.append(OneHotEncoder(num_classes=config.NUM_CLASSES))

    logger.debug("Preprocessing pipeline: %s", transforms)
    return Compose(transforms)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory (dataset split + DataLoader construction)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    train_transform: Optional[Compose] = None,
    val_transform:   Optional[Compose] = None,
    test_transform:  Optional[Compose] = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    pin_memory: bool = config.PIN_MEMORY,
    seed: int = config.RANDOM_SEED,
    cache: bool = False,
    use_drive: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / validation / test DataLoaders for the CBCT dataset.

    Split strategy
    --------------
    All indices are split deterministically:
        70 % train  |  15 % validation  |  15 % test

    Transforms
    ----------
    train_transform : Compose, optional
        Preprocessing + augmentation pipeline.  Defaults to preprocessing only.
    val_transform / test_transform : Compose, optional
        Preprocessing-only pipelines.  Default to preprocessing only.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """

    # ── Default transforms (preprocessing only) ──────────────────────────────
    preprocess = build_preprocessing_pipeline()
    if train_transform is None:
        train_transform = preprocess
    if val_transform is None:
        val_transform = preprocess
    if test_transform is None:
        test_transform = preprocess

    # ── Split indices deterministically ──────────────────────────────────────
    if use_drive:
        from drive_client import build_drive_manifest
        drive_manifest = build_drive_manifest()
        probe_dataset = DriveCBCTDataset(transform=None, cache=False, manifest=drive_manifest)
    else:
        drive_manifest = None
        probe_dataset = CBCTDataset(
            image_dir=config.IMAGE_DIR,
            mask_dir=config.MASK_DIR,
            transform=None,
            cache=False,
        )

    n = len(probe_dataset)
    indices = list(range(n))

    if n < 3:
        logger.warning(
            "Only %d sample(s) found — using all for training (minimal val/test).",
            n,
        )
        train_idx = indices
        val_idx = indices[:1]
        test_idx = indices[:1]
    else:
        n_val = max(1, round(n * config.VAL_RATIO))
        n_test = max(1, round(n * config.TEST_RATIO))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = max(1, n - 2)
            n_val = 1
            n_test = max(1, n - n_train - n_val)

        shuffled = indices.copy()
        rng = np.random.RandomState(seed)
        rng.shuffle(shuffled)
        train_idx = shuffled[:n_train]
        val_idx = shuffled[n_train:n_train + n_val]
        test_idx = shuffled[n_train + n_val:]
        if not test_idx:
            test_idx = val_idx[:1]

    logger.info(
        "Dataset split — train: %d | val: %d | test: %d",
        len(train_idx), len(val_idx), len(test_idx),
    )

    # ── Create per-split Dataset instances with correct transforms ───────────
    if use_drive:
        train_dataset = DriveCBCTDataset(
            transform=train_transform, cache=cache, manifest=drive_manifest,
        )
        val_dataset = DriveCBCTDataset(
            transform=val_transform, cache=cache, manifest=drive_manifest,
        )
        test_dataset = DriveCBCTDataset(
            transform=test_transform, cache=cache, manifest=drive_manifest,
        )
    else:
        train_dataset = CBCTDataset(
            image_dir=config.IMAGE_DIR,
            mask_dir=config.MASK_DIR,
            transform=train_transform,
            cache=cache,
        )
        val_dataset = CBCTDataset(
            image_dir=config.IMAGE_DIR,
            mask_dir=config.MASK_DIR,
            transform=val_transform,
            cache=cache,
        )
        test_dataset = CBCTDataset(
            image_dir=config.IMAGE_DIR,
            mask_dir=config.MASK_DIR,
            transform=test_transform,
            cache=cache,
        )

    # Slice each full dataset to the correct index subset
    train_subset = Subset(train_dataset, train_idx)
    val_subset   = Subset(val_dataset,   val_idx)
    test_subset  = Subset(test_dataset,  test_idx)

    # ── Wrap in DataLoaders ───────────────────────────────────────────────────
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_subset) > batch_size,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

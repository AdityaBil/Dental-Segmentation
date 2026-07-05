"""
augmentations.py — Spatial and intensity augmentations for training only.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import rotate, zoom

import config
from preprocessing import Compose

logger = logging.getLogger(__name__)


def _spatial_axes(ndim: int) -> List[int]:
    """Return spatial axis indices, skipping a leading channel dimension."""
    return [1, 2, 3] if ndim == 4 else [0, 1, 2]


def _rotation_axis_pairs(ndim: int) -> List[Tuple[int, int]]:
    axes = _spatial_axes(ndim)
    return [(axes[0], axes[1]), (axes[0], axes[2]), (axes[1], axes[2])]


def _crop_volume(
    array: np.ndarray,
    d0: int, h0: int, w0: int,
    cD: int, cH: int, cW: int,
) -> np.ndarray:
    if array.ndim == 4:
        return array[:, d0:d0 + cD, h0:h0 + cH, w0:w0 + cW]
    return array[d0:d0 + cD, h0:h0 + cH, w0:w0 + cW]


class RandomRotation3D:
    def __init__(
        self,
        max_degrees: float = config.AUG_ROTATION_DEGREES,
        p: float = 0.5,
    ) -> None:
        self.max_degrees = max_degrees
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.rand() > self.p:
            return sample

        image = sample["image"]
        mask = sample["mask"]

        for axes in _rotation_axis_pairs(image.ndim):
            angle = np.random.uniform(-self.max_degrees, self.max_degrees)
            image = rotate(image, angle, axes=axes, reshape=False, order=1, mode="nearest")
            mask = rotate(mask, angle, axes=axes, reshape=False, order=0, mode="nearest")

        sample["image"] = image.astype(np.float32)
        sample["mask"] = mask.astype(np.float32)
        return sample


class RandomFlip3D:
    def __init__(
        self,
        axes: Sequence[int] = config.AUG_FLIP_AXES,
        p: float = 0.5,
    ) -> None:
        self.axes = list(axes)
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image = sample["image"]
        mask = sample["mask"]
        flip_axes = _spatial_axes(image.ndim)

        for ax in flip_axes:
            if np.random.rand() < self.p:
                image = np.flip(image, axis=ax).copy()
                mask = np.flip(mask, axis=ax).copy()

        sample["image"] = image
        sample["mask"] = mask
        return sample


class RandomCrop3D:
    def __init__(
        self,
        crop_size: Tuple[int, int, int] = config.AUG_CROP_SIZE,
        target_size: Tuple[int, int, int] = config.IMAGE_SIZE,
        p: float = 0.5,
    ) -> None:
        self.crop_size = crop_size
        self.target_size = target_size
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.rand() > self.p:
            return sample

        image = sample["image"]
        mask = sample["mask"]

        spatial_shape = image.shape[1:] if image.ndim == 4 else image.shape
        D, H, W = spatial_shape
        cD, cH, cW = self.crop_size
        cD, cH, cW = min(cD, D), min(cH, H), min(cW, W)

        d0 = np.random.randint(0, D - cD + 1)
        h0 = np.random.randint(0, H - cH + 1)
        w0 = np.random.randint(0, W - cW + 1)

        cropped_img = _crop_volume(image, d0, h0, w0, cD, cH, cW)
        cropped_mask = _crop_volume(mask, d0, h0, w0, cD, cH, cW)

        tD, tH, tW = self.target_size
        factors = (tD / cD, tH / cH, tW / cW)
        channel_prefix = [1.0] if cropped_img.ndim == 4 else []

        resized_img = zoom(
            cropped_img,
            channel_prefix + list(factors),
            order=1,
            prefilter=False,
        )
        resized_mask = zoom(
            cropped_mask,
            channel_prefix + list(factors),
            order=0,
            prefilter=False,
        )

        target_img_shape = (image.shape[0], *self.target_size) if image.ndim == 4 else self.target_size
        target_mask_shape = (mask.shape[0], *self.target_size) if mask.ndim == 4 else self.target_size

        sample["image"] = _fit_shape(resized_img, target_img_shape).astype(np.float32)
        sample["mask"] = _fit_shape(resized_mask, target_mask_shape).astype(np.float32)
        return sample


def _fit_shape(array: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    if array.shape == target_shape:
        return array
    out = np.zeros(target_shape, dtype=array.dtype)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(array.shape, target_shape))
    out[slices] = array[slices]
    return out


class RandomGaussianNoise:
    def __init__(
        self,
        std: float = config.AUG_GAUSSIAN_STD,
        p: float = 0.5,
    ) -> None:
        self.std = std
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.rand() > self.p:
            return sample
        noise = np.random.normal(0.0, self.std, size=sample["image"].shape)
        sample["image"] = np.clip(sample["image"] + noise, 0.0, 1.0).astype(np.float32)
        return sample


class RandomContrastGamma:
    def __init__(
        self,
        gamma_range: Tuple[float, float] = config.AUG_CONTRAST_GAMMA,
        p: float = 0.5,
    ) -> None:
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.rand() > self.p:
            return sample
        gamma = np.random.uniform(*self.gamma_range)
        image = np.clip(sample["image"], 0.0, 1.0)
        sample["image"] = np.power(image, gamma).astype(np.float32)
        return sample


class RandomIntensityShift:
    def __init__(
        self,
        max_shift: float = config.AUG_INTENSITY_SHIFT,
        p: float = 0.4,
    ) -> None:
        self.max_shift = max_shift
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.rand() > self.p:
            return sample
        shift = np.random.uniform(-self.max_shift, self.max_shift)
        sample["image"] = np.clip(sample["image"] + shift, 0.0, 1.0).astype(np.float32)
        return sample


class RandomZoom3D:
    def __init__(self, min_zoom: float = 0.9, max_zoom: float = 1.1, p: float = 0.4) -> None:
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.rand() > self.p:
            return sample

        image = sample["image"]
        mask = sample["mask"]

        zfactor = float(np.random.uniform(self.min_zoom, self.max_zoom))

        # Zoom spatial dims only (preserve channel dim if present)
        if image.ndim == 4:
            channel_prefix = [1.0]
        else:
            channel_prefix = []

        image = zoom(image, channel_prefix + [zfactor, zfactor, zfactor], order=1, prefilter=False)
        mask = zoom(mask, channel_prefix + [zfactor, zfactor, zfactor], order=0, prefilter=False)

        # Fit back to original target shape
        target_img_shape = (image.shape[0], *config.IMAGE_SIZE) if image.ndim == 4 else config.IMAGE_SIZE
        target_mask_shape = (mask.shape[0], *config.IMAGE_SIZE) if mask.ndim == 4 else config.IMAGE_SIZE

        sample["image"] = _fit_shape(image, target_img_shape).astype(np.float32)
        sample["mask"] = _fit_shape(mask, target_mask_shape).astype(np.float32)
        return sample


def build_train_transform(
    preprocessing_pipeline: Optional[Compose] = None,
) -> Compose:
    from preprocessing import build_preprocessing_pipeline

    if preprocessing_pipeline is None:
        preprocessing_pipeline = build_preprocessing_pipeline()

    augmentation_transforms = [
        RandomRotation3D(max_degrees=config.AUG_ROTATION_DEGREES, p=0.5),
        RandomFlip3D(p=0.5),
        RandomZoom3D(min_zoom=0.92, max_zoom=1.08, p=0.4),
        RandomCrop3D(
            crop_size=config.AUG_CROP_SIZE,
            target_size=config.IMAGE_SIZE,
            p=0.4,
        ),
        RandomGaussianNoise(std=config.AUG_GAUSSIAN_STD, p=0.5),
        RandomContrastGamma(gamma_range=config.AUG_CONTRAST_GAMMA, p=0.5),
        RandomIntensityShift(max_shift=config.AUG_INTENSITY_SHIFT, p=0.4),
    ]

    all_transforms = list(preprocessing_pipeline.transforms) + augmentation_transforms
    logger.info(
        "Training transform pipeline: %d preprocessing + %d augmentation steps.",
        len(preprocessing_pipeline.transforms),
        len(augmentation_transforms),
    )
    return Compose(all_transforms)

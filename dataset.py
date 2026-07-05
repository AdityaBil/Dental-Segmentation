import os
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import config

logger = logging.getLogger(__name__)

try:
    import SimpleITK as sitk

    def _read_nifti(path: str) -> np.ndarray:
        return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)

except ImportError:
    import nibabel as nib

    logger.info("SimpleITK not installed; using nibabel for NIfTI I/O.")

    def _read_nifti(path: str) -> np.ndarray:
        return np.asanyarray(nib.load(path).dataobj).astype(np.float32)

def _image_case_id(stem: str) -> str:
    """Case id from an nnU-Net image filename (drops trailing ``_0000`` modality suffix)."""
    if stem.endswith("_0000"):
        return stem[:-5]
    return stem


def _label_case_id(stem: str) -> str:
    """Case id from a label filename (no modality suffix to remove)."""
    return stem


def _extract_id(stem: str) -> str:
    """Backward-compatible helper — assumes image-style naming."""
    return _image_case_id(stem)


def _discover_pairs(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    mask_dict: Dict[str, str] = {}
    for entry in os.scandir(mask_dir):
        if entry.name.endswith((".nii", ".nii.gz")):
            mask_id = _label_case_id(entry.name.split(".")[0])
            mask_dict[mask_id] = entry.path

    pairs: List[Tuple[str, str]] = []
    for img_entry in os.scandir(image_dir):
        if not img_entry.name.endswith((".nii", ".nii.gz")):
            continue
        img_id = _image_case_id(img_entry.name.split(".")[0])
        if img_id in mask_dict:
            pairs.append((img_entry.path, mask_dict[img_id]))
        else:
            logger.warning("No matching mask found for image: %s", img_entry.path)

    if not pairs:
        raise RuntimeError(
            f"No image/mask pairs found in:\n  images: {image_dir}\n  masks:  {mask_dir}"
        )

    pairs.sort(key=lambda pair: pair[0])
    logger.info("Discovered %d image/mask pairs.", len(pairs))
    return pairs


def _local_dataset_ready() -> bool:
    """True when both imagesTr and labelsTr exist with at least one pair."""
    if not os.path.isdir(config.IMAGE_DIR) or not os.path.isdir(config.MASK_DIR):
        return False
    try:
        return len(_discover_pairs(config.IMAGE_DIR, config.MASK_DIR)) > 0
    except (FileNotFoundError, RuntimeError):
        return False


class DriveCBCTDataset(Dataset):
    """
    CBCT dataset that lazily fetches volumes from Google Drive into a local cache.

    Each sample downloads its image + mask on first access (then reuses cache).
    """

    def __init__(
        self,
        transform: Optional[Callable] = None,
        cache: bool = False,
        manifest: Optional[dict] = None,
    ) -> None:
        from drive_client import build_drive_manifest, discover_drive_pairs, ensure_cached

        self.transform = transform
        self.cache = cache
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._ensure_cached = ensure_cached

        if manifest is None:
            manifest = build_drive_manifest()
        self.drive_pairs = discover_drive_pairs(manifest)

    def __len__(self) -> int:
        return len(self.drive_pairs)

    def _load_raw(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache and idx in self._cache:
            return self._cache[idx]

        img_name, img_id, mask_name, mask_id = self.drive_pairs[idx]
        img_path = self._ensure_cached("imagesTr", img_name, img_id)
        mask_path = self._ensure_cached("labelsTr", mask_name, mask_id)

        img = _read_nifti(img_path)
        mask = _read_nifti(mask_path)

        img = img[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        if self.cache:
            self._cache[idx] = (img, mask)
        return img, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = self._load_raw(idx)
        sample = {"image": img, "mask": mask}

        if self.transform:
            sample = self.transform(sample)

        mask_np = sample["mask"]
        if mask_np.max() > 1 or mask_np.min() < 0:
            sample["mask"] = (mask_np > 0).astype(np.float32)

        return torch.from_numpy(sample["image"]).float(), torch.from_numpy(sample["mask"]).float()


class CBCTDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        cache: bool = False,
    ) -> None:
        self.transform = transform
        self.pairs = _discover_pairs(image_dir, mask_dir)
        self.cache = cache
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_raw(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache and idx in self._cache:
            return self._cache[idx]

        img_path, mask_path = self.pairs[idx]
        img = _read_nifti(img_path)
        mask = _read_nifti(mask_path)

        img = img[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        if self.cache:
            self._cache[idx] = (img, mask)
        return img, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = self._load_raw(idx)
        sample = {"image": img, "mask": mask}

        if self.transform:
            sample = self.transform(sample)

        mask_np = sample["mask"]
        if mask_np.max() > 1 or mask_np.min() < 0:
            sample["mask"] = (mask_np > 0).astype(np.float32)

        return torch.from_numpy(sample["image"]).float(), torch.from_numpy(sample["mask"]).float()

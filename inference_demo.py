"""
inference_demo.py — Run UNet3D on a few images and save NIfTI + PNG overlays.

Usage:
    python inference_demo.py

Notes:
- Runs on CPU by default per `config.DEVICE`.
- If `checkpoints/best_model.pth` exists it will be loaded; otherwise the
  model runs with random weights (demo only).
"""

import os
from pathlib import Path
import logging
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt

# make project importable
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models.unet3d import UNet3D
from preprocessing import build_preprocessing_pipeline
from utils import load_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_demo")

OUT_DIR = Path("demo_outputs")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = config.DEVICE

# Instantiate model
model = UNet3D(
    in_channels=config.IN_CHANNELS,
    num_classes=config.NUM_CLASSES,
    base_filters=config.BASE_FILTERS,
    dropout_p=config.DROPOUT_P,
)
model.to(DEVICE)
model.eval()

# Optional checkpoint
if os.path.exists(config.BEST_MODEL_PATH):
    try:
        load_checkpoint(config.BEST_MODEL_PATH, model=model, device=DEVICE)
        logger.info("Loaded checkpoint: %s", config.BEST_MODEL_PATH)
    except Exception as e:
        logger.warning("Failed to load checkpoint: %s", e)
else:
    logger.warning("No checkpoint found at %s — running with random weights.", config.BEST_MODEL_PATH)

# Prepare preprocessing pipeline
preproc = build_preprocessing_pipeline()

# Find images
img_dir = Path(config.IMAGE_DIR)
imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix in ('.nii', '.gz', '.nii.gz') or p.name.endswith('.nii')]
if not imgs:
    logger.error("No images found in %s", img_dir)
    raise SystemExit(1)

# Select up to 3 images
imgs = imgs[:3]

results = []
with torch.no_grad():
    for p in imgs:
        logger.info("Processing %s", p.name)
        sitk_img = sitk.ReadImage(str(p))
        img_np = sitk.GetArrayFromImage(sitk_img).astype(np.float32)  # (D, H, W)

        # Create dummy mask (required by preprocessing pipeline)
        mask_dummy = np.zeros_like(img_np, dtype=np.float32)

        # Add channel dim to match dataset convention: (1, D, H, W)
        sample = {"image": img_np[np.newaxis, ...], "mask": mask_dummy[np.newaxis, ...]}

        # Apply preprocessing (normalise + resize). Returns numpy arrays.
        sample = preproc(sample)
        proc_img = sample["image"]  # (1, D, H, W)

        # Convert to torch tensor with batch dim: (1, 1, D, H, W)
        vol = torch.from_numpy(proc_img).unsqueeze(0).float().to(DEVICE)

        preds = model(vol)  # (1, C, D, H, W)
        if preds.shape[1] > 1:
            probs = torch.softmax(preds, dim=1)
            seg = probs.argmax(dim=1).cpu().numpy().squeeze(0).astype(np.uint8)  # (D, H, W)
        else:
            probs = torch.sigmoid(preds)
            seg = (probs > 0.5).cpu().numpy().squeeze(0).astype(np.uint8)

        # Save predicted mask as NIfTI (uint8)
        out_mask = sitk.GetImageFromArray(seg.astype(np.uint8))
        out_mask.SetOrigin(sitk_img.GetOrigin())
        out_mask.SetSpacing(sitk_img.GetSpacing())
        out_mask_path = OUT_DIR / f"pred_{p.stem}.nii.gz"
        sitk.WriteImage(out_mask, str(out_mask_path))

        # Save a middle axial slice PNG overlay
        D = seg.shape[0]
        mid = D // 2
        img_slice = proc_img.squeeze(0)[mid, ...]  # (H, W)
        mask_slice = seg[mid, ...]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_slice, cmap='gray')
        ax.imshow(mask_slice, cmap='Reds', alpha=0.4)
        ax.axis('off')
        png_path = OUT_DIR / f"pred_{p.stem}_slice.png"
        fig.savefig(str(png_path), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        logger.info("Saved: %s, %s", out_mask_path, png_path)
        results.append((out_mask_path, png_path))

print("Demo outputs:")
for m, s in results:
    print(m)
    print(s)

print("Done.")

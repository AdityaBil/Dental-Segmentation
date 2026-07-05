"""
losses.py — Loss functions for 3D dental CBCT segmentation.

Available losses
────────────────
  DiceLoss      — weighted soft Dice
  BCEDiceLoss   — 0.35·CrossEntropy + 0.65·Dice
  FocalDiceLoss — (1 - Dice)^γ · Dice  [recommended for sparse pathology]
  MonaiDiceLoss — wraps monai.losses.DiceLoss

Why FocalDiceLoss for dental CBCT
──────────────────────────────────
CBCT volumes are severely class-imbalanced: background can be 95–99% of
voxels. Plain Dice loss treats every voxel equally, so most gradient signal
comes from easy background predictions rather than from the hard, small
foreground structures (cavities, lesions, thin root edges).

Focal Dice modulates the per-sample Dice loss by (1 - Dice_i)^γ:
  • When Dice_i is already high (easy sample), (1-Dice_i)^γ → 0, reducing
    the contribution of well-segmented samples.
  • When Dice_i is low (hard sample or small structure), (1-Dice_i)^γ → 1,
    keeping full gradient.

This is the 3D volumetric analogue of Lin et al.'s Focal Loss (2017),
applied at the Dice level rather than the per-pixel BCE level.

All losses expect raw LOGITS from the model (no sigmoid/softmax applied).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert a single-channel integer mask to one-hot float32.

    Parameters
    ----------
    target      : (B, 1, D, H, W)  integer labels in [0, num_classes)
    num_classes : int

    Returns
    -------
    torch.Tensor : (B, num_classes, D, H, W) float32
    """
    idx = target.squeeze(1).long().clamp(min=0, max=num_classes - 1)
    oh  = F.one_hot(idx, num_classes)           # (B, D, H, W, C)
    return oh.permute(0, 4, 1, 2, 3).float()   # (B, C, D, H, W)


def _soft_dice_per_sample(
    probs:  torch.Tensor,   # (B, C, N) — probabilities
    target: torch.Tensor,   # (B, C, N) — one-hot float
    smooth: float,
    fg_weight: float,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute per-sample weighted soft Dice coefficients.

    Returns shape (B,) — one scalar per batch element.
    """
    inter  = (probs * target).sum(dim=2)               # (B, C)
    card   = probs.sum(dim=2) + target.sum(dim=2)      # (B, C)
    dice_c = (2.0 * inter + smooth) / (card + smooth)  # (B, C)

    if num_classes > 1 and fg_weight > 0.0:
        bg_w = 1.0 - fg_weight
        fg_w = fg_weight / max(num_classes - 1, 1)
        w = torch.tensor(
            [bg_w] + [fg_w] * (num_classes - 1),
            device=dice_c.device, dtype=dice_c.dtype,
        )
        return (dice_c * w).sum(dim=1)   # (B,)

    return dice_c.mean(dim=1)            # (B,)


def _prepare(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert logits → probs and ensure target is one-hot (B, C, N).
    """
    if num_classes > 1:
        probs = F.softmax(preds, dim=1)
    else:
        probs = torch.sigmoid(preds)

    if target.shape[1] != num_classes:
        target = _to_one_hot(target, num_classes)

    B, C = probs.shape[:2]
    return probs.reshape(B, C, -1), target.reshape(B, C, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Loss classes
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Weighted soft Dice loss.

    DiceLoss = 1 − Dice(pred, target)

    The foreground classes (class > 0) receive higher weight via
    `foreground_weight` to counter class imbalance.
    """

    def __init__(
        self,
        smooth: float = config.DICE_SMOOTH,
        foreground_weight: float = config.FOREGROUND_DICE_WEIGHT,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.fg_weight = foreground_weight

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = preds.shape[1]
        probs_flat, target_flat = _prepare(preds, target, num_classes)
        dice = _soft_dice_per_sample(
            probs_flat, target_flat, self.smooth, self.fg_weight, num_classes
        )
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Weighted sum of cross-entropy and Dice loss.

        L = α · CE(pred, target) + β · DiceLoss(pred, target)

    CE provides dense per-voxel gradient signal early in training when
    the Dice score is near zero (gradient vanishing); Dice directly
    optimises the overlap metric used at evaluation.
    """

    def __init__(
        self,
        bce_weight: float = 0.35,
        dice_weight: float = 0.65,
        smooth: float = config.DICE_SMOOTH,
        foreground_weight: float = config.FOREGROUND_DICE_WEIGHT,
    ) -> None:
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss   = DiceLoss(smooth=smooth, foreground_weight=foreground_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] > 1:
            ce = F.cross_entropy(pred, target.squeeze(1).long().clamp(min=0))
        else:
            ce = F.binary_cross_entropy_with_logits(pred, target)
        return self.bce_weight * ce + self.dice_weight * self.dice_loss(pred, target)


class FocalDiceLoss(nn.Module):
    """
    Focal Dice Loss — recommended for sparse dental pathology.

    Modulates the per-sample Dice loss by (1 - Dice_i)^γ:

        FocalDice = mean_i [ (1 - Dice_i)^γ · (1 - Dice_i) ]
                  = mean_i [ (1 - Dice_i)^(γ + 1) ]

    Effect
    ------
    • Well-segmented samples (Dice_i → 1): contribution → 0, focusing
      training on samples where the model is still struggling.
    • Poorly segmented or small-structure samples (Dice_i → 0):
      contribution → 1, keeping full gradient.

    This is particularly effective for CBCT where most volumes contain
    large background regions that are trivially easy to predict.

    Parameters
    ----------
    gamma : float
        Focusing parameter.  0.0 → plain Dice.  0.75 is a good default;
        increase to 1.0–1.5 if small structures (thin roots, micro-caries)
        are consistently under-segmented.
    smooth : float
        Smoothing constant.
    foreground_weight : float
        Extra weight given to foreground classes in the per-class Dice.
    """

    def __init__(
        self,
        gamma: float = getattr(config, "FOCAL_GAMMA", 0.75),
        smooth: float = config.DICE_SMOOTH,
        foreground_weight: float = config.FOREGROUND_DICE_WEIGHT,
    ) -> None:
        super().__init__()
        self.gamma     = gamma
        self.smooth    = smooth
        self.fg_weight = foreground_weight

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = preds.shape[1]
        probs_flat, target_flat = _prepare(preds, target, num_classes)

        # Per-sample Dice coefficient ∈ [0, 1]
        dice = _soft_dice_per_sample(
            probs_flat, target_flat, self.smooth, self.fg_weight, num_classes
        )   # (B,)

        # Focal modulation: (1 - Dice)^(γ+1)
        focal_loss = (1.0 - dice) ** (self.gamma + 1.0)

        return focal_loss.mean()

    def __repr__(self) -> str:
        return f"FocalDiceLoss(gamma={self.gamma}, fg_weight={self.fg_weight})"


class MonaiDiceLoss(nn.Module):
    """
    Thin wrapper around monai.losses.DiceLoss.
    Falls back to manual DiceLoss if MONAI is not installed.
    """

    def __init__(
        self,
        smooth_nr: float = config.DICE_SMOOTH,
        smooth_dr: float = config.DICE_SMOOTH,
    ) -> None:
        super().__init__()
        try:
            from monai.losses import DiceLoss as _MonaiDice
            self._loss = _MonaiDice(
                sigmoid=False,
                softmax=True,
                to_onehot_y=True,
                include_background=True,
                smooth_nr=smooth_nr,
                smooth_dr=smooth_dr,
                reduction="mean",
            )
            self._using_monai = True
            logger.info("Using MONAI DiceLoss.")
        except ImportError:
            logger.warning("MONAI not installed — falling back to manual DiceLoss.")
            self._loss = DiceLoss(smooth=smooth_nr)
            self._using_monai = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_loss_fn(name: str = config.LOSS_FN) -> nn.Module:
    """
    Construct a loss function by name string.

    Options
    -------
    "dice"        → DiceLoss
    "bce_dice"    → BCEDiceLoss  (CE + Dice)
    "focal_dice"  → FocalDiceLoss  [recommended]
    "monai_dice"  → MonaiDiceLoss
    """
    registry = {
        "dice":       DiceLoss,
        "bce_dice":   BCEDiceLoss,
        "focal_dice": FocalDiceLoss,
        "monai_dice": MonaiDiceLoss,
    }
    name = name.strip().lower()
    if name not in registry:
        raise ValueError(
            f"Unknown loss '{name}'. Choose from: {list(registry.keys())}"
        )
    loss_fn = registry[name]()
    logger.info("Loss function: %s", loss_fn)
    return loss_fn

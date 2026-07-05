"""
models/unet3d.py — Attention Residual 3D U-Net for CBCT dental segmentation.

Changes from previous version
──────────────────────────────
1. AttentionGate added to every decoder skip connection.
   Softly suppresses irrelevant encoder features using the decoder's gating
   signal — materially improves Dice on small, sparse structures (cavities,
   lesions) that are easily overwhelmed by background.

2. Dropout placement fixed.
   Previously Dropout3d was applied after EVERY decoder block, which
   randomly zeroed entire feature maps right after skip-connection fusion,
   breaking the spatial signal the skips just provided.
   Now: dropout only at the bottleneck, where overfitting risk is highest.

3. BASE_FILTERS default raised to 32 (config.py change).
   Channels [32, 64, 128, 256] → 22.9M params vs previous 5.7M.
   Still fits Colab T4 at batch=2; on CPU set BASE_FILTERS=16 in config.py.

Architecture
────────────
Input  : (B, 1, D, H, W)

Encoder:
  enc1  : ResConv(1→32)          96³  (no pool — keeps full-res skip)
  enc2  : ResConv(32→64)  + pool 48³
  enc3  : ResConv(64→128) + pool 24³
  enc4  : ResConv(128→256)+ pool 12³
  bn    : ResConv(256→512)       6³   ← bottleneck + Dropout3d

Decoder (each: upsample → AttentionGate(skip) → concat → ResConv):
  dec4  : 512 + 256(attn) → 256  12³
  dec3  : 256 + 128(attn) → 128  24³
  dec2  : 128 + 64(attn)  → 64   48³
  dec1  : 64  + 32(attn)  → 32   96³

Output : Conv1×1 → (B, num_classes, D, H, W)   [logits, no activation]
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResidualConvBlock(nn.Module):
    """
    Two Conv3d→BN→ReLU layers with a residual (identity or 1×1) shortcut.

    When in_channels ≠ out_channels the shortcut is a learned 1×1 projection,
    so the block can change channel width without losing the residual benefit.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels,  out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

        # Learnable shortcut when channel width changes; identity otherwise
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(x) + self.shortcut(x), inplace=True)


class EncoderBlock(nn.Module):
    """
    Encoder stage: ResidualConvBlock → MaxPool3d.

    Returns both the pre-pool feature map (used as skip connection in decoder)
    and the downsampled output (fed to the next encoder stage).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ResidualConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (downsampled, skip)."""
        skip = self.conv(x)
        return self.pool(skip), skip


class AttentionGate(nn.Module):
    """
    Soft attention gate for skip connections (Oktay et al., 2018).

    Computes a spatial attention map α ∈ [0,1] using:
      • g  — gating signal from the decoder (coarser scale, upsampled to match x)
      • x  — skip connection from the encoder (finer scale)

    α = σ( W_g·g + W_x·x + b )
    out = α ⊙ x

    The gate suppresses irrelevant background regions in the skip features so
    the decoder focuses gradient on the small, sparse pathology voxels.

    Parameters
    ----------
    in_channels  : int  Channels in the skip connection (x).
    gate_channels: int  Channels in the gating signal (g) — usually 2× in_channels.
    inter_channels: int Internal intermediate channels (defaults to in_channels // 2).
    """

    def __init__(
        self,
        in_channels:   int,
        gate_channels: int,
        inter_channels: int | None = None,
    ) -> None:
        super().__init__()

        if inter_channels is None:
            inter_channels = max(in_channels // 2, 1)

        # 1×1 linear projections (no spatial mixing — keep it lightweight)
        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels,   inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels),
        )
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : encoder skip feature  (B, in_channels,   D, H, W)
        g : decoder gating signal (B, gate_channels, d, h, w)  — coarser

        Returns
        -------
        torch.Tensor  Attention-weighted skip, shape same as x.
        """
        # Upsample g to match x's spatial dimensions if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="trilinear", align_corners=False)

        # Project both to inter_channels and add (element-wise)
        phi = F.relu(self.W_x(x) + self.W_g(g), inplace=True)

        # Scalar attention map per spatial location
        alpha = self.psi(phi)          # (B, 1, D, H, W)

        return x * alpha               # broadcast across channels


class DecoderBlock(nn.Module):
    """
    Decoder stage: upsample → attention gate → concat → ResidualConvBlock.

    Parameters
    ----------
    in_channels  : int  Channels from the previous decoder / bottleneck output.
    skip_channels: int  Channels in the matching encoder skip connection.
    out_channels : int  Output channels after the residual conv.
    use_attention: bool If True, an AttentionGate is applied to the skip
                        before concatenation.  If False, raw skip is used
                        (equivalent to standard U-Net decoder).
    """

    def __init__(
        self,
        in_channels:   int,
        skip_channels: int,
        out_channels:  int,
        use_attention: bool = True,
    ) -> None:
        super().__init__()

        # Learnable 2× upsampling; reduces channels by 2 before concat
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionGate(
                in_channels=skip_channels,
                gate_channels=out_channels,   # g comes from up(x)
            )

        # After concat: out_channels (from up) + skip_channels → out_channels
        self.conv = ResidualConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # 1. Upsample decoder feature map
        x = self.up(x)

        # 2. Pad to match skip spatial size (handles odd input dimensions)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

        # 3. Optionally gate the skip connection
        if self.use_attention:
            skip = self.attn(skip, g=x)

        # 4. Concat and refine
        return self.conv(torch.cat([x, skip], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    """
    Attention Residual 3D U-Net for volumetric dental CBCT segmentation.

    Parameters
    ----------
    in_channels  : int   Input channels (1 for grayscale CBCT).
    num_classes  : int   Output classes — 2 for binary (bg + pathology).
    base_filters : int   Channels at first encoder level; doubles each stage.
                         Default 32 gives [32, 64, 128, 256] (22.9M params).
                         Set to 16 for [16, 32, 64, 128] (5.7M) on CPU.
    dropout_p    : float Dropout probability at the bottleneck only.
    use_attention: bool  Toggle attention gates on skip connections.

    Output
    ------
    Raw logits of shape (B, num_classes, D, H, W).
    Apply softmax / sigmoid externally (losses expect logits).
    """

    def __init__(
        self,
        in_channels:   int   = config.IN_CHANNELS,
        num_classes:   int   = config.NUM_CLASSES,
        base_filters:  int   = config.BASE_FILTERS,
        dropout_p:     float = config.DROPOUT_P,
        use_attention: bool  = getattr(config, "USE_ATTENTION", True),
    ) -> None:
        super().__init__()

        C: List[int] = [
            base_filters,
            base_filters * 2,
            base_filters * 4,
            base_filters * 8,
        ]
        bn_ch = C[-1] * 2   # bottleneck output channels

        # ── Encoder ──────────────────────────────────────────────────────────
        # enc1 has no pooling: the skip at full resolution is critical for
        # preserving fine-grained dental surface detail.
        self.enc1 = ResidualConvBlock(in_channels, C[0])
        self.enc2 = EncoderBlock(C[0], C[1])
        self.enc3 = EncoderBlock(C[1], C[2])
        self.enc4 = EncoderBlock(C[2], C[3])

        # Shared max-pool for enc1 → enc2 transition (enc2/3/4 pool internally)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = ResidualConvBlock(C[3], bn_ch)
        # Dropout at bottleneck only — where overfitting risk is highest
        # and spatial structure is already heavily compressed.
        self.bottleneck_drop = (
            nn.Dropout3d(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec4 = DecoderBlock(bn_ch, C[3], C[3], use_attention)
        self.dec3 = DecoderBlock(C[3],  C[2], C[2], use_attention)
        self.dec2 = DecoderBlock(C[2],  C[1], C[1], use_attention)
        self.dec1 = DecoderBlock(C[1],  C[0], C[0], use_attention)

        # ── Output head ───────────────────────────────────────────────────────
        # 1×1 conv → logits (no activation; losses apply softmax/sigmoid)
        self.head = nn.Conv3d(C[0], num_classes, kernel_size=1)

        # ── Weight initialisation ─────────────────────────────────────────────
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "UNet3D (Attention=%s) | channels=%s | bottleneck=%d | params=%.2fM",
            use_attention, C, bn_ch, n_params / 1e6,
        )

    def _init_weights(self) -> None:
        """Kaiming He init for Conv3d/ConvTranspose3d; 1/0 for BatchNorm."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, D, H, W)

        Returns
        -------
        logits : (B, num_classes, D, H, W)
            Raw logits — apply softmax/sigmoid in the loss or for inference.
        """
        # ── Encoder ──────────────────────────────────────────────────────────
        s1 = self.enc1(x)           # (B, C0, 96, 96, 96)  — full-res skip
        e2, s2 = self.enc2(self.pool(s1))   # s2: (B, C1, 48, 48, 48)
        e3, s3 = self.enc3(e2)      # s3: (B, C2, 24, 24, 24)
        e4, s4 = self.enc4(e3)      # s4: (B, C3, 12, 12, 12)

        # ── Bottleneck ────────────────────────────────────────────────────────
        b = self.bottleneck(e4)     # (B, bn_ch, 6, 6, 6)
        b = self.bottleneck_drop(b)

        # ── Decoder (attention-gated skip fusion) ─────────────────────────────
        d4 = self.dec4(b,  s4)      # (B, C3, 12, 12, 12)
        d3 = self.dec3(d4, s3)      # (B, C2, 24, 24, 24)
        d2 = self.dec2(d3, s2)      # (B, C1, 48, 48, 48)
        d1 = self.dec1(d2, s1)      # (B, C0, 96, 96, 96)

        return self.head(d1)        # (B, num_classes, 96, 96, 96)

    # ── Convenience ───────────────────────────────────────────────────────────

    def freeze_encoder(self) -> None:
        """Freeze encoder weights — useful for fine-tuning on a new dataset."""
        for m in [self.enc1, self.enc2, self.enc3, self.enc4]:
            for p in m.parameters():
                p.requires_grad = False
        logger.info("Encoder frozen.")

    def unfreeze_encoder(self) -> None:
        for m in [self.enc1, self.enc2, self.enc3, self.enc4]:
            for p in m.parameters():
                p.requires_grad = True
        logger.info("Encoder unfrozen.")

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

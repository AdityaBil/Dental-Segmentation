import os
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Google Drive dataset folder:
# https://drive.google.com/drive/folders/154c_JD__HEN2j5R9ujYuDDIzom5hUbGi
# Download with: python download_dataset.py
DATASET_ROOT = os.environ.get(
    "TOOTHFAIRY_DATASET_ROOT",
    os.path.join(ROOT_DIR, "data", "toothfairy_dataset"),
)

IMAGE_DIR = os.path.join(DATASET_ROOT, "imagesTr")
MASK_DIR = os.path.join(DATASET_ROOT, "labelsTr")

# Google Drive lazy-load mode (downloads one file at a time into DRIVE_CACHE_DIR)
USE_DRIVE_DATASET = os.environ.get("TOOTHFAIRY_USE_DRIVE", "").lower() in ("1", "true", "yes")
DRIVE_CACHE_DIR = os.path.join(ROOT_DIR, "data", "drive_cache")
DRIVE_IMAGES_FOLDER_ID = "15t9i9JSkXMXe06njvY-c2tRMRJosMu3A"
DRIVE_LABELS_FOLDER_ID = "1Mb5jlcuxLbYCOBC5EVK5hhnXYo70SnoQ"
DRIVE_DOWNLOAD_RETRIES = 8
DRIVE_DOWNLOAD_DELAY = 5.0  # seconds between Drive requests (rate-limit friendly)

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LOG_PATH = os.path.join(ROOT_DIR, "training.log")

IMAGE_SIZE = (96, 96, 96)
NORMALIZATION = "minmax"
ONE_HOT_ENCODE = False

NUM_CLASSES = 2
CLASS_NAMES = {0: "Background", 1: "Pathology"}

BATCH_SIZE = 1
# Windows multiprocessing in DataLoader often hangs; use 0 workers locally.
NUM_WORKERS = 0 if os.name == "nt" else 2
PIN_MEMORY = torch.cuda.is_available()

# If GPU is available, prefer a small batch size increase for stability/perf
if torch.cuda.is_available() and BATCH_SIZE < 2:
    BATCH_SIZE = 2

IN_CHANNELS = 1

# ── Model architecture ────────────────────────────────────────────────────────
# BASE_FILTERS = 32 → channels [32, 64, 128, 256] → 22.9M params.
# On CPU or low-VRAM GPU (< 8 GB), set to 16 → [16, 32, 64, 128] → 5.7M params.
BASE_FILTERS = 32
ENCODER_CHANNELS = [BASE_FILTERS * (2 ** i) for i in range(4)]  # derived — do not edit
DROPOUT_P = 0.15

# Attention gates on skip connections (set False to revert to standard U-Net)
USE_ATTENTION = True

# ── Training ──────────────────────────────────────────────────────────────────
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
EPOCHS = 100
SCHEDULER_T0 = 20
GRAD_CLIP_NORM = 1.0

# Loss function: "dice" | "bce_dice" | "focal_dice" | "monai_dice"
# focal_dice is recommended for sparse dental pathology (severe class imbalance)
LOSS_FN = "focal_dice"
DICE_SMOOTH = 1e-6
FOREGROUND_DICE_WEIGHT = 0.8

# Focal Dice: γ ∈ [0.5, 2.0]. Higher γ = more focus on hard/small positives.
# 0.75 is a conservative start; increase to 1.0 if small lesions are missed.
FOCAL_GAMMA = 0.75

VALIDATE_EVERY = 1
SAVE_EVERY = 10
EARLY_STOPPING_PATIENCE = 15

# ── Augmentation ──────────────────────────────────────────────────────────────
AUG_ROTATION_DEGREES = 25
AUG_FLIP_AXES = [0, 1, 2]
# Must be SMALLER than IMAGE_SIZE to produce genuine cropping.
# With IMAGE_SIZE=(96,96,96), use 80 not 96.
AUG_CROP_SIZE = (80, 80, 80)
AUG_GAUSSIAN_STD = 0.08
AUG_CONTRAST_GAMMA = (0.75, 1.25)
AUG_INTENSITY_SHIFT = 0.15
AUG_ELASTIC_ALPHA = 2.0
AUG_ELASTIC_SIGMA = 8.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ToothFairy — 3D Dental CBCT Segmentation

Train a 3D U-Net on the [ToothFairy](https://toothfairy.grand-challenge.org/) CBCT dataset for pathology segmentation.

## Project layout

```
toothfairy/
├── config.py           # Paths and hyperparameters
├── train.py            # Training entry point
├── dataset.py          # NIfTI dataset loader
├── preprocessing.py    # Normalization, resize, dataloaders
├── augmentations.py    # Training augmentations
├── losses.py           # Dice / BCE+Dice losses
├── utils.py            # Metrics, logging, checkpoints
├── models/
│   └── unet3d.py       # 3D U-Net
├── data/
│   └── toothfairy_dataset/
│       ├── imagesTr/   # CBCT volumes (.nii / .nii.gz)
│       └── labelsTr/   # Segmentation masks
└── checkpoints/        # Created during training
```

## Setup

### 1. Install Python 3.10+

Download from [python.org](https://www.python.org/downloads/) and enable **"Add Python to PATH"** during install.

### 2. Run setup (recommended)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

Or manually create a virtual environment:

```powershell
cd C:\Users\bit\Downloads\toothfairy
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Download the dataset

**Google Drive (recommended):**

| Resource | Link |
|----------|------|
| Dataset (`imagesTr`, `labelsTr`, `dataset.json`) | [Open folder](https://drive.google.com/drive/folders/154c_JD__HEN2j5R9ujYuDDIzom5hUbGi) |
| Original Colab code backup | [Open folder](https://drive.google.com/drive/folders/1IeURyKfhkvzAKgk4f4J1AAsFbNBADzrF) |

**Option A — automatic download (after Python is installed):**

```powershell
pip install gdown
python download_dataset.py
```

This downloads into `data/toothfairy_dataset/`.

**Option B — manual download:**

1. Open the [dataset folder](https://drive.google.com/drive/folders/154c_JD__HEN2j5R9ujYuDDIzom5hUbGi)
2. Download `imagesTr`, `labelsTr`, and `dataset.json`
3. Place them here:

```
data/toothfairy_dataset/
├── imagesTr/
├── labelsTr/
└── dataset.json
```

**Custom path:**

```powershell
$env:TOOTHFAIRY_DATASET_ROOT = "D:\datasets\toothfairy_dataset"
```

### 4. Verify the setup

```powershell
python verify_setup.py
```

### 5. Train

```powershell
python train.py
```

Useful options:

```powershell
python train.py --epochs 100 --batch-size 2 --lr 0.0001
python train.py --resume checkpoints/best_model.pth
python train.py --no-cache
```

### Train from Google Drive (no full local download)

If you cannot download the full dataset locally, use **lazy Drive mode**. It lists files on Drive, then downloads one image + mask at a time into `data/drive_cache/` as training needs them:

```powershell
python train.py --drive
```

This is selected **automatically** when `labelsTr/` is missing locally.

**Alternatives:**
- **Google Drive Desktop:** Sync the [dataset folder](https://drive.google.com/drive/folders/154c_JD__HEN2j5R9ujYuDDIzom5hUbGi) and set:
  ```powershell
  $env:TOOTHFAIRY_DRIVE_MOUNT = "G:\My Drive\toothfairy_dataset"
  python train.py
  ```
- **Google Colab:** Open `colab_train.ipynb`, mount Drive, and run training there.

## Configuration

Edit `config.py` or use environment variables:

| Setting | Default |
|---------|---------|
| `IMAGE_SIZE` | `(64, 64, 64)` |
| `BATCH_SIZE` | `1` |
| `EPOCHS` | `100` |
| `LEARNING_RATE` | `0.0001` |
| `LOSS_FN` | `"dice"` |
| Train / val / test split | 80% / 10% / 10% |

## Notes

- GPU training is recommended; the code falls back to CPU automatically.
- Best model is saved to `checkpoints/best_model.pth` based on validation Dice score.
- Logs are written to `training.log`.

## Google Drive links

- **Dataset:** https://drive.google.com/drive/folders/154c_JD__HEN2j5R9ujYuDDIzom5hUbGi  
  Contains `imagesTr/`, `labelsTr/`, and `dataset.json`
- **Original Colab code backup:** https://drive.google.com/drive/folders/1IeURyKfhkvzAKgk4f4J1AAsFbNBADzrF  
  Contains the `project/` folder from Colab (this repo is the fixed local version)

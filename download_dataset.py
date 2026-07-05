"""
download_dataset.py — Download ToothFairy dataset from Google Drive.

Requires:
    pip install gdown

Usage:
    python download_dataset.py
    python download_dataset.py --output data/toothfairy_dataset
"""

import argparse
import os
import sys

# Public Google Drive folder shared by the user
DATASET_FOLDER_ID = "154c_JD__HEN2j5R9ujYuDDIzom5hUbGi"
DATASET_FOLDER_URL = f"https://drive.google.com/drive/folders/{DATASET_FOLDER_ID}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ToothFairy dataset from Google Drive.")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "data", "toothfairy_dataset"),
        help="Local folder for imagesTr/, labelsTr/, and dataset.json",
    )
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        print("gdown is required. Install it with:\n  pip install gdown")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"Downloading dataset from Google Drive…")
    print(f"  Source : {DATASET_FOLDER_URL}")
    print(f"  Target : {args.output}")
    print("This may take a while depending on your connection.\n")

    # Some networks produce intermittent errors when downloading many files.
    # Retry the folder download several times before giving up.
    import time
    import traceback

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Folder download attempt {attempt}/{max_attempts}...")
            gdown.download_folder(
                url=DATASET_FOLDER_URL,
                output=args.output,
                quiet=False,
                use_cookies=False,
                resume=True,
            )
            break
        except Exception as e:
            print(f"Download attempt {attempt} failed: {e}")
            traceback.print_exc()
            if attempt < max_attempts:
                wait = 10 * attempt
                print(f"Retrying after {wait}s...")
                time.sleep(wait)
            else:
                print("All download attempts failed. Please retry manually.")
                raise

    images_dir = os.path.join(args.output, "imagesTr")
    labels_dir = os.path.join(args.output, "labelsTr")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(
            "\nDownload finished, but expected folders were not found.\n"
            "Please verify the Drive folder contains:\n"
            "  imagesTr/\n"
            "  labelsTr/\n"
            "You can also download manually from:\n"
            f"  {DATASET_FOLDER_URL}"
        )
        sys.exit(1)

    n_images = sum(
        1 for name in os.listdir(images_dir)
        if name.endswith((".nii", ".nii.gz"))
    )
    n_labels = sum(
        1 for name in os.listdir(labels_dir)
        if name.endswith((".nii", ".nii.gz"))
    )

    print("\nDownload complete.")
    print(f"  imagesTr : {n_images} files")
    print(f"  labelsTr : {n_labels} files")
    print("\nNext steps:")
    print("  python verify_setup.py")
    print("  python train.py")


if __name__ == "__main__":
    main()

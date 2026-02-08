"""
Full pipeline: download real dataset -> extract features -> build sequences -> train -> model ready for app.

Two modes:
  With Kaggle (real exercise videos, best): set up kaggle.json once (docs/CREDENTIALS.md), then run this script.
  Without Kaggle: run with --no-kaggle to try a direct-download dataset; if none, uses demo data and trains.

Run from project root:
  python scripts/run_full_pipeline.py              # uses Kaggle (needs credentials)
  python scripts/run_full_pipeline.py --no-kaggle  # direct download or demo data

Then start the app (serves the trained model):
  streamlit run main.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_VIDEOS = DATA_DIR / "kaggle_exercise"
REAL_DATASET_DIR = DATA_DIR / "real_dataset"
FEATURES_DIR = DATA_DIR / "features"
SEQUENCES_NPZ = DATA_DIR / "data_sequences.npz"
MODELS_DIR = PROJECT_ROOT / "models"


def run(cmd, desc, required=True):
    print("\n" + "=" * 60)
    print(desc)
    print("=" * 60)
    r = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    if required and r.returncode != 0:
        print("Failed:", cmd)
        sys.exit(r.returncode)
    return r.returncode


def has_videos(folder):
    if not folder.exists():
        return False
    return bool(list(folder.rglob("*.mp4")) + list(folder.rglob("*.avi")) + list(folder.rglob("*.mov")))


def already_downloaded():
    """True if real_dataset folder or zip already exists (skip re-download)."""
    if REAL_DATASET_DIR.exists() and any(REAL_DATASET_DIR.iterdir()):
        return True
    zip_path = DATA_DIR / "real_dataset.zip"
    return zip_path.exists()


def main():
    parser = argparse.ArgumentParser(description="Download dataset, train model, ready for app")
    parser.add_argument("--no-kaggle", action="store_true", help="Use direct-download dataset (or demo if no videos)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    video_dir = None

    if args.no_kaggle:
        # Skip download if data folder or zip already exists
        if already_downloaded():
            print("\n" + "=" * 60)
            print("Step 1: Already downloaded from source link. Skipping download.")
            print("=" * 60)
        else:
            run(
                f"python scripts/download_real_dataset.py --output_dir {REAL_DATASET_DIR}",
                "Step 1: Download dataset (direct URL, no Kaggle)",
                required=False,
            )
        if has_videos(REAL_DATASET_DIR):
            video_dir = REAL_DATASET_DIR
        if not video_dir:
            print("No videos from direct download. Using demo data to train so you can still run the app.")
            run(f"python scripts/generate_demo_data.py --output {SEQUENCES_NPZ}", "Step 1b: Generate demo data")
            run(
                f"python scripts/train_bidirectionallstm.py --data {SEQUENCES_NPZ} --output_dir {MODELS_DIR} --source demo",
                "Step 2: Train model (saved to models/)",
            )
            print("\n" + "=" * 60)
            print("Done. Model is in models/. Start the app: streamlit run main.py")
            print("=" * 60)
            return
    else:
        # Kaggle path
        if not has_videos(KAGGLE_VIDEOS):
            run(
                f"python scripts/download_dataset.py --output_dir {KAGGLE_VIDEOS}",
                "Step 1: Download real dataset from Kaggle (requires kaggle.json; see docs/CREDENTIALS.md)",
            )
        video_dir = KAGGLE_VIDEOS

    # 2. Extract 22-D features from videos
    run(
        f"python scripts/extract_features.py --video_dir {video_dir} --output_dir {FEATURES_DIR}",
        "Step 2: Extract pose features from videos",
    )

    # 3. Build 30-frame sequences
    run(
        f"python scripts/create_sequence_of_features.py --input_dir {FEATURES_DIR} --output {SEQUENCES_NPZ}",
        "Step 3: Build 30-frame sequences",
    )

    # 4. Train improved model (saves to models/)
    source = "kaggle" if video_dir == KAGGLE_VIDEOS else "direct"
    run(
        f"python scripts/train_bidirectionallstm.py --data {SEQUENCES_NPZ} --output_dir {MODELS_DIR} --source {source}",
        "Step 4: Train model (saved to models/)",
    )

    print("\n" + "=" * 60)
    print("Pipeline done. Model is in models/. Start the app:")
    print("  streamlit run main.py")
    print("Then use 'Auto Classify' to run with the trained model.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Run full demo: generate synthetic data, train BASELINE and IMPROVED models, print comparison.
No Kaggle or credentials needed. Use this to show your performance increase.

Run from project root:
  python scripts/run_demo_training.py
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "data_sequences.npz"
MODELS_DIR = PROJECT_ROOT / "models"


def run(cmd, desc):
    print("\n" + "=" * 60)
    print(desc)
    print("=" * 60)
    r = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        print("Command failed:", cmd)
        sys.exit(r.returncode)


def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate demo data (no Kaggle)
    run(
        f"python scripts/generate_demo_data.py --output {DATA_PATH}",
        "Step 1: Generate synthetic demo dataset (no credentials)",
    )

    # 2. Train BASELINE (original: 64,64, no aug, 30 epochs)
    run(
        f"python scripts/train_bidirectionallstm.py --data {DATA_PATH} --output_dir {MODELS_DIR} "
        "--lstm_units 64,64 --augment 0 --epochs 30 --no_batchnorm --source demo",
        "Step 2: Train BASELINE (64-64, no augmentation)",
    )
    print("\n>>> BASELINE model saved. Test accuracy is printed above.")

    # 3. Train IMPROVED (128,64, aug, early stopping) - this overwrites models/ with improved; app will use it
    run(
        f"python scripts/train_bidirectionallstm.py --data {DATA_PATH} --output_dir {MODELS_DIR} "
        "--lstm_units 128,64 --augment 0.02 --epochs 80 --source demo",
        "Step 3: Train IMPROVED (128-64, augmentation, early stopping)",
    )
    print("\n>>> IMPROVED model saved. Test accuracy is printed above.")
    print("\n>>> Compare the two 'Test accuracy' lines to see performance increase.")
    print(">>> The IMPROVED model is now in models/ and used by the app (Auto Classify).")


if __name__ == "__main__":
    main()
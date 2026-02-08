"""
Download the Real-Time Exercise Recognition dataset from Kaggle for training.
Requires Kaggle API credentials (see docs/CREDENTIALS.md).

Easiest: put KAGGLE_USERNAME and KAGGLE_KEY in a .env file in the project root.
  The script loads .env automatically so you don't need to create ~/.config/kaggle/.

Usage (from project root):
  pip install kaggle python-dotenv
  # Add to .env: KAGGLE_USERNAME=your_username, KAGGLE_KEY=your_key
  python scripts/download_dataset.py [--output_dir data/kaggle_exercise]
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root so KAGGLE_USERNAME and KAGGLE_KEY work without creating kaggle.json
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

KAGGLE_DATASET = "riccardoriccio/real-time-exercise-recognition-dataset"


def main():
    parser = argparse.ArgumentParser(description="Download exercise recognition dataset from Kaggle")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "kaggle_exercise"),
        help="Directory to download and unzip the dataset",
    )
    args = parser.parse_args()
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Kaggle authenticates on import; credentials missing raise OSError/IOError
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except ImportError:
        print("Install the Kaggle API: pip install kaggle")
        print("Then set up credentials: see docs/CREDENTIALS.md")
        sys.exit(1)
    except (OSError, IOError) as e:
        print("Kaggle credentials not found or invalid.")
        print("Do this:")
        print("  1. Go to https://www.kaggle.com/settings → Create New Token (downloads kaggle.json)")
        print("  2. Place kaggle.json in one of:")
        print("       ~/.config/kaggle/kaggle.json   (Linux)")
        print("       ~/.kaggle/kaggle.json          (alternative)")
        print("  3. Easiest: create a .env file in the project root with:")
        print("       KAGGLE_USERNAME=your_username")
        print("       KAGGLE_KEY=your_key")
        print("     (No need to create ~/.config/kaggle/ - the script loads .env automatically.)")
        print("  4. If using a file: chmod 600 ~/.config/kaggle/kaggle.json")
        print("See docs/CREDENTIALS.md for details.")
        sys.exit(1)
    except Exception as e:
        print("Kaggle authentication failed:", e)
        print("See docs/CREDENTIALS.md for credential setup.")
        sys.exit(1)

    print(f"Downloading {KAGGLE_DATASET} to {out_path} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(out_path), unzip=True)
    print("Done. Next: run extract_features.py on the video folders, then create_sequence_of_features.py, then train.")


if __name__ == "__main__":
    main()

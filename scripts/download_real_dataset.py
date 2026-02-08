"""
Download a real exercise dataset (no Kaggle). Uses direct URL.
Saves to data/real_dataset/; folder structure should be .../label/video.mp4 for best results.
"""
import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Direct download (no login). MM-Fit: multimodal exercise data (may include sensor + refs; videos on Zenodo).
# Alternative: a small public sample if available.
DEFAULT_URL = "https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip"


def main():
    parser = argparse.ArgumentParser(description="Download real dataset from direct URL")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Direct download URL (zip)")
    parser.add_argument("--output_dir", type=str, default=None, help="Unzip here (default: data/real_dataset)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir or PROJECT_ROOT / "data" / "real_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir.parent / "real_dataset.zip"

    print(f"Downloading from {args.url} ...")
    try:
        urlretrieve(args.url, zip_path)
    except Exception as e:
        print("Download failed:", e)
        print("For the Real-Time Exercise Recognition dataset (videos), use Kaggle:")
        print("  1. See docs/CREDENTIALS.md to set up kaggle.json")
        print("  2. Run: python scripts/download_dataset.py --output_dir data/kaggle_exercise")
        sys.exit(1)

    print("Unzipping ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
    except Exception as e:
        print("Unzip failed:", e)
        sys.exit(1)

    videos = list(out_dir.rglob("*.mp4")) + list(out_dir.rglob("*.avi")) + list(out_dir.rglob("*.mov"))
    print(f"Done. Found {len(videos)} video file(s) under {out_dir}")
    if not videos:
        print("No videos in this zip. For exercise videos use Kaggle:")
        print("  See docs/CREDENTIALS.md then: python scripts/download_dataset.py --output_dir data/kaggle_exercise")
    else:
        print("Next: python scripts/extract_features.py --video_dir", out_dir, "--output_dir data/features")


if __name__ == "__main__":
    main()

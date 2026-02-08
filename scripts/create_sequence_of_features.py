"""
Build sequences of 30 consecutive frames with features for BiLSTM training.
Reads per-video feature CSVs (from extract_features.py) and outputs (X, y) with X shape (N, 30, 22).
Run from project root: python scripts/create_sequence_of_features.py --input_dir <dir> --output <output.npz>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WINDOW_SIZE = 30
NUM_FEATURES = 22

# Map common dataset folder/label names to app-expected class names (for Auto Classify)
LABEL_NORMALIZE = {
    "squat": "squat",
    "push_up": "push-up",
    "push-up": "push-up",
    "bicep_curl": "barbell biceps curl",
    "bicep curl": "barbell biceps curl",
    "barbell biceps curl": "barbell biceps curl",
    "shoulder_press": "shoulder press",
    "shoulder press": "shoulder press",
}


def normalize_label(label):
    """Normalize label to app-expected class name."""
    key = label.strip().lower().replace(" ", "_").replace("-", "_")
    for k, v in LABEL_NORMALIZE.items():
        if k.replace(" ", "_").replace("-", "_") == key:
            return v
    return label


def load_features_csv(csv_path):
    """Load frame_idx, label, f0..f21 from CSV. Return (features array, labels list)."""
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"CSV must have 'label' column: {csv_path}")
    feat_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    if len(feat_cols) != NUM_FEATURES:
        feat_cols = [f"f{i}" for i in range(NUM_FEATURES)]
    X = df[feat_cols].values.astype(np.float32)
    y_raw = df["label"].iloc[0] if "label" in df.columns else "unknown"
    y = normalize_label(str(y_raw))
    return X, y


def build_sequences(X, y, window_size=WINDOW_SIZE):
    """Slide window over X; each window is one sample. y is scalar label for whole video."""
    if len(X) < window_size:
        return [], []
    X_seq = []
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i : i + window_size])
    y_seq = [y] * len(X_seq)
    return X_seq, y_seq


def main():
    parser = argparse.ArgumentParser(description="Create 30-frame sequences from feature CSVs")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing *_features.csv files")
    parser.add_argument("--output", type=str, default="data_sequences.npz", help="Output .npz file (X, y, labels)")
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE, help="Sequence length (default 30)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Not a directory: {input_dir}")
        return

    all_X = []
    all_y = []
    label_set = set()

    for csv_path in sorted(input_dir.glob("*_features.csv")):
        try:
            X, y = load_features_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path}: {e}")
            continue
        if len(X) < args.window_size:
            continue
        X_seq, y_seq = build_sequences(X, y, args.window_size)
        all_X.extend(X_seq)
        all_y.extend(y_seq)
        label_set.add(y)

    if not all_X:
        print("No sequences created. Ensure CSVs have at least 30 frames each.")
        return

    X_arr = np.array(all_X, dtype=np.float32)
    # y: keep as string list; encoding to int is done in training
    labels = sorted(label_set)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[y] for y in all_y], dtype=np.int64)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X_arr, y=y_idx, labels=np.array(labels, dtype=object))
    print(f"Saved {out_path}: X.shape={X_arr.shape}, y.shape={y_idx.shape}, labels={labels}")


if __name__ == "__main__":
    main()

"""
Generate a synthetic demo dataset (no Kaggle, no credentials).
Same format as create_sequence_of_features.py output: (X, y, labels).
Use this to run the full training pipeline and see baseline vs improved performance.

Run from project root:
  python scripts/generate_demo_data.py --output data/data_sequences.npz
Then train:
  python scripts/train_bidirectionallstm.py --data data/data_sequences.npz --output_dir models
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WINDOW_SIZE = 30
NUM_FEATURES = 22

# App-expected class names (for Auto Classify)
LABELS = ["push-up", "squat", "shoulder press", "barbell biceps curl"]


def make_class_sequence(class_idx, n_frames=WINDOW_SIZE, n_feat=NUM_FEATURES, seed=None):
    """
    Generate one 30-frame x 22-feature sequence with class-specific pattern.
    Intentionally noisy and overlapping so the model does not trivially reach 100%
    (avoids overfitting / unrealistic metrics on demo data).
    """
    rng = np.random.default_rng(seed)
    # Class-specific base means - kept close so classes overlap (no trivial separation)
    bases = {
        0: np.linspace(0.35, 0.75, n_frames),
        1: np.linspace(0.30, 0.70, n_frames),
        2: np.linspace(0.40, 0.70, n_frames),
        3: np.linspace(0.35, 0.65, n_frames),
    }
    base = bases.get(class_idx, np.linspace(0.35, 0.70, n_frames))
    # Strong per-frame and per-feature noise so accuracy is realistic (not 100%)
    X = np.zeros((n_frames, n_feat), dtype=np.float32)
    for t in range(n_frames):
        # Larger noise range so classes overlap
        X[t] = base[t] + rng.uniform(-0.25, 0.25, n_feat)
    return X


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic demo dataset (no Kaggle)")
    parser.add_argument("--output", type=str, default="data/data_sequences.npz", help="Output .npz path")
    parser.add_argument("--samples_per_class", type=int, default=200, help="Sequences per class (default 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    all_X = []
    all_y = []

    for c in range(len(LABELS)):
        for _ in range(args.samples_per_class):
            seq = make_class_sequence(c, seed=rng.integers(0, 100000))
            all_X.append(seq)
            all_y.append(c)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y, labels=np.array(LABELS, dtype=object))
    print(f"Saved {out_path}: X.shape={X.shape}, y.shape={y.shape}, labels={LABELS}")
    print("Next: python scripts/train_bidirectionallstm.py --data", args.output, "--output_dir models")


if __name__ == "__main__":
    main()

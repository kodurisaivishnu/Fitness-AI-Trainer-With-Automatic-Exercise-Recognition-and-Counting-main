"""
Extract pose landmarks and derived features (angles, normalized distances) from exercise videos.
Output: per-frame feature vectors (22 dims) compatible with the BiLSTM exercise classifier.
Run from project root: python scripts/extract_features.py --video_dir <path> [--output_dir <path>]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Add project root so we can optionally share constants (here we duplicate to keep script standalone)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Map normalized folder name -> app-expected label (so similar_dataset/push-up/ -> push-up)
def _normalize_folder_name(name):
    return name.strip().lower().replace(" ", "_").replace("-", "_")


FOLDER_TO_EXERCISE_LABEL = {
    "push_up": "push-up",
    "squat": "squat",
    "barbell_biceps_curl": "barbell biceps curl",
    "bicep_curl": "barbell biceps curl",
    "shoulder_press": "shoulder press",
}


def _label_from_path(rel_path):
    """Use deepest folder that matches a known exercise; else first folder. E.g. similar_dataset/push-up/video.mp4 -> push-up."""
    parts = rel_path.parts[:-1] if len(rel_path.parts) > 1 else []
    if not parts:
        return None
    for part in reversed(parts):
        key = _normalize_folder_name(part)
        if key in FOLDER_TO_EXERCISE_LABEL:
            return FOLDER_TO_EXERCISE_LABEL[key]
    return parts[0]


mp_pose = mp.solutions.pose
RELEVANT_LANDMARKS_INDICES = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value,
]
NUM_FEATURES = 22


def calculate_angle(a, b, c):
    if np.any(np.array([a, b, c]) == 0):
        return -1.0
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculate_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0
    return np.linalg.norm(np.array(a) - np.array(b))


def calculate_y_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0
    return np.abs(a[1] - b[1])


def extract_features_from_landmarks(landmarks):
    """Compute 22-dim feature vector (angles + normalized distances) from flat landmark list [x,y,z]*12."""
    if len(landmarks) != len(RELEVANT_LANDMARKS_INDICES) * 3:
        return None
    features = []
    # Angles (8)
    features.append(calculate_angle(landmarks[0:3], landmarks[6:9], landmarks[12:15]))
    features.append(calculate_angle(landmarks[3:6], landmarks[9:12], landmarks[15:18]))
    features.append(calculate_angle(landmarks[18:21], landmarks[24:27], landmarks[30:33]))
    features.append(calculate_angle(landmarks[21:24], landmarks[27:30], landmarks[33:36]))
    features.append(calculate_angle(landmarks[0:3], landmarks[18:21], landmarks[24:27]))
    features.append(calculate_angle(landmarks[3:6], landmarks[21:24], landmarks[27:30]))
    features.append(calculate_angle(landmarks[18:21], landmarks[0:3], landmarks[6:9]))
    features.append(calculate_angle(landmarks[21:24], landmarks[3:6], landmarks[9:12]))
    # Distances (12)
    distances = [
        calculate_distance(landmarks[0:3], landmarks[3:6]),
        calculate_distance(landmarks[18:21], landmarks[21:24]),
        calculate_distance(landmarks[18:21], landmarks[24:27]),
        calculate_distance(landmarks[21:24], landmarks[27:30]),
        calculate_distance(landmarks[0:3], landmarks[18:21]),
        calculate_distance(landmarks[3:6], landmarks[21:24]),
        calculate_distance(landmarks[6:9], landmarks[24:27]),
        calculate_distance(landmarks[9:12], landmarks[27:30]),
        calculate_distance(landmarks[12:15], landmarks[0:3]),
        calculate_distance(landmarks[15:18], landmarks[3:6]),
        calculate_distance(landmarks[12:15], landmarks[18:21]),
        calculate_distance(landmarks[15:18], landmarks[21:24]),
    ]
    y_distances = [
        calculate_y_distance(landmarks[6:9], landmarks[0:3]),
        calculate_y_distance(landmarks[9:12], landmarks[3:6]),
    ]
    norm_factors = [
        calculate_distance(landmarks[0:3], landmarks[18:21]),
        calculate_distance(landmarks[3:6], landmarks[21:24]),
        calculate_distance(landmarks[18:21], landmarks[24:27]),
        calculate_distance(landmarks[21:24], landmarks[27:30]),
    ]
    norm = next((d for d in norm_factors if d > 0), 0.5)
    normalized_d = [d / norm if d != -1.0 else d for d in distances]
    normalized_y = [d / norm if d != -1.0 else d for d in y_distances]
    features.extend(normalized_d)
    features.extend(normalized_y)
    return features


def process_video(video_path, pose, label=None):
    """Yield (frame_index, feature_list) for each frame with valid pose."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        landmarks = []
        if results.pose_landmarks:
            for idx in RELEVANT_LANDMARKS_INDICES:
                lm = results.pose_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
        feats = extract_features_from_landmarks(landmarks)
        if feats is not None:
            yield frame_idx, feats, label
        frame_idx += 1
    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Extract features from exercise videos")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory containing videos (optional subdirs = labels)")
    parser.add_argument("--video", type=str, default=None, help="Single video file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for CSV files (default: same as video_dir or current)")
    parser.add_argument("--label", type=str, default=None, help="Label for single video (e.g. squat, push-up)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or args.video_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return
        label = args.label or video_path.stem
        out_csv = output_dir / f"{video_path.stem}_features.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "label"] + [f"f{i}" for i in range(NUM_FEATURES)])
            for frame_idx, feats, _ in process_video(video_path, pose, label):
                writer.writerow([frame_idx, label] + feats)
        print(f"Wrote {out_csv}")

    elif args.video_dir:
        video_dir = Path(args.video_dir)
        if not video_dir.is_dir():
            print(f"Not a directory: {video_dir}")
            return
        # Support: video_dir/label/video.mp4 or video_dir/dataset/push-up/video.mp4 (use exercise name when present)
        for path in sorted(video_dir.rglob("*")):
            if path.suffix.lower() not in (".mp4", ".avi", ".mov", ".m4v"):
                continue
            rel = path.relative_to(video_dir)
            label = _label_from_path(rel)
            if label is None:
                label = rel.parts[0] if len(rel.parts) > 1 else path.stem
            out_csv = output_dir / f"{path.stem}_features.csv"
            with open(out_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_idx", "label"] + [f"f{i}" for i in range(NUM_FEATURES)])
                count = 0
                for frame_idx, feats, _ in process_video(path, pose, label):
                    writer.writerow([frame_idx, label] + feats)
                    count += 1
                if count > 0:
                    print(f"Wrote {out_csv} ({count} frames)")
    else:
        print("Provide --video_dir or --video. Use --help for options.")
        return

    print("Done.")


if __name__ == "__main__":
    main()

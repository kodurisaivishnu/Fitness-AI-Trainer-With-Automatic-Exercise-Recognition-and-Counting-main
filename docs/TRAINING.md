# Training Pipeline: Improved Exercise Recognition Model

This document describes how to download the dataset, extract features, build sequences, and train the improved BiLSTM model. The resulting model is used by the app for **Auto Classify** mode.

---

## One command: download, train, then serve

**With Kaggle (real exercise videos, best):**

1. Set up Kaggle once: [docs/CREDENTIALS.md](CREDENTIALS.md) (put `kaggle.json` in `~/.config/kaggle/` or `~/.kaggle/`).
2. From project root run:
   ```bash
   python scripts/run_full_pipeline.py
   ```
3. Start the app (serves the trained model):
   ```bash
   streamlit run main.py
   ```

**Without Kaggle (direct download or demo data):**

```bash
python scripts/run_full_pipeline.py --no-kaggle
streamlit run main.py
```

If the direct-download dataset has no videos, the script falls back to demo data and still trains a model so the app runs.

---

## 1. Prerequisites

- Python 3.9+ with conda env `fitness-ai-trainer` (or install from `requirements.txt`)
- **Kaggle account** and API credentials for dataset download (see [CREDENTIALS.md](CREDENTIALS.md))

---

## 2. Dataset

### 2.1 Download from Kaggle

The pipeline uses the **Real-Time Exercise Recognition Dataset** (Riccardo Riccio et al.):

- **Kaggle:** [riccardoriccio/real-time-exercise-recognition-dataset](https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset)
- **Contents:** Videos (or frames) of four exercises: squat, push-up, shoulder press, bicep curl.

**Steps:**

1. Set up Kaggle API credentials (see [docs/CREDENTIALS.md](CREDENTIALS.md)).
2. Install and run the download script:

```bash
pip install kaggle
python scripts/download_dataset.py --output_dir data/kaggle_exercise
```

The dataset is extracted under `data/kaggle_exercise/`. The exact folder structure may vary; typically you get one folder per exercise or per video.

### 2.2 Optional: Use your own videos

Place videos in a directory, optionally grouped by exercise:

```
data/my_videos/
  squat/
    video1.mp4
    video2.mp4
  push_up/
    ...
  shoulder_press/
    ...
  bicep_curl/
    ...
```

Labels are taken from the **folder name** (or from `--label` when processing a single file). The pipeline normalizes common names (e.g. `push_up` → `push-up`, `bicep_curl` → `barbell biceps curl`) so the app’s Auto Classify mode works correctly.

---

## 3. Feature extraction

Extract 22-D pose features (angles + normalized distances) from each video. Output: one CSV per video.

**Single video:**

```bash
python scripts/extract_features.py --video path/to/video.mp4 --label "squat" --output_dir data/features
```

**Whole directory (labels from subfolder names):**

```bash
python scripts/extract_features.py --video_dir data/kaggle_exercise --output_dir data/features
```

CSVs are written as `data/features/<videoname>_features.csv`. Each row: `frame_idx`, `label`, `f0`…`f21`.

---

## 4. Build sequences

Build 30-frame sliding-window sequences for the BiLSTM (input shape `(N, 30, 22)`).

```bash
python scripts/create_sequence_of_features.py --input_dir data/features --output data/data_sequences.npz
```

Optional: `--window_size 30` (default). Labels are normalized to app-expected names (e.g. `barbell biceps curl`, `push-up`).

---

## 5. Train the model

Train either **BiLSTM** (default) or **1D CNN**. Saved model and encoders are **app-compatible** (same filenames in `models/`). The app loads whichever architecture you last trained.

**Default (BiLSTM):**

```bash
python scripts/train_bidirectionallstm.py --data data/data_sequences.npz --output_dir models
```

**Alternative – 1D CNN (simple, easy to explain to a mentor):**

```bash
python scripts/train_bidirectionallstm.py --data data/data_sequences.npz --output_dir models --model cnn1d
```

- **BiLSTM:** Processes the 30-frame sequence in both directions; good for long-range temporal context. Easy to describe: "Two LSTM layers read the sequence forward and backward, then a dense layer classifies."
- **1D CNN:** Sliding filters along the time axis detect local motion patterns, then global pooling and a dense layer classify. Easy to describe: "Convolutions over time act like short motion detectors; we pool them and classify the exercise."

**Options:**

| Option           | Default | Description                              |
| ---------------- | ------- | ---------------------------------------- |
| `--model`        | bilstm  | Architecture: `bilstm` or `cnn1d`        |
| `--epochs`       | 80      | Max epochs                               |
| `--batch_size`   | 32      | Batch size                               |
| `--val_split`    | 0.15    | Validation fraction                      |
| `--test_split`   | 0.15    | Hold-out test fraction                   |
| `--lstm_units`   | 128,64  | LSTM units per layer (BiLSTM only)       |
| `--dropout`      | 0.4     | Dropout rate                             |
| `--l2`           | 1e-3    | L2 regularization                        |
| `--augment`      | 0.02    | Gaussian noise std (0 = no augmentation) |
| `--no_batchnorm` | false   | Disable BatchNorm (BiLSTM only)          |

**Example – train 1D CNN with demo data:**

```bash
python scripts/generate_demo_data.py --output data/data_sequences.npz
python scripts/train_bidirectionallstm.py --data data/data_sequences.npz --output_dir models --model cnn1d --source demo
```

**Outputs in `models/`:**

- `final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5` – Keras model
- `thesis_bidirectionallstm_scaler.pkl` – feature scaler
- `thesis_bidirectionallstm_label_encoder.pkl` – label list (`.classes_`)
- `train_metrics.txt` – test accuracy and classification report (for the paper)

---

## 6. Improvements over the baseline

| Aspect          | Baseline (original) | Improved pipeline                               |
| --------------- | ------------------- | ----------------------------------------------- |
| Architecture    | 2× BiLSTM(64)       | 2× BiLSTM(128, 64), BatchNorm, dropout 0.35     |
| Training        | Fixed 50 epochs     | Early stopping (patience 15), ReduceLROnPlateau |
| Data            | Train/val only      | Train/val/test (stratified), test metrics       |
| Augmentation    | None                | Optional Gaussian noise on features             |
| Model selection | Last epoch          | Best validation accuracy (checkpoint)           |
| Labels          | As in CSV           | Normalized to app class names                   |

These changes aim at **higher accuracy and more stable training**, and provide **test-set metrics** for the research paper.

---

## 7. Full pipeline summary

```bash
# 1. Credentials: set Kaggle (see CREDENTIALS.md)
# 2. Download
python scripts/download_dataset.py --output_dir data/kaggle_exercise

# 3. Extract features (adjust path to your extracted folders)
python scripts/extract_features.py --video_dir data/kaggle_exercise --output_dir data/features

# 4. Build sequences
python scripts/create_sequence_of_features.py --input_dir data/features --output data/data_sequences.npz

# 5. Train (saves app-compatible model to models/)
python scripts/train_bidirectionallstm.py --data data/data_sequences.npz --output_dir models
```



 python scripts/train_bidirectionallstm.py --data data/data_sequences.npz --output_dir models --source kaggle --epochs 1 --batch_size 128^C

Then run the app: `streamlit run main.py` and use **Auto Classify** to try the new model.

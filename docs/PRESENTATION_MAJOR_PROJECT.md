# Major Project Presentation Guide

## Real-Time Fitness Exercise Recognition and Repetition Counting

Use this document to prepare your presentation: it answers typical panel questions and gives flowcharts, commands, and a detailed Kaggle pipeline example.

---

## 1. Why did we choose this project idea?

**Short answer:**  
We wanted a **real-world application** that combines **Computer Vision**, **Pose Estimation**, and **Sequence Modeling** to help people exercise safely and consistently—without needing a gym or a coach.

**Points to say:**

- **Problem:** People exercise at home but lack feedback on _which_ exercise they are doing and _how many_ reps they did. Wrong form or wrong exercise selection can reduce effectiveness or cause injury.
- **Goal:** Build a system that (1) **recognizes** the exercise from video in real time and (2) **counts repetitions** and can give simple form tips.
- **Why this approach:**
  - Uses **pose (skeleton)** instead of raw pixels → more robust to clothing, lighting, camera angle.
  - Uses **temporal sequences** (e.g. 30 frames) → captures _movement_, not just a single pose.
  - Based on a **published method** (Riccio, 2024) so we can compare and improve.
- **Outcome:** A web app (Streamlit) with **Video upload**, **WebCam**, and **Auto Classify** (recognize + count) plus an optional fitness chatbot.

---

## 2. What research paper did you use? What model and metrics did they use? What are your metrics?

### Research paper

- **Title:** _Real-Time Fitness Exercise Classification and Counting from Video Frames_
- **Author:** Riccardo Riccio
- **Year:** 2024
- **Link:** https://arxiv.org/abs/2411.11548

### Their approach (from the paper)

| Aspect              | Paper (Riccio)                                                                                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Model**           | BiLSTM (Bidirectional LSTM) on sequences of **invariant features** (joint angles + normalized distances), **no raw (x,y,z)**. |
| **Input**           | **30-frame** windows of **22-D** feature vectors per frame.                                                                   |
| **Features**        | 8 angles (e.g. elbow, knee) + 12 normalized distances + 2 vertical distances; derived from 12 body landmarks (MediaPipe).     |
| **Classes**         | 4 exercises: squat, push-up, shoulder press, bicep curl.                                                                      |
| **Reported metric** | **>99% test accuracy** on their dataset (Kaggle + synthetic / InfiniteRep, etc.).                                             |
| **Rep counting**    | Rule-based (angle thresholds) per exercise.                                                                                   |

### Our model and metrics

| Aspect                | Our implementation                                                                                                                                                                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model**             | **Improved BiLSTM**: 2 layers with **128 and 64 units** per direction (wider/deeper than typical 64–64), **BatchNormalization**, **Dropout (0.4)**, **L2 regularization (1e-3)**. Optional **1D CNN** (`--model cnn1d`) for a simpler, mentor-friendly alternative. |
| **Input**             | Same: **30 frames × 22 features**; StandardScaler fit on training set only.                                                                                                                                                                                         |
| **Training**          | **Stratified** train/validation/**test** (e.g. 70% / 15% / 15%); **early stopping** (patience 12); **ReduceLROnPlateau**; **Gaussian noise** augmentation (σ=0.02) on features. Best model by **validation accuracy**; final numbers on **held-out test set**.      |
| **Metrics we report** | **Test accuracy**, **precision**, **recall**, **F1** (per class and macro). Example from a run: **Test accuracy 98.33%**; see `models/train_metrics.txt` for the full classification report.                                                                        |

**What to say:**  
“We followed the paper’s idea of BiLSTM on 30-frame invariant features, but we **improved** the architecture (128–64 units, BatchNorm, dropout, L2), added **data augmentation**, **early stopping**, and a **proper test set**. We report **test accuracy** and full **precision/recall/F1** in `models/train_metrics.txt`.”

---

## 3. How did you achieve this? What is the difference from the research paper (what did you change)?

**Summary table:**

| Aspect              | Paper (baseline)       | Our changes                                                                                                                                                            |
| ------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Architecture**    | BiLSTM (e.g. 64–64)    | BiLSTM **128–64**, **BatchNorm**, **Dropout 0.4**, **L2**                                                                                                              |
| **Training**        | Fixed epochs (e.g. 50) | **Early stopping** (patience 12), **ReduceLROnPlateau**                                                                                                                |
| **Data**            | Train/val              | **Stratified train/val/test**; report on **test set**                                                                                                                  |
| **Augmentation**    | Not specified          | **Gaussian noise** (σ=0.02) on feature sequences                                                                                                                       |
| **Model selection** | Often “last epoch”     | **Best checkpoint** by validation accuracy                                                                                                                             |
| **Labels**          | As in dataset          | **Normalized** to app names (e.g. push-up, barbell biceps curl) and **exercise-level** labels when folder structure allows (e.g. `similar_dataset/push-up/` → push-up) |
| **Pipeline**        | —                      | **Documented**: download → extract → sequences → train; **Kaggle** and **no-Kaggle** / **demo** flows with commands                                                    |
| **App**             | —                      | **Video**, **WebCam**, **Auto Classify** (LSTM + counters), optional **chatbot**; **train_info.json** for class mapping                                                |

**What to say:**  
“We kept the core idea—BiLSTM on 30-frame invariant features—and **improved** training (early stopping, LR schedule, augmentation, test set) and **architecture** (deeper BiLSTM, BatchNorm, regularization). We also built a **full pipeline** (Kaggle/demo), **label normalization**, and a **web app** that loads the trained model for Auto Classify.”

---

## 4. Architecture – flowcharts

### 4.1 High-level system flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FITNESS AI COACH (Web App)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Video Upload │ WebCam │ Auto Classify │ Chatbot                             │
└───────┬───────┴────┬───┴───────┬───────┴────────────────────────────────────┘
        │            │           │
        ▼            ▼           ▼
   ┌─────────┐  ┌─────────┐  ┌──────────────────────────────────────────────┐
   │  Video  │  │ WebCam  │  │ Auto Classify                                │
   │  frame  │  │ frame   │  │  • Pose (MediaPipe) → 22-D features          │
   └────┬────┘  └────┬────┘  │  • Buffer 30 frames → BiLSTM → exercise      │
        │            │       │  • Rule-based rep counter for that exercise   │
        │            │       └──────────────────────────────────────────────┘
        ▼            ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  Pose (MediaPipe) → 12 landmarks → angles + distances → 22-D/frame      │
   │  Exercise-specific rep logic (angle thresholds: up/down states)          │
   └─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Auto Classify (real-time) flow

```
  Webcam → Frame → MediaPipe Pose → 12 landmarks
       → Extract 22-D (angles + norm distances)
       → Append to 30-frame buffer
       → When buffer full: Scale → BiLSTM → argmax → exercise class
       → Map class to display name (train_info / index)
       → Run rep counter for that exercise + form tips
       → Show on screen (exercise name, counts, tip)
```

### 4.3 Training pipeline (how the model is built)

```
  Videos (Kaggle / local / demo)
       │
       ▼
  extract_features.py  →  One CSV per video: frame_idx, label, f0..f21
       │
       ▼
  create_sequence_of_features.py  →  Sliding 30-frame windows → data_sequences.npz (X, y, labels)
       │
       ▼
  train_bidirectionallstm.py  →  Train/val/test split → BiLSTM (or 1D CNN) → train
       │
       ▼
  models/  →  .h5/.keras, scaler.pkl, label_encoder.pkl, train_metrics.txt, train_info.json
       │
       ▼
  App loads model from models/ for Auto Classify (no training in app)
```

### 4.4 BiLSTM model (simplified)

```
  Input: (batch, 30, 22)
       │
       ▼
  Bidirectional LSTM(128)  →  BatchNorm  →  Dropout(0.4)
       │
       ▼
  Bidirectional LSTM(64)   →  BatchNorm  →  Dropout(0.4)
       │
       ▼
  Dense(4, softmax)  →  [push-up, squat, barbell biceps curl, shoulder press]
```

---

## 5. How many flows do we have? (Kaggle / no-Kaggle / demo) – how to start

We have **three** main ways to get a model and run the app.

### Flow 1: With Kaggle (real exercise videos – recommended)

**When:** You have Kaggle credentials and want to train on the real exercise dataset.

**One command (download + extract + sequences + train):**

```bash
# 1) Set up Kaggle once (see docs/CREDENTIALS.md): place kaggle.json in ~/.config/kaggle/
# 2) From project root:
python scripts/run_full_pipeline.py
streamlit run main.py
```

**Step-by-step (if you want to run each stage yourself):**

```bash
# Download dataset to data/kaggle_exercise/
python scripts/download_dataset.py --output_dir data/kaggle_exercise

# Extract 22-D features → one CSV per video in data/features/
python scripts/extract_features.py --video_dir data/kaggle_exercise --output_dir data/features

# Build 30-frame sequences → data/data_sequences.npz
python scripts/create_sequence_of_features.py --input_dir data/features --output data/data_sequences.npz

# Train BiLSTM (saves to models/)
python scripts/train_bidirectionallstm.py --data data/data_sequences.npz --output_dir models

# Run app
streamlit run main.py
```

---

### Flow 2: Without Kaggle (direct download or demo data)

**When:** No Kaggle; uses direct-download dataset if available, otherwise **synthetic demo data**.

**One command:**

```bash
python scripts/run_full_pipeline.py --no-kaggle
streamlit run main.py
```

- If `data/real_dataset/` (or the direct-download folder) has **videos** (.mp4/.avi/.mov), they are used.
- If there are **no videos**, the script uses **demo data** (synthetic sequences) and still trains a model so the app runs.

---

### Flow 3: Demo only (no download, no Kaggle)

**When:** You only want to see “baseline vs improved” or run the app without any dataset download.

```bash
# Generate synthetic sequences and train (saves improved model to models/)
python scripts/run_demo_training.py
streamlit run main.py
```

Or manually:

```bash
python scripts/generate_demo_data.py --output data/data_sequences.npz
python scripts/train_bidirectionallstm.py --data data/data_sequences.npz --output_dir models --source demo
streamlit run main.py
```

---

### Summary table

| Flow          | Command(s)                                        | Data used                        |
| ------------- | ------------------------------------------------- | -------------------------------- |
| **Kaggle**    | `python scripts/run_full_pipeline.py`             | `data/kaggle_exercise/` (videos) |
| **No Kaggle** | `python scripts/run_full_pipeline.py --no-kaggle` | `data/real_dataset/` or **demo** |
| **Demo only** | `python scripts/run_demo_training.py`             | **Synthetic** (no CSV/video)     |

**Start the app (same for all):**  
`streamlit run main.py`

---

## 6. If we use the dataset from Kaggle – process in detail (why CSV, train; small example)

### 6.1 Why do we generate CSV features?

- The **BiLSTM** needs **fixed-size numeric input**: sequences of **22-D vectors**, not raw video pixels.
- **MediaPipe** gives **body landmarks** (e.g. 33 points); we use **12** (shoulders, elbows, wrists, hips, knees, ankles) and compute:
  - **8 angles** (e.g. elbow angle: shoulder–elbow–wrist),
  - **12 normalized distances** (e.g. shoulder–shoulder, hip–knee),
  - **2 vertical distances** (e.g. elbow–shoulder in y).
- These **22 numbers per frame** are **invariant** to camera position and scale, so the model generalizes better.
- We **precompute** them once per video and save as **CSV** (one file per video): fast to reload, no need to run MediaPipe again during training. Training then only reads **CSVs → sequences → train**.

So: **Videos → Pose → 22-D per frame → CSV per video** so that the next step (sequences + training) is simple and reproducible.

### 6.2 End-to-end Kaggle process (with a small example)

**Step 0 – Kaggle credentials**

- Create `~/.config/kaggle/kaggle.json` with your API key (see `docs/CREDENTIALS.md`).

**Step 1 – Download**

```bash
python scripts/download_dataset.py --output_dir data/kaggle_exercise
```

- Dataset goes to `data/kaggle_exercise/`.
- Example structure:  
  `data/kaggle_exercise/similar_dataset/push-up/video1.mp4`,  
  `data/kaggle_exercise/similar_dataset/squat/video2.mp4`, …

**Step 2 – Extract features (why CSV)**

- For **each video**, we:
  - Read frames → run **MediaPipe Pose** → get 12 landmarks per frame.
  - Compute **22-D** (8 angles + 12 distances + 2 y-distances).
  - Write one CSV: `data/features/<videoname>_features.csv`.

Command:

```bash
python scripts/extract_features.py --video_dir data/kaggle_exercise --output_dir data/features
```

- **Example CSV** (`data/features/video1_features.csv`):

  ```text
  frame_idx,label,f0,f1,f2,...,f21
  0,push-up,45.2,120.1,...,0.3
  1,push-up,47.1,118.5,...,0.31
  ...
  ```

- **Label:** From folder name; if path is `.../similar_dataset/push-up/video1.mp4`, label is **push-up** (exercise-level label when folder matches).

**Step 3 – Build sequences**

- BiLSTM expects **30 consecutive frames** per sample.
- We **slide a window** of size 30 over each video’s feature rows and assign the **video label** to every window from that video.

Command:

```bash
python scripts/create_sequence_of_features.py --input_dir data/features --output data/data_sequences.npz
```

- **Example:**
  - Video has 100 frames → we get **71** windows (frames 0–29, 1–30, …, 70–99), all with the same label (e.g. push-up).
  - All videos’ windows are collected; labels are normalized (e.g. push-up, squat, barbell biceps curl, shoulder press).
  - Output: **data/data_sequences.npz** with:
    - `X`: shape `(N, 30, 22)` (N = total number of windows),
    - `y`: class index (0..3),
    - `labels`: list of 4 class names.

**Step 4 – Train**

```bash
python scripts/train_bidirectionallstm.py --data data/data_sequences.npz --output_dir models
```

- **Inside:**
  - Stratified split: e.g. 70% train, 15% val, 15% test.
  - StandardScaler fit on **train** only; apply to train/val/test.
  - Optional Gaussian noise on **train** sequences.
  - BiLSTM (128–64), BatchNorm, Dropout, L2; early stopping; ReduceLROnPlateau.
  - Best model by **validation accuracy**; evaluate on **test** set.
- **Outputs in `models/`:**
  - `final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5` (or .keras),
  - `thesis_bidirectionallstm_scaler.pkl`,
  - `thesis_bidirectionallstm_label_encoder.pkl`,
  - `train_metrics.txt` (test accuracy + classification report),
  - `train_info.json` (source, classes, optional class_index_to_exercise).

**Step 5 – Run app**

```bash
streamlit run main.py
```

- App loads model, scaler, and label encoder from `models/` and uses them in **Auto Classify** (no training in the app).

### 6.3 Small example summary

| Stage     | Input                | Output                                                 |
| --------- | -------------------- | ------------------------------------------------------ |
| Download  | —                    | `data/kaggle_exercise/` (videos)                       |
| Extract   | 1 video              | 1 CSV: rows = frames, cols = frame_idx, label, f0..f21 |
| Sequences | Many CSVs            | 1 file: `X (N, 30, 22)`, `y`, `labels`                 |
| Train     | `data_sequences.npz` | Model + scaler + label encoder + metrics in `models/`  |
| App       | `models/`            | Auto Classify uses model to predict and count reps     |

---

## 7. Quick reference – commands

| Goal                      | Command                                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Full pipeline (Kaggle)    | `python scripts/run_full_pipeline.py`                                                                        |
| Full pipeline (no Kaggle) | `python scripts/run_full_pipeline.py --no-kaggle`                                                            |
| Demo training only        | `python scripts/run_demo_training.py`                                                                        |
| Run app                   | `streamlit run main.py`                                                                                      |
| Download only             | `python scripts/download_dataset.py --output_dir data/kaggle_exercise`                                       |
| Extract only              | `python scripts/extract_features.py --video_dir data/kaggle_exercise --output_dir data/features`             |
| Sequences only            | `python scripts/create_sequence_of_features.py --input_dir data/features --output data/data_sequences.npz`   |
| Train only                | `python scripts/train_bidirectionallstm.py --data data/data_sequences.npz --output_dir models`               |
| Train 1D CNN              | `python scripts/train_bidirectionallstm.py --data data/data_sequences.npz --output_dir models --model cnn1d` |

Use this document to prepare your slides and answers for the major project presentation.

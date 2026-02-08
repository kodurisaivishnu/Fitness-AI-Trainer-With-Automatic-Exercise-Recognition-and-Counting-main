# Real-Time Fitness Exercise Recognition and Repetition Counting Using an Improved BiLSTM and Pose Estimation

**Draft for extension and submission**

---

## Abstract

We present a system for real-time fitness exercise classification and repetition counting from video. The pipeline combines pose estimation (MediaPipe) with a Bidirectional LSTM (BiLSTM) classifier that operates on sequences of invariant geometric features—joint angles and normalized distances—derived from body landmarks. Using 30-frame windows, the model classifies four exercises (squat, push-up, shoulder press, bicep curl) and drives rule-based repetition counters. We extend the approach of Riccio [1] with a deeper BiLSTM (128–64 units), BatchNormalization, feature-space augmentation, early stopping, and learning-rate scheduling. The system is integrated into a web application supporting video upload, webcam, and an “auto classify” mode. On the Real-Time Exercise Recognition dataset, the improved model achieves [X]% test accuracy (baseline: [Y]%), with improved robustness and training stability.

**Keywords:** Exercise recognition, repetition counting, BiLSTM, pose estimation, MediaPipe, real-time classification, data augmentation.

---

## 1. Introduction

Automated fitness coaching from video can improve adherence and form feedback. Two core tasks are (1) **exercise recognition**—identifying which exercise is being performed—and (2) **repetition counting**. Challenges include varying camera angles, lighting, and body types; real-time constraints; and the need for temporal modeling of movement.

We adopt a BiLSTM-based classifier that uses **invariant features** (joint angles and normalized distances) over **30-frame sequences**, improving robustness to camera and user variation [1]. We **improve upon the baseline** by (i) a deeper and wider BiLSTM (128–64 units per direction), (ii) BatchNormalization and tuned dropout, (iii) Gaussian noise augmentation in feature space, (iv) early stopping and learning-rate reduction on the validation set, and (v) a held-out test set for unbiased evaluation. The classifier output drives exercise-specific repetition logic based on joint angles. The full pipeline—dataset download, feature extraction, sequence building, and training—is documented and reproducible (see TRAINING.md and CREDENTIALS.md in the repository).

**Contributions:** (1) Reproducible training pipeline with Kaggle dataset download and credential documentation. (2) Improved BiLSTM architecture and training (augmentation, early stopping, LR schedule). (3) Label normalization so trained models align with the app’s Auto Classify mode. (4) Test-set metrics and saved artifacts for the paper and deployment.

---

## 2. Related Work

- **Pose-based exercise recognition:** Many systems use skeleton/landmark features rather than raw pixels. Recurrent models (LSTM/BiLSTM) are common for temporal modeling [1].
- **Repetition counting:** Typically rule-based (angle thresholds) or learned counters; we use angle-based rules per exercise type.
- **Riccio [1]:** BiLSTM on 30-frame sequences of angles and normalized distances; >99% test accuracy on four exercises; combined synthetic (InfiniteRep) and real (Kaggle) data.

See **REFERENCES.md** in this repository for BibTeX and links.

---

## 3. Method

### 3.1 Pose estimation and feature extraction

- **Pose:** MediaPipe Pose returns 33 body landmarks (x, y, z).
- **Landmarks used:** 12 keypoints (shoulders, elbows, wrists, hips, knees, ankles).
- **Features per frame (22-D):**
  - **8 angles:** e.g. left/right elbow angle (shoulder–elbow–wrist), knee angles (hip–knee–ankle), torso–limb angles.
  - **12 normalized distances:** e.g. shoulder–shoulder, hip–knee, wrist–shoulder; normalized by a body-length scale (e.g. shoulder–hip or hip–knee).
  - **2 y-coordinate distances:** elbow–shoulder (vertical), normalized.

This set is invariant to global position and scale, improving generalization [1].

### 3.2 Sequence-based classification (improved model)

- **Input:** Consecutive 30-frame windows of 22-D feature vectors, standardized per feature dimension (scaler fit on training set only).
- **Model (improved):**
  - Two Bidirectional LSTM layers with **128** and **64** units per direction (wider and deeper than the 64–64 baseline).
  - **BatchNormalization** after each BiLSTM layer to stabilize training.
  - **Dropout** (0.35) after each layer to reduce overfitting.
  - Dense output layer with softmax for four classes.
- **Training:**
  - **Stratified** train/validation/test split (e.g. 70% / 15% / 15%).
  - **Augmentation:** additive Gaussian noise (σ = 0.02) on the feature sequence for training samples only.
  - **Early stopping** on validation accuracy (patience 15); **ReduceLROnPlateau** on validation loss (factor 0.5, patience 7).
  - Best model (by validation accuracy) is saved and evaluated on the **held-out test set**.
- **Output:** Exercise class (squat, push-up, shoulder press, barbell biceps curl). Label names are normalized so they match the app’s Auto Classify logic.

### 3.3 Repetition counting

- For each classified exercise, rule-based counters use joint angles (e.g. elbow angle for bicep curl, knee angle for squat) and state machines (“up”/“down”) to increment rep count. Logic is implemented per exercise in the codebase.

### 3.4 System integration

- **Web app:** Streamlit; modes: Video (upload), WebCam, Auto Classify (BiLSTM + counter), Chatbot (fitness Q&A).
- **Auto Classify:** Live webcam → pose → 30-frame buffer → BiLSTM → exercise label → corresponding counter.

---

## 4. Experiments and results

### 4.1 Dataset

- **Primary:** Real-Time Exercise Recognition Dataset (Riccardo Riccio et al.) from Kaggle [2], containing four exercises: squat, push-up, shoulder press, bicep curl. Downloaded via Kaggle API (see CREDENTIALS.md).
- **Preprocessing:** Videos are processed with MediaPipe; 22-D features are extracted per frame; 30-frame sliding windows form sequences. Labels from folder names are normalized to the app’s class names.

### 4.2 Setup

- **Splits:** Stratified 70% train, 15% validation, 15% test.
- **Metrics:** Accuracy, precision, recall, F1 (per class and macro) on the **test set**.
- **Baseline:** Original 2-layer BiLSTM (64 units), no augmentation, no early stopping, 50 epochs.
- **Improved:** 2-layer BiLSTM (128, 64), BatchNorm, dropout 0.35, augmentation σ=0.02, early stopping (patience 15), ReduceLROnPlateau; best model by val accuracy.

### 4.3 Results

| Model    | Test accuracy | Macro F1 | Notes                   |
| -------- | ------------- | -------- | ----------------------- |
| Baseline | [Y]%          | [Y_f1]   | 64–64, no aug, 50 ep    |
| Improved | [X]%          | [X_f1]   | 128–64, aug, early stop |

_Table 1: Replace [X], [Y] with your run results. Run `scripts/train_bidirectionallstm.py` and use `models/train_metrics.txt` for the improved model. For baseline, run with `--lstm_units 64,64 --augment 0 --epochs 50` and no early stopping (or use the original script)._

**Classification report (improved model):** After training, see `models/train_metrics.txt` for the full per-class precision, recall, and F1. Example (fill with your numbers):

```
              precision  recall  f1-score  support
squat            0.xx     0.xx    0.xx      ...
push-up          0.xx     0.xx    0.xx      ...
shoulder press   0.xx     0.xx    0.xx      ...
barbell biceps curl 0.xx  0.xx    0.xx      ...
```

### 4.4 Ablations (optional)

- **Augmentation:** Training with `--augment 0.02` vs `--augment 0` to show gain from noise.
- **Architecture:** `--lstm_units 128,64` vs `64,64` to show gain from wider/deeper layers.
- **Early stopping:** Compare final test accuracy when using best checkpoint vs last epoch.

---

## 5. Conclusion

We implemented a real-time exercise recognition and repetition counting system based on BiLSTM and invariant pose features, building on [1]. We improved the model with a deeper BiLSTM (128–64), BatchNorm, dropout, feature augmentation, early stopping, and learning-rate scheduling, and we documented the full pipeline (dataset download, credentials, feature extraction, sequence building, training). The system runs in a web application with video upload, webcam, and auto-classify modes. Test-set evaluation and saved metrics support reproducibility and the write-up of the paper. Future work may include more exercises, synthetic data (e.g. InfiniteRep), and learned repetition counters.

---

## References

[1] R. Riccio, “Real-Time Fitness Exercise Classification and Counting from Video Frames,” arXiv:2411.11548 [cs.CV], 2024. https://arxiv.org/abs/2411.11548

[2] Riccardo Riccio et al., “Real-Time Exercise Recognition Dataset,” Kaggle. https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset

---

_This draft is in the `docs/` folder. Fill in [X], [Y] and the classification report with your training runs; see `docs/TRAINING.md` and `models/train_metrics.txt` after training._

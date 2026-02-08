# Personal AI Trainer With Automatic Exercise Recognition and Counting

This project is an AI-powered application that leverages Computer Vision, Pose Estimation, and Machine Learning to accurately track exercise repetitions during workouts. The goal is to enhance fitness routines by providing real-time tracking through an easy-to-use web interface.

Datasets available at "https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset".

This project is based on the paper: "Real-Time Fitness Exercise Classification and Counting from Video Frames"

Link to the paper: "https://arxiv.org/abs/2411.11548"

<!--
Feel free to Contact me at: riccardopersonalmail@gmail.com

LinkedIn: https://www.linkedin.com/in/riccardo-riccio-bb7163296/

(Give a star ⭐ to the repository if it was useful. Thank you! 😊)

## Demo

Watch the Fitness AI Coach in action:
[![Watch the video](https://img.youtube.com/vi/GPmDPB1bSmc/hqdefault.jpg)](https://www.youtube.com/watch?v=GPmDPB1bSmc)
-->

---

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Overview of the WebApp](#overview-of-the-webapp)
  - [App Navigation](#app-navigation)
- [Implementation Details](#implementation-details)
  - [Exercise Classifier](#exercise-classifier)
  - [Repetition Counting](#repetition-counting)
  - [Chatbot Integration](#chatbot-integration)
- [Technologies Used](#technologies-used)

---

# WARNING

1. In the current repository, the model used in the application is the "BiLSTM Invariant," which is a model trained without utilizing the raw (x, y, z) coordinates. Instead, it relies solely on angles and normalized distances. The best-performing model is described in the code file "train_bidirectionallstm.py", which incorporates both raw coordinates and angles.

2. In the current repository, the instructional videos are not included (except for shoulder_press_form.mp4). Consequently, after running the current repository, the other instructional videos will not be available.

## Project Structure

```
├── main.py                 # Entry point: runs the Streamlit app
├── requirements.txt        # Python dependencies
├── environment.yml        # Conda environment (see below)
├── src/                   # Core application modules
│   ├── ExerciseAiTrainer.py   # Exercise logic, BiLSTM classifier, repetition counting
│   ├── AiTrainer_utils.py     # Image resize, FPS, distance helpers
│   ├── PoseModule2.py         # MediaPipe pose detection
│   └── chatbot.py             # Fitness chatbot (OpenAI / LangChain)
├── scripts/               # Training and feature extraction
│   ├── download_dataset.py        # Download Kaggle dataset (needs credentials)
│   ├── extract_features.py       # Extract 22-D features from videos → CSV
│   ├── create_sequence_of_features.py  # Build 30-frame sequences → .npz
│   └── train_bidirectionallstm.py # Train improved BiLSTM → model + scaler + label encoder
├── models/                # Pre-trained artifacts (place .h5 and .pkl here)
│   ├── final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5
│   ├── thesis_bidirectionallstm_scaler.pkl
│   └── thesis_bidirectionallstm_label_encoder.pkl
├── assets/videos/         # Demo / form videos (e.g. demo_2.mp4, shoulder_press_form.mp4)
├── static/                # Web assets (e.g. styles.css)
└── docs/                  # Research references and paper draft
    ├── REFERENCES.md      # Links and BibTeX for the paper
    ├── CREDENTIALS.md    # Kaggle and OpenAI API setup
    ├── TRAINING.md       # Full pipeline: download → extract → train
    └── ResearchPaper_Draft.md  # Draft for your thesis/paper
```

### Download a dataset, train, and serve (one pipeline)

To **download a real dataset, train the model, and then run the app** with that model:

1. **Kaggle (real exercise videos):** Set up once: [docs/CREDENTIALS.md](docs/CREDENTIALS.md) (put `kaggle.json` in `~/.config/kaggle/`). Then:

   ```bash
   python scripts/run_full_pipeline.py
   streamlit run main.py
   ```

2. **Without Kaggle:** Tries a direct-download dataset; if none, uses demo data and trains:
   ```bash
   python scripts/run_full_pipeline.py --no-kaggle
   streamlit run main.py
   ```

The trained model is saved in `models/`; the app loads it automatically for **Auto Classify**.

**For your mentor:** Short answers to "Where did you get the data? What format? How did you train?" are in [docs/FOR_MENTOR_FAQ.md](docs/FOR_MENTOR_FAQ.md).  
**"Is the model trained from /data or pre-existing?"** → [docs/DATA_AND_MODEL.md](docs/DATA_AND_MODEL.md).

### Demo: show performance increase (no Kaggle)

To **see baseline vs improved performance** without using Kaggle or any credentials:

```bash
python scripts/run_demo_training.py
```

This generates a synthetic dataset, trains the **baseline** and **improved** models, and prints test accuracy for both. The improved model is saved to `models/` and used by the app. See [docs/DEMO.md](docs/DEMO.md).

### Improving model performance (train your own model)

To **download the dataset**, **retrain** with the improved pipeline, and **reproduce the paper results**:

1. **Credentials:** Set up Kaggle API (see [docs/CREDENTIALS.md](docs/CREDENTIALS.md)).
2. **Pipeline:** Follow [docs/TRAINING.md](docs/TRAINING.md):
   - `python scripts/download_dataset.py` → download dataset
   - `python scripts/extract_features.py --video_dir ...` → extract features
   - `python scripts/create_sequence_of_features.py --input_dir ...` → build sequences
   - `python scripts/train_bidirectionallstm.py --data data_sequences.npz` → train improved BiLSTM

The improved trainer uses a **deeper BiLSTM (128–64 units)**, **BatchNorm**, **augmentation**, **early stopping**, and **test-set evaluation**. Saved model/scaler/label encoder are **app-compatible** (drop them in `models/` and run the app). Results are written to `models/train_metrics.txt` for the research paper.

---

## Getting Started

### Prerequisites

- Python 3.7+ must be installed on your machine.
- It's recommended to use a virtual environment to manage dependencies.

### Installation

**Option A – Conda (recommended)**

1. Clone the repository and enter the project folder:

   ```bash
   git clone https://github.com/yourusername/Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting.git
   cd Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting
   ```

2. Create and activate the conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate fitness-ai-trainer
   ```

3. Install Python dependencies (if not already installed via env):

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app from the **project root**:
   ```bash
   streamlit run main.py
   ```

**Option B – venv**

1. Clone the repo and `cd` into it, then:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run main.py
   ```

## Overview of the WebApp

The Fitness AI Coach is a web application built with Streamlit, aimed at providing users with tools for fitness tracking, real-time exercise classification, repetition counting, and chatbot support.

### App Navigation

The main navigation sidebar allows users to access the following features:

1. **Video Analysis**: Upload exercise videos to count repetitions based on pose estimation.
2. **Webcam Mode**: Perform exercises in front of a webcam for real-time repetition counting.
3. **Auto Classify Mode**: Automatically identifies exercises in real-time and counts repetitions accordingly.
4. **Chatbot**: Acts as a fitness coach to provide fitness guidance using the OpenAI API.

The application is designed to be modular and user-friendly, with visual cues and instructional videos to assist users with exercise form and repetition counts.

## Implementation Details

### Exercise Classifier

The exercise classifier is built using a combination of real and synthetic datasets, including:

- **Kaggle Workout Dataset**: Real-world exercise videos.
- **InfiniteRep Dataset**: Synthetic videos of avatars performing exercises.
- **Similar Dataset**: Videos sourced from online to cover diverse exercise variations.

The classification model employs LSTM and BiLSTM networks to process body landmarks and classify exercises based on joint angles and movement patterns. The model was optimized using accuracy, precision, recall, and F1-score metrics.

### Repetition Counting

Repetition counting is implemented in two modes:

1. **Manual Mode**: Users manually select the exercise, and repetitions are counted using angle-based thresholds.
2. **Automatic Mode**: A BiLSTM model classifies exercises and applies counting logic based on identified body angles. The system tracks "up" and "down" movements to ensure accurate repetition counting.

### Chatbot Integration

The chatbot feature utilizes OpenAI's GPT-3.5-turbo model to answer fitness-related questions. It is integrated using LangChain’s ConversationChain to maintain context and provide meaningful responses. Users are advised to verify critical information with professionals as the chatbot may occasionally provide incorrect information.

## Technologies Used

- **Pose Estimation**: Utilizes MediaPipe to extract key body landmarks and monitor movement.
- **Machine Learning**: LSTM and BiLSTM models for real-time exercise classification.
- **Streamlit**: Provides the web interface for user interaction.
- **Python Libraries**: Includes OpenCV, MediaPipe, Streamlit, and others for backend processing.

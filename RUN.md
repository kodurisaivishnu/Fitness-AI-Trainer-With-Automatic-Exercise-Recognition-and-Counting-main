# How to Run - Fitness AI Coach

Complete guide to set up, run, and deploy the project.

---

## Prerequisites

- **Python 3.9 or 3.10** (recommended). Python 3.11+ may have TensorFlow compatibility issues.
- **Webcam** (for WebCam and Auto Classify modes when running locally)
- **Git** (to clone the repo)
- **Conda** (recommended) or **pip** with virtualenv

---

## Option A: Run Locally (Conda - Recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting.git
cd Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting-main
```

### Step 2: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate fitness-ai-trainer
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If `pyttsx3` (voice feedback) fails on Linux, also run:

```bash
sudo apt-get install espeak-ng libespeak-dev
```

### Step 4: Verify Model Files

The `models/` folder must contain these files (they come pre-included in the repo):

```
models/
  ├── final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.keras
  ├── final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5
  ├── thesis_bidirectionallstm_scaler.pkl
  ├── thesis_bidirectionallstm_label_encoder.pkl
  └── train_info.json
```

If these are missing, re-train using:

```bash
python scripts/run_demo_training.py
```

### Step 5: Set Up Environment Variables

Create a `.env` file in the project root (or edit the existing one):

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```
# Required only for downloading Kaggle datasets (training)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

# Optional: Enables AI-powered chatbot (works without it using built-in knowledge)
OPENAI_API_KEY=your_openai_api_key
```

**The app works fully without an OpenAI key** - the chatbot falls back to built-in fitness knowledge.

### Step 6: Run the App

```bash
streamlit run main.py
```

The terminal will show:

```
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open the **Local URL** in your browser.

---

## Option B: Run Locally (pip + virtualenv)

```bash
cd Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting-main

python3.10 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt

streamlit run main.py
```

---

## Option C: Run with Docker

### Step 1: Build the Docker Image

```bash
docker build -t fitness-ai-coach .
```

### Step 2: Run the Container

```bash
docker run -p 8501:8501 fitness-ai-coach
```

With OpenAI chatbot:

```bash
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key_here fitness-ai-coach
```

Open http://localhost:8501 in your browser.

---

## Option D: Deploy to Render (Cloud)

### Step 1: Push to GitHub

```bash
git add -A
git commit -m "Deploy to Render"
git push origin main
```

**Important:** Make sure `.env` is in `.gitignore` (it is by default) so your API keys are not exposed.

### Step 2: Create Render Web Service

1. Go to [render.com](https://render.com) and sign in
2. Click **New** -> **Web Service**
3. Connect your GitHub repository
4. Render auto-detects the `Dockerfile` and `render.yaml`
5. Settings will auto-fill:
   - **Name:** `fitness-ai-coach`
   - **Runtime:** Docker
   - **Plan:** Standard (minimum - TensorFlow needs ~2GB RAM)
   - **Port:** 8501

### Step 3: Set Environment Variables in Render Dashboard

Go to **Environment** tab and add:

| Key | Value | Required? |
|-----|-------|-----------|
| `OPENAI_API_KEY` | Your OpenAI key | Optional (chatbot works without it) |

### Step 4: Deploy

Click **Deploy**. First build takes 5-10 minutes (TensorFlow is large).

Your app will be live at: `https://fitness-ai-coach.onrender.com`

### How WebCam Works on Cloud

The app uses **WebRTC** (`streamlit-webrtc`) so webcam modes work even when deployed:

```
User's Browser (webcam) --WebRTC--> Render Server (AI processing) --WebRTC--> Browser (results)
```

The webcam video never leaves the WebRTC connection - it streams directly between the browser and server.

---

## Using the App

### 4 Modes

| Mode | What it does | How to use |
|------|-------------|------------|
| **Video** | Analyze uploaded video | Select exercise -> Upload MP4 -> See rep count + form tips |
| **WebCam** | Real-time webcam analysis | Select exercise -> Allow camera -> Exercise in front of camera |
| **Auto Classify** | AI detects exercise automatically | Allow camera -> Start exercising -> AI identifies exercise + counts reps |
| **Chatbot** | Fitness Q&A assistant | Type questions about exercises, nutrition, form, etc. |

### Sidebar Settings

- **Voice Feedback:** Toggle on/off - speaks form corrections aloud (local mode)
- **Your Weight (kg):** Used for calorie estimation
- **Model Status:** Shows loaded model accuracy and source

### During Exercise

- **Rep counter** (top-left): Shows current rep count and calories burned
- **Exercise name** (top-center): Current exercise detected
- **Form tips** (bottom, cyan): Suggestions like "Keep both arms even"
- **Injury alerts** (bottom, red): Warnings like "DANGER: Squat too deep!"
- **Workout Summary:** Shown after each session with total reps, calories, duration

### Auto Classify - Supported Exercises

The BiLSTM model recognizes 4 exercises:

1. **Push-up** (98.33% accuracy)
2. **Squat**
3. **Bicep Curl**
4. **Shoulder Press**

---

## Re-training the Model (Optional)

### Quick Demo Training (No Dataset Needed)

```bash
python scripts/run_demo_training.py
```

This generates synthetic data and trains a model for testing purposes.

### Full Training Pipeline (With Kaggle Dataset)

```bash
# Make sure KAGGLE_USERNAME and KAGGLE_KEY are set in .env

# Run the complete pipeline: download -> extract features -> train
python scripts/run_full_pipeline.py
```

### Manual Step-by-Step Training

```bash
# 1. Download exercise videos from Kaggle
python scripts/download_dataset.py

# 2. Extract 22-D pose features from videos
python scripts/extract_features.py --video_dir data/videos --output_dir data/features

# 3. Create 30-frame sliding window sequences
python scripts/create_sequence_of_features.py --input_dir data/features --output data_sequences.npz

# 4. Train BiLSTM model
python scripts/train_bidirectionallstm.py --data data_sequences.npz --output_dir models
```

Trained model files are saved to `models/`.

---

## Project Structure

```
├── main.py                    # Streamlit app entry point
├── src/
│   ├── ExerciseAiTrainer.py   # Core: pose detection, counting, form feedback, BiLSTM
│   ├── PoseModule2.py         # MediaPipe pose estimation wrapper
│   ├── AiTrainer_utils.py     # Image utilities (resize, FPS, distance)
│   ├── chatbot.py             # AI chatbot (OpenAI or built-in fallback)
│   └── webrtc_processor.py    # WebRTC video processors for cloud webcam
├── models/                    # Pre-trained BiLSTM model + scaler + encoder
├── scripts/                   # Training pipeline scripts
├── assets/videos/             # Demo videos
├── static/styles.css          # Custom CSS styling
├── .streamlit/config.toml     # Streamlit theme and server config
├── Dockerfile                 # Docker image for deployment
├── render.yaml                # Render.com deployment blueprint
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment
└── .env                       # API keys (not committed to git)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'mediapipe'` | Run `pip install mediapipe==0.10.14` |
| `Could not open camera` | Check webcam is connected and not used by another app |
| WebCam not working on cloud | Ensure `streamlit-webrtc` is installed; check browser allows camera access |
| Model not loading | Verify all files exist in `models/`. Re-run `python scripts/run_demo_training.py` |
| Chatbot not responding | Works without OpenAI key (built-in mode). For AI mode, set `OPENAI_API_KEY` in `.env` |
| `pyttsx3` error on Linux | Run `sudo apt-get install espeak-ng libespeak-dev` |
| TensorFlow GPU errors | The app runs on CPU by default. GPU is optional. |
| Render deploy fails (OOM) | Use **Standard plan** or higher (2GB+ RAM needed for TensorFlow) |
| Slow first prediction | First BiLSTM inference is slow (~2s). Subsequent predictions are fast. |

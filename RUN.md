# How to Run the Project (Conda)

## 1. Conda environment

Create the environment (if not already created):

```bash
cd /path/to/Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting-main
conda env create -f environment.yml
```

Activate and install Python dependencies:

```bash
conda activate fitness-ai-trainer
pip install -r requirements.txt
```

(List of envs: `conda env list` — you should see `fitness-ai-trainer`.)

## 2. Run the Streamlit app

From the **project root** (same folder as `main.py`):

```bash
conda activate fitness-ai-trainer
streamlit run main.py
```

Then open the URL shown in the terminal (e.g. http://localhost:8501).

## 3. Optional: Chatbot

For the fitness chatbot, set your OpenAI API key:

```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

## 4. Training pipeline (optional)

If you have exercise videos and want to re-train the BiLSTM:

```bash
conda activate fitness-ai-trainer

# Extract features from videos (e.g. folder with subdirs = labels, or single video)
python scripts/extract_features.py --video_dir path/to/videos --output_dir path/to/features

# Build 30-frame sequences
python scripts/create_sequence_of_features.py --input_dir path/to/features --output data_sequences.npz

# Train and save model to models/
python scripts/train_bidirectionallstm.py --data data_sequences.npz --output_dir models
```

## 5. Project layout (reminder)

- **models/** — Pre-trained `.h5` and `.pkl` (must be present to run the app).
- **assets/videos/** — Demo videos (`demo_2.mp4`, `shoulder_press_form.mp4`).
- **src/** — App code; **scripts/** — Training and feature extraction.

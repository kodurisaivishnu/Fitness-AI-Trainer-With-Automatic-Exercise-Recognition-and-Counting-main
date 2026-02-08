# Demo: Show Performance Increase (No Kaggle Needed)

You **don’t need to go to Kaggle or download anything manually**. You can run everything locally with a **synthetic demo dataset** and see the **baseline vs improved** performance.

---

## Option A: One command (recommended)

From the **project root**:

```bash
python scripts/run_demo_training.py
```

This will:

1. **Generate** a synthetic dataset (4 classes, 30-frame × 22-feature sequences).
2. **Train BASELINE** (2-layer BiLSTM 64–64, no augmentation, 30 epochs) and print **test accuracy**.
3. **Train IMPROVED** (2-layer BiLSTM 128–64, augmentation, early stopping) and print **test accuracy**.
4. **Save the improved model** into `models/` so the app uses it for Auto Classify.

Compare the two “Test accuracy” lines in the output to see the **performance increase**. No Kaggle account or credentials are required.

---

## Option B: Step by step

```bash
# 1. Generate demo data (no credentials)
python scripts/generate_demo_data.py --output data/data_sequences.npz

# 2. Train improved model (saves to models/)
python scripts/train_bidirectionallstm.py --data data/data_sequences.npz --output_dir models
```

Then run the app: `streamlit run main.py` and use **Auto Classify**.

---

## Option C: Use the real Kaggle dataset (optional)

If you want to train on the **real exercise videos** from Kaggle:

1. **One-time setup:** Go to [kaggle.com](https://www.kaggle.com) → sign in → **Account** → **Create New Token** (downloads `kaggle.json`).
2. Put the file where the script expects it:
   ```bash
   mkdir -p ~/.config/kaggle
   mv ~/Downloads/kaggle.json ~/.config/kaggle/
   chmod 600 ~/.config/kaggle/kaggle.json
   ```
3. Run the pipeline (see [TRAINING.md](TRAINING.md)):
   ```bash
   python scripts/download_dataset.py --output_dir data/kaggle_exercise
   python scripts/extract_features.py --video_dir data/kaggle_exercise --output_dir data/features
   python scripts/create_sequence_of_features.py --input_dir data/features --output data/data_sequences.npz
   python scripts/train_bidirectionallstm.py --data data/data_sequences.npz --output_dir models
   ```

---

**Summary:** Use **Option A** if you just want to show a performance increase without Kaggle. Use **Option C** when you want to train on real videos and cite the Kaggle dataset in your paper.

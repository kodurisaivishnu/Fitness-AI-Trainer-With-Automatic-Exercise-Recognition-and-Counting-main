# Credentials for Dataset and Optional Services

## Kaggle (for downloading the training dataset)

To download the **Real-Time Exercise Recognition** dataset used for training, you need a Kaggle API token.

### 1. Get your API token

1. Log in at [Kaggle](https://www.kaggle.com).
2. Open **Account** → [API](https://www.kaggle.com/settings) → **Create New Token**.
3. This downloads `kaggle.json` (contains `username` and `key`).

### 2. Configure the API

**Option A – Easiest (.env in project root)**  
Create a `.env` file in the **project root** (same folder as `main.py`). The download script loads it automatically — no need to create `~/.config/kaggle/` or export variables.

```bash
# From project root: copy the example and edit
cp .env.example .env
# Edit .env and set:
# KAGGLE_USERNAME=your_kaggle_username
# KAGGLE_KEY=your_kaggle_api_key
```

Or create `.env` manually with:

```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

**Option B – Local file (kaggle.json)**  
Place the file so the Kaggle CLI can find it:

- **Linux:** `~/.config/kaggle/kaggle.json` or `~/.kaggle/kaggle.json`
- **macOS:** `~/.kaggle/kaggle.json`
- **Windows:** `C:\Users\<You>\.kaggle\kaggle.json`

```bash
mkdir -p ~/.config/kaggle
mv ~/Downloads/kaggle.json ~/.config/kaggle/
chmod 600 ~/.config/kaggle/kaggle.json
```

**Option C – Environment variables**  
Set before running (or use Option A with `.env`):

```bash
export KAGGLE_USERNAME="your-kaggle-username"
export KAGGLE_KEY="your-kaggle-api-key"
```

### 3. Install and run

```bash
pip install kaggle
python scripts/download_dataset.py --output_dir data/kaggle_exercise
```

The dataset will be extracted under `data/kaggle_exercise/`. See **docs/TRAINING.md** for the full training pipeline.

---

## OpenAI (for the fitness chatbot)

The in-app chatbot uses OpenAI’s API. To enable it:

1. Create an API key at [OpenAI API keys](https://platform.openai.com/api-keys).
2. In the project root, create a `.env` file (this file is typically in `.gitignore`):

```bash
OPENAI_API_KEY=sk-your-key-here
```

The app loads this via `python-dotenv`. Do **not** commit `.env` or your API key.

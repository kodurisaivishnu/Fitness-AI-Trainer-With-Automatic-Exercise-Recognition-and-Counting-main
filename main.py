import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import cv2
import tempfile
import ExerciseAiTrainer as exercise
from chatbot import chat_ui
import time

ASSETS_VIDEOS = ROOT / "assets" / "videos"
DEMO_VIDEO = ASSETS_VIDEOS / "demo_2.mp4"
MODELS_DIR = ROOT / "models"


def _read_accuracy_from_metrics_file(path):
    if not path.exists():
        return None
    try:
        text = path.read_text()
        m = re.search(r"Test accuracy:\s*([\d.]+)", text)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _show_model_metrics_in_sidebar():
    """Show current vs previous model metrics in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model status")
    info_path = MODELS_DIR / "train_info.json"
    metrics_path = MODELS_DIR / "train_metrics.txt"
    metrics_prev_path = MODELS_DIR / "train_metrics_previous.txt"
    model_h5 = MODELS_DIR / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5"
    model_keras = MODELS_DIR / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.keras"
    has_model = model_h5.exists() or model_keras.exists()
    if not has_model:
        st.sidebar.warning("No model in **models/**.\nRun training first:\n- `python scripts/run_full_pipeline.py` (Kaggle)\n- or `python scripts/run_full_pipeline.py --no-kaggle`\n- or `python scripts/run_demo_training.py`")
        return
    # Show which file the app loads (.keras takes precedence over .h5; see docs/DATA_AND_MODEL.md)
    loaded_file = model_keras.name if model_keras.exists() else model_h5.name
    st.sidebar.success("Model loaded from **models/**")
    st.sidebar.caption(f"**Model file:** `{loaded_file}`")
    source = "pre-trained / unknown"
    architecture = None
    current_acc = None
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            source = info.get("source", source)
            architecture = info.get("architecture")
            current_acc = info.get("test_accuracy")
        except Exception:
            pass
    current_acc = current_acc or _read_accuracy_from_metrics_file(metrics_path)
    previous_acc = _read_accuracy_from_metrics_file(metrics_prev_path)
    if architecture:
        st.sidebar.caption(f"**Architecture:** {architecture}")
    st.sidebar.caption(f"**Source:** {source}")
    if current_acc is not None:
        st.sidebar.metric("Current model (test acc.)", f"{current_acc * 100:.2f}%")
    if previous_acc is not None and current_acc is not None:
        diff = (current_acc - previous_acc) * 100
        st.sidebar.metric("Previous model (test acc.)", f"{previous_acc * 100:.2f}%", delta=f"{diff:+.2f}%" if diff != 0 else None)
    elif previous_acc is not None:
        st.sidebar.caption(f"Previous: {previous_acc * 100:.2f}%")
    st.sidebar.markdown("---")


def main():
    st.set_page_config(page_title='Fitness AI Coach', layout='centered')

    css_path = ROOT / "static" / "styles.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.title('Fitness AI Coach')
    _show_model_metrics_in_sidebar()
    options = st.sidebar.selectbox('Select Option', ('Video', 'WebCam', 'Auto Classify', 'chatbot'))

    if options == 'chatbot':
        st.caption("The chatbot can make mistakes. Check important info.")
        chat_ui()

    if options == 'Video':
        exercise_options = st.sidebar.selectbox(
            'Select Exercise', ('Bicept Curl', 'Push Up', 'Squat', 'Shoulder Press')
        )
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "asf", "m4v"])

        cap = None
        video_path_for_display = None

        if video_file_buffer:
            tfflie = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfflie.write(video_file_buffer.read())
            tfflie.close()
            video_path_for_display = tfflie.name
            cap = cv2.VideoCapture(video_path_for_display)
        elif DEMO_VIDEO.exists():
            video_path_for_display = str(DEMO_VIDEO)
            cap = cv2.VideoCapture(video_path_for_display)
        else:
            st.info("Upload a video or add demo at assets/videos/demo_2.mp4")

        if video_path_for_display:
            st.sidebar.video(video_path_for_display)
            st.video(video_path_for_display)

        if cap is not None and cap.isOpened():
            if exercise_options == 'Bicept Curl':
                exer = exercise.Exercise()
                exer.bicept_curl(cap, is_video=True, counter=0, stage_right=None, stage_left=None)
            elif exercise_options == 'Push Up':
                exer = exercise.Exercise()
                exer.push_up(cap, is_video=True, counter=0, stage=None)
            elif exercise_options == 'Squat':
                exer = exercise.Exercise()
                exer.squat(cap, is_video=True, counter=0, stage=None)
            elif exercise_options == 'Shoulder Press':
                exer = exercise.Exercise()
                exer.shoulder_press(cap, is_video=True, counter=0, stage=None)

    elif options == 'Auto Classify':
        st.caption("Join hands in front of camera to stop.")
        if st.button('Start Auto Classification'):
            time.sleep(1)
            exer = exercise.Exercise()
            exer.auto_classify_and_count()

    elif options == 'WebCam':
        exercise_general = st.sidebar.selectbox(
            'Select Exercise', ('Bicept Curl', 'Push Up', 'Squat', 'Shoulder Press')
        )
        st.caption("Click Start. Webcam runs until you close or refresh the page.")
        start_button = st.button('Start Exercise')

        if start_button:
            time.sleep(1)
            exer = exercise.Exercise()
            if exercise_general == 'Bicept Curl':
                exer.bicept_curl(None, counter=0, stage_right=None, stage_left=None)
            elif exercise_general == 'Push Up':
                exer.push_up(None, counter=0, stage=None)
            elif exercise_general == 'Squat':
                exer.squat(None, counter=0, stage=None)
            elif exercise_general == 'Shoulder Press':
                exer.shoulder_press(None, counter=0, stage=None)


if __name__ == '__main__':
    main()

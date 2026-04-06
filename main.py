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

# WebRTC for browser webcam (works on cloud deployments)
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    from webrtc_processor import WebRTCExerciseProcessor, WebRTCAutoClassifyProcessor
    _WEBRTC_AVAILABLE = True
except ImportError:
    _WEBRTC_AVAILABLE = False

ASSETS_VIDEOS = ROOT / "assets" / "videos"
DEMO_VIDEO = ASSETS_VIDEOS / "demo_2.mp4"
MODELS_DIR = ROOT / "models"

# TURN/STUN servers for WebRTC (needed for cloud deployment behind NAT)
RTC_CONFIGURATION = None
if _WEBRTC_AVAILABLE:
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]}
    )

EXERCISE_NAME_MAP = {
    'Bicep Curl': 'barbell biceps curl',
    'Push Up': 'push-up',
    'Squat': 'squat',
    'Shoulder Press': 'shoulder press',
}


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
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Status")
    info_path = MODELS_DIR / "train_info.json"
    metrics_path = MODELS_DIR / "train_metrics.txt"
    metrics_prev_path = MODELS_DIR / "train_metrics_previous.txt"
    model_h5 = MODELS_DIR / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5"
    model_keras = MODELS_DIR / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.keras"
    has_model = model_h5.exists() or model_keras.exists()
    if not has_model:
        st.sidebar.warning("No model in **models/**.\nRun training first:\n- `python scripts/run_full_pipeline.py`\n- or `python scripts/run_demo_training.py`")
        return
    loaded_file = model_keras.name if model_keras.exists() else model_h5.name
    st.sidebar.success("Model loaded")
    st.sidebar.caption(f"**File:** `{loaded_file}`")
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
        st.sidebar.metric("Test Accuracy", f"{current_acc * 100:.2f}%")
    if previous_acc is not None and current_acc is not None:
        diff = (current_acc - previous_acc) * 100
        st.sidebar.metric("Previous Accuracy", f"{previous_acc * 100:.2f}%", delta=f"{diff:+.2f}%" if diff != 0 else None)
    st.sidebar.markdown("---")


def _webrtc_exercise_mode(exercise_display_name):
    """WebCam mode using WebRTC - works on cloud deployments."""
    exercise_canonical = EXERCISE_NAME_MAP.get(exercise_display_name, "push-up")

    ctx = webrtc_streamer(
        key=f"exercise-{exercise_canonical}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=WebRTCExerciseProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.set_exercise(exercise_canonical)

        # Live stats display below video
        stats_placeholder = st.empty()
        while ctx.state.playing:
            state = ctx.video_processor.get_state()
            with stats_placeholder.container():
                cols = st.columns(4)
                cols[0].metric("Reps", state["counter"])
                cols[1].metric("Calories", f"{state['calories']:.1f}")
                cols[2].metric("Duration", f"{state['duration'] / 60:.1f} min")
                cols[3].metric("Exercise", exercise.canonical_to_display_name(state["exercise"]))

                if state["injury"]:
                    st.error(f"**{state['injury']}**")
                if state["tip"]:
                    st.info(f"**Tip:** {state['tip']}")
            time.sleep(0.5)

        # Show final summary
        if ctx.video_processor:
            final = ctx.video_processor.get_state()
            if final["counter"] > 0:
                st.markdown("---")
                st.subheader("Workout Summary")
                cols = st.columns(3)
                cols[0].metric("Total Reps", final["counter"])
                cols[1].metric("Calories Burned", f"{final['calories']:.1f} cal")
                cols[2].metric("Duration", f"{final['duration'] / 60:.1f} min")


def _webrtc_auto_classify_mode():
    """Auto Classify mode using WebRTC."""
    ctx = webrtc_streamer(
        key="auto-classify",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=WebRTCAutoClassifyProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        if not ctx.video_processor.model_ready:
            st.error("Model not loaded. Ensure model files are in the **models/** directory.")
            return

        stats_placeholder = st.empty()
        while ctx.state.playing:
            state = ctx.video_processor.get_state()
            with stats_placeholder.container():
                cols = st.columns(4)
                cols[0].metric("Detected Exercise", state["prediction"])
                cols[1].metric("Total Reps", state["total_reps"])
                cols[2].metric("Calories", f"{state['total_calories']}")
                cols[3].metric("Duration", f"{state['duration'] / 60:.1f} min")

                if state["injury"]:
                    st.error(f"**{state['injury']}**")
                if state["tip"]:
                    st.info(f"**Tip:** {state['tip']}")

                if state["breakdown"]:
                    st.markdown("**Live breakdown:**")
                    for name, data in state["breakdown"].items():
                        st.write(f"- **{name}**: {data['reps']} reps ({data['calories']} cal)")
            time.sleep(0.5)

        # Final summary
        if ctx.video_processor:
            final = ctx.video_processor.get_state()
            if final["total_reps"] > 0:
                st.markdown("---")
                st.subheader("Workout Summary")
                cols = st.columns(3)
                cols[0].metric("Total Reps", final["total_reps"])
                cols[1].metric("Calories Burned", f"{final['total_calories']} cal")
                cols[2].metric("Duration", f"{final['duration'] / 60:.1f} min")
                if final["breakdown"]:
                    for name, data in final["breakdown"].items():
                        st.write(f"- **{name}**: {data['reps']} reps ({data['calories']} cal)")


def main():
    st.set_page_config(page_title='Fitness AI Coach', layout='centered', page_icon="💪")

    css_path = ROOT / "static" / "styles.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.title('Fitness AI Coach')
    st.caption("AI-powered exercise recognition, counting, and form feedback")

    _show_model_metrics_in_sidebar()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    voice_enabled = st.sidebar.checkbox("Voice Feedback", value=True, help="Speak form corrections and injury alerts aloud (local mode only)")
    user_weight = st.sidebar.number_input("Your Weight (kg)", min_value=30, max_value=200, value=70, step=1, help="Used for calorie estimation")

    options = st.sidebar.selectbox('Select Mode', ('Video', 'WebCam', 'Auto Classify', 'Chatbot'))

    if options == 'Chatbot':
        st.caption("Ask fitness questions - works with or without OpenAI API key.")
        chat_ui()

    elif options == 'Video':
        exercise_options = st.sidebar.selectbox(
            'Select Exercise', ('Bicep Curl', 'Push Up', 'Squat', 'Shoulder Press')
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
            exer = exercise.Exercise()
            exer.voice.enabled = voice_enabled
            if exercise_options == 'Bicep Curl':
                exer.bicept_curl(cap, is_video=True, counter=0, stage_right=None, stage_left=None)
            elif exercise_options == 'Push Up':
                exer.push_up(cap, is_video=True, counter=0, stage=None)
            elif exercise_options == 'Squat':
                exer.squat(cap, is_video=True, counter=0, stage=None)
            elif exercise_options == 'Shoulder Press':
                exer.shoulder_press(cap, is_video=True, counter=0, stage=None)

    elif options == 'WebCam':
        exercise_general = st.sidebar.selectbox(
            'Select Exercise', ('Bicep Curl', 'Push Up', 'Squat', 'Shoulder Press')
        )

        if _WEBRTC_AVAILABLE:
            st.caption("Your browser webcam is used via WebRTC. Allow camera access when prompted.")
            _webrtc_exercise_mode(exercise_general)
        else:
            # Fallback to local cv2.VideoCapture (only works when running locally)
            st.caption("Click Start. Webcam runs until you close or refresh the page.")
            st.warning("For cloud deployment, install `streamlit-webrtc` for browser webcam support.")
            start_button = st.button('Start Exercise', type="primary")
            if start_button:
                time.sleep(1)
                exer = exercise.Exercise()
                exer.voice.enabled = voice_enabled
                canonical = EXERCISE_NAME_MAP.get(exercise_general, "push-up")
                if exercise_general == 'Bicep Curl':
                    exer.bicept_curl(None, counter=0, stage_right=None, stage_left=None)
                elif exercise_general == 'Push Up':
                    exer.push_up(None, counter=0, stage=None)
                elif exercise_general == 'Squat':
                    exer.squat(None, counter=0, stage=None)
                elif exercise_general == 'Shoulder Press':
                    exer.shoulder_press(None, counter=0, stage=None)

    elif options == 'Auto Classify':
        if _WEBRTC_AVAILABLE:
            st.caption("The AI will automatically detect which exercise you're doing and count reps.")
            _webrtc_auto_classify_mode()
        else:
            # Fallback to local mode
            st.caption("Join hands in front of camera to stop.")
            st.warning("For cloud deployment, install `streamlit-webrtc` for browser webcam support.")
            if st.button('Start Auto Classification', type="primary"):
                time.sleep(1)
                exer = exercise.Exercise()
                exer.voice.enabled = voice_enabled
                exer.auto_classify_and_count()


if __name__ == '__main__':
    main()

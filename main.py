import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title='Fitness AI Coach', layout='centered', page_icon="💪")

import cv2
import tempfile
import time

# Lazy imports - don't load TF/mediapipe until needed
_exercise_module = None
def _get_exercise():
    global _exercise_module
    if _exercise_module is None:
        import ExerciseAiTrainer as ex
        _exercise_module = ex
    return _exercise_module

from chatbot import chat_ui

# WebRTC
_WEBRTC_AVAILABLE = False
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    _WEBRTC_AVAILABLE = True
except Exception:
    pass

ASSETS_VIDEOS = ROOT / "assets" / "videos"
DEMO_VIDEO = ASSETS_VIDEOS / "demo_2.mp4"
MODELS_DIR = ROOT / "models"

def _get_rtc_configuration():
    """Build RTC config from env vars. TURN server required for cloud deployment."""
    if not _WEBRTC_AVAILABLE:
        return None
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    # Read TURN credentials from env (set in Render dashboard)
    turn_url = os.environ.get("TURN_URL", "")
    turn_user = os.environ.get("TURN_USERNAME", "")
    turn_cred = os.environ.get("TURN_CREDENTIAL", "")
    if turn_url and turn_user and turn_cred:
        ice_servers.append({
            "urls": [turn_url],
            "username": turn_user,
            "credential": turn_cred,
        })
        # Also add TCP and TLS variants
        if ":" in turn_url:
            base = turn_url.split("?")[0]
            ice_servers.append({"urls": [f"{base}?transport=tcp"], "username": turn_user, "credential": turn_cred})
            tls_url = base.replace("turn:", "turns:", 1)
            if ":443" not in tls_url:
                tls_url = tls_url.rsplit(":", 1)[0] + ":443"
            ice_servers.append({"urls": [f"{tls_url}?transport=tcp"], "username": turn_user, "credential": turn_cred})
    return RTCConfiguration({"iceServers": ice_servers})

RTC_CONFIGURATION = _get_rtc_configuration()
_TURN_CONFIGURED = bool(os.environ.get("TURN_URL"))

def _show_turn_setup_help():
    """Show instructions to set up TURN server for WebCam on cloud."""
    st.warning("**WebCam needs a TURN server to work on cloud deployment.**")
    st.markdown("""
**Free setup (1 minute):**
1. Go to [metered.ca/tools/openrelay](https://www.metered.ca/tools/openrelay/) and sign up free
2. Copy your **TURN URL**, **Username**, and **Credential** from the dashboard
3. In Render dashboard → **Environment** → add these 3 variables:

| Key | Example Value |
|-----|---------------|
| `TURN_URL` | `turn:global.relay.metered.ca:80` |
| `TURN_USERNAME` | `your_username_from_metered` |
| `TURN_CREDENTIAL` | `your_credential_from_metered` |

4. Redeploy - WebCam will work instantly!

**Meanwhile, use Video mode** - upload any exercise video and the AI will analyze it.
""")


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
    model_h5 = MODELS_DIR / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5"
    model_keras = MODELS_DIR / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.keras"
    has_model = model_h5.exists() or model_keras.exists()
    if not has_model:
        st.sidebar.warning("No model found. Run training first.")
        return
    st.sidebar.success("Model ready")
    info_path = MODELS_DIR / "train_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            acc = info.get("test_accuracy")
            if acc:
                st.sidebar.metric("Accuracy", f"{acc * 100:.2f}%")
        except Exception:
            pass
    st.sidebar.markdown("---")


def _webrtc_exercise_mode(exercise_display_name):
    exercise_canonical = EXERCISE_NAME_MAP.get(exercise_display_name, "push-up")
    from webrtc_processor import WebRTCExerciseProcessor

    ctx = webrtc_streamer(
        key=f"exercise-{exercise_canonical}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=WebRTCExerciseProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.set_exercise(exercise_canonical)

    if ctx.state.playing and ctx.video_processor:
        exercise = _get_exercise()
        state = ctx.video_processor.get_state()
        cols = st.columns(4)
        cols[0].metric("Reps", state["counter"])
        cols[1].metric("Calories", f"{state['calories']:.1f}")
        cols[2].metric("Duration", f"{state['duration'] / 60:.1f} min")
        cols[3].metric("Exercise", exercise.canonical_to_display_name(state["exercise"]))
        if state["injury"]:
            st.error(f"**{state['injury']}**")
        if state["tip"]:
            st.info(f"**Tip:** {state['tip']}")


def _webrtc_auto_classify_mode():
    from webrtc_processor import WebRTCAutoClassifyProcessor

    ctx = webrtc_streamer(
        key="auto-classify",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=WebRTCAutoClassifyProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing and ctx.video_processor:
        if not ctx.video_processor.model_ready:
            st.error("Model not loaded. Check **models/** directory.")
            return
        state = ctx.video_processor.get_state()
        cols = st.columns(4)
        cols[0].metric("Detected", state["prediction"])
        cols[1].metric("Total Reps", state["total_reps"])
        cols[2].metric("Calories", f"{state['total_calories']}")
        cols[3].metric("Duration", f"{state['duration'] / 60:.1f} min")
        if state["injury"]:
            st.error(f"**{state['injury']}**")
        if state["tip"]:
            st.info(f"**Tip:** {state['tip']}")
        if state["breakdown"]:
            for name, data in state["breakdown"].items():
                st.write(f"- **{name}**: {data['reps']} reps ({data['calories']} cal)")


def main():
    css_path = ROOT / "static" / "styles.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.title('Fitness AI Coach')
    st.caption("AI-powered exercise recognition, counting, and form feedback")

    _show_model_metrics_in_sidebar()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    voice_enabled = st.sidebar.checkbox("Voice Feedback", value=False, help="Local mode only")
    user_weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70, step=1)

    options = st.sidebar.selectbox('Mode', ('Video', 'WebCam', 'Auto Classify', 'Chatbot'))

    if options == 'Chatbot':
        chat_ui()

    elif options == 'Video':
        exercise_options = st.sidebar.selectbox(
            'Exercise', ('Bicep Curl', 'Push Up', 'Squat', 'Shoulder Press')
        )
        video_file_buffer = st.sidebar.file_uploader("Upload video", type=["mp4", "mov", "avi", "asf", "m4v"])

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
            exercise = _get_exercise()
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
            'Exercise', ('Bicep Curl', 'Push Up', 'Squat', 'Shoulder Press')
        )

        if _WEBRTC_AVAILABLE and _TURN_CONFIGURED:
            st.caption("Allow camera access when prompted. Click START to begin.")
            _webrtc_exercise_mode(exercise_general)
        elif _WEBRTC_AVAILABLE and not _TURN_CONFIGURED:
            _show_turn_setup_help()
        else:
            st.caption("Click Start. Webcam runs until you refresh.")
            if st.button('Start Exercise', type="primary"):
                time.sleep(1)
                exercise = _get_exercise()
                exer = exercise.Exercise()
                exer.voice.enabled = voice_enabled
                if exercise_general == 'Bicep Curl':
                    exer.bicept_curl(None, counter=0, stage_right=None, stage_left=None)
                elif exercise_general == 'Push Up':
                    exer.push_up(None, counter=0, stage=None)
                elif exercise_general == 'Squat':
                    exer.squat(None, counter=0, stage=None)
                elif exercise_general == 'Shoulder Press':
                    exer.shoulder_press(None, counter=0, stage=None)

    elif options == 'Auto Classify':
        if _WEBRTC_AVAILABLE and _TURN_CONFIGURED:
            st.caption("AI automatically detects your exercise and counts reps. Click START.")
            _webrtc_auto_classify_mode()
        elif _WEBRTC_AVAILABLE and not _TURN_CONFIGURED:
            _show_turn_setup_help()
        else:
            st.caption("Join hands to stop.")
            if st.button('Start Auto Classification', type="primary"):
                time.sleep(1)
                exercise = _get_exercise()
                exer = exercise.Exercise()
                exer.voice.enabled = voice_enabled
                exer.auto_classify_and_count()


if __name__ == '__main__':
    main()

import os
import cv2
import PoseModule2 as pm
import numpy as np
import streamlit as st
from AiTrainer_utils import *
import joblib
import mediapipe as mp
import time
import json
import threading
from pathlib import Path

# Use TFLite interpreter - try multiple packages (only need ~2-30MB, not full TF 400MB)
_TFLITE_AVAILABLE = False
tflite = None
try:
    # Option 1: tflite-runtime (official lightweight package)
    import tflite_runtime.interpreter as _tfl
    tflite = _tfl
    _TFLITE_AVAILABLE = True
except ImportError:
    try:
        # Option 2: ai-edge-litert (Google's new replacement)
        from ai_edge_litert import interpreter as _tfl
        tflite = _tfl
        _TFLITE_AVAILABLE = True
    except ImportError:
        try:
            # Option 3: Full TF (fallback, uses more RAM)
            import tensorflow as tf
            tflite = tf.lite
            _TFLITE_AVAILABLE = True
        except ImportError:
            pass

# Optional voice feedback
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

# Project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"


class LabelEncoderCompat:
    """Compatible with pickles saved by train script (expects .classes_)."""
    def __init__(self, classes):
        self.classes_ = np.array(classes)

_MODEL_TFLITE = _MODELS_DIR / "exercise_classifier.tflite"
_SCALER_PKL = _MODELS_DIR / "thesis_bidirectionallstm_scaler.pkl"
_LABEL_ENCODER_PKL = _MODELS_DIR / "thesis_bidirectionallstm_label_encoder.pkl"

# MediaPipe pose references (no global Pose() instance - saves ~50MB)
mp_pose = mp.solutions.pose

# Define relevant landmarks indices
relevant_landmarks_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value
]

# --- Calorie estimation (MET values per exercise) ---
# MET = Metabolic Equivalent of Task
EXERCISE_MET = {
    'push-up': 8.0,
    'squat': 5.5,
    'barbell biceps curl': 3.5,
    'shoulder press': 4.0,
}
# Average seconds per rep (approximate)
EXERCISE_SEC_PER_REP = {
    'push-up': 3.0,
    'squat': 4.0,
    'barbell biceps curl': 3.5,
    'shoulder press': 3.0,
}

def estimate_calories(exercise_name, rep_count, user_weight_kg=70):
    """Estimate calories burned: MET * weight_kg * time_hours * 1.05"""
    met = EXERCISE_MET.get(exercise_name, 4.0)
    sec_per_rep = EXERCISE_SEC_PER_REP.get(exercise_name, 3.5)
    time_hours = (rep_count * sec_per_rep) / 3600.0
    return met * user_weight_kg * time_hours * 1.05


# --- Voice Feedback Engine ---
class VoiceFeedback:
    """Non-blocking voice feedback using pyttsx3 in a background thread."""
    def __init__(self, enabled=True):
        self.enabled = enabled and _TTS_AVAILABLE
        self._lock = threading.Lock()
        self._speaking = False
        self._last_spoken = ""
        self._last_spoken_time = 0
        self._cooldown = 4.0  # seconds between voice tips

    def speak(self, text):
        if not self.enabled or not text:
            return
        now = time.time()
        with self._lock:
            if self._speaking:
                return
            if text == self._last_spoken and (now - self._last_spoken_time) < self._cooldown:
                return
            self._speaking = True
            self._last_spoken = text
            self._last_spoken_time = now

        def _do_speak():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)
                engine.setProperty('volume', 0.9)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                pass
            finally:
                with self._lock:
                    self._speaking = False

        t = threading.Thread(target=_do_speak, daemon=True)
        t.start()


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    if np.any(np.array([a, b, c]) == 0):
        return -1.0
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def calculate_y_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0
    return np.abs(a[1] - b[1])

def draw_styled_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.55, font_color=(255, 255, 255), font_thickness=2, bg_color=(0, 0, 0), padding=5):
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x, text_y = position
    box_coords = ((text_x - padding, text_y + padding), (text_x + text_size[0] + padding, text_y - text_size[1] - padding))
    cv2.rectangle(frame, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)


def _angle_from_landmarks(landmark_list, p1, p2, p3):
    """Angle at p2 (landmark index) between p1-p2-p3, in degrees [0, 360)."""
    if not landmark_list or max(p1, p2, p3) >= len(landmark_list):
        return -1.0
    x1, y1 = landmark_list[p1][1], landmark_list[p1][2]
    x2, y2 = landmark_list[p2][1], landmark_list[p2][2]
    x3, y3 = landmark_list[p3][1], landmark_list[p3][2]
    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle


# --- Injury Prevention Alerts ---
def get_injury_alerts(landmark_list, exercise_name):
    """Return a critical injury prevention warning if dangerous form is detected."""
    if not landmark_list or len(landmark_list) < 29:
        return None

    if exercise_name == "squat":
        # Check if knees go too far past toes (knee x vs ankle x from front view)
        left_knee = landmark_list[25][1:]  # x, y
        left_ankle = landmark_list[27][1:]
        right_knee = landmark_list[26][1:]
        right_ankle = landmark_list[28][1:]
        # Check knee angle - too deep squat is dangerous
        left_knee_angle = _angle_from_landmarks(landmark_list, 23, 25, 27)
        right_knee_angle = _angle_from_landmarks(landmark_list, 24, 26, 28)
        if left_knee_angle >= 0 and right_knee_angle >= 0:
            if left_knee_angle < 50 or right_knee_angle < 50:
                return "DANGER: Squat too deep! Risk of knee injury"
        # Check if back is rounding (shoulder-hip-knee alignment)
        left_back_angle = _angle_from_landmarks(landmark_list, 11, 23, 25)
        if left_back_angle >= 0 and (left_back_angle < 60 or left_back_angle > 300):
            return "WARNING: Keep your back straight!"

    elif exercise_name == "push-up":
        # Check hip sag (shoulder-hip-ankle should be roughly straight)
        left_body = _angle_from_landmarks(landmark_list, 11, 23, 27)
        right_body = _angle_from_landmarks(landmark_list, 12, 24, 28)
        if left_body >= 0 and right_body >= 0:
            # Very bent body = hip sag
            avg_body = (left_body + right_body) / 2
            if 120 < avg_body < 150:
                return "WARNING: Hips sagging! Engage your core"

    elif exercise_name == "shoulder press":
        # Check for excessive back arch
        left_align = _angle_from_landmarks(landmark_list, 11, 23, 25)
        if left_align >= 0 and left_align < 150 and left_align > 0:
            return "WARNING: Don't arch your back too much"

    return None


def get_form_suggestions(landmark_list, exercise_name):
    """Return a short live form suggestion only when form is clearly wrong."""
    if not landmark_list or len(landmark_list) < 29:
        return None
    suggestion = None

    if exercise_name == "push-up":
        left_arm = _angle_from_landmarks(landmark_list, 11, 13, 15)
        right_arm = _angle_from_landmarks(landmark_list, 12, 14, 16)
        if left_arm >= 0 and right_arm >= 0:
            diff = abs(left_arm - right_arm)
            if diff > 360 - diff:
                diff = 360 - diff
            if diff > 55:
                suggestion = "Keep both arms even"

    elif exercise_name == "squat":
        left_knee = _angle_from_landmarks(landmark_list, 23, 25, 27)
        right_knee = _angle_from_landmarks(landmark_list, 24, 26, 28)
        if left_knee >= 0 and right_knee >= 0:
            if left_knee > 175 and right_knee > 175:
                suggestion = "Bend your knees more - go lower"
            elif (left_knee < 65 or right_knee < 65) and (left_knee >= 0 and right_knee >= 0):
                suggestion = "Don't squat too deep - protect your knees"
            elif abs(left_knee - right_knee) > 35:
                suggestion = "Keep both knees at similar depth"

    elif exercise_name == "shoulder press":
        left_arm = _angle_from_landmarks(landmark_list, 11, 13, 15)
        right_arm = _angle_from_landmarks(landmark_list, 12, 14, 16)
        if left_arm >= 0 and right_arm >= 0:
            diff = abs(left_arm - right_arm)
            if diff > 360 - diff:
                diff = 360 - diff
            if diff > 55:
                suggestion = "Press both arms evenly"

    elif exercise_name == "barbell biceps curl":
        left_arm = _angle_from_landmarks(landmark_list, 11, 13, 15)
        right_arm = _angle_from_landmarks(landmark_list, 12, 14, 16)
        if left_arm >= 0 and right_arm >= 0:
            diff = abs(left_arm - right_arm)
            if diff > 360 - diff:
                diff = 360 - diff
            if diff > 55:
                suggestion = "Curl both arms together"

    return suggestion


def count_repetition_push_up(detector, img, landmark_list, stage, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    right_shoulder = landmark_list[12][1:]
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    left_shoulder = landmark_list[11][1:]
    exercise_instance.visualize_angle(img, right_arm_angle, right_shoulder)
    exercise_instance.visualize_angle(img, left_arm_angle, left_shoulder)

    if left_arm_angle < 220:
        stage = "down"
    if left_arm_angle > 240 and stage == "down":
        stage = "up"
        counter += 1

    return stage, counter


def count_repetition_squat(detector, img, landmark_list, stage, counter, exercise_instance):
    right_leg_angle = detector.find_angle(img, 24, 26, 28)
    left_leg_angle = detector.find_angle(img, 23, 25, 27)
    right_leg = landmark_list[26][1:]
    exercise_instance.visualize_angle(img, right_leg_angle, right_leg)

    if right_leg_angle > 160 and left_leg_angle < 220:
        stage = "down"
    if right_leg_angle < 140 and left_leg_angle > 210 and stage == "down":
        stage = "up"
        counter += 1

    return stage, counter

def count_repetition_bicep_curl(detector, img, landmark_list, stage_right, stage_left, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    exercise_instance.visualize_angle(img, right_arm_angle, landmark_list[14][1:])
    exercise_instance.visualize_angle(img, left_arm_angle, landmark_list[13][1:])

    if right_arm_angle > 160 and right_arm_angle < 200:
        stage_right = "down"
    if left_arm_angle < 200 and left_arm_angle > 140:
        stage_left = "down"

    if stage_right == "down" and (right_arm_angle > 310 or right_arm_angle < 60) and (left_arm_angle > 310 or left_arm_angle < 60) and stage_left == "down":
        stage_right = "up"
        stage_left = "up"
        counter += 1

    return stage_right, stage_left, counter

def count_repetition_shoulder_press(detector, img, landmark_list, stage, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    right_elbow = landmark_list[14][1:]
    exercise_instance.visualize_angle(img, right_arm_angle, right_elbow)

    if right_arm_angle > 280 and left_arm_angle < 80:
        stage = "down"
    if right_arm_angle < 240 and left_arm_angle > 120 and stage == "down":
        stage = "up"
        counter += 1

    return stage, counter


class Exercise:
    def __init__(self):
        self.lstm_model = None       # TFLite interpreter
        self._input_details = None
        self._output_details = None
        self._load_error = []
        self.voice = VoiceFeedback(enabled=True)
        self.session_start = None
        self.session_data = {}

        # Load TFLite model (~400KB, uses ~30MB RAM vs ~400MB for full TF)
        if _TFLITE_AVAILABLE and _MODEL_TFLITE.exists():
            try:
                self.lstm_model = tflite.Interpreter(model_path=str(_MODEL_TFLITE))
                self.lstm_model.allocate_tensors()
                self._input_details = self.lstm_model.get_input_details()
                self._output_details = self.lstm_model.get_output_details()
            except Exception as e:
                self.lstm_model = None
                self._load_error.append(("model", str(e)))
        elif not _MODEL_TFLITE.exists():
            self._load_error.append(("model", f"TFLite model not found at {_MODEL_TFLITE}"))
        elif not _TFLITE_AVAILABLE:
            self._load_error.append(("model", "tflite-runtime not installed"))

        try:
            self.scaler = joblib.load(str(_SCALER_PKL))
        except Exception as e:
            self.scaler = None
            self._load_error.append(("scaler", str(e)))

        try:
            import sys
            main_mod = sys.modules.get("__main__")
            if main_mod is not None:
                setattr(main_mod, "LabelEncoderCompat", LabelEncoderCompat)
            self.label_encoder = joblib.load(str(_LABEL_ENCODER_PKL))
            raw = self.label_encoder.classes_
            self.exercise_classes = [str(x) for x in (raw.tolist() if hasattr(raw, "tolist") else raw)]
        except Exception as e:
            self.label_encoder = None
            self.exercise_classes = []
            self._load_error.append(("label_encoder", str(e)))

    def extract_features(self, landmarks):
        features = []
        if len(landmarks) == len(relevant_landmarks_indices) * 3:
            features.append(calculate_angle(landmarks[0:3], landmarks[6:9], landmarks[12:15]))
            features.append(calculate_angle(landmarks[3:6], landmarks[9:12], landmarks[15:18]))
            features.append(calculate_angle(landmarks[18:21], landmarks[24:27], landmarks[30:33]))
            features.append(calculate_angle(landmarks[21:24], landmarks[27:30], landmarks[33:36]))
            features.append(calculate_angle(landmarks[0:3], landmarks[18:21], landmarks[24:27]))
            features.append(calculate_angle(landmarks[3:6], landmarks[21:24], landmarks[27:30]))
            features.append(calculate_angle(landmarks[18:21], landmarks[0:3], landmarks[6:9]))
            features.append(calculate_angle(landmarks[21:24], landmarks[3:6], landmarks[9:12]))

            distances = [
                calculate_distance(landmarks[0:3], landmarks[3:6]),
                calculate_distance(landmarks[18:21], landmarks[21:24]),
                calculate_distance(landmarks[18:21], landmarks[24:27]),
                calculate_distance(landmarks[21:24], landmarks[27:30]),
                calculate_distance(landmarks[0:3], landmarks[18:21]),
                calculate_distance(landmarks[3:6], landmarks[21:24]),
                calculate_distance(landmarks[6:9], landmarks[24:27]),
                calculate_distance(landmarks[9:12], landmarks[27:30]),
                calculate_distance(landmarks[12:15], landmarks[0:3]),
                calculate_distance(landmarks[15:18], landmarks[3:6]),
                calculate_distance(landmarks[12:15], landmarks[18:21]),
                calculate_distance(landmarks[15:18], landmarks[21:24])
            ]

            y_distances = [
                calculate_y_distance(landmarks[6:9], landmarks[0:3]),
                calculate_y_distance(landmarks[9:12], landmarks[3:6])
            ]

            normalization_factor = -1
            distances_to_check = [
                calculate_distance(landmarks[0:3], landmarks[18:21]),
                calculate_distance(landmarks[3:6], landmarks[21:24]),
                calculate_distance(landmarks[18:21], landmarks[24:27]),
                calculate_distance(landmarks[21:24], landmarks[27:30])
            ]

            for distance in distances_to_check:
                if distance > 0:
                    normalization_factor = distance
                    break

            if normalization_factor == -1:
                normalization_factor = 0.5

            normalized_distances = [d / normalization_factor if d != -1.0 else d for d in distances]
            normalized_y_distances = [d / normalization_factor if d != -1.0 else d for d in y_distances]

            features.extend(normalized_distances)
            features.extend(normalized_y_distances)
        else:
            features = [-1.0] * 22
        return features

    def preprocess_frame(self, frame, pose):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        landmarks = []
        if results.pose_landmarks:
            for idx in relevant_landmarks_indices:
                landmark = results.pose_landmarks.landmark[idx]
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks

    def visualize_angle(self, img, angle, landmark):
        h, w = img.shape[:2]
        if len(landmark) >= 2:
            x, y = float(landmark[0]), float(landmark[1])
            if 0 <= x <= 1 and 0 <= y <= 1:
                pt = (int(x * w), int(y * h))
            else:
                pt = (int(x), int(y))
            pt = (max(0, min(pt[0], w - 1)), max(0, min(pt[1], h - 1)))
        else:
            pt = (0, 0)
        cv2.putText(img, str(int(angle)), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def predict(self, input_data):
        """Run inference using TFLite. input_data shape: (1, 30, 22)."""
        if self.lstm_model is None:
            return None
        self.lstm_model.set_tensor(self._input_details[0]['index'], input_data)
        self.lstm_model.invoke()
        return self.lstm_model.get_tensor(self._output_details[0]['index'])

    def _update_session(self, exercise_name, counter):
        """Track session data for summary."""
        if exercise_name not in self.session_data:
            self.session_data[exercise_name] = {'reps': 0, 'calories': 0.0}
        self.session_data[exercise_name]['reps'] = counter
        self.session_data[exercise_name]['calories'] = estimate_calories(exercise_name, counter)

    def get_session_summary(self):
        """Return session summary dict."""
        total_reps = sum(d['reps'] for d in self.session_data.values())
        total_cal = sum(d['calories'] for d in self.session_data.values())
        duration = time.time() - self.session_start if self.session_start else 0
        return {
            'exercises': dict(self.session_data),
            'total_reps': total_reps,
            'total_calories': round(total_cal, 1),
            'duration_sec': round(duration, 1),
        }

    def auto_classify_and_count(self):
        if self.lstm_model is None or self.scaler is None or self.label_encoder is None:
            msg = "Model not loaded. Ensure files in **models/** are present (model .h5 or .keras, scaler .pkl, label_encoder .pkl)."
            if self._load_error:
                details = "; ".join(f"{c}: {e}" for c, e in self._load_error)
                msg += f" Load error(s): {details}"
            st.error(msg)
            return

        self.session_start = time.time()
        stframe = st.empty()
        status_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open camera.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        frame_interval = 1.0 / 30
        last_draw = time.time()

        window_size = 30
        landmarks_window = []
        frame_count = 0
        current_prediction = "No prediction yet"
        canonical_prediction = None
        last_predicted_class = -1
        prediction_history = []
        stable_exercise = None
        tip_cooldown_sec = 2.0
        last_tip_text = None
        last_tip_time = 0.0
        last_injury_text = None
        last_injury_time = 0.0
        counters = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        stages = {'push_up': None, 'squat': None, 'left_bicep_curl': None, 'right_bicep_curl': None, 'shoulder_press': None}
        known_exercises = {'push-up', 'squat', 'shoulder press', 'barbell biceps curl'}
        canonical_by_index = ['push-up', 'squat', 'barbell biceps curl', 'shoulder press']
        class_name_to_exercise = {}

        try:
            info_path = _MODELS_DIR / "train_info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text())
                if isinstance(info.get("class_index_to_exercise"), list) and len(info["class_index_to_exercise"]) == len(self.exercise_classes):
                    canonical_by_index = info["class_index_to_exercise"]
                if isinstance(info.get("class_to_exercise"), dict):
                    class_name_to_exercise = info["class_to_exercise"]
        except Exception:
            pass
        canonical_to_display = {'push-up': 'Push-up', 'squat': 'Squat', 'barbell biceps curl': 'Curl', 'shoulder press': 'Press'}
        counter_key_map = {'push-up': 'push_up', 'squat': 'squat', 'barbell biceps curl': 'bicep_curl', 'shoulder press': 'shoulder_press'}

        detector = pm.posture_detector()
        pose_local = mp.solutions.pose.Pose()

        try:
            while True:
                try:
                    ret, frame = cap.read()
                except Exception:
                    break
                if not ret or frame is None:
                    break

                landmarks = self.preprocess_frame(frame, pose_local)
                if len(landmarks) == len(relevant_landmarks_indices) * 3:
                    features = self.extract_features(landmarks)
                    if len(features) == 22:
                        landmarks_window.append(features)

                frame_count += 1

                if len(landmarks_window) == window_size:
                    landmarks_window_np = np.array(landmarks_window, dtype=np.float32)
                    scaled_landmarks_window = self.scaler.transform(landmarks_window_np)
                    scaled_landmarks_window = scaled_landmarks_window.reshape(1, window_size, 22).astype(np.float32)

                    prediction = self.predict(scaled_landmarks_window)

                    if prediction.shape[1] != len(self.exercise_classes):
                        break
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    if predicted_class >= len(self.exercise_classes):
                        break
                    current_prediction = self.exercise_classes[predicted_class]
                    last_predicted_class = predicted_class
                    if current_prediction in known_exercises:
                        canonical_prediction = current_prediction
                    elif current_prediction in class_name_to_exercise and class_name_to_exercise[current_prediction] in known_exercises:
                        canonical_prediction = class_name_to_exercise[current_prediction]
                    elif len(canonical_by_index) == len(self.exercise_classes) and predicted_class < len(canonical_by_index):
                        canonical_prediction = canonical_by_index[predicted_class]
                    else:
                        canonical_prediction = current_prediction if current_prediction in known_exercises else None
                    prediction_history.append(canonical_prediction)
                    if len(prediction_history) > 3:
                        prediction_history.pop(0)
                    if len(prediction_history) >= 2 and prediction_history[-1] == prediction_history[-2] and canonical_prediction in known_exercises:
                        stable_exercise = canonical_prediction
                    else:
                        stable_exercise = None

                    landmarks_window = []
                    frame_count = 0

                # Repetition counting + injury detection
                detector.find_person(frame, draw=True)
                landmark_list = detector.find_landmarks(frame, draw=True)
                if len(landmark_list) > 0:
                    # Hand-join detection to stop
                    if self.are_hands_joined(landmark_list, stop=False):
                        break

                    exercise_for_count = canonical_prediction if canonical_prediction in known_exercises else None
                    if exercise_for_count == 'push-up':
                        stages['push_up'], counters['push_up'] = count_repetition_push_up(detector, frame, landmark_list, stages['push_up'], counters['push_up'], self)
                        self._update_session('push-up', counters['push_up'])
                    elif exercise_for_count == 'squat':
                        stages['squat'], counters['squat'] = count_repetition_squat(detector, frame, landmark_list, stages['squat'], counters['squat'], self)
                        self._update_session('squat', counters['squat'])
                    elif exercise_for_count == 'barbell biceps curl':
                        stages['right_bicep_curl'], stages['left_bicep_curl'], counters['bicep_curl'] = count_repetition_bicep_curl(detector, frame, landmark_list, stages['right_bicep_curl'], stages['left_bicep_curl'], counters['bicep_curl'], self)
                        self._update_session('barbell biceps curl', counters['bicep_curl'])
                    elif exercise_for_count == 'shoulder press':
                        stages['shoulder_press'], counters['shoulder_press'] = count_repetition_shoulder_press(detector, frame, landmark_list, stages['shoulder_press'], counters['shoulder_press'], self)
                        self._update_session('shoulder press', counters['shoulder_press'])

                    # Injury prevention alert (highest priority)
                    now = time.time()
                    injury_alert = None
                    if exercise_for_count:
                        injury_alert = get_injury_alerts(landmark_list, exercise_for_count)
                    if injury_alert:
                        if injury_alert != last_injury_text or (now - last_injury_time) >= 3.0:
                            last_injury_text = injury_alert
                            last_injury_time = now
                            self.voice.speak(injury_alert)

                    # Form tip (lower priority, only when stable)
                    form_tip = None
                    if stable_exercise and stable_exercise in known_exercises:
                        form_tip = get_form_suggestions(landmark_list, stable_exercise)
                    if form_tip is None:
                        last_tip_text = None
                    elif form_tip != last_tip_text:
                        if last_tip_text is None or (now - last_tip_time) >= tip_cooldown_sec:
                            last_tip_text = form_tip
                            last_tip_time = now
                            if not injury_alert:
                                self.voice.speak(form_tip)

                exercise_name_map = {
                    'push_up': 'Push-up',
                    'squat': 'Squat',
                    'bicep_curl': 'Curl',
                    'shoulder_press': 'Press'
                }

                height, width, _ = frame.shape
                num_exercises = len(counters)
                vertical_spacing = height // (num_exercises + 1)

                cv2.rectangle(frame, (0, 0), (140, height), (0, 0, 0), -1)
                cv2.rectangle(frame, (0, 0), (width, 30), (0, 0, 0), -1)

                display_name = canonical_to_display.get(canonical_prediction, "Detecting...") if canonical_prediction else "Detecting..."
                draw_styled_text(frame, f"Exercise: {display_name}", ((width - 290) // 2 + 100, 20))

                for idx, (exercise, count) in enumerate(counters.items()):
                    short_name = exercise_name_map.get(exercise, exercise)
                    draw_styled_text(frame, f"{short_name}: {count}", (10, (idx + 1) * vertical_spacing))

                # Show calorie estimate
                total_cal = sum(d['calories'] for d in self.session_data.values())
                draw_styled_text(frame, f"Cal: {total_cal:.1f}", (10, height - 15), font_scale=0.45, font_color=(0, 255, 0), bg_color=(30, 30, 30))

                # Draw injury alert (red, prominent)
                if last_injury_text and (time.time() - last_injury_time) < 3.0:
                    draw_styled_text(frame, last_injury_text, (130, height - 60), font_scale=0.55, font_color=(0, 0, 255), bg_color=(255, 255, 255))

                # Draw form tip
                if last_tip_text:
                    draw_styled_text(frame, f"Tip: {last_tip_text}", (130, height - 35), font_scale=0.55, font_color=(0, 255, 255), bg_color=(40, 40, 40))

                if time.time() - last_draw >= frame_interval:
                    try:
                        stframe.image(frame, channels='BGR', use_column_width=True)
                    except Exception:
                        pass
                    last_draw = time.time()
                time.sleep(0.01)
        except Exception:
            pass
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            cv2.destroyAllWindows()

        # Show session summary
        self._show_session_summary(status_placeholder)

    def _show_session_summary(self, placeholder=None):
        """Display workout session summary in Streamlit."""
        summary = self.get_session_summary()
        if summary['total_reps'] == 0:
            return
        target = placeholder if placeholder else st
        target.markdown("---")
        target.subheader("Workout Summary")
        cols = target.columns(3)
        cols[0].metric("Total Reps", summary['total_reps'])
        cols[1].metric("Calories Burned", f"{summary['total_calories']} cal")
        duration_min = summary['duration_sec'] / 60
        cols[2].metric("Duration", f"{duration_min:.1f} min")

        if summary['exercises']:
            target.markdown("**Breakdown by exercise:**")
            for name, data in summary['exercises'].items():
                display = canonical_to_display_name(name)
                target.write(f"- **{display}**: {data['reps']} reps ({data['calories']:.1f} cal)")

    def are_hands_joined(self, landmark_list, stop, is_video=False):
        if len(landmark_list) < 17:
            return False
        left_wrist = landmark_list[15][1:]
        right_wrist = landmark_list[16][1:]
        distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
        if distance < 30 and not is_video:
            return True
        return False

    def repetitions_counter(self, img, counter, exercise_name=None):
        cv2.rectangle(img, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.putText(img, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        # Show calories inline
        if exercise_name:
            cal = estimate_calories(exercise_name, counter)
            cv2.putText(img, f'{cal:.1f} cal', (120, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    def push_up(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_push_up, counter=counter, stage=stage, exercise_name="push-up")

    def squat(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_squat, counter=counter, stage=stage, exercise_name="squat")

    def bicept_curl(self, cap, is_video=False, counter=0, stage_right=None, stage_left=None):
        self.exercise_method(cap, is_video, count_repetition_bicep_curl, multi_stage=True, counter=counter, stage_right=stage_right, stage_left=stage_left, exercise_name="barbell biceps curl")

    def shoulder_press(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_shoulder_press, counter=counter, stage=stage, exercise_name="shoulder press")

    def exercise_method(self, cap, is_video, count_repetition_function, multi_stage=False, counter=0, stage=None, stage_right=None, stage_left=None, exercise_name=None):
        self.session_start = time.time()
        if is_video:
            stframe = st.empty()
            summary_placeholder = st.empty()
            detector = pm.posture_detector()

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            if original_fps <= 0:
                original_fps = 30.0  # fallback FPS
            frame_time = 1 / original_fps

            frame_count = 0
            start_time = time.time()
            last_update_time = start_time
            update_interval = 0.1
            img = None

            while cap.isOpened():
                current_time = time.time()
                elapsed_time = current_time - start_time
                target_frame = int(elapsed_time * original_fps)

                while frame_count < target_frame:
                    ret, frame = cap.read()
                    if not ret:
                        # Show summary before returning
                        if exercise_name:
                            self._update_session(exercise_name, counter)
                        self._show_session_summary(summary_placeholder)
                        return

                    frame_count += 1

                    if frame_count == target_frame:
                        img = detector.find_person(frame)
                        landmark_list = detector.find_landmarks(img, draw=False)

                        if len(landmark_list) != 0:
                            if multi_stage:
                                stage_right, stage_left, counter = count_repetition_function(detector, img, landmark_list, stage_right, stage_left, counter, self)
                            else:
                                stage, counter = count_repetition_function(detector, img, landmark_list, stage, counter, self)

                            if self.are_hands_joined(landmark_list, stop=False, is_video=is_video):
                                if exercise_name:
                                    self._update_session(exercise_name, counter)
                                self._show_session_summary(summary_placeholder)
                                return

                            # Injury alert
                            if exercise_name:
                                injury = get_injury_alerts(landmark_list, exercise_name)
                                if injury:
                                    h, w = img.shape[:2]
                                    draw_styled_text(img, injury, (10, h - 55), font_scale=0.5, font_color=(0, 0, 255), bg_color=(255, 255, 255))
                                    self.voice.speak(injury)

                            # Form suggestion
                            if exercise_name:
                                form_tip = get_form_suggestions(landmark_list, exercise_name)
                                if form_tip:
                                    h, w = img.shape[:2]
                                    draw_styled_text(img, f"Tip: {form_tip}", (10, h - 25), font_scale=0.5, font_color=(0, 255, 255), bg_color=(40, 40, 40))
                                    self.voice.speak(form_tip)

                        self.repetitions_counter(img, counter, exercise_name)

                if img is not None and current_time - last_update_time >= update_interval:
                    stframe.image(img, channels='BGR', use_column_width=True)
                    last_update_time = current_time

                time.sleep(0.001)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            if exercise_name:
                self._update_session(exercise_name, counter)
            self._show_session_summary(summary_placeholder)

        else:
            # Webcam mode
            stframe = st.empty()
            summary_placeholder = st.empty()
            webcam_start_time = time.time()
            if cap is None:
                cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open camera.")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            # Warmup
            for _ in range(60):
                cap.read()
                time.sleep(0.033)
            detector = pm.posture_detector()
            frame_interval = 1.0 / 30
            last_update = time.time()
            read_retries = 15
            last_injury_text = None
            last_injury_time = 0.0

            try:
                while cap.isOpened():
                    ret, frame = None, None
                    for _ in range(read_retries):
                        try:
                            ret, frame = cap.read()
                        except Exception:
                            time.sleep(0.05)
                            continue
                        if ret and frame is not None and frame.size > 0:
                            break
                        time.sleep(0.03)
                    if not ret or frame is None or frame.size == 0:
                        break
                    now = time.time()
                    if now - last_update < frame_interval:
                        time.sleep(0.01)
                        continue
                    last_update = now

                    try:
                        img = detector.find_person(frame)
                        landmark_list = detector.find_landmarks(img, draw=False)

                        if len(landmark_list) != 0:
                            if multi_stage:
                                stage_right, stage_left, counter = count_repetition_function(detector, img, landmark_list, stage_right, stage_left, counter, self)
                            else:
                                stage, counter = count_repetition_function(detector, img, landmark_list, stage, counter, self)

                            # Injury prevention
                            if exercise_name:
                                injury = get_injury_alerts(landmark_list, exercise_name)
                                if injury:
                                    if injury != last_injury_text or (now - last_injury_time) >= 3.0:
                                        last_injury_text = injury
                                        last_injury_time = now
                                        self.voice.speak(injury)
                                    h, w = img.shape[:2]
                                    draw_styled_text(img, injury, (10, h - 55), font_scale=0.5, font_color=(0, 0, 255), bg_color=(255, 255, 255))

                            # Form tips
                            if exercise_name:
                                form_tip = get_form_suggestions(landmark_list, exercise_name)
                                if form_tip:
                                    h, w = img.shape[:2]
                                    draw_styled_text(img, f"Tip: {form_tip}", (10, h - 25), font_scale=0.5, font_color=(0, 255, 255), bg_color=(40, 40, 40))
                                    if not last_injury_text or (now - last_injury_time) >= 3.0:
                                        self.voice.speak(form_tip)

                        self.repetitions_counter(img, counter, exercise_name)
                        stframe.image(img, channels='BGR', use_column_width=True)
                    except Exception:
                        continue
            except Exception:
                pass
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cv2.destroyAllWindows()
                if exercise_name:
                    self._update_session(exercise_name, counter)
                self._show_session_summary(summary_placeholder)
                try:
                    ran_sec = time.time() - webcam_start_time
                    if ran_sec < 3:
                        stframe.warning("Camera not ready or stopped quickly. Click **Start Exercise** to try again.")
                except Exception:
                    pass


def canonical_to_display_name(name):
    """Convert canonical exercise name to display name."""
    mapping = {
        'push-up': 'Push-up',
        'squat': 'Squat',
        'barbell biceps curl': 'Bicep Curl',
        'shoulder press': 'Shoulder Press',
    }
    return mapping.get(name, name.title())


# --- Cached model loading (call from main.py / webrtc_processor.py) ---
_cached_exercise = None

def get_cached_exercise():
    """Return a singleton Exercise instance so the model loads only once."""
    global _cached_exercise
    if _cached_exercise is None:
        _cached_exercise = Exercise()
    return _cached_exercise

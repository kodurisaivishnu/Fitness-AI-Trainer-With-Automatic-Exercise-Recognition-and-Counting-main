import cv2
import PoseModule2 as pm
import numpy as np
import streamlit as st
from AiTrainer_utils import *
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
import mediapipe as mp
import time
import json
from pathlib import Path

# Project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"


class LabelEncoderCompat:
    """Compatible with pickles saved by train script (expects .classes_)."""
    def __init__(self, classes):
        self.classes_ = np.array(classes)
_MODEL_H5 = _MODELS_DIR / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5"
_MODEL_KERAS = _MODELS_DIR / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.keras"
_SCALER_PKL = _MODELS_DIR / "thesis_bidirectionallstm_scaler.pkl"
_LABEL_ENCODER_PKL = _MODELS_DIR / "thesis_bidirectionallstm_label_encoder.pkl"

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

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

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    if np.any(np.array([a, b, c]) == 0):
        return -1.0  # Placeholder for missing landmarks
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to calculate Euclidean distance between two points
def calculate_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0  # Placeholder for missing landmarks
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

# Function to calculate Y-coordinate distance between two points
def calculate_y_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0  # Placeholder for missing landmarks
    return np.abs(a[1] - b[1])

def draw_styled_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.55, font_color=(255, 255, 255), font_thickness=2, bg_color=(0, 0, 0), padding=5):
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x, text_y = position
    box_coords = ((text_x - padding, text_y + padding), (text_x + text_size[0] + padding, text_y - text_size[1] - padding))
    cv2.rectangle(frame, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)


def _angle_from_landmarks(landmark_list, p1, p2, p3):
    """Angle at p2 (landmark index) between p1-p2-p3, in degrees [0, 360). Returns -1 if missing."""
    if not landmark_list or max(p1, p2, p3) >= len(landmark_list):
        return -1.0
    x1, y1 = landmark_list[p1][1], landmark_list[p1][2]
    x2, y2 = landmark_list[p2][1], landmark_list[p2][2]
    x3, y3 = landmark_list[p3][1], landmark_list[p3][2]
    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle


def get_form_suggestions(landmark_list, exercise_name):
    """
    Return a short live form suggestion only when form is clearly wrong.
    Conservative thresholds to avoid wrong tips; only one suggestion at a time.
    """
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
    right_wrist = landmark_list[16][1:]
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



# Define the class that handles the analysis of the exercises
class Exercise:
    def __init__(self):
        self.lstm_model = None
        self._load_error = []  # list of (component, error_message) for debugging

        # Try .keras first (Keras 3 loads it reliably); then .h5
        for path in (_MODEL_KERAS, _MODEL_H5):
            if not path.exists():
                continue
            last_err = None
            try:
                self.lstm_model = load_model(str(path), compile=False)
            except Exception as e:
                last_err = e
                try:
                    self.lstm_model = load_model(str(path))
                except Exception as e2:
                    last_err = e2
                    try:
                        self.lstm_model = keras.saving.load_model(str(path), compile=False)
                    except Exception as e3:
                        last_err = e3
                        try:
                            self.lstm_model = keras.saving.load_model(str(path))
                        except Exception as e4:
                            last_err = e4
            if self.lstm_model is not None:
                break
            if last_err is not None:
                self._load_error.append(("model", str(last_err)))

        try:
            self.scaler = joblib.load(str(_SCALER_PKL))
        except Exception as e:
            self.scaler = None
            self._load_error.append(("scaler", str(e)))

        try:
            # Pickle was saved from train script run as __main__, so it looks for __main__.LabelEncoderCompat.
            # Patch __main__ so joblib can resolve the class when loading.
            import sys
            main_mod = sys.modules.get("__main__")
            if main_mod is not None:
                setattr(main_mod, "LabelEncoderCompat", LabelEncoderCompat)
            self.label_encoder = joblib.load(str(_LABEL_ENCODER_PKL))
            # Ensure list of Python strings for comparison with 'push-up', etc.
            raw = self.label_encoder.classes_
            self.exercise_classes = [str(x) for x in (raw.tolist() if hasattr(raw, "tolist") else raw)]
        except Exception as e:
            self.label_encoder = None
            self.exercise_classes = []
            self._load_error.append(("label_encoder", str(e)))

    def extract_features(self, landmarks):
        features = []
        if len(landmarks) == len(relevant_landmarks_indices) * 3:
            # Angles
            features.append(calculate_angle(landmarks[0:3], landmarks[6:9], landmarks[12:15]))  # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
            features.append(calculate_angle(landmarks[3:6], landmarks[9:12], landmarks[15:18]))  # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
            features.append(calculate_angle(landmarks[18:21], landmarks[24:27], landmarks[30:33]))  # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
            features.append(calculate_angle(landmarks[21:24], landmarks[27:30], landmarks[33:36]))  # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
            features.append(calculate_angle(landmarks[0:3], landmarks[18:21], landmarks[24:27]))  # LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE
            features.append(calculate_angle(landmarks[3:6], landmarks[21:24], landmarks[27:30]))  # RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE

            # New angles
            features.append(calculate_angle(landmarks[18:21], landmarks[0:3], landmarks[6:9]))  # LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW
            features.append(calculate_angle(landmarks[21:24], landmarks[3:6], landmarks[9:12]))  # RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW

            # Distances
            distances = [
                calculate_distance(landmarks[0:3], landmarks[3:6]),  # LEFT_SHOULDER, RIGHT_SHOULDER
                calculate_distance(landmarks[18:21], landmarks[21:24]),  # LEFT_HIP, RIGHT_HIP
                calculate_distance(landmarks[18:21], landmarks[24:27]),  # LEFT_HIP, LEFT_KNEE
                calculate_distance(landmarks[21:24], landmarks[27:30]),  # RIGHT_HIP, RIGHT_KNEE
                calculate_distance(landmarks[0:3], landmarks[18:21]),  # LEFT_SHOULDER, LEFT_HIP
                calculate_distance(landmarks[3:6], landmarks[21:24]),  # RIGHT_SHOULDER, RIGHT_HIP
                calculate_distance(landmarks[6:9], landmarks[24:27]),  # LEFT_ELBOW, LEFT_KNEE
                calculate_distance(landmarks[9:12], landmarks[27:30]),  # RIGHT_ELBOW, RIGHT_KNEE
                calculate_distance(landmarks[12:15], landmarks[0:3]),  # LEFT_WRIST, LEFT_SHOULDER
                calculate_distance(landmarks[15:18], landmarks[3:6]),  # RIGHT_WRIST, RIGHT_SHOULDER
                calculate_distance(landmarks[12:15], landmarks[18:21]),  # LEFT_WRIST, LEFT_HIP
                calculate_distance(landmarks[15:18], landmarks[21:24])   # RIGHT_WRIST, RIGHT_HIP
            ]

            # Y-coordinate distances
            y_distances = [
                calculate_y_distance(landmarks[6:9], landmarks[0:3]),  # LEFT_ELBOW, LEFT_SHOULDER
                calculate_y_distance(landmarks[9:12], landmarks[3:6])   # RIGHT_ELBOW, RIGHT_SHOULDER
            ]

            # Normalization factor based on shoulder-hip or hip-knee distance
            normalization_factor = -1
            distances_to_check = [
                calculate_distance(landmarks[0:3], landmarks[18:21]),  # LEFT_SHOULDER, LEFT_HIP
                calculate_distance(landmarks[3:6], landmarks[21:24]),  # RIGHT_SHOULDER, RIGHT_HIP
                calculate_distance(landmarks[18:21], landmarks[24:27]),  # LEFT_HIP, LEFT_KNEE
                calculate_distance(landmarks[21:24], landmarks[27:30])   # RIGHT_HIP, RIGHT_KNEE
            ]

            for distance in distances_to_check:
                if distance > 0:
                    normalization_factor = distance
                    break
            
            if normalization_factor == -1:
                normalization_factor = 0.5  # Fallback normalization factor
            
            # Normalize distances
            normalized_distances = [d / normalization_factor if d != -1.0 else d for d in distances]
            normalized_y_distances = [d / normalization_factor if d != -1.0 else d for d in y_distances]

            # Combine features
            features.extend(normalized_distances)
            features.extend(normalized_y_distances)

        else:
            features = [-1.0] * 22  # Placeholder for missing landmarks
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

    # Auto classify and count method with repetition counting logic
    def auto_classify_and_count(self):
        if self.lstm_model is None or self.scaler is None or self.label_encoder is None:
            msg = "Model not loaded. Ensure files in **models/** are present (model .h5 or .keras, scaler .pkl, label_encoder .pkl)."
            if self._load_error:
                details = "; ".join(f"{c}: {e}" for c, e in self._load_error)
                msg += f" Load error(s): {details}"
            msg += " If the model file fails to load, re-run training once to create a .keras copy: `python scripts/run_demo_training.py`"
            st.error(msg)
            return
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
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
        canonical_prediction = None  # exercise name used for counting/display (maps model output to push-up/squat/etc.)
        last_predicted_class = -1
        prediction_history = []  # last N predictions for stability
        stable_exercise = None  # only show tips when same exercise predicted 2+ times
        tip_cooldown_sec = 2.0
        last_tip_text = None
        last_tip_time = 0.0
        counters = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        stages = {'push_up': None, 'squat': None, 'left_bicep_curl': None, 'right_bicep_curl': None, 'shoulder_press': None}
        known_exercises = {'push-up', 'squat', 'shoulder press', 'barbell biceps curl'}
        # When model was trained with dataset/source names, map class index or class name to exercise (see train_info.json)
        canonical_by_index = ['push-up', 'squat', 'barbell biceps curl', 'shoulder press']
        class_name_to_exercise = {}  # model class name -> exercise name (from train_info "class_to_exercise")
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

        detector = pm.posture_detector()
        pose = mp.solutions.pose.Pose()

        try:
            while True:
                try:
                    ret, frame = cap.read()
                except Exception:
                    break
                if not ret:
                    break

                landmarks = self.preprocess_frame(frame, pose)
                if len(landmarks) == len(relevant_landmarks_indices) * 3:
                    features = self.extract_features(landmarks)
                    if len(features) == 22:
                        landmarks_window.append(features)

                frame_count += 1

                if len(landmarks_window) == window_size:
                    # Shape (window_size, 22); scaler expects (N, 22) per frame
                    landmarks_window_np = np.array(landmarks_window, dtype=np.float32)
                    scaled_landmarks_window = self.scaler.transform(landmarks_window_np)
                    scaled_landmarks_window = scaled_landmarks_window.reshape(1, window_size, 22).astype(np.float32)

                    prediction = self.lstm_model.predict(scaled_landmarks_window)

                    if prediction.shape[1] != len(self.exercise_classes):
                        break
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    if predicted_class >= len(self.exercise_classes):
                        break
                    current_prediction = self.exercise_classes[predicted_class]
                    last_predicted_class = predicted_class
                    # Use canonical exercise name (from train_info or index fallback when model uses dataset names)
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

                # Repetition counting logic based on current prediction
                detector.find_person(frame, draw=True)
                landmark_list = detector.find_landmarks(frame, draw=True)
                if len(landmark_list) > 0:
                    exercise_for_count = canonical_prediction if canonical_prediction in known_exercises else None
                    if exercise_for_count == 'push-up':
                        stages['push_up'], counters['push_up'] = count_repetition_push_up(detector, frame, landmark_list, stages['push_up'], counters['push_up'], self)

                    elif exercise_for_count == 'squat':
                        stages['squat'], counters['squat'] = count_repetition_squat(detector, frame, landmark_list, stages['squat'], counters['squat'], self)

                    elif exercise_for_count == 'barbell biceps curl':
                        stages['right_bicep_curl'], stages['left_bicep_curl'], counters['bicep_curl'] = count_repetition_bicep_curl(detector, frame, landmark_list, stages['right_bicep_curl'], stages['left_bicep_curl'], counters['bicep_curl'], self)

                    elif exercise_for_count == 'shoulder press':
                        stages['shoulder_press'], counters['shoulder_press'] = count_repetition_shoulder_press(detector, frame, landmark_list, stages['shoulder_press'], counters['shoulder_press'], self)

                    # Form tip only when prediction is stable
                    form_tip = None
                    if stable_exercise and stable_exercise in known_exercises:
                        form_tip = get_form_suggestions(landmark_list, stable_exercise)
                    now = time.time()
                    if form_tip is None:
                        last_tip_text = None
                    elif form_tip != last_tip_text:
                        if last_tip_text is None or (now - last_tip_time) >= tip_cooldown_sec:
                            last_tip_text = form_tip
                            last_tip_time = now
                exercise_name_map = {
                    'push_up': 'Push-up',
                    'squat': 'Squat',
                    'bicep_curl': 'Curl',
                    'shoulder_press': 'Press'
                }

                height, width, _ = frame.shape
                num_exercises = len(counters)
                vertical_spacing = height // (num_exercises + 1)

                cv2.rectangle(frame, (0, 0), (120, height), (0, 0, 0), -1)
                cv2.rectangle(frame, (0, 0), (width, 30), (0, 0, 0), -1)

                # Show exercise name (Push-up, Squat, Curl, Press); never show raw model/dataset name
                display_name = canonical_to_display.get(canonical_prediction, "Detecting...") if canonical_prediction else "Detecting..."
                draw_styled_text(frame, f"Exercise: {display_name}", ((width - 290) // 2 + 100, 20))

                for idx, (exercise, count) in enumerate(counters.items()):
                    short_name = exercise_name_map.get(exercise, exercise)
                    draw_styled_text(frame, f"{short_name}: {count}", (10, (idx + 1) * vertical_spacing))

                # Draw tip last and to the right of the left sidebar (x=130) so it is not covered
                if last_tip_text:
                    draw_styled_text(frame, f"Tip: {last_tip_text}", (130, height - 35), font_scale=0.55, font_color=(0, 255, 255), bg_color=(40, 40, 40))

                if time.time() - last_draw >= frame_interval:
                    try:
                        stframe.image(frame, channels='BGR', use_container_width=True)
                    except Exception:
                        pass  # skip this frame, keep going
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

    # Check if hands are joined together in a 'prayer' gesture
    def are_hands_joined(self, landmark_list, stop, is_video=False):
        # Extract wrist coordinates
        left_wrist = landmark_list[15][1:]  # (x, y) for left wrist
        right_wrist = landmark_list[16][1:]  # (x, y) for right wrist

        # Calculate the Euclidean distance between the wrists
        distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
        # Consider hands joined if the distance is below a certain threshold, e.g., 50 pixels
        if distance < 30 and not is_video:
            return True
        
        return False


    # Visualize repetitions of the exercise on screen
    def repetitions_counter(self, img, counter):
        cv2.rectangle(img, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(img, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Define push-up method
    def push_up(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_push_up, counter=counter, stage=stage, exercise_name="push-up")

    # Define squat method
    def squat(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_squat, counter=counter, stage=stage, exercise_name="squat")

    # Define bicep curl method
    def bicept_curl(self, cap, is_video=False, counter=0, stage_right=None, stage_left=None):
        self.exercise_method(cap, is_video, count_repetition_bicep_curl, multi_stage=True, counter=counter, stage_right=stage_right, stage_left=stage_left, exercise_name="barbell biceps curl")

    # Define shoulder press method
    def shoulder_press(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_shoulder_press, counter=counter, stage=stage, exercise_name="shoulder press")

    # Generic exercise method (with optional live form suggestions)
    def exercise_method(self, cap, is_video, count_repetition_function, multi_stage=False, counter=0, stage=None, stage_right=None, stage_left=None, exercise_name=None):
        if is_video:
            stframe = st.empty()
            detector = pm.posture_detector()

            # Get the original video's FPS
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_time = 1 / original_fps

            frame_count = 0
            start_time = time.time()
            last_update_time = start_time

            update_interval = 0.1  # Update display every 100ms

            while cap.isOpened():
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Determine how many frames should have been processed by now
                target_frame = int(elapsed_time * original_fps)

                # Process frames until we catch up to where we should be
                while frame_count < target_frame:
                    ret, frame = cap.read()
                    if not ret:
                        return

                    frame_count += 1

                    # Process the last frame we read
                    if frame_count == target_frame:
                        img = detector.find_person(frame)
                        landmark_list = detector.find_landmarks(img, draw=False)

                        if len(landmark_list) != 0:
                            if multi_stage:
                                stage_right, stage_left, counter = count_repetition_function(detector, img, landmark_list, stage_right, stage_left, counter, self)
                            else:
                                stage, counter = count_repetition_function(detector, img, landmark_list, stage, counter, self)

                            if self.are_hands_joined(landmark_list, stop=False, is_video=is_video):
                                return

                            # Live form suggestion
                            if exercise_name:
                                form_tip = get_form_suggestions(landmark_list, exercise_name)
                                if form_tip:
                                    h, w = img.shape[:2]
                                    draw_styled_text(img, f"Tip: {form_tip}", (10, h - 25), font_scale=0.5, font_color=(0, 255, 255), bg_color=(40, 40, 40))

                        self.repetitions_counter(img, counter)

                # Update display at regular intervals
                if current_time - last_update_time >= update_interval:
                    stframe.image(img, channels='BGR', use_container_width=True)
                    last_update_time = current_time

                # Small sleep to prevent busy-waiting
                time.sleep(0.001)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            # Webcam exercise: warmup camera properly, then loop (retry reads so we don't exit after 1 sec)
            stframe = st.empty()
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
            # Warmup: read and discard ~2 sec of frames so camera is stable before we start
            for _ in range(60):
                cap.read()
                time.sleep(0.033)
            detector = pm.posture_detector()
            frame_interval = 1.0 / 30
            last_update = time.time()
            read_retries = 15  # retry this many times if read fails before giving up

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
                            if exercise_name:
                                form_tip = get_form_suggestions(landmark_list, exercise_name)
                                if form_tip:
                                    h, w = img.shape[:2]
                                    draw_styled_text(img, f"Tip: {form_tip}", (10, h - 25), font_scale=0.5, font_color=(0, 255, 255), bg_color=(40, 40, 40))

                        self.repetitions_counter(img, counter)
                        stframe.image(img, channels='BGR', use_container_width=True)
                    except Exception:
                        continue  # skip bad frame, keep exercising
            except Exception:
                pass
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cv2.destroyAllWindows()
                try:
                    ran_sec = time.time() - webcam_start_time
                    if ran_sec < 3:
                        stframe.warning("Camera not ready or stopped quickly. Click **Start Exercise** to try again.")
                except Exception:
                    pass

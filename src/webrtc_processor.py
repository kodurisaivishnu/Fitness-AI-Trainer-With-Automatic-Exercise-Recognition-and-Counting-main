"""
WebRTC video processors for browser-based webcam access.
Uses streamlit-webrtc so webcam works even when deployed to cloud (Render, etc.)
"""
import cv2
import numpy as np
import time
import threading
import av

try:
    import mediapipe as mp
    import PoseModule2 as pm
    from ExerciseAiTrainer import (
        get_cached_exercise, relevant_landmarks_indices,
        count_repetition_push_up, count_repetition_squat,
        count_repetition_bicep_curl, count_repetition_shoulder_press,
        get_form_suggestions, get_injury_alerts, draw_styled_text,
        estimate_calories, canonical_to_display_name,
    )
    _IMPORTS_OK = True
except Exception:
    _IMPORTS_OK = False


class WebRTCExerciseProcessor:
    """Processes webcam frames for a selected exercise (manual mode)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.exercise_name = "push-up"
        self.multi_stage = False

        if _IMPORTS_OK:
            self.detector = pm.posture_detector()
            self.exercise_instance = get_cached_exercise()
        else:
            self.detector = None
            self.exercise_instance = None

        self.counter = 0
        self.stage = None
        self.stage_right = None
        self.stage_left = None
        self.last_tip = ""
        self.last_injury = ""
        self.last_tip_time = 0.0
        self.last_injury_time = 0.0
        self.session_start = time.time()

        self._count_funcs = {}
        if _IMPORTS_OK:
            self._count_funcs = {
                "push-up": count_repetition_push_up,
                "squat": count_repetition_squat,
                "barbell biceps curl": count_repetition_bicep_curl,
                "shoulder press": count_repetition_shoulder_press,
            }

    def set_exercise(self, exercise_name):
        with self._lock:
            self.exercise_name = exercise_name
            self.multi_stage = (exercise_name == "barbell biceps curl")
            self.counter = 0
            self.stage = None
            self.stage_right = None
            self.stage_left = None

    def get_state(self):
        with self._lock:
            cal = estimate_calories(self.exercise_name, self.counter) if _IMPORTS_OK else 0
            return {
                "counter": self.counter,
                "exercise": self.exercise_name,
                "tip": self.last_tip,
                "injury": self.last_injury,
                "calories": cal,
                "duration": time.time() - self.session_start,
            }

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if not _IMPORTS_OK or not self.detector:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            with self._lock:
                exercise_name = self.exercise_name
                multi_stage = self.multi_stage

            img = self.detector.find_person(img, draw=True)
            landmark_list = self.detector.find_landmarks(img, draw=False)

            if len(landmark_list) > 0:
                count_func = self._count_funcs.get(exercise_name)
                if count_func:
                    with self._lock:
                        if multi_stage:
                            self.stage_right, self.stage_left, self.counter = count_func(
                                self.detector, img, landmark_list,
                                self.stage_right, self.stage_left, self.counter,
                                self.exercise_instance
                            )
                        else:
                            self.stage, self.counter = count_func(
                                self.detector, img, landmark_list,
                                self.stage, self.counter,
                                self.exercise_instance
                            )

                now = time.time()
                injury = get_injury_alerts(landmark_list, exercise_name)
                if injury:
                    with self._lock:
                        if injury != self.last_injury or (now - self.last_injury_time) >= 3.0:
                            self.last_injury = injury
                            self.last_injury_time = now

                tip = get_form_suggestions(landmark_list, exercise_name)
                if tip:
                    with self._lock:
                        if tip != self.last_tip or (now - self.last_tip_time) >= 2.0:
                            self.last_tip = tip
                            self.last_tip_time = now

            # Draw overlays
            h, w = img.shape[:2]
            with self._lock:
                counter = self.counter
                tip_text = self.last_tip
                injury_text = self.last_injury
                injury_time = self.last_injury_time
                cal = estimate_calories(exercise_name, counter)

            cv2.rectangle(img, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(img, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f'{cal:.1f} cal', (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            display = canonical_to_display_name(exercise_name)
            draw_styled_text(img, f"Exercise: {display}", ((w - 290) // 2 + 100, 20))

            if injury_text and (time.time() - injury_time) < 3.0:
                draw_styled_text(img, injury_text, (10, h - 55), font_scale=0.55, font_color=(0, 0, 255), bg_color=(255, 255, 255))
            if tip_text:
                draw_styled_text(img, f"Tip: {tip_text}", (10, h - 25), font_scale=0.55, font_color=(0, 255, 255), bg_color=(40, 40, 40))

        except Exception:
            pass  # Return frame as-is on any error

        return av.VideoFrame.from_ndarray(img, format="bgr24")


class WebRTCAutoClassifyProcessor:
    """Processes webcam frames with automatic exercise classification via BiLSTM."""

    def __init__(self):
        self._lock = threading.Lock()
        self.model_ready = False

        if _IMPORTS_OK:
            self.detector = pm.posture_detector()
            self.exercise_instance = get_cached_exercise()
            self.pose_for_features = mp.solutions.pose.Pose()
            self.model_ready = (
                self.exercise_instance.lstm_model is not None
                and self.exercise_instance.scaler is not None
                and self.exercise_instance.label_encoder is not None
            )
        else:
            self.detector = None
            self.exercise_instance = None
            self.pose_for_features = None

        self.window_size = 30
        self.landmarks_window = []
        self.canonical_prediction = None
        self.prediction_history = []
        self.stable_exercise = None
        self.counters = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        self.stages = {'push_up': None, 'squat': None, 'left_bicep_curl': None, 'right_bicep_curl': None, 'shoulder_press': None}
        self.known_exercises = {'push-up', 'squat', 'shoulder press', 'barbell biceps curl'}
        self.last_tip = ""
        self.last_injury = ""
        self.last_tip_time = 0.0
        self.last_injury_time = 0.0
        self.session_start = time.time()
        self.canonical_by_index = ['push-up', 'squat', 'barbell biceps curl', 'shoulder press']
        self.class_name_to_exercise = {}
        self._load_train_info()

    def _load_train_info(self):
        try:
            from pathlib import Path
            import json
            models_dir = Path(__file__).resolve().parent.parent / "models"
            info_path = models_dir / "train_info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text())
                classes = self.exercise_instance.exercise_classes if self.exercise_instance else []
                if isinstance(info.get("class_index_to_exercise"), list) and len(info["class_index_to_exercise"]) == len(classes):
                    self.canonical_by_index = info["class_index_to_exercise"]
                if isinstance(info.get("class_to_exercise"), dict):
                    self.class_name_to_exercise = info["class_to_exercise"]
        except Exception:
            pass

    def get_state(self):
        with self._lock:
            exercise_name_map = {'push_up': 'Push-up', 'squat': 'Squat', 'bicep_curl': 'Curl', 'shoulder_press': 'Press'}
            exercise_cal_map = {'push_up': 'push-up', 'squat': 'squat', 'bicep_curl': 'barbell biceps curl', 'shoulder_press': 'shoulder press'}
            total_reps = sum(self.counters.values())
            total_cal = 0
            breakdown = {}
            for key, count in self.counters.items():
                if count > 0:
                    ex_name = exercise_cal_map.get(key, key)
                    cal = estimate_calories(ex_name, count) if _IMPORTS_OK else 0
                    total_cal += cal
                    breakdown[exercise_name_map.get(key, key)] = {"reps": count, "calories": round(cal, 1)}
            pred_name = "Detecting..."
            if _IMPORTS_OK and self.canonical_prediction:
                pred_name = canonical_to_display_name(self.canonical_prediction)
            return {
                "prediction": pred_name,
                "counters": dict(self.counters),
                "total_reps": total_reps,
                "total_calories": round(total_cal, 1),
                "tip": self.last_tip,
                "injury": self.last_injury,
                "duration": time.time() - self.session_start,
                "breakdown": breakdown,
                "model_ready": self.model_ready,
            }

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if not _IMPORTS_OK or not self.model_ready:
            h, w = img.shape[:2]
            cv2.putText(img, "Model not loaded", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            ex = self.exercise_instance

            # Feature extraction for classification
            landmarks = ex.preprocess_frame(img, self.pose_for_features)
            if len(landmarks) == len(relevant_landmarks_indices) * 3:
                features = ex.extract_features(landmarks)
                if len(features) == 22:
                    with self._lock:
                        self.landmarks_window.append(features)

            with self._lock:
                window_ready = len(self.landmarks_window) == self.window_size

            if window_ready:
                with self._lock:
                    lw = list(self.landmarks_window)
                    self.landmarks_window = []

                lw_np = np.array(lw, dtype=np.float32)
                scaled = ex.scaler.transform(lw_np)
                scaled = scaled.reshape(1, self.window_size, 22).astype(np.float32)
                prediction = ex.lstm_model.predict(scaled, verbose=0)

                if prediction.shape[1] == len(ex.exercise_classes):
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    if predicted_class < len(ex.exercise_classes):
                        current_pred = ex.exercise_classes[predicted_class]
                        if current_pred in self.known_exercises:
                            canon = current_pred
                        elif current_pred in self.class_name_to_exercise and self.class_name_to_exercise[current_pred] in self.known_exercises:
                            canon = self.class_name_to_exercise[current_pred]
                        elif len(self.canonical_by_index) == len(ex.exercise_classes) and predicted_class < len(self.canonical_by_index):
                            canon = self.canonical_by_index[predicted_class]
                        else:
                            canon = current_pred if current_pred in self.known_exercises else None

                        with self._lock:
                            self.canonical_prediction = canon
                            self.prediction_history.append(canon)
                            if len(self.prediction_history) > 3:
                                self.prediction_history.pop(0)
                            if len(self.prediction_history) >= 2 and self.prediction_history[-1] == self.prediction_history[-2] and canon in self.known_exercises:
                                self.stable_exercise = canon
                            else:
                                self.stable_exercise = None

            # Pose detection for counting
            img = self.detector.find_person(img, draw=True)
            landmark_list = self.detector.find_landmarks(img, draw=True)

            with self._lock:
                canon = self.canonical_prediction
                stable = self.stable_exercise

            if len(landmark_list) > 0 and canon in self.known_exercises:
                with self._lock:
                    if canon == 'push-up':
                        self.stages['push_up'], self.counters['push_up'] = count_repetition_push_up(self.detector, img, landmark_list, self.stages['push_up'], self.counters['push_up'], ex)
                    elif canon == 'squat':
                        self.stages['squat'], self.counters['squat'] = count_repetition_squat(self.detector, img, landmark_list, self.stages['squat'], self.counters['squat'], ex)
                    elif canon == 'barbell biceps curl':
                        self.stages['right_bicep_curl'], self.stages['left_bicep_curl'], self.counters['bicep_curl'] = count_repetition_bicep_curl(self.detector, img, landmark_list, self.stages['right_bicep_curl'], self.stages['left_bicep_curl'], self.counters['bicep_curl'], ex)
                    elif canon == 'shoulder press':
                        self.stages['shoulder_press'], self.counters['shoulder_press'] = count_repetition_shoulder_press(self.detector, img, landmark_list, self.stages['shoulder_press'], self.counters['shoulder_press'], ex)

                now = time.time()
                injury = get_injury_alerts(landmark_list, canon)
                if injury:
                    with self._lock:
                        if injury != self.last_injury or (now - self.last_injury_time) >= 3.0:
                            self.last_injury = injury
                            self.last_injury_time = now
                if stable and stable in self.known_exercises:
                    tip = get_form_suggestions(landmark_list, stable)
                    if tip:
                        with self._lock:
                            if tip != self.last_tip or (now - self.last_tip_time) >= 2.0:
                                self.last_tip = tip
                                self.last_tip_time = now

            # Draw overlays
            h, w = img.shape[:2]
            with self._lock:
                counters = dict(self.counters)
                tip_text = self.last_tip
                injury_text = self.last_injury
                injury_time = self.last_injury_time

            exercise_name_map = {'push_up': 'Push-up', 'squat': 'Squat', 'bicep_curl': 'Curl', 'shoulder_press': 'Press'}
            v_spacing = h // (len(counters) + 1)
            cv2.rectangle(img, (0, 0), (140, h), (0, 0, 0), -1)
            cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)

            display_name = canonical_to_display_name(canon) if canon else "Detecting..."
            draw_styled_text(img, f"Exercise: {display_name}", ((w - 290) // 2 + 100, 20))
            for idx, (ex_key, count) in enumerate(counters.items()):
                short = exercise_name_map.get(ex_key, ex_key)
                draw_styled_text(img, f"{short}: {count}", (10, (idx + 1) * v_spacing))

            if injury_text and (time.time() - injury_time) < 3.0:
                draw_styled_text(img, injury_text, (130, h - 60), font_scale=0.55, font_color=(0, 0, 255), bg_color=(255, 255, 255))
            if tip_text:
                draw_styled_text(img, f"Tip: {tip_text}", (130, h - 35), font_scale=0.55, font_color=(0, 255, 255), bg_color=(40, 40, 40))

        except Exception:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

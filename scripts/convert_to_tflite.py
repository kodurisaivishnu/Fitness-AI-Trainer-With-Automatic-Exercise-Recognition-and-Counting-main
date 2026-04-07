#!/usr/bin/env python3
"""Convert the Keras/H5 BiLSTM model to TFLite format for low-memory deployment.
Run ONCE locally, commit the .tflite file.
Usage: python scripts/convert_to_tflite.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_KERAS = MODELS_DIR / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.keras"
MODEL_H5 = MODELS_DIR / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5"
OUTPUT = MODELS_DIR / "exercise_classifier.tflite"

import tensorflow as tf

model = None
for path in [MODEL_KERAS, MODEL_H5]:
    if path.exists():
        try:
            model = tf.keras.models.load_model(str(path), compile=False)
            print(f"Loaded from {path.name}")
            break
        except Exception as e:
            print(f"Failed {path.name}: {e}")

if model is None:
    print("ERROR: No model found"); sys.exit(1)

# BiLSTM needs SELECT_TF_OPS for TensorListReserve op
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
OUTPUT.write_bytes(tflite_model)
print(f"Saved {OUTPUT} ({len(tflite_model)/1024:.0f} KB)")

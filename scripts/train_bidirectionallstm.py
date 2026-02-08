"""
Train BiLSTM exercise classifier on 30-frame feature sequences.
Improved: deeper model, augmentation, early stopping, LR schedule, test metrics.
Saves model compatible with the app (same filenames in models/).
Run from project root: python scripts/train_bidirectionallstm.py --data data_sequences.npz [--output_dir models]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, Input, BatchNormalization,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, ReLU,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WINDOW_SIZE = 30
NUM_FEATURES = 22


class LabelEncoderCompat:
    """Compatible with app: has .classes_ (numpy array of label strings)."""
    def __init__(self, classes):
        self.classes_ = np.array(classes)


def augment_sequence(X_seq, noise_std=0.02):
    """Add Gaussian noise to features (excluding invalid -1)."""
    X = X_seq.copy()
    mask = X != -1.0
    X[mask] = X[mask] + np.random.randn(*X.shape)[mask].astype(np.float32) * noise_std
    return X


def build_bilstm_model(
    num_classes,
    input_shape=(WINDOW_SIZE, NUM_FEATURES),
    lstm_units=(128, 64),
    dropout=0.4,
    use_batchnorm=True,
    l2=1e-3,
):
    """
    BiLSTM with regularization to reduce overfitting: L2 on weights, dropout.
    """
    reg = L2(l2) if l2 > 0 else None
    inp = Input(shape=input_shape)
    x = inp
    for i, units in enumerate(lstm_units):
        x = Bidirectional(LSTM(units, return_sequences=(i < len(lstm_units) - 1), kernel_regularizer=reg))(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
    x = Dense(num_classes, activation="softmax", kernel_regularizer=reg)(x)
    model = Model(inp, x)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn1d_model(
    num_classes,
    input_shape=(WINDOW_SIZE, NUM_FEATURES),
    filters=(64, 128, 128),
    kernel_size=3,
    dropout=0.4,
    l2=1e-3,
):
    """
    Simple 1D CNN: convolutions along the time axis to detect local motion patterns,
    then global pooling and dense layer. Easy to explain: "sliding filters over the
    sequence, then classify."
    """
    reg = L2(l2) if l2 > 0 else None
    inp = Input(shape=input_shape)
    x = inp
    for i, f in enumerate(filters):
        x = Conv1D(f, kernel_size, padding="same", kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling1D(2, padding="same")(x)
        x = Dropout(dropout)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(num_classes, activation="softmax", kernel_regularizer=reg)(x)
    model = Model(inp, x)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM exercise classifier (improved)")
    parser.add_argument("--data", type=str, default="data_sequences.npz", help="Path to .npz from create_sequence_of_features.py")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save model and encoders (default: project models/)")
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split")
    parser.add_argument("--test_split", type=float, default=0.15, help="Test split (hold-out)")
    parser.add_argument("--lstm_units", type=str, default="128,64", help="LSTM units per layer, comma-separated (e.g. 128,64)")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate (higher reduces overfitting)")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 regularization (e.g. 1e-3 to reduce overfitting)")
    parser.add_argument("--augment", type=float, default=0.02, help="Gaussian noise std for augmentation (0 to disable)")
    parser.add_argument("--no_batchnorm", action="store_true", help="Disable BatchNorm layers")
    parser.add_argument("--source", type=str, default="unknown", help="Model source: kaggle, demo, direct, or pre-trained (for app display)")
    parser.add_argument("--model", type=str, default="bilstm", choices=("bilstm", "cnn1d"), help="Architecture: bilstm (BiLSTM) or cnn1d (1D CNN, simple and explainable)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data not found: {data_path}. Run create_sequence_of_features.py first.")
        return

    out_dir = Path(args.output_dir or PROJECT_ROOT / "models")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    labels = data["labels"].tolist()
    num_classes = len(labels)

    # Train / val / test split (stratified)
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=args.test_split, stratify=y, random_state=42
    )
    val_ratio = args.val_split / (1 - args.test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_ratio, stratify=y_rest, random_state=42
    )

    # Scale on train only
    n_samples, n_time, n_feat = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_flat)
    X_train = scaler.transform(X_train_flat).reshape(X_train.shape).astype(np.float32)
    X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape).astype(np.float32)
    X_test = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape).astype(np.float32)

    # Optional augmentation (on training data only)
    if args.augment > 0:
        X_train_aug = np.array([augment_sequence(x, args.augment) for x in X_train], dtype=np.float32)
        # Combine original + augmented for more data
        X_train = np.concatenate([X_train, X_train_aug], axis=0)
        y_train = np.concatenate([y_train, y_train], axis=0)

    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    if args.model == "cnn1d":
        model = build_cnn1d_model(
            num_classes,
            dropout=args.dropout,
            l2=args.l2,
        )
        best_path = out_dir / "best_cnn1d_weights.weights.h5"
        architecture = "1D CNN"
    else:
        lstm_units = [int(u) for u in args.lstm_units.split(",")]
        model = build_bilstm_model(
            num_classes,
            lstm_units=lstm_units,
            dropout=args.dropout,
            use_batchnorm=not args.no_batchnorm,
            l2=args.l2,
        )
        best_path = out_dir / "best_bilstm_weights.weights.h5"
        architecture = "BiLSTM"

    print("Architecture:", architecture)

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            str(best_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
    ]

    model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Load best weights and evaluate on test set
    if best_path.exists():
        model.load_weights(str(best_path))
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_acc = accuracy_score(y_test, y_pred)
    print("\n--- Test set results ---")
    print("Test accuracy: {:.4f}".format(test_acc))
    print(classification_report(y_test, y_pred, target_names=labels, digits=4))

    # Save final artifacts (app-compatible names)
    model_path = out_dir / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5"
    model_keras_path = out_dir / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.keras"
    scaler_path = out_dir / "thesis_bidirectionallstm_scaler.pkl"
    label_path = out_dir / "thesis_bidirectionallstm_label_encoder.pkl"

    model.save(model_path)
    try:
        model.save(model_keras_path)
    except Exception:
        pass
    joblib.dump(scaler, scaler_path)
    label_encoder = LabelEncoderCompat(labels)
    joblib.dump(label_encoder, label_path)

    print("\nSaved (app-compatible):")
    print("  Model:", model_path)
    print("  Scaler:", scaler_path)
    print("  Label encoder:", label_path)
    print("  Classes:", labels)

    # Save metrics and info for app (current vs previous comparison)
    metrics_path = out_dir / "train_metrics.txt"
    metrics_previous = out_dir / "train_metrics_previous.txt"
    info_path = out_dir / "train_info.json"
    if metrics_path.exists():
        import shutil
        shutil.copy(metrics_path, metrics_previous)
    with open(metrics_path, "w") as f:
        f.write("Test accuracy: {:.4f}\n".format(test_acc))
        f.write("\nClassification report:\n")
        f.write(classification_report(y_test, y_pred, target_names=labels, digits=4))
    info = {
        "source": args.source,
        "architecture": architecture,
        "test_accuracy": float(test_acc),
        "timestamp": datetime.now().isoformat(),
        "classes": labels,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print("  Metrics:", metrics_path)
    print("  Info (source, accuracy):", info_path)


if __name__ == "__main__":
    main()

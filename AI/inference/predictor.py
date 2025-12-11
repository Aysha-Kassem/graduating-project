"""AI/inference/predictor.py

Production-ready inference helper for the Fall Detection project.

Provides:
- model/scaler loading
- preprocessing (scaling + sliding-window)
- batch prediction (ensemble)
- a small RealTimePredictor class for streaming 1-row-per-second data

Paths: file expects to live at AI/inference/predictor.py and loads models from ../models and scaler from ../scaler

Usage (examples -- see bottom of file):
- from AI.inference.predictor import load_resources, predict_from_array, RealTimePredictor

"""

import os
import joblib
import numpy as np
from collections import deque
from typing import List, Tuple, Optional

# TF import deferred inside functions to avoid heavy import at module load in some contexts

# -----------------------------
# Default configuration
# -----------------------------
TIME_STEPS = 50
STEP_SIZE = 1
FEATURES = None  # will be inferred from input data

# -----------------------------
# Helpers for path discovery
# -----------------------------
def _get_ai_root() -> str:
    """Return the AI folder root assuming this file is at AI/inference/predictor.py"""
    this_file = os.path.abspath(__file__)
    ai_dir = os.path.dirname(os.path.dirname(this_file))  # .. /AI
    return ai_dir

# -----------------------------
# Resource loading
# -----------------------------

def load_resources(
    model_lstm_path: Optional[str] = None,
    model_bigru_path: Optional[str] = None,
    model_bigru_lstm_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    verbose: bool = True,
):
    """Load the three trained models and the scaler.

    If paths are not provided, looks in the AI folder relative to this file:
      ../models/FINAL_*.keras and ../scaler/scaler_all.save

    Returns: (model_now, model_bigru, model_bigru_lstm, scaler)
    """
    import tensorflow as tf

    ai_root = _get_ai_root()

    if model_lstm_path is None:
        model_lstm_path = os.path.join(ai_root, "models", "FINAL_LSTM_Attention.keras")
    if model_bigru_path is None:
        model_bigru_path = os.path.join(ai_root, "models", "FINAL_BiGRU_Attention.keras")
    if model_bigru_lstm_path is None:
        model_bigru_lstm_path = os.path.join(ai_root, "models", "FINAL_BiGRU_Attention_LSTM.keras")
    if scaler_path is None:
        scaler_path = os.path.join(ai_root, "scaler", "scaler_all.save")

    for p in (model_lstm_path, model_bigru_path, model_bigru_lstm_path, scaler_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    if verbose:
        print("Loading models and scaler from:\n",
              model_lstm_path, '\n', model_bigru_path, '\n', model_bigru_lstm_path, '\n', scaler_path)

    model_now = tf.keras.models.load_model(model_lstm_path, compile=False)
    model_bigru = tf.keras.models.load_model(model_bigru_path, compile=False)
    model_bigru_lstm = tf.keras.models.load_model(model_bigru_lstm_path, compile=False)
    scaler = joblib.load(scaler_path)

    return model_now, model_bigru, model_bigru_lstm, scaler


# -----------------------------
# Preprocessing (sliding windows + scaling)
# -----------------------------

def create_sliding_windows(
    raw_array: np.ndarray,
    time_steps: int = TIME_STEPS,
    step_size: int = STEP_SIZE,
) -> np.ndarray:
    """Convert 2D raw_array (n_rows, n_features) into 3D windows (n_windows, time_steps, n_features).

    Notes:
    - assumes raw_array is numeric and already in shape (n_rows, n_features)
    - returns np.array of shape (n_windows, time_steps, n_features)
    """
    if raw_array.ndim != 2:
        raise ValueError("raw_array must be 2D: (n_rows, n_features)")

    n_rows, n_features = raw_array.shape
    if n_rows < time_steps:
        return np.zeros((0, time_steps, n_features))  # not enough rows yet

    windows = []
    for i in range(0, n_rows - time_steps + 1, step_size):
        windows.append(raw_array[i:i + time_steps])

    return np.stack(windows, axis=0)


def scale_windows(windows: np.ndarray, scaler) -> np.ndarray:
    """Scale windows using a fitted sklearn scaler (joblib loaded).

    The scaler expects 2D data (n_samples, n_features). We reshape all windows into 2D,
    transform, then reshape back to 3D.
    """
    if windows.size == 0:
        return windows

    n_windows, time_steps, n_features = windows.shape
    flat = windows.reshape(-1, n_features)  # (n_windows * time_steps, n_features)
    scaled_flat = scaler.transform(flat)
    scaled = scaled_flat.reshape(n_windows, time_steps, n_features)
    return scaled


# -----------------------------
# Ensemble prediction
# -----------------------------

def weighted_ensemble_predict(models_tuple, X: np.ndarray, weights: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """Run the three models and return weighted ensemble probabilities for (fall_now, fall_soon).

    models_tuple: (model_now, model_bigru, model_bigru_lstm)
    X: shape (n_windows, time_steps, n_features)

    Returns: (ensemble_now_probs, ensemble_soon_probs) each shape (n_windows, 1)
    """
    model_now, model_bigru, model_bigru_lstm = models_tuple
    if weights is None:
        weights = {"m1": 0.5, "m2": 0.2, "m3": 0.3}

    p1_now, p1_soon = model_now.predict(X)
    p2_now, p2_soon = model_bigru.predict(X)
    p3_now, p3_soon = model_bigru_lstm.predict(X)

    ensemble_now = (p1_now * weights["m1"]) + (p2_now * weights["m2"]) + (p3_now * weights["m3"])
    ensemble_soon = (p1_soon * weights["m1"]) + (p2_soon * weights["m2"]) + (p3_soon * weights["m3"])

    return ensemble_now, ensemble_soon


# -----------------------------
# High-level prediction API
# -----------------------------

def predict_from_array(
    raw_array: np.ndarray,
    models_tuple,
    scaler,
    time_steps: int = TIME_STEPS,
    step_size: int = STEP_SIZE,
    threshold: float = 0.5,
) -> dict:
    """Full pipeline: raw 2D array -> windows -> scaled -> model predictions -> binary results.

    raw_array: np.ndarray, shape (n_rows, n_features)

    Returns dict with keys:
      - 'probs_now', 'probs_soon' : lists of probabilities (per window)
      - 'pred_now', 'pred_soon' : binary lists (per window)
      - 'n_windows'
    """
    if raw_array.ndim != 2:
        raise ValueError("raw_array must be 2D: (n_rows, n_features)")

    windows = create_sliding_windows(raw_array, time_steps=time_steps, step_size=step_size)
    if windows.shape[0] == 0:
        return {"probs_now": [], "probs_soon": [], "pred_now": [], "pred_soon": [], "n_windows": 0}

    scaled = scale_windows(windows, scaler)

    probs_now, probs_soon = weighted_ensemble_predict(models_tuple, scaled)

    preds_now = (probs_now > threshold).astype(int).reshape(-1).tolist()
    preds_soon = (probs_soon > threshold).astype(int).reshape(-1).tolist()

    return {
        "probs_now": probs_now.reshape(-1).tolist(),
        "probs_soon": probs_soon.reshape(-1).tolist(),
        "pred_now": preds_now,
        "pred_soon": preds_soon,
        "n_windows": int(windows.shape[0]),
    }


# -----------------------------
# Real-time predictor (1-row-at-a-time)
# -----------------------------

class RealTimePredictor:
    """Maintain a sliding buffer of the last `time_steps` rows and predict when buffer is full.

    Usage:
      rt = RealTimePredictor(models_tuple, scaler)
      out = rt.add_row(row)  # row: 1D array/list of features
      if out is not None:
          # out is same dict as predict_from_array but for the newest window only

    The class keeps a deque buffer; every call to add_row returns prediction for the current
    buffered window (when available) or None if not enough rows yet.
    """

    def __init__(self, models_tuple, scaler, time_steps: int = TIME_STEPS, threshold: float = 0.5):
        self.models_tuple = models_tuple
        self.scaler = scaler
        self.time_steps = time_steps
        self.threshold = threshold
        self.buffer = deque(maxlen=time_steps)

    def add_row(self, row: List[float]):
        arr = np.array(row, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("row must be 1D sequence of features")

        self.buffer.append(arr)
        if len(self.buffer) < self.time_steps:
            return None

        raw_window = np.stack(list(self.buffer), axis=0)  # (time_steps, n_features)
        # add batch dim
        windows = raw_window[np.newaxis, ...]  # (1, time_steps, n_features)
        scaled = scale_windows(windows, self.scaler)
        probs_now, probs_soon = weighted_ensemble_predict(self.models_tuple, scaled)
        pred_now = int((probs_now > self.threshold).astype(int).reshape(-1)[0])
        pred_soon = int((probs_soon > self.threshold).astype(int).reshape(-1)[0])

        return {
            "probs_now": float(probs_now.reshape(-1)[0]),
            "probs_soon": float(probs_soon.reshape(-1)[0]),
            "pred_now": pred_now,
            "pred_soon": pred_soon,
        }


# -----------------------------
# Example quick test (only runs when executed directly)
# -----------------------------
if __name__ == "__main__":
    print("Quick self-test for predictor.py")
    try:
        model_now, model_bigru, model_bigru_lstm, scaler = load_resources(verbose=True)
    except Exception as e:
        print("Skipping model load in quick test (files may be missing):", e)
        raise

    models_tuple = (model_now, model_bigru, model_bigru_lstm)

    # Create fake data to validate shapes (50 timesteps, infer features from model input)
    # We attempt to infer feature size from model input shape
    input_shape = model_now.input_shape  # (None, time_steps, features)
    _, ts, nf = input_shape
    sample = np.random.randn(ts * 3, nf).astype(np.float32)  # 3 windows worth of rows

    out = predict_from_array(sample, models_tuple, scaler)
    print("Prediction output example:", out)

    # Real-time example
    rt = RealTimePredictor(models_tuple, scaler)
    for r in sample[:60]:
        res = rt.add_row(r.tolist())
        if res is not None:
            print("Realtime prediction ->", res)

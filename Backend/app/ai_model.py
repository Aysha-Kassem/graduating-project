import os
import joblib
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_DIR = os.path.join(BASE_DIR, "..", "AI")

MODEL_NOW_PATH = os.path.join(AI_DIR, "models", "FINAL_LSTM_Attention.keras")
MODEL_BiGRU_PATH = os.path.join(AI_DIR, "models", "FINAL_BiGRU_Attention.keras")
MODEL_BiGRU_LSTM_PATH = os.path.join(AI_DIR, "models", "FINAL_BiGRU_Attention_LSTM.keras")
SCALER_PATH = os.path.join(AI_DIR, "scaler", "scaler_all.save")

print("Loading AI models and scaler...")
model_now = tf.keras.models.load_model(MODEL_NOW_PATH, compile=False)
model_bigru = tf.keras.models.load_model(MODEL_BiGRU_PATH, compile=False)
model_bigru_lstm = tf.keras.models.load_model(MODEL_BiGRU_LSTM_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

def ensemble_predict(X):
    # X shape: [samples, timesteps, 6]

    # Compute acc_mag + gyro_mag
    acc_mag = np.sqrt(X[:, :, 0]**2 + X[:, :, 1]**2 + X[:, :, 2]**2)
    gyro_mag = np.sqrt(X[:, :, 3]**2 + X[:, :, 4]**2 + X[:, :, 5]**2)

    # Append new features → shape becomes: [samples, timesteps, 8]
    X_augmented = np.concatenate(
        [X, acc_mag[..., None], gyro_mag[..., None]],
        axis=-1
    )

    # Scale
    X_scaled = scaler.transform(
        X_augmented.reshape(-1, X_augmented.shape[-1])
    ).reshape(X_augmented.shape)

    # Predict
    p1_now, p1_soon = model_now.predict(X_scaled)
    p2_now, p2_soon = model_bigru.predict(X_scaled)
    p3_now, p3_soon = model_bigru_lstm.predict(X_scaled)

    weights = {"m1": 0.5, "m2": 0.2, "m3": 0.3}

    ensemble_now = (p1_now * weights["m1"]) + (p2_now * weights["m2"]) + (p3_now * weights["m3"])
    ensemble_soon = (p1_soon * weights["m1"]) + (p2_soon * weights["m2"]) + (p3_soon * weights["m3"])

    return ensemble_now, ensemble_soon

import os
import joblib
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "AI", "models", "FINAL_LSTM_Attention.keras")
SCALER_PATH = os.path.join(BASE_DIR, "AI", "scaler", "scaler_all.save")

print("Loading LSTM model and scaler...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

def predict_lstm(X):
    # X shape: [samples, timesteps, 6]
    acc_mag = np.sqrt(X[:, :, 0]**2 + X[:, :, 1]**2 + X[:, :, 2]**2)
    gyro_mag = np.sqrt(X[:, :, 3]**2 + X[:, :, 4]**2 + X[:, :, 5]**2)

    X_aug = np.concatenate([X, acc_mag[..., None], gyro_mag[..., None]], axis=-1)
    X_scaled = scaler.transform(X_aug.reshape(-1, X_aug.shape[-1])).reshape(X_aug.shape)

    probs_now, probs_soon = model.predict(X_scaled)
    return probs_now, probs_soon

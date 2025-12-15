import time
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from AI.train_scripts.train_all import prepare_common_data

print("Preparing test data...")
_, X_test, _, y_test_now, _, y_test_soon = prepare_common_data()

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODELS = {
    "LSTM_Attention": os.path.join(MODELS_DIR, "FINAL_LSTM_Attention.keras"),
    "BiGRU_Attention": os.path.join(MODELS_DIR, "FINAL_BiGRU_Attention.keras"),
    "BiGRU_LSTM_Attention": os.path.join(MODELS_DIR, "FINAL_BiGRU_Attention_LSTM.keras"),
}


results = {}

def evaluate_model(name, path):
    model = tf.keras.models.load_model(path, compile=False)

    start = time.time()
    p_now, p_soon = model.predict(X_test, verbose=0)
    end = time.time()

    time_per_sample = (end - start) / len(X_test)

    y_pred_now = (p_now > 0.5).astype(int)
    y_pred_soon = (p_soon > 0.5).astype(int)

    acc_now = accuracy_score(y_test_now, y_pred_now)
    f1_now = f1_score(y_test_now, y_pred_now)

    acc_soon = accuracy_score(y_test_soon, y_pred_soon)
    f1_soon = f1_score(y_test_soon, y_pred_soon)

    results[name] = {
        "acc_now": acc_now,
        "f1_now": f1_now,
        "acc_soon": acc_soon,
        "f1_soon": f1_soon,
        "time": time_per_sample
    }

    print(f"\n{name}")
    print(f" Accuracy Now : {acc_now:.4f}")
    print(f" F1 Now       : {f1_now:.4f}")
    print(f" Inference    : {time_per_sample*1000:.2f} ms/sample")


for name, path in MODELS.items():
    evaluate_model(name, path)

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

best = min(results.items(), key=lambda x: (1 - x[1]["f1_now"], x[1]["time"]))

for k, v in results.items():
    print(f"{k} -> F1: {v['f1_now']:.4f}, Time: {v['time']*1000:.2f} ms")

print(f"\n🏆 BEST MODEL: {best[0]}")

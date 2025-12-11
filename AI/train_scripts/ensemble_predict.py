import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from train_all import prepare_common_data

def print_metrics(y_true, y_pred, title):
    y_pred_label = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred_label)
    prec = precision_score(y_true, y_pred_label, zero_division=0)
    rec = recall_score(y_true, y_pred_label, zero_division=0)
    f1 = f1_score(y_true, y_pred_label, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_label)

    print(f"\n=== Metrics for {title} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return acc, prec, rec, f1


# ===============================
# Load data
# ===============================
X_train, X_test, y_train_now, y_test_now, y_train_soon, y_test_soon = prepare_common_data()

# ===============================
# Load best models
# ===============================
print("Loading models...")
model1 = tf.keras.models.load_model("../models/FINAL_LSTM_Attention.keras", compile=False)
model2 = tf.keras.models.load_model("../models/FINAL_BiGRU_Attention.keras", compile=False)
model3 = tf.keras.models.load_model("../models/FINAL_BiGRU_Attention_LSTM.keras", compile=False)

# ===============================
# Predict using all models
# ===============================
p1_now, p1_soon = model1.predict(X_test)
p2_now, p2_soon = model2.predict(X_test)
p3_now, p3_soon = model3.predict(X_test)

# ===============================
# Weighted Ensemble
# ===============================
weights = {"m1": 0.5, "m2": 0.2, "m3": 0.3}

ensemble_now = (p1_now * weights["m1"]) + (p2_now * weights["m2"]) + (p3_now * weights["m3"])
ensemble_soon = (p1_soon * weights["m1"]) + (p2_soon * weights["m2"]) + (p3_soon * weights["m3"])

# ===============================
# Evaluate ensemble
# ===============================
print("\n\n==============================")
print("Evaluating WEIGHTED ENSEMBLE (best system)")
print("==============================")

print_metrics(y_test_now, ensemble_now, "Weighted Ensemble — Fall Now")
print_metrics(y_test_soon, ensemble_soon, "Weighted Ensemble — Fall Soon")

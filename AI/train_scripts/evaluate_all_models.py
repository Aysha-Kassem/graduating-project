import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from train_all import prepare_common_data

# ===============================
# Function to print metrics
# ===============================
def print_metrics(y_true, y_pred, title):
    y_pred = np.array(y_pred).reshape(-1)
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
# Load models
# ===============================
print("Loading models...")
model1 = tf.keras.models.load_model("../models/FINAL_LSTM_Attention.keras", compile=False)
model2 = tf.keras.models.load_model("../models/FINAL_BiGRU_Attention.keras", compile=False)
model3 = tf.keras.models.load_model("../models/FINAL_BiGRU_Attention_LSTM.keras", compile=False)

models = {
    "LSTM + Attention": model1,
    "BiGRU + Attention": model2,
    "BiGRU + LSTM + Attention": model3
}

# ===============================
# Evaluate each model separately
# ===============================
results = {}

for name, model in models.items():
    print(f"\n\n==============================")
    print(f"Evaluating: {name}")
    print(f"==============================")

    pred_now, pred_soon = model.predict(X_test)

    print_metrics(y_test_now, pred_now, f"{name} — Fall Now")
    print_metrics(y_test_soon, pred_soon, f"{name} — Fall Soon")

    # Save results
    results[name] = {
        "fall_now": pred_now,
        "fall_soon": pred_soon
    }

# ===============================
# Ensemble Evaluation
# ===============================
print("\n\n==============================")
print("Evaluating ENSEMBLE (average of models)")
print("==============================")

ensemble_now = (
    results["LSTM + Attention"]["fall_now"] +
    results["BiGRU + Attention"]["fall_now"] +
    results["BiGRU + LSTM + Attention"]["fall_now"]
) / 3

ensemble_soon = (
    results["LSTM + Attention"]["fall_soon"] +
    results["BiGRU + Attention"]["fall_soon"] +
    results["BiGRU + LSTM + Attention"]["fall_soon"]
) / 3

print_metrics(y_test_now, ensemble_now, "ENSEMBLE — Fall Now")
print_metrics(y_test_soon, ensemble_soon, "ENSEMBLE — Fall Soon")

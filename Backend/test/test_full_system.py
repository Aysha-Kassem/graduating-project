import sys
import os
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# =================================================
#    Configure import paths
# =================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from app.database import SessionLocal
from app import crud, schemas, models
from app.ai_model import ensemble_predict

# =================================================
#   Simulation Settings
# =================================================
NUM_BATCHES = 5
TIMESTEPS = 50
FEATURES = 6
DELAY_SECONDS = 0.5

# =================================================
#   Start DB Session
# =================================================
db = SessionLocal()

# Prepare plot
plt.ion()
fig, ax = plt.subplots()
ax.set_ylim(0, 1)
ax.set_xlim(1, NUM_BATCHES)
ax.set_xlabel("Batch")
ax.set_ylabel("Probability")
line_now, = ax.plot([], [], 'ro-', label="Fall NOW")
line_soon, = ax.plot([], [], 'bo-', label="Fall SOON")
ax.legend()
now_probs_list, soon_probs_list = [], []

try:
    print("\n========== STARTING FULL SYSTEM TEST WITH LIVE PLOTS ==========\n")

    # =================================================
    # 1. Create Simulation User
    # =================================================
    user_data = schemas.UserCreate(name="Sensor Simulation User", age=25, gender="female")
    user = crud.create_user(db, user_data)
    print(f"[INFO] Created Simulation User → ID = {user.id}\n")

    # =================================================
    # 2. Start Sending Batches of Sensor Data
    # =================================================
    for batch_id in range(1, NUM_BATCHES + 1):
        print(f"\n---------------- BATCH {batch_id}/{NUM_BATCHES} ----------------")

        motion_array = np.random.rand(1, TIMESTEPS, FEATURES)
        motion_records = []

        for t in range(TIMESTEPS):
            motion_data = schemas.MotionDataCreate(
                user_id=user.id,
                acc_x=float(motion_array[0, t, 0]),
                acc_y=float(motion_array[0, t, 1]),
                acc_z=float(motion_array[0, t, 2]),
                gyro_x=float(motion_array[0, t, 3]),
                gyro_y=float(motion_array[0, t, 4]),
                gyro_z=float(motion_array[0, t, 5]),
            )
            record = crud.create_motion_data(db, motion_data)
            motion_records.append(record)

        print(f"[INFO] Saved {len(motion_records)} motion points with acc_mag & gyro_mag.")

        # =================================================
        # 3. Run AI Prediction (Ensemble)
        # =================================================
        now_probs, soon_probs = ensemble_predict(motion_array)
        fall_now = int(now_probs[0] > 0.5)
        fall_soon = int(soon_probs[0] > 0.5)

        now_probs_list.append(float(now_probs[0]))
        soon_probs_list.append(float(soon_probs[0]))

        # Update plot
        line_now.set_data(range(1, len(now_probs_list)+1), now_probs_list)
        line_soon.set_data(range(1, len(soon_probs_list)+1), soon_probs_list)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)

        print(f"[MODEL] Fall NOW Probability:  {float(now_probs[0]):.4f}")
        print(f"[MODEL] Fall SOON Probability: {float(soon_probs[0]):.4f}")
        print(f"[MODEL] Classified NOW:  {fall_now}")
        print(f"[MODEL] Classified SOON: {fall_soon}")

        # =================================================
        # 4. Save Prediction to DB
        # =================================================
        pred_record = schemas.PredictionCreate(
            user_id=user.id,
            motion_batch_id=motion_records[0].id,
            predicted_label=fall_now,
            probability=float(now_probs[0])
        )
        saved_pred = crud.create_prediction(db, pred_record)
        print(f"[INFO] Saved Prediction ID → {saved_pred.id}")

        # =================================================
        # 5. Wait before next batch
        # =================================================
        time.sleep(DELAY_SECONDS)

    # =================================================
    # 6. Fetch all predictions for final review
    # =================================================
    all_preds = db.query(models.Prediction).filter_by(user_id=user.id).all()
    print("\n=========== SUMMARY ===========")
    print(f"Total Predictions Saved: {len(all_preds)}")
    for p in all_preds:
        print(f"• Prediction ID {p.id} | Fall:{p.predicted_label} | Prob:{p.probability:.4f}")

    plt.ioff()
    plt.show()

finally:
    db.close()
    print("\n========== TEST COMPLETED ==========\n")

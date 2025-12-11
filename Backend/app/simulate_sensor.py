import numpy as np
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app import ai_model, crud, schemas

db: Session = SessionLocal()

sample_data = np.random.rand(1, 10, 6).astype(np.float32)
user_id = 1

ensemble_now, ensemble_soon = ai_model.ensemble_predict(sample_data)
result_now = (ensemble_now > 0.5).astype(int).tolist()
result_soon = (ensemble_soon > 0.5).astype(int).tolist()

print("Predictions:")
print("Fall Now Prob:", ensemble_now)
print("Fall Soon Prob:", ensemble_soon)
print("Fall Now:", result_now)
print("Fall Soon:", result_soon)

for i in range(len(result_now)):
    pred = schemas.PredictionCreate(
        user_id=user_id,
        motion_batch_id=1,
        predicted_label=result_now[i],
        probability=float(ensemble_now[i])
    )
    crud.create_prediction(db, pred)

if result_now[0] == 1:
    print(f"🚨 ALERT: Fall detected for user {user_id}!")

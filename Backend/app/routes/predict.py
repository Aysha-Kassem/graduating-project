from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import database, schemas, ai_model, crud
import numpy as np

router = APIRouter(prefix="/predict", tags=["AI Predictions"])

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/")
def predict(sensor_data: schemas.SensorDataBase, db: Session = Depends(get_db)):
    data_array = np.array(sensor_data.data, dtype=np.float32)
    if data_array.ndim != 3:
        raise HTTPException(status_code=400, detail="Data must be 3D: [samples, timesteps, features]")
    
    probs_now, probs_soon = ai_model.ensemble_predict(data_array)
    preds_now = (probs_now > 0.5).astype(int).tolist()
    preds_soon = (probs_soon > 0.5).astype(int).tolist()

    # Save predictions to DB (example for first batch/sample)
    for i in range(len(preds_now)):
        pred = schemas.PredictionCreate(
            user_id=1,  # مثال: استخدم user_id فعلي
            motion_batch_id=1,
            predicted_label=preds_now[i],
            probability=float(probs_now[i])
        )
        crud.create_prediction(db, pred)

    return {
        "pred_now": preds_now,
        "pred_soon": preds_soon,
        "probs_now": probs_now.tolist(),
        "probs_soon": probs_soon.tolist()
    }

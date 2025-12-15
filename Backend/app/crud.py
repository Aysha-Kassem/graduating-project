from sqlalchemy.orm import Session
from . import models, schemas
import numpy as np
from .ai_model import predict_lstm

# -------------------- USERS --------------------
def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# -------------------- MOTION DATA --------------------
def create_motion_data(db: Session, motion: schemas.MotionDataCreate):
    acc_mag = float(np.sqrt(motion.acc_x**2 + motion.acc_y**2 + motion.acc_z**2))
    gyro_mag = float(np.sqrt(motion.gyro_x**2 + motion.gyro_y**2 + motion.gyro_z**2))
    db_motion = models.MotionSensorData(
        **motion.dict(),
        acc_mag=acc_mag,
        gyro_mag=gyro_mag
    )
    db.add(db_motion)
    db.commit()
    db.refresh(db_motion)
    return db_motion

# -------------------- VITAL DATA --------------------
def create_vital_data(db: Session, vital: schemas.VitalDataCreate):
    db_vital = models.VitalSensorData(**vital.dict())
    db.add(db_vital)
    db.commit()
    db.refresh(db_vital)
    return db_vital

# -------------------- PREDICTION --------------------
def create_prediction(db: Session, pred: schemas.PredictionCreate):
    db_pred = models.Prediction(**pred.dict())
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred

# -------------------- PREDICT USING LSTM --------------------
def predict_and_save(db: Session, data: schemas.SensorData):
    X = np.array(data.data, dtype=np.float32)
    probs_now, probs_soon = predict_lstm(X)

    pred_now = (probs_now > 0.5).astype(int).tolist()
    pred_soon = (probs_soon > 0.5).astype(int).tolist()

    for i in range(len(pred_now)):
        pred_schema = schemas.PredictionCreate(
            user_id=data.user_id,
            motion_batch_id=data.motion_batch_id,
            predicted_label=pred_now[i],
            probability=float(probs_now[i])
        )
        create_prediction(db, pred_schema)

    return {"pred_now": pred_now, "pred_soon": pred_soon, "probs_now": probs_now.tolist(), "probs_soon": probs_soon.tolist()}

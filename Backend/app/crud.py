from sqlalchemy.orm import Session
from . import models, schemas
import numpy as np

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

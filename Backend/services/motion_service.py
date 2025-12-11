from sqlalchemy.orm import Session
from models import MotionSensorData
from utils.preprocess import prepare_motion_data

def get_last_motion_window(db: Session, user_id: int, window_size: int = 100):
    rows = db.query(MotionSensorData)\
             .filter(MotionSensorData.user_id == user_id)\
             .order_by(MotionSensorData.timestamp.desc())\
             .limit(window_size)\
             .all()

    rows = rows[::-1]  # من القديم للجديد

    if len(rows) < window_size:
        return None

    return prepare_motion_data(rows)

from pydantic import BaseModel
from datetime import datetime
from typing import List

# -------------------- USERS --------------------
class UserBase(BaseModel):
    name: str
    age: int
    gender: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    class Config:
        from_attributes = True

# -------------------- MOTION DATA --------------------
class MotionDataBase(BaseModel):
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float

class MotionDataCreate(MotionDataBase):
    user_id: int

class MotionData(MotionDataBase):
    id: int
    user_id: int
    timestamp: datetime
    acc_mag: float
    gyro_mag: float
    class Config:
        from_attributes = True

# -------------------- VITAL DATA --------------------
class VitalDataBase(BaseModel):
    heart_rate: float
    blood_pressure: float
    oxygen_saturation: float

class VitalDataCreate(VitalDataBase):
    user_id: int

class VitalData(VitalDataBase):
    id: int
    user_id: int
    timestamp: datetime
    class Config:
        from_attributes = True

# -------------------- PREDICTION --------------------
class PredictionBase(BaseModel):
    predicted_label: int
    probability: float

class PredictionCreate(PredictionBase):
    user_id: int
    motion_batch_id: int

class Prediction(PredictionBase):
    id: int
    user_id: int
    motion_batch_id: int
    timestamp: datetime
    class Config:
        from_attributes = True

# -------------------- PREDICT ROUTER --------------------
class SensorData(BaseModel):
    data: List[List[List[float]]]  # shape: [samples, timesteps, 6]
    user_id: int
    motion_batch_id: int

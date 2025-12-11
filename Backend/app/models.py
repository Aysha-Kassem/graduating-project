from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50))
    age = Column(Integer)
    gender = Column(String(10))

    motions = relationship("MotionSensorData", back_populates="user")
    vitals = relationship("VitalSensorData", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")

class MotionSensorData(Base):
    __tablename__ = "motion_sensor_data"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    acc_x = Column(Float)
    acc_y = Column(Float)
    acc_z = Column(Float)
    gyro_x = Column(Float)
    gyro_y = Column(Float)
    gyro_z = Column(Float)
    acc_mag = Column(Float)
    gyro_mag = Column(Float)

    user = relationship("User", back_populates="motions")

class VitalSensorData(Base):
    __tablename__ = "vital_sensor_data"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    heart_rate = Column(Float)
    blood_pressure = Column(Float)
    oxygen_saturation = Column(Float)

    user = relationship("User", back_populates="vitals")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    motion_batch_id = Column(Integer, ForeignKey("motion_sensor_data.id"))
    predicted_label = Column(Integer)
    probability = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")

# app/routes/motions.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app import crud, schemas

router = APIRouter(prefix="/motions", tags=["Motion Data"])

@router.post("/", response_model=schemas.MotionData)
def create_motion(motion: schemas.MotionDataCreate, db: Session = Depends(get_db)):
    return crud.create_motion_data(db, motion)

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import crud, schemas, database

router = APIRouter(prefix="/motions", tags=["Motions"])

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=schemas.MotionData)
def add_motion(motion: schemas.MotionDataCreate, db: Session = Depends(get_db)):
    return crud.create_motion_data(db, motion)

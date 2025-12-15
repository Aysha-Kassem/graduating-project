from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import schemas, crud
from ..database import get_db

router = APIRouter(prefix="/predict", tags=["predict"])

@router.post("/")
def predict(data: schemas.SensorData, db: Session = Depends(get_db)):
    return crud.predict_and_save(db, data)

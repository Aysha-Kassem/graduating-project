# app/routes/predictions.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app import crud, schemas

router = APIRouter(prefix="/predictions", tags=["Predictions"])

@router.get("/")
def get_predictions(db: Session = Depends(get_db)):
    return db.query(schemas.Prediction).all()

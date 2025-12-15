# app/routes/vitals.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app import crud, schemas

router = APIRouter(prefix="/vitals", tags=["Vital Data"])

@router.post("/", response_model=schemas.VitalData)
def create_vital(vital: schemas.VitalDataCreate, db: Session = Depends(get_db)):
    return crud.create_vital_data(db, vital)

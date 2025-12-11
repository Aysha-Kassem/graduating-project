from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import crud, schemas, database

router = APIRouter(prefix="/vitals", tags=["Vitals"])

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=schemas.VitalData)
def add_vital(vital: schemas.VitalDataCreate, db: Session = Depends(get_db)):
    return crud.create_vital_data(db, vital)

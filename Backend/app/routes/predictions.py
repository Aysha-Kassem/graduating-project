from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import crud, schemas, database

router = APIRouter(prefix="/predictions", tags=["Predictions"])

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/{user_id}")
def get_user_predictions(user_id: int, db: Session = Depends(get_db)):
    preds = db.query(crud.models.Prediction).filter(crud.models.Prediction.user_id == user_id).all()
    return preds

from fastapi import FastAPI
from .routes import users, motions, vitals, predictions, predict

app = FastAPI(title="Fall Detection API")

app.include_router(users.router)
app.include_router(motions.router)
app.include_router(vitals.router)
app.include_router(predictions.router)
app.include_router(predict.router)

@app.get("/")
def root():
    return {"message": "Fall Detection API is running!"}

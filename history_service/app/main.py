from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from . import crud, schemas, models, database

app = FastAPI(title="History Service")

models.Base.metadata.create_all(bind=database.engine)

@app.post("/save", response_model=schemas.RecognitionOut)
def save_recognition(rec: schemas.RecognitionCreate, db: Session = Depends(database.get_db)):
    return crud.save_recognition(db, rec)

@app.get("/stats/{session_id}", response_model=schemas.StatsResponse)
def get_stats(session_id: int, db: Session = Depends(database.get_db)):
    return crud.get_stats_by_session(db, session_id)

@app.get("/all-sessions")
def get_all_sessions(db: Session = Depends(database.get_db)):
    records = db.query(models.Recognition.session_id, models.Recognition.predicted_digit, models.Recognition.confidence).all()
    return [{"session_id": r[0], "predicted_digit": r[1], "confidence": r[2]} for r in records]

@app.get("/health")
def health():
    return {"status": "ok"}
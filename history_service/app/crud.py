from sqlalchemy.orm import Session
from . import models, schemas
from typing import Dict, Any

def save_recognition(db: Session, rec: schemas.RecognitionCreate) -> models.Recognition:
    db_rec = models.Recognition(
        session_id=rec.session_id,
        pixels_json=rec.pixels,
        predicted_digit=rec.predicted_digit,
        confidence=rec.confidence,
        actual_digit=rec.actual_digit
    )
    db.add(db_rec)
    db.commit()
    db.refresh(db_rec)
    return db_rec

def get_stats_by_session(db: Session, session_id: int) -> Dict[str, Any]:
    records = db.query(models.Recognition).filter(models.Recognition.session_id == session_id).all()
    if not records:
        return {"total": 0, "avg_confidence": 0.0, "by_digit": {}}
    total = len(records)
    avg_conf = sum(r.confidence for r in records) / total
    by_digit = {}
    for r in records:
        d = r.predicted_digit
        by_digit[d] = by_digit.get(d, 0) + 1
    return {
        "total": total,
        "avg_confidence": round(avg_conf, 4),
        "by_digit": by_digit
    }
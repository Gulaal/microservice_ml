from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class RecognitionCreate(BaseModel):
    session_id: int
    pixels: List[int]
    predicted_digit: int
    confidence: float
    actual_digit: Optional[int] = None

class RecognitionOut(BaseModel):
    id: int
    session_id: int
    predicted_digit: int
    confidence: float
    actual_digit: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True

class StatsResponse(BaseModel):
    total: int
    avg_confidence: float
    by_digit: dict
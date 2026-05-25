from pydantic import BaseModel
from typing import List

class ImageData(BaseModel):
    pixels: List[int]

class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: List[float]
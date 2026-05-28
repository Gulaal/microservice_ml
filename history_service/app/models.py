from sqlalchemy import Column, BigInteger, Integer, Float, DateTime, JSON
from sqlalchemy.sql import func
from .database import Base

class Recognition(Base):
    __tablename__ = "recognitions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(BigInteger, index=True, nullable=False)
    pixels_json = Column(JSON, nullable=False)
    predicted_digit = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    actual_digit = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
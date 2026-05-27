import httpx
from typing import Dict, Any

ML_SERVICE_URL = "http://ml-service:8001"
HISTORY_SERVICE_URL = "http://history-service:8002"

async def predict_digit(pixels: list) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{ML_SERVICE_URL}/predict",
            json={"pixels": pixels}
        )
        response.raise_for_status()
        return response.json()

async def save_recognition(session_id: int, pixels: list, predicted_digit: int, confidence: float):
    async with httpx.AsyncClient(timeout=5.0) as client:
        await client.post(
            f"{HISTORY_SERVICE_URL}/save",
            json={
                "session_id": session_id,
                "pixels": pixels,
                "predicted_digit": predicted_digit,
                "confidence": confidence
            }
        )

async def get_stats(session_id: int) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{HISTORY_SERVICE_URL}/stats/{session_id}")
        response.raise_for_status()
        return response.json()
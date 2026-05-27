from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from .gateway.client import predict_digit, save_recognition, get_stats
import os

app = FastAPI(title="Web UI + Gateway")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "app/static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_path = os.path.join(BASE_DIR, "app/templates/index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/predict")
async def predict_endpoint(data: dict):
    pixels = data.get("pixels")
    session_id = data.get("session_id")
    if not pixels or len(pixels) != 784:
        raise HTTPException(status_code=400, detail="Invalid pixels array")
    
    ml_result = await predict_digit(pixels)
    await save_recognition(session_id, pixels, ml_result["digit"], ml_result["confidence"])
    return ml_result

@app.get("/api/stats/{session_id}")
async def stats_endpoint(session_id: int):
    return await get_stats(session_id)
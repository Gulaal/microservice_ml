import torch
from fastapi import FastAPI
from .model import MNISTCNN
from .preprocess import preprocess
from .schemas import ImageData, PredictionResponse

app = FastAPI()

model = MNISTCNN()
model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=torch.device('cpu')))
model.eval()

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: ImageData):
    tensor = preprocess(data.pixels)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)
        confidence = probs[0][pred].item()
    return PredictionResponse(digit=pred.item(), confidence=confidence, probabilities=probs[0].tolist())
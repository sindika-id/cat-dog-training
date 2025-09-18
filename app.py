import io
import os
from typing import Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

# ---- Config ----
MODEL_PATH = os.getenv("MODEL_PATH", "models/catdog_model.pth")
LABELS = ["cat", "dog"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Preprocess & Model (loaded once) ----
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model(path: str) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

def read_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

def predict_image(img: Image.Image, model: torch.nn.Module) -> Tuple[str, float]:
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        return LABELS[idx], float(probs[idx].item())

app = FastAPI(title="CatDog Inference API", version="1.0.0")

# Load model at startup
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    # Defer failing hard—surface clear message at /health
    model = None
    load_error = str(e)
else:
    load_error = None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename.")

    try:
        content = await file.read()
        img = read_image(content)
        label, confidence = predict_image(img, model)
        return {"label": label, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

if __name__ == "__main__":
    # For local testing; in production, use: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

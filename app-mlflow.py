# app-mlflow.py
import io
import os
from contextlib import asynccontextmanager
from typing import Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

import mlflow
import mlflow.pytorch

# ---- Config ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://165.232.169.40:5000")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/catdog-v1.0@cherry")
LABELS = os.getenv("LABELS", "cat,dog").split(",")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optional: Artifact store credentials (MinIO/S3/etc.)
os.environ.setdefault("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", "minio"))
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"))
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://165.232.169.40:9000"))

# ---- Preprocess ----
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def read_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

def predict_image(img: Image.Image, model: nn.Module) -> Tuple[str, float]:
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        return LABELS[idx], float(probs[idx].item())

# ---- Lifespan (replaces @app.on_event) ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.load_error = None
    try:
        if not MLFLOW_MODEL_URI:
            raise RuntimeError(
                "Missing MLFLOW_MODEL_URI (expected models:/<name>@<alias> or models:/<name>/<version>)"
            )

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.pytorch.load_model(MLFLOW_MODEL_URI, map_location="cpu")
        model.to(DEVICE).eval()

        app.state.model = model
        yield
    except Exception as e:
        app.state.load_error = str(e)
        # Still yield so /health can report the failure
        yield
    finally:
        # Optional cleanup
        if getattr(app.state, "model", None) is not None:
            try:
                app.state.model.to("cpu")
            except Exception:
                pass
        app.state.model = None

# ---- FastAPI ----
app = FastAPI(
    title="CatDog Inference API (MLflow)",
    version="1.1.0",
    lifespan=lifespan,
)

@app.get("/health")
def health(request: Request):
    model = request.app.state.model
    load_error = request.app.state.load_error
    if model is None:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Model not loaded: {load_error}",
                "tracking_uri": MLFLOW_TRACKING_URI,
                "model_uri": MLFLOW_MODEL_URI,
                "env": {
                    "AWS_ACCESS_KEY_ID": bool(os.getenv("AWS_ACCESS_KEY_ID")),
                    "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL") or None,
                },
            },
        )
    return {
        "status": "ok",
        "device": str(DEVICE),
        "labels": LABELS,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_uri": MLFLOW_MODEL_URI,
    }

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    model = request.app.state.model
    load_error = request.app.state.load_error

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
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

if __name__ == "__main__":
    uvicorn.run("app-mlflow:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

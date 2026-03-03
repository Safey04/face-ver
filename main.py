"""
ArcFace Face Embedding API
---------------------------
Receives a face image, sends it to a Roboflow Workflow for
face detection + cropping, then returns 512-d ArcFace embeddings
for each detected face.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000

Environment variables:
    ROBOFLOW_API_KEY      - Your Roboflow API key
    ROBOFLOW_WORKSPACE    - Your Roboflow workspace name
    ROBOFLOW_WORKFLOW_ID  - Your Roboflow workflow ID
"""

import base64
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "TqKdLCjMBWTHVC8CNhN7")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "validationmodel")
ROBOFLOW_WORKFLOW_ID = os.environ.get("ROBOFLOW_WORKFLOW_ID", "custom-workflow-6")

# ---------------------------------------------------------------------------
# Global model reference (loaded once at startup)
# ---------------------------------------------------------------------------
face_app = None


def _load_model():
    from insightface.app import FaceAnalysis

    logger.info("Loading InsightFace buffalo_l model ...")
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("Model loaded successfully.")
    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_app
    face_app = _load_model()
    if not ROBOFLOW_API_KEY:
        logger.warning("ROBOFLOW_API_KEY is not set!")
    yield
    logger.info("Shutting down ...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ArcFace Face Embedding API",
    version="1.0.0",
    description="Sends image to Roboflow for face cropping, returns ArcFace embeddings.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _get_embedding(img: np.ndarray) -> Optional[np.ndarray]:
    """Get embedding, first trying detection, then falling back to direct recognition."""
    # Try normal detection + recognition
    faces = face_app.get(img)
    if faces:
        best = max(faces, key=lambda f: f.det_score)
        return best.normed_embedding

    # Face already cropped by Roboflow — detector can't find it again.
    # Resize to 112x112 and run recognition model directly.
    resized = cv2.resize(img, (112, 112))
    blob = np.transpose(resized, (2, 0, 1)).astype(np.float32)
    blob = (blob - 127.5) / 127.5
    blob = np.expand_dims(blob, axis=0)

    rec_model = face_app.models["recognition"]
    embedding = rec_model.session.run(None, {rec_model.session.get_inputs()[0].name: blob})[0][0]
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


async def _call_roboflow(image_bytes: bytes) -> list[dict]:
    """Send image to Roboflow Workflow and return list of face bounding boxes."""
    url = (
        f"https://serverless.roboflow.com/{ROBOFLOW_WORKSPACE}/workflows/{ROBOFLOW_WORKFLOW_ID}"
    )
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {"type": "base64", "value": image_b64},
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload)

    if resp.status_code != 200:
        logger.error("Roboflow error %s: %s", resp.status_code, resp.text)
        raise HTTPException(
            status_code=502,
            detail=f"Roboflow returned {resp.status_code}: {resp.text}",
        )

    data = resp.json()
    outputs = data.get("outputs", [])
    if not outputs:
        return []

    predictions = outputs[0].get("predictions", {}).get("predictions", [])
    return predictions


def _crop_face(img: np.ndarray, det: dict) -> np.ndarray:
    """Crop a face from the image using Roboflow's center-x/y/w/h bbox."""
    h, w = img.shape[:2]
    cx, cy = det["x"], det["y"]
    bw, bh = det["width"], det["height"]

    x1 = max(0, int(cx - bw / 2))
    y1 = max(0, int(cy - bh / 2))
    x2 = min(w, int(cx + bw / 2))
    y2 = min(h, int(cy + bh / 2))

    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class FaceEmbedding(BaseModel):
    embedding: list[float]
    dimension: int


class EmbedResponse(BaseModel):
    faces: list[FaceEmbedding]
    face_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=face_app is not None)


@app.post("/embed", response_model=EmbedResponse)
async def embed(
    image: UploadFile = File(..., description="Image containing faces"),
):
    """
    Send an image to Roboflow for face detection/cropping,
    then return ArcFace embeddings for each detected face.
    """
    image_bytes = await image.read()

    # Step 1: Decode the uploaded image
    try:
        img = _decode_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}")

    # Step 2: Send to Roboflow for face detection
    try:
        detections = await _call_roboflow(image_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Roboflow call failed: {exc}")

    if not detections:
        raise HTTPException(status_code=422, detail="No faces detected by Roboflow")

    # Step 3: Crop each face and compute embeddings
    results = []
    for i, det in enumerate(detections):
        crop = _crop_face(img, det)
        emb = _get_embedding(crop)
        if emb is None:
            logger.warning("No embedding for face %d, skipping", i)
            continue

        results.append(FaceEmbedding(embedding=emb.tolist(), dimension=len(emb)))

    return EmbedResponse(faces=results, face_count=len(results))

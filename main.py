"""
ArcFace Face Embedding API
---------------------------
Receives a face image and returns 512-d ArcFace embeddings
for each detected face. Detection and recognition are handled
entirely by InsightFace (buffalo_l) — no external API calls.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

import cv2
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
    app.prepare(ctx_id=0, det_size=(320, 320))
    logger.info("Model loaded successfully.")
    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_app
    face_app = _load_model()
    yield
    logger.info("Shutting down ...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ArcFace Face Embedding API",
    version="1.0.0",
    description="Detects faces and returns 512-d ArcFace embeddings.",
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
    """Detect faces and return ArcFace embeddings for each one."""
    try:
        img = _decode_image(await image.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}")

    faces = face_app.get(img)
    if not faces:
        raise HTTPException(status_code=422, detail="No faces detected")

    results = []
    for face in sorted(faces, key=lambda f: f.det_score, reverse=True):
        emb = face.normed_embedding
        results.append(FaceEmbedding(embedding=emb.tolist(), dimension=len(emb)))

    return EmbedResponse(faces=results, face_count=len(results))

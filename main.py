"""
ArcFace Face Verification API
------------------------------
Lightweight FastAPI microservice that receives two face images
and returns a cosine-similarity score + match decision.

Designed to pair with a Roboflow Workflow that handles
face detection and cropping upstream.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
    """Load InsightFace model. Called once during app lifespan."""
    from insightface.app import FaceAnalysis

    logger.info("Loading InsightFace buffalo_l model ...")
    app = FaceAnalysis(
        name="buffalo_l",
        # Only load detection + recognition -- skip landmarks & gender/age
        # This saves memory and speeds up inference
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("Model loaded successfully.")
    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook -- load model once into memory."""
    global face_app
    face_app = _load_model()
    yield
    logger.info("Shutting down ...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ArcFace Face Verification API",
    version="1.0.0",
    description=(
        "Receives two face images and returns a cosine-similarity score. "
        "Designed to work with Roboflow Workflows for face detection + cropping."
    ),
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
def _read_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR numpy array."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _get_embedding(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Run InsightFace detection + recognition on an image.
    Returns the 512-d normalized embedding of the best-detected face,
    or None if no face is found.
    """
    faces = face_app.get(img)
    if not faces:
        return None
    # Pick the face with the highest detection confidence
    best = max(faces, key=lambda f: f.det_score)
    return best.normed_embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class VerifyResponse(BaseModel):
    similarity: float
    match: bool
    threshold: float
    time_ms: float
    error: Optional[str] = None


class EmbedResponse(BaseModel):
    embedding: list
    dimension: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", model_loaded=face_app is not None)


@app.post("/verify", response_model=VerifyResponse)
async def verify(
    image_1: UploadFile = File(..., description="Reference face image"),
    image_2: UploadFile = File(..., description="Probe face image"),
    threshold: float = Form(0.45, description="Cosine-similarity threshold"),
):
    """
    Compare two face images and return a similarity score.

    - **image_1**: Reference / enrolled face (JPEG or PNG).
    - **image_2**: Probe / query face (JPEG or PNG).
    - **threshold**: Score >= threshold --> MATCH (default 0.45).
    """
    t0 = time.perf_counter()

    # --- Read images ---
    try:
        bytes_1 = await image_1.read()
        bytes_2 = await image_2.read()
        img1 = _read_image(bytes_1)
        img2 = _read_image(bytes_2)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}")

    # --- Extract embeddings ---
    emb1 = _get_embedding(img1)
    emb2 = _get_embedding(img2)

    elapsed = (time.perf_counter() - t0) * 1000

    if emb1 is None or emb2 is None:
        missing = []
        if emb1 is None:
            missing.append("image_1")
        if emb2 is None:
            missing.append("image_2")
        return VerifyResponse(
            similarity=0.0,
            match=False,
            threshold=threshold,
            time_ms=round(elapsed, 1),
            error=f"No face detected in: {', '.join(missing)}",
        )

    # --- Compare ---
    sim = cosine_similarity(emb1, emb2)

    return VerifyResponse(
        similarity=round(sim, 4),
        match=sim >= threshold,
        threshold=threshold,
        time_ms=round(elapsed, 1),
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(
    image: UploadFile = File(..., description="Face image to embed"),
):
    """
    Return the 512-d ArcFace embedding for a single face image.
    Useful for pre-computing and storing reference embeddings.
    """
    try:
        img = _read_image(await image.read())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}")

    emb = _get_embedding(img)
    if emb is None:
        raise HTTPException(status_code=422, detail="No face detected in image")

    return EmbedResponse(embedding=emb.tolist(), dimension=len(emb))

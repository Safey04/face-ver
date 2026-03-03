"""
Face Verification Client
-------------------------
Orchestrates the full pipeline:
  1. Send both images to the Roboflow hosted workflow for face detection + cropping
  2. Send the cropped faces to the ArcFace API for verification
  3. Return the final match result

Usage:
    python verify_faces.py --image1 reference.jpg --image2 probe.jpg

Environment variables:
    ROBOFLOW_API_KEY     - Your Roboflow API key
    ROBOFLOW_WORKSPACE   - Your Roboflow workspace name
    ROBOFLOW_WORKFLOW_ID - Your Roboflow workflow ID
    ARCFACE_API_URL      - URL of your deployed ArcFace API (e.g. https://your-app.railway.app)
    SIMILARITY_THRESHOLD - Cosine similarity threshold (default: 0.45)
"""

import argparse
import base64
import io
import os
import sys
import time

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "YOUR_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "your-workspace")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "face-detect-crop")
ARCFACE_API_URL = os.getenv("ARCFACE_API_URL", "http://localhost:8000")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))

# Roboflow hosted API base
ROBOFLOW_API_BASE = "https://serverless.roboflow.com"


# ---------------------------------------------------------------------------
# Step 1: Detect and crop faces via Roboflow Workflow
# ---------------------------------------------------------------------------
def detect_and_crop_face(image_path: str) -> bytes:
    """
    Send an image to the Roboflow hosted workflow.
    Returns the cropped face image as JPEG bytes.

    The workflow does:
      - Face detection (using a pre-trained model from Roboflow Universe)
      - Dynamic crop (isolate the detected face region)
    """
    url = (
        f"{ROBOFLOW_API_BASE}/{ROBOFLOW_WORKSPACE}/workflows/{ROBOFLOW_WORKFLOW_ID}"
    )

    # Read and base64-encode the image
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {"type": "base64", "value": image_b64},
            "confidence": 0.5,
        },
    }

    print(f"  Sending {image_path} to Roboflow workflow ...")
    resp = requests.post(url, json=payload, timeout=20)
    resp.raise_for_status()
    result = resp.json()

    # Extract the cropped face image
    # The workflow returns crops as base64-encoded images
    outputs = result.get("outputs", [{}])
    if not outputs:
        raise RuntimeError(f"No outputs returned from workflow for {image_path}")

    first_output = outputs[0] if isinstance(outputs, list) else outputs

    crops = first_output.get("face_crops", [])
    if not crops:
        raise RuntimeError(f"No face detected in {image_path}")

    # Take the first (highest confidence) crop
    crop_data = crops[0]

    # Depending on workflow output format, crop may be base64 or dict
    if isinstance(crop_data, dict):
        crop_b64 = crop_data.get("value", crop_data.get("image", ""))
    elif isinstance(crop_data, str):
        crop_b64 = crop_data
    else:
        raise RuntimeError(f"Unexpected crop format: {type(crop_data)}")

    return base64.b64decode(crop_b64)


# ---------------------------------------------------------------------------
# Step 2: Verify faces via ArcFace API
# ---------------------------------------------------------------------------
def verify_faces(crop1_bytes: bytes, crop2_bytes: bytes) -> dict:
    """
    Send two cropped face images to the ArcFace API for verification.
    Returns the API response as a dict.
    """
    url = f"{ARCFACE_API_URL}/verify"

    files = {
        "image_1": ("face1.jpg", io.BytesIO(crop1_bytes), "image/jpeg"),
        "image_2": ("face2.jpg", io.BytesIO(crop2_bytes), "image/jpeg"),
    }
    data = {"threshold": str(SIMILARITY_THRESHOLD)}

    print(f"  Sending cropped faces to ArcFace API ...")
    resp = requests.post(url, files=files, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_verification(image1_path: str, image2_path: str) -> dict:
    """
    Full face verification pipeline:
      1. Detect + crop face in image1 via Roboflow
      2. Detect + crop face in image2 via Roboflow
      3. Compare both crops via ArcFace API
    """
    print("\n=== Face Verification Pipeline ===\n")
    t0 = time.perf_counter()

    # Step 1: Detect and crop faces
    print("[Step 1/3] Detecting face in reference image ...")
    crop1 = detect_and_crop_face(image1_path)
    print(f"  -> Cropped face: {len(crop1):,} bytes")

    print("[Step 2/3] Detecting face in probe image ...")
    crop2 = detect_and_crop_face(image2_path)
    print(f"  -> Cropped face: {len(crop2):,} bytes")

    # Step 2: Verify
    print("[Step 3/3] Comparing faces via ArcFace ...")
    result = verify_faces(crop1, crop2)

    elapsed = (time.perf_counter() - t0) * 1000
    result["total_pipeline_ms"] = round(elapsed, 1)

    # Print results
    print(f"\n{'=' * 42}")
    print(f"  Similarity:  {result['similarity']:.4f}")
    print(f"  Threshold:   {result['threshold']}")
    print(f"  Match:       {'YES' if result['match'] else 'NO'}")
    print(f"  ArcFace ms:  {result['time_ms']}")
    print(f"  Total ms:    {result['total_pipeline_ms']}")
    if result.get("error"):
        print(f"  Error:       {result['error']}")
    print(f"{'=' * 42}\n")

    return result


# ---------------------------------------------------------------------------
# Alternative: Skip Roboflow, send full images directly to ArcFace
# ---------------------------------------------------------------------------
def verify_direct(image1_path: str, image2_path: str) -> dict:
    """
    Skip Roboflow entirely -- send full images directly to ArcFace API.
    InsightFace has its own face detector built in, so this works too.

    Use this for simpler setups where you don't need
    Roboflow's visualization or filtering.
    """
    print("\n=== Direct ArcFace Verification (no Roboflow) ===\n")
    t0 = time.perf_counter()

    with open(image1_path, "rb") as f1, open(image2_path, "rb") as f2:
        files = {
            "image_1": ("face1.jpg", f1, "image/jpeg"),
            "image_2": ("face2.jpg", f2, "image/jpeg"),
        }
        data = {"threshold": str(SIMILARITY_THRESHOLD)}
        resp = requests.post(
            f"{ARCFACE_API_URL}/verify", files=files, data=data, timeout=30
        )
        resp.raise_for_status()
        result = resp.json()

    elapsed = (time.perf_counter() - t0) * 1000
    result["total_pipeline_ms"] = round(elapsed, 1)

    print(f"  Similarity:  {result['similarity']:.4f}")
    print(f"  Match:       {'YES' if result['match'] else 'NO'}")
    print(f"  Total ms:    {result['total_pipeline_ms']}\n")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Face Verification: Roboflow + ArcFace pipeline"
    )
    parser.add_argument("--image1", required=True, help="Path to reference face image")
    parser.add_argument("--image2", required=True, help="Path to probe face image")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Skip Roboflow, send full images directly to ArcFace",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override similarity threshold",
    )
    args = parser.parse_args()

    if args.threshold is not None:
        global SIMILARITY_THRESHOLD
        SIMILARITY_THRESHOLD = args.threshold

    # Validate files exist
    for path in [args.image1, args.image2]:
        if not os.path.isfile(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)

    if args.direct:
        verify_direct(args.image1, args.image2)
    else:
        run_verification(args.image1, args.image2)


if __name__ == "__main__":
    main()

# Face Verification Pipeline

**Roboflow Workflows + InsightFace ArcFace — Hybrid Architecture**

A production-grade face verification (1:1 matching) system that combines
Roboflow's hosted computer vision workflows with InsightFace's ArcFace
model for accurate identity verification.

---

## Architecture

```
                    ROBOFLOW HOSTED WORKFLOW
                    (Serverless API - managed)
┌─────────────────────────────────────────────────┐
│                                                 │
│  Input Image ──► Face Detection ──► Dynamic Crop│
│                  (RF Universe        (isolate   │
│                   model)              face)     │
│                                                 │
└──────────────────────┬──────────────────────────┘
                       │ cropped face image
                       ▼
              ┌─────────────────────┐
              │  YOUR ARCFACE API   │
              │  (Railway / Render  │
              │   / Fly.io)         │
              │                     │
              │  ► detect face      │
              │  ► 512-d embedding  │
              │  ► cosine similarity│
              │  ► match decision   │
              └─────────────────────┘
```

**Why this split?**

- Roboflow handles the managed CV pipeline (detection, cropping, visualization)
  with zero infrastructure on your end.
- The ArcFace API is a tiny stateless microservice (~$5/month) that provides
  99%+ face verification accuracy — something CLIP cannot achieve.

---

## Project Structure

```
face-verification/
├── arcface-api/              # The ArcFace microservice
│   ├── main.py               # FastAPI app with /verify and /embed endpoints
│   ├── requirements.txt      # Python dependencies
│   └── Dockerfile            # Ready to deploy to Railway/Render/Fly.io
│
├── roboflow-client/          # Client that orchestrates the pipeline
│   ├── verify_faces.py       # Full pipeline: Roboflow → ArcFace → result
│   ├── workflow_definition.json  # Roboflow Workflow JSON (import into UI)
│   ├── requirements.txt      # Client dependencies
│   └── .env.example          # Environment variable template
│
└── README.md                 # This file
```

---

## Setup Guide

### Part 1: Deploy the ArcFace API

#### Option A: Railway (recommended, simplest)

1. Push the `arcface-api/` folder to a GitHub repo.
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub.
3. Railway auto-detects the Dockerfile and deploys.
4. Note your app URL (e.g., `https://arcface-api-production.up.railway.app`).

#### Option B: Render

1. Push to GitHub.
2. Go to [render.com](https://render.com) → New Web Service → Connect repo.
3. Set:
   - **Build Command**: `docker build -t arcface .`
   - **Start Command**: auto-detected from Dockerfile
4. Choose a plan with at least 1 GB RAM (the model needs ~800 MB).

#### Option C: Local Docker

```bash
cd arcface-api/
docker build -t arcface-api .
docker run -p 8000:8000 arcface-api
```

#### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Verify two faces
curl -X POST http://localhost:8000/verify \
  -F "image_1=@reference.jpg" \
  -F "image_2=@probe.jpg" \
  -F "threshold=0.45"
```

Expected response:
```json
{
  "similarity": 0.7234,
  "match": true,
  "threshold": 0.45,
  "time_ms": 87.3,
  "error": null
}
```

---

### Part 2: Set Up the Roboflow Workflow

1. Go to [app.roboflow.com](https://app.roboflow.com) → your workspace → **Workflows**.
2. Click **Create Workflow** → **Build your own**.
3. **Option A (UI):** Add blocks manually:
   - Add **Object Detection Model** block → select `face-detection-mik1i/5`
     from Roboflow Universe (or any face detection model you prefer).
   - Add **Dynamic Crop** block → connect predictions from the detection block.
   - Add **Bounding Box Visualization** block (optional, for debugging).
4. **Option B (JSON):** Click the JSON editor icon and paste the contents
   of `roboflow-client/workflow_definition.json`.
5. Click **Save Workflow**. Note your workspace name and workflow ID.

---

### Part 3: Run the Client

```bash
cd roboflow-client/

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ROBOFLOW_API_KEY="your_key"
export ROBOFLOW_WORKSPACE="your-workspace"
export ROBOFLOW_WORKFLOW_ID="face-detect-crop"
export ARCFACE_API_URL="https://your-arcface-app.railway.app"

# Run full pipeline (Roboflow detection → ArcFace verification)
python verify_faces.py --image1 reference.jpg --image2 probe.jpg

# Or skip Roboflow and send full images directly to ArcFace
# (InsightFace has its own detector built in)
python verify_faces.py --image1 reference.jpg --image2 probe.jpg --direct

# Override threshold
python verify_faces.py --image1 ref.jpg --image2 probe.jpg --threshold 0.50
```

---

## API Reference

### ArcFace API Endpoints

#### `POST /verify`

Compare two face images.

| Parameter  | Type        | Default | Description                          |
|------------|-------------|---------|--------------------------------------|
| image_1    | File (JPEG/PNG) | required | Reference face image           |
| image_2    | File (JPEG/PNG) | required | Probe face image               |
| threshold  | float       | 0.45    | Similarity cutoff for MATCH          |

**Response:**
```json
{
  "similarity": 0.7234,
  "match": true,
  "threshold": 0.45,
  "time_ms": 87.3,
  "error": null
}
```

#### `POST /embed`

Get the 512-dimensional embedding for a single face.

| Parameter | Type            | Description          |
|-----------|-----------------|----------------------|
| image     | File (JPEG/PNG) | Face image to embed  |

**Response:**
```json
{
  "embedding": [0.0123, -0.0456, ...],
  "dimension": 512
}
```

#### `GET /health`

Health check.

---

## Threshold Tuning Guide

ArcFace cosine similarity ranges from -1 to 1 for normalized embeddings:

| Threshold | Behavior                                                  |
|-----------|-----------------------------------------------------------|
| 0.30      | Very lenient — allows more false positives                |
| 0.40      | Lenient — good for low-security use cases                 |
| **0.45**  | **Balanced (recommended starting point)**                 |
| 0.50      | Moderate — fewer false positives, some false negatives    |
| 0.60      | Strict — high confidence required                         |
| 0.70+     | Very strict — may reject valid matches with poor images   |

**Tips:**
- Start at 0.45 and test with your actual data.
- Photo ID vs. selfie: use 0.40 (lighting/angle differences are large).
- Same-session photos: use 0.55+ (should be very similar).
- Collect a test set of known match/non-match pairs and plot the score
  distribution to pick the optimal threshold for your use case.

---

## Performance

| Component               | Typical Latency       |
|-------------------------|-----------------------|
| Roboflow face detection | 200-500ms (hosted)    |
| ArcFace embedding (CPU) | 50-150ms per image    |
| Cosine similarity       | < 1ms                 |
| **Total pipeline**      | **500ms - 1.2s**      |

For faster performance:
- Deploy ArcFace API on a GPU instance (~2ms per face).
- Use `buffalo_s` instead of `buffalo_l` (3x faster, slightly less accurate).
- Pre-compute reference embeddings using the `/embed` endpoint.

---

## Licensing

- **InsightFace buffalo_l models**: Available for non-commercial research.
  For commercial use, contact recognition-oss-pack@insightface.ai.
- **Roboflow**: Usage is metered per your Roboflow plan.
- **This code**: MIT License.

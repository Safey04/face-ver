FROM python:3.11-slim

# Install system deps for opencv + build tools for insightface
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the buffalo_l model at build time
# so first request doesn't have a cold-start delay
RUN python -c "from insightface.app import FaceAnalysis; \
    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection','recognition'], \
    providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640,640)); \
    print('Model pre-downloaded successfully')"

COPY main.py .

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

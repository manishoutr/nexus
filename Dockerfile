# Build args: WITH_ML=1 installs sentence-transformers (pulls torch, large image). WITH_ONNX=1 is the small alternative.
# Default is the light image that uses the hashing embedding fallback.
FROM python:3.11-slim
ARG WITH_ML=0
# WITH_ONNX=1: torch-free semantic embeddings (onnxruntime). The model files are downloaded at build time.
ARG WITH_ONNX=0
ENV HF_HOME=/app/.hf-cache
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# The app runs as a non-root user, so the default SQLite file must live in /app/data.
ENV DATABASE_URL=sqlite:////app/data/nexus.db DATA_DIR=/app/data
WORKDIR /app
COPY requirements.txt requirements-ml.txt requirements-onnx.txt ./
RUN pip install -r requirements.txt && \
    if [ "$WITH_ML" = "1" ]; then pip install -r requirements-ml.txt; fi
RUN if [ "$WITH_ONNX" = "1" ]; then pip install -r requirements-onnx.txt && \
    python -c "from huggingface_hub import hf_hub_download as d; r='sentence-transformers/all-MiniLM-L6-v2'; d(r,'onnx/model.onnx'); d(r,'tokenizer.json')"; fi
COPY . .
RUN useradd -m nexus && mkdir -p /app/data /app/.hf-cache && chown -R nexus /app/data /app/.hf-cache
USER nexus
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')"
# Migrations run inside the app on startup (init_db -> alembic upgrade head).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

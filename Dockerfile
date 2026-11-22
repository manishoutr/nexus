# Build args: WITH_ML=1 installs sentence-transformers (pulls torch, large image).
# Default is the light image that uses the hashing embedding fallback.
FROM python:3.11-slim
ARG WITH_ML=0
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt && \
    if [ "$WITH_ML" = "1" ]; then pip install -r requirements-ml.txt; fi
COPY . .
RUN useradd -m nexus && mkdir -p /app/data && chown -R nexus /app/data
USER nexus
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')"
# Migrations run inside the app on startup (init_db -> alembic upgrade head).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

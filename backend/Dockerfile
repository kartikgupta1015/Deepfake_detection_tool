# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps: ffmpeg + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies to an isolated prefix
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Final image ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy system libraries needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create a non-root user for security
RUN useradd -m deepshield && chown -R deepshield:deepshield /app
USER deepshield

# Persist the SQLite database in a volume
VOLUME ["/app/data"]
ENV DB_PATH=/app/data/deepshield.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

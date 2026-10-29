# Two stages: build wheels once, then ship a slim runtime without compilers.
FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip build \
    && pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="ScamShield" \
      org.opencontainers.image.description="Scam detection for WhatsApp conversations"

# Never run the web app as root.
RUN useradd --create-home --uid 10001 scamshield
WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels scamshield \
    && rm -rf /wheels

COPY data ./data
COPY models ./models

ENV SCAMSHIELD_HOST=0.0.0.0 \
    SCAMSHIELD_PORT=8000 \
    SCAMSHIELD_DATA_DIR=/app/data \
    SCAMSHIELD_MODELS_DIR=/app/models \
    SCAMSHIELD_CORS_ORIGINS=http://localhost:8000 \
    SCAMSHIELD_LOG_JSON=true \
    PYTHONUNBUFFERED=1

USER scamshield
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

CMD ["scamshield-serve"]

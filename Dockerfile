FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SEMSEARCH_EMBEDDER=hashing \
    SEMSEARCH_DATA_DIR=/app/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# INSTALL_ML=1 also installs sentence-transformers and downloads nothing until
# the process is started with SEMSEARCH_EMBEDDER=minilm.
ARG INSTALL_ML=0
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
COPY scripts ./scripts

RUN pip install --upgrade pip \
    && if [ "$INSTALL_ML" = "1" ]; then \
         pip install --no-cache-dir ".[ml]"; \
       else \
         pip install --no-cache-dir .; \
       fi \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "uvicorn", "semsearch.api:app", "--host", "0.0.0.0", "--port", "8000"]

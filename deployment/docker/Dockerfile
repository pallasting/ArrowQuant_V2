# Multi-stage build for minimal image size
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements-arrow.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-arrow.txt

# Copy application code
COPY llm_compression/ ./llm_compression/
COPY setup.py ./
RUN pip install -e .

# Copy models (if available)
COPY llm_compression/models/ ./llm_compression/models/

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Run the service
CMD ["uvicorn", "llm_compression.inference.server:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]

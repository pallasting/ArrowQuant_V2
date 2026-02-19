# ArrowEngine Production Dockerfile
# Optimized for CPU inference with minimal image size

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY llm_compression/ ./llm_compression/
COPY setup.py .
COPY README.md .

# Install package in editable mode
RUN pip install -e .

# Create model directory
RUN mkdir -p /app/models

# Set environment variables
ENV MODEL_PATH=/app/models/minilm
ENV DEVICE=cpu
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "llm_compression.api:app", "--host", "0.0.0.0", "--port", "8000"]

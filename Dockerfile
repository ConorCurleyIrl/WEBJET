FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/data /app/logs

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
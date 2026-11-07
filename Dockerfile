# ============================================================
# Multi-Stage Production Dockerfile for Retail ML System
# ============================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install minimal OS dependencies required for compilation (C++ for XGBoost/LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy packaging metadata first to leverage Docker layer caching
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Copy application source code and configurations
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY train_model.py .

# Create directories for persistent volume mounts
RUN mkdir -p data/raw data/processed models/plots mlruns

# Expose Streamlit default port
EXPOSE 8501

# Health check to ensure service responsiveness
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default startup command launches Streamlit application
CMD ["streamlit", "run", "app/streamlit_app.py"]

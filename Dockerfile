# ---- Python image -----------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for LightGBM / CatBoost wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (layered for better caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install the project package
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-deps -e .

# Copy application code & default entrypoints
COPY scripts/ ./scripts/

# Default model source (override with MODEL_URI for MLflow registry)
ENV MODEL_SOURCE=local

EXPOSE 8000

# Default command: REST API. Override to launch the dashboard:
#   docker run -p 8501:8501 <image> streamlit
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

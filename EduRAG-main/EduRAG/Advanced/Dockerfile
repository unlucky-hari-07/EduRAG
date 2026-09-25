# EduRAG Pro - single image shared by both the backend (FastAPI) and
# frontend (Streamlit) services. docker-compose.yml runs two containers
# from this same image with different startup commands.

FROM python:3.11-slim

WORKDIR /app

# System dependencies required by PyMuPDF and sentence-transformers/faiss.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No secrets are baked into the image - OLLAMA_BASE_URL, OLLAMA_MODEL_NAME
# and other configuration are supplied at runtime via environment variables
# (see docker-compose.yml and .env.example). No API key is required since
# Ollama runs as its own service/container.

EXPOSE 8000 8501

# Default command; overridden per-service in docker-compose.yml.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

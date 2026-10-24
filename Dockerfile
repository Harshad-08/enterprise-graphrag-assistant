FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only PyTorch keeps the image far smaller than the default CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY frontend ./frontend
COPY .streamlit ./.streamlit
COPY scripts ./scripts
COPY data ./data
COPY tests ./tests
COPY pyproject.toml .

# Bake the local models into the image so containers start without downloading.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
CrossEncoder('cross-encoder/nli-deberta-v3-xsmall')"

RUN useradd --create-home appuser && mkdir -p /app/data/uploads /app/data/evaluation/results \
    && chown -R appuser /app /models
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

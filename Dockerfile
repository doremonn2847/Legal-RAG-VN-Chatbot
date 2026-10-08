FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5
WORKDIR /app
RUN useradd --create-home --uid 10001 app

FROM base AS ui
COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt
COPY app.py .
COPY css ./css
USER app
EXPOSE 7860
CMD ["python", "app.py"]

FROM base AS api
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 gcc g++ \
    && rm -rf /var/lib/apt/lists/*
# CPU embedding/reranking; Ollama inference runs on the host.
RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
COPY api.py config.py ./
COPY main ./main
COPY utils ./utils
RUN mkdir -p index data/utils /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app/.cache
ENV HF_HOME=/home/app/.cache/huggingface
USER app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

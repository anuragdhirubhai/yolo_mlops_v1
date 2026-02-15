FROM python:3.11-slim

WORKDIR /app

COPY api/ api/
COPY database/ database/
COPY model_registry/ model_registry/
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt


CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-10000}"]

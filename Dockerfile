FROM python:3.11-slim

WORKDIR /app

COPY api/ api/
COPY database/ database/
COPY model_registry/ model_registry/
COPY yolov8n.pt yolov8n.pt
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

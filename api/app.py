from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import mlflow
import mlflow.pyfunc

from database.db import init_db, insert_prediction

app = FastAPI(title="YOLO Object Detection API")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_URI = "models:/YOLO_Object_Detector@production"
model = mlflow.pyfunc.load_model(MODEL_URI)
MODEL_VERSION = "registry_production"

init_db()

@app.get("/")
def home():
    return {"message": "YOLO Object Detection API is running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # IMPORTANT: use predict()
    results = model.predict(image)

    detections = []

    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls[0])]
        confidence = float(box.conf[0])

        detections.append({
            "class_name": class_name,
            "confidence": confidence
        })

        insert_prediction(
            filename=file.filename,
            detected_class=class_name,
            confidence=confidence,
            model_version=MODEL_VERSION
        )

    return {"detections": detections}


import sqlite3
from collections import Counter

@app.get("/stats")
def get_stats():
    conn = sqlite3.connect("database/predictions.db")
    cursor = conn.cursor()

    cursor.execute("SELECT detected_class, confidence FROM predictions")
    rows = cursor.fetchall()

    conn.close()

    total_predictions = len(rows)
    class_counts = Counter([row[0] for row in rows])
    avg_confidence = (
        sum([row[1] for row in rows]) / total_predictions
        if total_predictions > 0 else 0
    )

    return {
        "total_predictions": total_predictions,
        "class_distribution": class_counts,
        "average_confidence": round(avg_confidence, 3)
    }


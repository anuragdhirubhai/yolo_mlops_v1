import os
import random
import mlflow
import mlflow.pyfunc
from ultralytics import YOLO
from mlflow.tracking import MlflowClient

MODEL_NAME = "YOLO_Object_Detector"


class YOLOWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = YOLO("yolov8n.pt")

    def predict(self, context, model_input):
        results = self.model(model_input)
        return results


# -------- Tracking Configuration --------
if os.getenv("CI") == "true":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
else:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("Tracking URI:", mlflow.get_tracking_uri())


# -------- Training + Registration --------
with mlflow.start_run(run_name="yolo_mlops_auto_pipeline") as run:

    # Log model
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=YOLOWrapper()
    )

    # Log parameters
    mlflow.log_param("model_type", "yolov8n")
    mlflow.log_param("framework", "ultralytics")

    # Simulated validation metric
    validation_score = random.uniform(0.6, 0.95)
    mlflow.log_metric("validation_score", validation_score)

    print("Validation Score:", validation_score)

    # Register model
    model_uri = f"runs:/{run.info.run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print("Registered Model Version:", registered_model.version)

    # -------- Conditional Promotion --------
    if validation_score > 0.75:
        client = MlflowClient()
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="production",
            version=registered_model.version
        )
        print("Production alias updated.")
    else:
        print("Model did not meet promotion threshold.")


print("Pipeline execution completed.")

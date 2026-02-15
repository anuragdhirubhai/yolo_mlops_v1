import mlflow
import mlflow.pyfunc
from ultralytics import YOLO

MODEL_NAME = "YOLO_Object_Detector"

class YOLOWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # Load lightweight YOLO model
        self.model = YOLO("yolov8n.pt")

    def predict(self, context, model_input):
        results = self.model(model_input)
        return results


mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("Tracking URI:", mlflow.get_tracking_uri())

with mlflow.start_run(run_name="v2_power_move"):


    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=YOLOWrapper()
    )

    mlflow.log_param("model_type", "yolov8n")
    mlflow.log_param("framework", "ultralytics")
    mlflow.log_param("improvement", "v2_upgrade_simulation")

print("Model logged successfully!")

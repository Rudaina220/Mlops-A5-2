import os
import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

with mlflow.start_run() as run:
    accuracy = 0.91  # replace with your real model accuracy

    mlflow.log_metric("accuracy", accuracy)

    run_id = run.info.run_id
    print("Run ID:", run_id)
    print("Accuracy:", accuracy)

    with open("model_info.txt", "w", encoding="utf-8") as f:
        f.write(run_id)

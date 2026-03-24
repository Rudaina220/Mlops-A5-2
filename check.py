import os
import sys
import mlflow

THRESHOLD = 0.85

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Error: MLFLOW_TRACKING_URI is not set.")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)

    if not os.path.exists("model_info.txt"):
        print("Error: model_info.txt does not exist.")
        sys.exit(1)

    with open("model_info.txt", "r", encoding="utf-8") as f:
        run_id = f.read().strip()

    if not run_id:
        print("Error: model_info.txt is empty.")
        sys.exit(1)

    print(f"Checking MLflow run: {run_id}")

    run = mlflow.get_run(run_id)
    metrics = run.data.metrics

    if "accuracy" not in metrics:
        print("Error: accuracy metric not found in MLflow run.")
        sys.exit(1)

    accuracy = metrics["accuracy"]
    print(f"Accuracy = {accuracy}")
    print(f"Threshold = {THRESHOLD}")

    if accuracy < THRESHOLD:
        print("Pipeline failed: accuracy is below threshold.")
        sys.exit(1)

    print("Threshold passed. Deployment can continue.")

if __name__ == "__main__":
    main()

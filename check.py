import mlflow
import sys
import os

print("Checking model quality...")

mlflow.set_tracking_uri("file:./mlruns")

with open('model_info.txt', 'r') as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

try:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get('accuracy', 0.0)
    
    print(f"Measured Accuracy: {accuracy:.3f}")
    
    if accuracy < 0.85:
        print("FAILED: Accuracy below threshold (0.85)")
        sys.exit(1)
    else:
        print("PASSED: Quality gate cleared!")
        sys.exit(0)
        
except Exception as e:
    print(f"Error checking run: {e}")
    sys.exit(1)

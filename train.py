import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os

print(" Starting training...")

print("Loading mock data...")
data = pd.read_csv('data/train.csv')
X = data[['feature1', 'feature2']].values
y = data['target'].values
print(f"Dataset shape: {X.shape}")

np.random.seed(123)  

print("Training model...")
mlflow.set_experiment("model-validation")
mlflow.set_tracking_uri("file:./mlruns")  

with mlflow.start_run(run_name="github-actions-run"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=3)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Final Accuracy: {accuracy:.3f}")
    
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("features", 2)
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("test_size", len(X_test))
    
    mlflow.sklearn.log_model(model, "model")
    mlflow.sklearn.log_model(scaler, "preprocessor")
    
    run_id = mlflow.active_run().info.run_id
    print(f"MLflow Run ID: {run_id}")
    
    with open('model_info.txt', 'w') as f:
        f.write(run_id)
    
    print("Training complete!")

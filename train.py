import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os

data = pd.read_csv('data/train.csv')
X = data[['feature1', 'feature2']].values
y = data['target'].values

np.random.seed(42)

mlflow.set_experiment("model-validation")
mlflow.set_tracking_uri("file:./mlruns")

with mlflow.start_run(run_name="high-accuracy-run"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5, 
        min_samples_split=2,
        random_state=123
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Final Accuracy: {accuracy}")
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", float(accuracy))
    
    mlflow.sklearn.log_model(model, "model")
    mlflow.sklearn.log_model(scaler, "preprocessor")
    
    run_id = mlflow.active_run().info.run_id
    with open('model_info.txt', 'w') as f:
        f.write(run_id)

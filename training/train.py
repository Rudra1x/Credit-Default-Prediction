"""
Model Training Pipeline
Explainable Credit Default Prediction System

- Trains LightGBM model
- Logs metrics and artifacts to MLflow
- Saves model for production inference
"""

import os
import mlflow
import mlflow.lightgbm
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score
)

# Config
DATA_PATH = "data/processed/credit_data.csv"
TARGET_COL = "default"
EXPERIMENT_NAME = "credit-risk-explainable-model"

RANDOM_STATE = 42

# Utility Functions
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)

def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    return metrics

# Training Pipeline
def train():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lightgbm_credit_risk"):

        print("Loading data...")
        df = load_data(DATA_PATH)

        print("Splitting data...")
        X_train, X_test, y_train, y_test = split_data(df)

        print("Training LightGBM model...")
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE
        )

        model.fit(X_train, y_train)

        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        print("Logging model to MLflow...")
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name="CreditRiskLightGBM"
        )

        # Save feature list (used later in inference & explainability)
        feature_path = "models/features.txt"
        os.makedirs("models", exist_ok=True)
        with open(feature_path, "w") as f:
            for col in X_train.columns:
                f.write(f"{col}\n")

        mlflow.log_artifact(feature_path)

        print("Training completed successfully!")
        print("Logged Metrics:", metrics)

if __name__ == "__main__":
    train()
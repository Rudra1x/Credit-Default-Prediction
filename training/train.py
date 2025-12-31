"""
Model Training Pipeline
Explainable Credit Default Prediction System
"""

import os
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score
)

# Configuration
DATA_PATH = "data/processed/credit_data.csv"
TARGET_COL = "default"
EXPERIMENT_NAME = "credit-risk-explainable-model"
MODEL_NAME = "CreditRiskLightGBM"
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
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }


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
            random_state=RANDOM_STATE,
            verbosity=-1,
        )

        model.fit(X_train, y_train)

        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # CRITICAL: Log feature schema as MLflow metadata
        feature_list = list(X_train.columns)
        mlflow.log_param("features", ",".join(feature_list))

        print("Logging model to MLflow registry...")
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        print("Training complete")
        print("Metrics:", metrics)


if __name__ == "__main__":
    train()
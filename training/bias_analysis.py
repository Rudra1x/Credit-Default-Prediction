"""
Bias Analysis & Fairness Audit
Explainable Credit Default Prediction System
"""

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
    selection_rate,
    true_positive_rate,
)

# Configuration
DATA_PATH = "data/processed/credit_data.csv"
TARGET_COL = "default"
MODEL_NAME = "CreditRiskLightGBM"

SENSITIVE_FEATURES = {
    "gender": "gender",
    "age_group": "age_group",
}

# Utilities
def load_model_and_features():
    """
    Load MLflow model and its training feature schema
    """
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME)

    if not versions:
        raise RuntimeError("No registered model versions found")

    run_id = versions[0].run_id
    run = client.get_run(run_id)

    features = run.data.params.get("features")
    if features is None:
        raise RuntimeError("Feature schema missing in MLflow params")

    feature_list = features.split(",")

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")

    return model, feature_list


def create_age_groups(df: pd.DataFrame):
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 50, 100],
        labels=["young", "middle", "senior"],
    )
    return df


# Bias Analysis
def run_bias_audit():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("Creating sensitive groups...")
    df = create_age_groups(df)

    print("Loading trained model and feature schema...")
    model, feature_list = load_model_and_features()

    # STRICT FEATURE ALIGNMENT (CRITICAL) 
    X = df[feature_list]
    y_true = df[TARGET_COL]

    print(" Running bias metrics...")

    results = {}

    for name, col in SENSITIVE_FEATURES.items():
        print(f"\n Bias metrics for: {name}")

        mf = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "tpr": true_positive_rate,
            },
            y_true=y_true,
            y_pred=model.predict(X),
            sensitive_features=df[col],
        )

        dp = demographic_parity_difference(
            y_true,
            model.predict(X),
            sensitive_features=df[col],
        )

        eo = equalized_odds_difference(
            y_true,
            model.predict(X),
            sensitive_features=df[col],
        )

        print(mf.by_group)
        print(f"Demographic Parity Difference: {dp:.3f}")
        print(f"Equalized Odds Difference: {eo:.3f}")

        results[name] = {
            "demographic_parity_difference": dp,
            "equalized_odds_difference": eo,
            "group_metrics": mf.by_group.to_dict(),
        }

    print("\n Bias audit complete.")
    return results


if __name__ == "__main__":
    run_bias_audit()
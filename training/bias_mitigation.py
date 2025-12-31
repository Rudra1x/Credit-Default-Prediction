"""
Bias Mitigation using Fairlearn Reweighing
Explainable Credit Default Prediction System
"""

import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)

# Configuration
DATA_PATH = "data/processed/credit_data.csv"
TARGET_COL = "default"
MODEL_NAME = "CreditRiskLightGBM_Fair"
EXPERIMENT_NAME = "credit-risk-bias-mitigation"
SENSITIVE_COL = "gender"   # primary protected attribute
RANDOM_STATE = 42


# Utilities
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def evaluate_fairness(y_true, y_pred, sensitive):
    return {
        "dp_diff": demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive
        ),
        "eo_diff": equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive
        ),
    }


# Bias Mitigation Pipeline
def run_bias_mitigation():
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(" Loading data...")
    df = load_data()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    sensitive = df[SENSITIVE_COL]

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Baseline Model
    print("⚙️ Training baseline model...")
    baseline_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )

    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)

    baseline_auc = roc_auc_score(y_test, y_pred_base)
    baseline_fairness = evaluate_fairness(
        y_test, y_pred_base, s_test
    )

    # Fairness-Constrained Model (Reweighing)
    print(" Training fairness-constrained model...")

    estimator = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )

    mitigator = ExponentiatedGradient(
        estimator,
        constraints=DemographicParity(),
    )

    mitigator.fit(
        X_train,
        y_train,
        sensitive_features=s_train,
    )

    y_pred_fair = mitigator.predict(X_test)

    fair_auc = roc_auc_score(y_test, y_pred_fair)
    fair_fairness = evaluate_fairness(
        y_test, y_pred_fair, s_test
    )

    # Logging Results
    print("\n Results Comparison")
    print("Baseline AUC:", round(baseline_auc, 3))
    print("Fair Model AUC:", round(fair_auc, 3))

    print("\nBaseline Fairness:", baseline_fairness)
    print("Fair Model Fairness:", fair_fairness)

    with mlflow.start_run(run_name="bias_mitigation_comparison"):
        mlflow.log_metric("baseline_auc", baseline_auc)
        mlflow.log_metric("fair_auc", fair_auc)

        mlflow.log_metric(
            "baseline_dp_diff",
            baseline_fairness["dp_diff"],
        )
        mlflow.log_metric(
            "fair_dp_diff",
            fair_fairness["dp_diff"],
        )

        mlflow.log_metric(
            "baseline_eo_diff",
            baseline_fairness["eo_diff"],
        )
        mlflow.log_metric(
            "fair_eo_diff",
            fair_fairness["eo_diff"],
        )

    print("\n Bias mitigation experiment complete.")


if __name__ == "__main__":
    run_bias_mitigation()
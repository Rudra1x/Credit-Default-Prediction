"""
Inference Module
Explainable Credit Default Prediction System
"""

import mlflow
import pandas as pd
from typing import Dict, List
from mlflow.tracking import MlflowClient

# Configuration
MODEL_NAME = "CreditRiskLightGBM"
MODEL_URI = f"models:/{MODEL_NAME}/latest"
DEFAULT_THRESHOLD = 0.5


class CreditRiskPredictor:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.model = self._load_model()
        self.features = self._load_features()

    # Model Loading
    def _load_model(self):
        print(f"📦 Loading model from MLflow: {MODEL_URI}")
        return mlflow.pyfunc.load_model(MODEL_URI)

    def _load_features(self) -> List[str]:
        client = MlflowClient()

        versions = client.get_latest_versions(MODEL_NAME)
        if not versions:
            raise RuntimeError("No registered model versions found")

        run_id = versions[0].run_id
        run = client.get_run(run_id)

        features = run.data.params.get("features")
        if features is None:
            raise RuntimeError("Feature schema missing in MLflow params")

        return features.split(",")

    # Prediction
    def _prepare_input(self, input_data: Dict) -> pd.DataFrame:
        df = pd.DataFrame([input_data])

        missing = set(self.features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        return df[self.features]

    def predict(self, input_data: Dict) -> Dict:
        X = self._prepare_input(input_data)

        prob = float(self.model.predict(X)[0])
        decision = "APPROVED" if prob < self.threshold else "REJECTED"

        return {
            "default_probability": round(prob, 4),
            "risk_label": "HIGH_RISK" if prob >= self.threshold else "LOW_RISK",
            "decision": decision,
        }

"""
Inference Module
Explainable Credit Default Prediction System

Responsibilities:
- Load production model from MLflow
- Validate input features
- Generate predictions & business decisions
"""

import mlflow
import pandas as pd
from typing import Dict, List

# Config
MODEL_NAME = "CreditRiskLightGBM"
MODEL_STAGE = "None"  # Later: "Staging" or "Production"
DEFAULT_THRESHOLD = 0.5

# Predictor Class
class CreditRiskPredictor:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.model = self._load_model()
        self.features = self._load_features()

    def _load_model(self):
        """
        Load model from MLflow Model Registry
        """
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f" Loading model from MLflow: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)

    def _load_features(self) -> List[str]:
        """
        Load feature schema used during training
        """
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}/features.txt"
        )
        with open(local_path, "r") as f:
            features = [line.strip() for line in f.readlines()]
        return features

    def _prepare_input(self, input_data: Dict) -> pd.DataFrame:
        """
        Convert raw input dict into model-ready DataFrame
        """
        df = pd.DataFrame([input_data])

        missing = set(self.features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        return df[self.features]

    def predict(self, input_data: Dict) -> Dict:
        """
        Generate prediction and business decision
        """
        X = self._prepare_input(input_data)

        probability = self.model.predict(X)[0]
        decision = "APPROVED" if probability < self.threshold else "REJECTED"

        return {
            "default_probability": round(float(probability), 4),
            "risk_label": "HIGH_RISK" if probability >= self.threshold else "LOW_RISK",
            "decision": decision
        }
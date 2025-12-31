"""
Explainability Module
SHAP-based local explanations + counterfactuals
"""


import shap
import pandas as pd
from typing import Dict, List
import mlflow.lightgbm
from inference.predictor import CreditRiskPredictor, MODEL_URI



class CreditRiskExplainer:
    def __init__(self, predictor: CreditRiskPredictor):
        self.predictor = predictor
        self.features = predictor.features

        # Load the underlying LightGBM model directly for SHAP
        try:
            lgbm_model = mlflow.lightgbm.load_model(MODEL_URI)
        except Exception as e:
            raise TypeError(f"Failed to load LightGBM model for SHAP: {e}")

        self.explainer = shap.TreeExplainer(lgbm_model)

    def explain(self, input_data: Dict, top_k: int = 5) -> Dict:
        X = pd.DataFrame([input_data])[self.features]

        shap_values = self.explainer.shap_values(X)

        # Binary classification → class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = shap_values[0]

        feature_imp = sorted(
            zip(self.features, shap_values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_features = [
            {
                "feature": f,
                "impact": round(float(v), 4),
                "direction": "increases_risk" if v > 0 else "reduces_risk",
            }
            for f, v in feature_imp[:top_k]
        ]

        return {
            "top_contributing_factors": top_features,
            "counterfactual_suggestions": self._counterfactuals(top_features),
        }

    def _counterfactuals(self, top_features: List[Dict]) -> List[str]:
        suggestions = []

        for item in top_features:
            if item["direction"] == "increases_risk":
                f = item["feature"]

                if "bill_amt" in f:
                    suggestions.append("Reduce outstanding bill amounts")
                elif "pay_amt" in f:
                    suggestions.append("Increase recent repayment amounts")
                elif "repayment_status" in f:
                    suggestions.append("Avoid payment delays")
                elif f == "limit_bal":
                    suggestions.append("Maintain higher available credit limit")
                elif f == "age":
                    suggestions.append("Longer credit history improves risk profile")

        return list(set(suggestions))
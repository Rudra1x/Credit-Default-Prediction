"""
FastAPI Application
Explainable Credit Default Prediction System
"""

from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from inference.predictor import CreditRiskPredictor
from inference.explain import CreditRiskExplainer
from api.schemas import (
    CreditRequest,
    CreditResponse,
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
)

# App Initialization
app = FastAPI(
    title="Explainable Credit Risk API",
    description="Real-time credit default prediction with governance readiness",
    version="1.0.0",
)

# Load Model + Explainer ONCE at Startup
try:
    predictor = CreditRiskPredictor(threshold=0.4)
    explainer = CreditRiskExplainer(predictor)

    MODEL_LOADED = True
except Exception as e:
    predictor = None
    explainer = None
    MODEL_LOADED = False
    LOAD_ERROR = str(e)

# Routes
@app.get("/health", response_model=HealthResponse)
def health_check():
    if not MODEL_LOADED:
        return HealthResponse(
            status=f"error: {LOAD_ERROR}",
            model_loaded=False,
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
    )


@app.post("/predict", response_model=CreditResponse)
def predict_credit_risk(request: CreditRequest):
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable.",
        )

    try:
        prediction = predictor.predict(request.dict())
        return CreditResponse(**prediction)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/explain", response_model=ExplainResponse)
def explain_credit_decision(request: CreditRequest):
    if not explainer:
        raise HTTPException(
            status_code=503,
            detail="Explainability service unavailable",
        )

    try:
        explanation = explainer.explain(request.dict())
        return ExplainResponse(**explanation)

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Explanation generation failed",
        )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    client = MlflowClient()
    versions = client.get_latest_versions("CreditRiskLightGBM")
    version = versions[0].version if versions else "unknown"

    return ModelInfoResponse(
        model_name="CreditRiskLightGBM",
        model_version=str(version),
        threshold=predictor.threshold if predictor else 0.0,
    )
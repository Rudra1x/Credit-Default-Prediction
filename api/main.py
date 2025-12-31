"""
FastAPI Application
Explainable Credit Default Prediction System
"""

from fastapi import FastAPI, HTTPException
from inference.predictor import CreditRiskPredictor
from api.schemas import (
    CreditRequest,
    CreditResponse,
    HealthResponse,
    ModelInfoResponse,
)
from mlflow.tracking import MlflowClient

# App Initialization
app = FastAPI(
    title="Explainable Credit Risk API",
    description="Real-time credit default prediction with governance readiness",
    version="1.0.0",
)

# Load Model ONCE at Startup
try:
    predictor = CreditRiskPredictor(threshold=0.4)
    MODEL_LOADED = True
except Exception as e:
    predictor = None
    MODEL_LOADED = False
    LOAD_ERROR = str(e)

# Routes
@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint for monitoring & orchestration
    """
    return HealthResponse(
        status="ok" if MODEL_LOADED else "error",
        model_loaded=MODEL_LOADED,
    )


@app.post("/predict", response_model=CreditResponse)
def predict_credit_risk(request: CreditRequest):
    """
    Predict credit default risk and loan decision
    """
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

    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """
    Governance & transparency endpoint
    """
    client = MlflowClient()
    versions = client.get_latest_versions("CreditRiskLightGBM")
    version = versions[0].version if versions else "unknown"

    return ModelInfoResponse(
        model_name="CreditRiskLightGBM",
        model_version=version,
        threshold=predictor.threshold if predictor else 0.0,
    )
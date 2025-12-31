# Explainable Credit Default Prediction System (AI Governance Ready)

## Overview
This project implements a production-grade, explainable credit risk prediction system designed with regulatory, ethical, and governance considerations in mind. It demonstrates the full lifecycle of a real-world machine learning system: training, governance, deployment, explainability, fairness evaluation, and user-facing interaction.

The system follows a **clean separation of concerns**:
- FastAPI serves as the production inference layer
- Streamlit provides a lightweight demonstration UI
- MLflow manages model versioning and governance artifacts

---

## Key Capabilities
- Real-time credit default prediction via FastAPI
- MLflow-based model registry and versioning
- SHAP-based local explainability with counterfactual reasoning
- Fairness audits and bias mitigation using Fairlearn
- Governance-ready documentation aligned with EU AI Act principles
- Interactive demo application for non-technical users

---

## System Architecture
Training → MLflow Registry → Inference Layer → FastAPI → Governance & Monitoring

---

## Applications in This Project

### 1️⃣ FastAPI Backend (Production Inference Layer)
The FastAPI service exposes REST endpoints for:
- Credit risk prediction
- Model explainability
- Health checks and metadata

The backend enforces:
- Strict numeric feature schemas
- Model version consistency via MLflow
- Safe handling of metadata fields

**Endpoints include:**
- `/predict`
- `/explain`
- `/health`
- `/model-info`

---

### 2️⃣ Streamlit Demo Application (User Interface)

The Streamlit app is a **demonstration-only client** that interacts with the FastAPI backend.

**Purpose of the Streamlit app:**
- Provide a simple, human-friendly interface
- Allow business users to test scenarios
- Visualize predictions and explanations
- Showcase the system during interviews or demos

**Important Design Principle:**
> Streamlit does NOT load models or perform inference.  
> All predictions are served by the FastAPI backend.

This mirrors real-world deployments where:
- Backend = production logic
- Frontend = presentation layer

---

## How to Run the System

### 1️⃣ Start the FastAPI Backend
From the project root:

```bash
uvicorn api.main:app --reload

Verify:

http://127.0.0.1:8000/health

Stream lit demo app:

streamlit run app/demo_app.py
http://localhost:8501

```
## Governance Documentation
- **Model Card:** [`MODEL_CARD.md`](MODEL_CARD.md)
- **Bias Audit Report:** [`BIAS_AUDIT.md`](BIAS_AUDIT.md)
- **Deployment Guide:** [`DEPLOYMENT.md`](DEPLOYMENT.md)

## Explainability
The system provides per-decision explanations using SHAP, highlighting:
- Top risk-increasing and risk-reducing factors
- Human-readable counterfactual suggestions

Accessible via the `/explain` API endpoint.

## Fairness & Responsible AI
Fairness is evaluated across sensitive attributes (gender, age groups) using:
- Demographic Parity Difference
- Equalized Odds Difference

Bias mitigation strategies were tested and documented.

## Disclaimer
This system is intended for educational and demonstration purposes. It should not be used as a standalone decision-maker in real-world lending without additional validation and oversight.

## Author
Built by a Rudraksh Sharma focusing on responsible and explainable AI systems.
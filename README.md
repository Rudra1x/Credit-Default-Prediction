# Explainable Credit Default Prediction System (AI Governance Ready)

## Overview
This project implements a production-grade, explainable credit risk prediction system designed with regulatory and ethical considerations in mind. It demonstrates the full lifecycle of a real-world machine learning system: training, governance, deployment, explainability, and fairness evaluation.

## Key Capabilities
- Real-time credit default prediction (FastAPI)
- MLflow-based model registry and versioning
- SHAP-based local explainability with counterfactual reasoning
- Fairness audits and bias mitigation using Fairlearn
- Governance-ready documentation aligned with EU AI Act principles

## Architecture
Training → MLflow Registry → Inference Layer → FastAPI → Governance & Monitoring

## How to Run the Application
See [`DEPLOYMENT.md`](DEPLOYMENT.md) for detailed setup and execution instructions.

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
Built by a Data Scientist / MLOps practitioner focusing on responsible and explainable AI systems.
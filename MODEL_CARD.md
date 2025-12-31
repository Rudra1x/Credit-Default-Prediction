# Model Card – Explainable Credit Default Prediction System

## 1. Model Overview

**Model Name:** CreditRiskLightGBM  
**Model Type:** Gradient Boosted Decision Trees (LightGBM)  
**Task:** Binary classification (Credit Default Prediction)  
**Deployment Interface:** FastAPI (Real-time inference)

This model predicts the probability of loan default for individual applicants and supports automated or semi-automated loan approval decisions.

---

## 2. Business Objective

The model supports:
- Loan approval automation
- Risk-based credit decisions
- Regulatory-compliant decision explanations
- Bias and fairness monitoring

It is designed to assist human decision-makers and **not** to replace final credit approval authority.

---

## 3. Training Data

- **Source:** Public credit dataset (processed and cleaned)
- **Target Variable:** `default` (1 = default, 0 = non-default)
- **Key Feature Categories:**
  - Credit limits
  - Billing amounts
  - Repayment amounts
  - Repayment status history
  - Demographic indicators (used for fairness analysis only)

Sensitive attributes such as gender and age are **not explicitly optimized for**, but are monitored for bias.

---

## 4. Model Architecture & Training

- Algorithm: LightGBM Classifier
- Training Strategy:
  - Stratified train-test split
  - Feature schema logged in MLflow
  - Model registered and versioned using MLflow Model Registry

Evaluation metrics:
- ROC-AUC
- Accuracy
- Precision
- Recall

---

## 5. Explainability

The model provides **local, per-decision explanations** using:
- SHAP (TreeExplainer)
- Feature contribution direction (risk increasing / reducing)
- Counterfactual suggestions (actionable insights)

Explainability is exposed via the `/explain` API endpoint.

---

## 6. Ethical Considerations & Limitations

- Predictions are probabilistic, not deterministic.
- Model reflects historical patterns and may inherit societal biases.
- Outputs must be reviewed within the broader credit policy framework.
- The model should **not** be used as the sole decision-maker for high-stakes lending.

---

## 7. Intended Use

**Intended:**
- Credit risk screening
- Decision support for loan officers
- Scenario analysis with explainability

**Not Intended:**
- Fully autonomous loan approval
- Use without fairness monitoring
- Use outside the population distribution it was trained on

---

## 8. Governance & Monitoring

- Model versioning via MLflow
- Fairness audits conducted periodically
- Performance and bias metrics reviewed before redeployment
- Thresholds configurable based on risk appetite

---

## 9. Regulatory Alignment

This model aligns with:
- EU AI Act (High-Risk AI transparency principles)
- Banking Model Risk Management (MRM) expectations
- Responsible AI best practices

---

## 10. Model Owner

**Owner:** Data Science / ML Engineering Team  
**Review Cycle:** Quarterly or upon data drift detection
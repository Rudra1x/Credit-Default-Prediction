# Bias & Fairness Audit Report

## 1. Objective

This report documents fairness evaluations performed on the Credit Default Prediction model to identify potential bias across sensitive demographic groups.

---

## 2. Sensitive Attributes Evaluated

- **Gender**
- **Age Group**
  - Young (< 30)
  - Middle (30–50)
  - Senior (> 50)

Sensitive attributes were used **only for evaluation**, not for training objectives.

---

## 3. Fairness Metrics Used

The following Fairlearn metrics were evaluated:

- **Demographic Parity Difference (DPD)**
- **Equalized Odds Difference (EOD)**
- **Selection Rate**
- **True Positive Rate (TPR)**

These metrics capture both outcome distribution and error rate disparities.

---

## 4. Results Summary

### Gender

| Metric | Value |
|------|------|
| Demographic Parity Difference | 0.034 |
| Equalized Odds Difference | 0.042 |

Observation:
- Slightly higher approval rate for male applicants
- Differences fall within acceptable operational thresholds

---

### Age Group

| Metric | Value |
|------|------|
| Demographic Parity Difference | 0.056 |
| Equalized Odds Difference | 0.115 |

Observation:
- Senior applicants receive higher approval rates
- Younger applicants show moderate disadvantage
- Pattern consistent with credit history length effects

---

## 5. Bias Interpretation

- No evidence of prohibited discrimination
- Observed disparities are explainable by financial behavior patterns
- Bias is classified as **moderate and monitorable**

---

## 6. Bias Mitigation Experiment

A fairness-constrained model was trained using Fairlearn reweighing.

### Trade-off Results (Illustrative)

- Bias reduced by ~50–65%
- Small decrease in predictive performance observed

Decision:
- Mitigation strategy documented
- Baseline model retained pending policy requirements

---

## 7. Governance Decision

- Bias levels documented and approved
- Monitoring recommended on a periodic basis
- Mitigation available if regulatory or policy thresholds change

---

## 8. Conclusion

The model demonstrates acceptable fairness characteristics with transparent measurement, interpretation, and mitigation options in place.

This satisfies Responsible AI and regulatory documentation requirements.
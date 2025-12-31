import streamlit as st
import requests


# Configuration

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Explainable Credit Risk Demo",
    page_icon="🏦",
    layout="wide",
)


# Encoding Maps (UI → Model)

GENDER_MAP = {"Male": 1, "Female": 2}

EDUCATION_MAP = {
    "Graduate": 1,
    "University": 2,
    "High School": 3,
    "Other": 4,
}

MARITAL_MAP = {
    "Married": 1,
    "Single": 2,
    "Other": 3,
}

REPAY_STATUS_MAP = {
    "On Time": 0,
    "1 Month Delay": 1,
    "2 Months Delay": 2,
    "3+ Months Delay": 3,
}


# UI Header

st.title(" Explainable Credit Default Prediction")
st.markdown(
    """
**Production-style demo UI** for a governed credit risk system.

- FastAPI backend
- MLflow-registered LightGBM model
- SHAP-based explainability
- Fairness-aware design

 Demonstration only — not for real lending use.
"""
)


# Input Form

st.subheader("Applicant Information")

with st.form("credit_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        limit_bal = st.number_input(
            "Credit Limit", min_value=0, max_value=1_000_000, value=200_000
        )
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender_label = st.selectbox("Gender", list(GENDER_MAP.keys()))

    with col2:
        education_label = st.selectbox(
            "Education Level", list(EDUCATION_MAP.keys())
        )
        marital_label = st.selectbox(
            "Marital Status", list(MARITAL_MAP.keys())
        )
        pay_amt = st.number_input(
            "Recent Monthly Payment", min_value=0, max_value=100_000, value=5000
        )

    with col3:
        bill_amt = st.number_input(
            "Outstanding Bill Amount", min_value=0, max_value=200_000, value=30000
        )
        repay_label = st.selectbox(
            "Recent Repayment Status", list(REPAY_STATUS_MAP.keys())
        )

    submit = st.form_submit_button(" Evaluate Credit Risk")


# Build NUMERIC Payload (MODEL-SAFE)

payload = {
    "id": 1,
    "limit_bal": float(limit_bal),
    "gender": int(GENDER_MAP[gender_label]),
    "education": int(EDUCATION_MAP[education_label]),
    "marital_status": int(MARITAL_MAP[marital_label]),
    "age": int(age),

    "pay_amt1": float(pay_amt),
    "pay_amt2": float(pay_amt),
    "pay_amt3": float(pay_amt),
    "pay_amt4": float(pay_amt),
    "pay_amt5": float(pay_amt),
    "pay_amt6": float(pay_amt),

    "bill_amt1": float(bill_amt),
    "bill_amt2": float(bill_amt),
    "bill_amt3": float(bill_amt),
    "bill_amt4": float(bill_amt),
    "bill_amt5": float(bill_amt),
    "bill_amt6": float(bill_amt),

    "repayment_status_sep": int(REPAY_STATUS_MAP[repay_label]),
    "repayment_status_aug": int(REPAY_STATUS_MAP[repay_label]),
    "repayment_status_jul": int(REPAY_STATUS_MAP[repay_label]),
    "repayment_status_jun": int(REPAY_STATUS_MAP[repay_label]),
    "repayment_status_may": int(REPAY_STATUS_MAP[repay_label]),
    "repayment_status_apr": int(REPAY_STATUS_MAP[repay_label]),
}


# API Call & Display Results

if submit:
    try:
        with st.spinner("Evaluating credit risk..."):
            pred_resp = requests.post(
                f"{API_URL}/predict", json=payload, timeout=5
            )
            exp_resp = requests.post(
                f"{API_URL}/explain", json=payload, timeout=5
            )

        if pred_resp.status_code != 200:
            st.error(f"Prediction failed: {pred_resp.text}")
            st.stop()

        prediction = pred_resp.json()

        
        # Prediction Results
        
        st.subheader(" Credit Decision")

        col1, col2, col3 = st.columns(3)
        col1.metric("Default Probability", prediction["default_probability"])
        col2.metric("Risk Category", prediction["risk_label"])
        col3.metric("Decision", prediction["decision"])

        
        # Explainability
        
        if exp_resp.status_code == 200:
            explanation = exp_resp.json()

            st.subheader(" Explanation")

            st.markdown("**Key Risk Drivers:**")
            for f in explanation["top_contributing_factors"]:
                st.write(
                    f"- **{f['feature']}** → {f['direction']} "
                    f"(impact: {f['impact']})"
                )

            st.markdown("**What Could Improve This Outcome:**")
            for s in explanation["counterfactual_suggestions"]:
                st.write(f"- {s}")

        else:
            st.warning("Explainability service unavailable.")

    except requests.exceptions.ConnectionError:
        st.error(" FastAPI backend is not running.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
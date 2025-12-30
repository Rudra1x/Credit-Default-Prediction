"""
Data Preprocessing Script
Explainable Credit Default Prediction System

- Reads raw UCI Credit Card XLS data
- Cleans column names
- Creates ML-ready CSV
- Preserves sensitive attributes for bias analysis
"""

import pandas as pd
from pathlib import Path


# Paths
RAW_DATA_PATH = Path("data/processed/default of credit card clients.xls")
OUTPUT_PATH = Path("data/processed/credit_data.csv")


# Column Mapping
COLUMN_RENAME_MAP = {
    "default payment next month": "default",
    "SEX": "gender",
    "AGE": "age",
    "EDUCATION": "education",
    "MARRIAGE": "marital_status",
    "PAY_0": "repayment_status_sep",
    "PAY_2": "repayment_status_aug",
    "PAY_3": "repayment_status_jul",
    "PAY_4": "repayment_status_jun",
    "PAY_5": "repayment_status_may",
    "PAY_6": "repayment_status_apr"
}

# Main Preprocessing Logic
def preprocess():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}")

    print("Reading XLS file...")
    df = pd.read_excel(RAW_DATA_PATH, header=1)

    print("Cleaning column names...")
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    # Normalize column names
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    print("Checking target distribution...")
    print(df["default"].value_counts(normalize=True))

    # Basic sanity checks
    assert "default" in df.columns, "Target column missing!"
    assert "gender" in df.columns, "Sensitive feature missing!"
    assert "age" in df.columns, "Age column missing!"

    print("Saving processed CSV...")
    df.to_csv(OUTPUT_PATH, index=False)

    print("Preprocessing complete!")
    print(f"Processed data saved at: {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()
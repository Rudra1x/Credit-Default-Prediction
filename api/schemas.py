"""
Pydantic Schemas for Credit Risk API
"""

from pydantic import BaseModel, Field
from typing import Optional


class CreditRequest(BaseModel):
    id: Optional[int] = Field(None, description="Customer ID")

    limit_bal: float
    gender: int = Field(..., description="1=Male, 2=Female")
    education: int
    marital_status: int
    age: int

    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float

    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float

    repayment_status_sep: int
    repayment_status_aug: int
    repayment_status_jul: int
    repayment_status_jun: int
    repayment_status_may: int
    repayment_status_apr: int


class CreditResponse(BaseModel):
    default_probability: float
    risk_label: str
    decision: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


class CustomerData(BaseModel):
    """
    Schema for raw Telco customer input features.
    """

    gender: Literal["Male", "Female"] = Field(
        ..., description="Gender of the customer"
    )
    SeniorCitizen: int = Field(
        ..., ge=0, le=1, description="Whether the customer is a senior citizen (0 or 1)"
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has a partner"
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has dependents"
    )
    tenure: float = Field(
        ..., ge=0, description="Number of months the customer has stayed with the company"
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has phone service"
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Whether the customer has multiple phone lines"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Type of internet service subscribed"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether online security is enabled"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether online backup is enabled"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether device protection is enabled"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether technical support is enabled"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether streaming TV is enabled"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether streaming movies are enabled"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Contract term type"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Whether paperless billing is enabled"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        ...,
        description="Payment method used by the customer",
    )
    MonthlyCharges: float = Field(
        ..., ge=0, description="The amount charged to the customer monthly"
    )
    TotalCharges: Union[float, str] = Field(
        ..., description="The total amount charged to the customer over tenure"
    )

    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, v):
        if isinstance(v, str):
            v_clean = v.strip()
            if v_clean == "":
                return 0.0
            try:
                v_float = float(v_clean)
                if v_float < 0:
                    raise ValueError("TotalCharges cannot be negative.")
                return v_float
            except ValueError as err:
                raise ValueError(f"Invalid TotalCharges string value: {v}") from err
        if isinstance(v, (int, float)):
            if v < 0:
                raise ValueError("TotalCharges cannot be negative.")
            return float(v)
        raise ValueError("TotalCharges must be a float or numeric string.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Standard churn prediction response format as specified in the assignment.
    """

    prediction: Literal["Yes", "No"] = Field(
        ..., description="Predicted churn status ('Yes' or 'No')"
    )
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability of churn between 0.0 and 1.0",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": "Yes",
                "churn_probability": 0.82,
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """
    Batch customer inference payload.
    """

    customers: List[CustomerData]


class BatchPredictionResponse(BaseModel):
    """
    Batch customer inference response.
    """

    predictions: List[PredictionResponse]
    total_records: int


class HealthResponse(BaseModel):
    """
    API Health check and model metadata response.
    """

    status: str = Field(...)
    model_version: str = Field(...)
    model_type: str = Field(...)
    features_count: int = Field(...)
    model_loaded: bool = Field(...)

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_version": "1.0.0",
                "model_type": "DecisionTreeClassifier",
                "features_count": 19,
                "model_loaded": True,
            }
        }
    }

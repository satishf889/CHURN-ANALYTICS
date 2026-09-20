import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.data_pipeline import (
    FeatureEngineeringTransformer,
    build_preprocessor,
    create_full_pipeline,
)


@pytest.fixture
def sample_raw_dataframe():
    return pd.DataFrame(
        [
            {
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
                "TotalCharges": "29.85",
            },
            {
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 0,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 105.50,
                "TotalCharges": " ",  # Blank space edge case
            },
        ]
    )


def test_feature_engineering_transformer(sample_raw_dataframe):
    transformer = FeatureEngineeringTransformer()
    transformed_df = transformer.transform(sample_raw_dataframe)

    # 1. TotalCharges clean
    assert transformed_df.loc[0, "TotalCharges"] == 29.85
    assert transformed_df.loc[1, "TotalCharges"] == 0.0

    # 2. Tenure cohort
    assert transformed_df.loc[0, "tenure_cohort"] == "0-12m"
    assert transformed_df.loc[1, "tenure_cohort"] == "0-12m"

    # 3. Average monthly charges ratio
    assert transformed_df.loc[0, "avg_monthly_charges_ratio"] == pytest.approx(29.85 / 2.0)

    # 4. Total services count
    assert transformed_df.loc[0, "total_services_count"] == 1.0  # Only OnlineBackup='Yes'
    assert transformed_df.loc[1, "total_services_count"] == 6.0  # All 6 services='Yes'


def test_preprocessor_transformation(sample_raw_dataframe):
    fe = FeatureEngineeringTransformer()
    transformed_df = fe.transform(sample_raw_dataframe)

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(transformed_df)

    assert isinstance(X_processed, np.ndarray)
    assert X_processed.shape[0] == 2
    assert X_processed.shape[1] > 10


def test_full_pipeline_fit_predict(sample_raw_dataframe):
    y = np.array(["Yes", "No"])
    pipeline = create_full_pipeline(DecisionTreeClassifier(max_depth=2, random_state=42))

    pipeline.fit(sample_raw_dataframe, y)
    preds = pipeline.predict(sample_raw_dataframe)
    probs = pipeline.predict_proba(sample_raw_dataframe)

    assert len(preds) == 2
    assert preds[0] in ["Yes", "No"]
    assert probs.shape == (2, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

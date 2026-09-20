import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for Telco Churn feature engineering:
    1. Converts TotalCharges to numeric, replacing empty spaces with NaN.
    2. tenure_cohort: Groups tenure into 4 lifecycle buckets (0-12m, 13-24m, 25-48m, 49-72m).
    3. avg_monthly_charges_ratio: TotalCharges / (tenure + 1) ratio.
    4. total_services_count: Count of active value-added add-on services.
    """

    def __init__(self):
        self.service_columns = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy()
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df)

        # 1. Clean TotalCharges (handle strings / whitespace)
        if "TotalCharges" in X_df.columns:
            X_df["TotalCharges"] = pd.to_numeric(
                X_df["TotalCharges"].astype(str).str.strip(), errors="coerce"
            )
            # If tenure is 0, TotalCharges is typically 0
            if "tenure" in X_df.columns:
                zero_tenure_mask = (X_df["tenure"] == 0) & (X_df["TotalCharges"].isna())
                X_df.loc[zero_tenure_mask, "TotalCharges"] = 0.0

        if "MonthlyCharges" in X_df.columns:
            X_df["MonthlyCharges"] = pd.to_numeric(X_df["MonthlyCharges"], errors="coerce")

        if "tenure" in X_df.columns:
            X_df["tenure"] = pd.to_numeric(X_df["tenure"], errors="coerce")

        # 2. Feature 1: Tenure Cohort
        if "tenure" in X_df.columns:
            bins = [-1, 12, 24, 48, 72, np.inf]
            labels = ["0-12m", "13-24m", "25-48m", "49-72m", "72m+"]
            X_df["tenure_cohort"] = pd.cut(
                X_df["tenure"].fillna(0), bins=bins, labels=labels
            ).astype(str)

        # 3. Feature 2: Average Monthly Charges Ratio (Charge Velocity)
        if "TotalCharges" in X_df.columns and "tenure" in X_df.columns:
            tenure_safe = X_df["tenure"].fillna(0) + 1.0
            total_charges_safe = X_df["TotalCharges"].fillna(0)
            X_df["avg_monthly_charges_ratio"] = total_charges_safe / tenure_safe

        # 4. Feature 3: Total Services Count
        available_service_cols = [c for c in self.service_columns if c in X_df.columns]
        if available_service_cols:
            X_df["total_services_count"] = (
                (X_df[available_service_cols] == "Yes").sum(axis=1).astype(float)
            )

        return X_df


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs the preprocessor ColumnTransformer for numerical and categorical features.
    """
    numerical_cols = NUMERICAL_FEATURES + ["avg_monthly_charges_ratio", "total_services_count"]
    categorical_cols = CATEGORICAL_FEATURES + ["tenure_cohort"]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def create_full_pipeline(model_estimator) -> Pipeline:
    """
    Builds the complete end-to-end scikit-learn pipeline combining:
    1. Feature Engineering
    2. Column-level Preprocessing (Imputation, Scaling, One-Hot Encoding)
    3. Final Estimator / Classifier
    """
    pipeline = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineeringTransformer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", model_estimator),
        ]
    )
    return pipeline

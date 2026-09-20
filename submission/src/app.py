import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.config import MODEL_FILE, PIPELINE_METADATA_FILE
from src.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerData,
    HealthResponse,
    PredictionResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global model state
app_state: Dict[str, Any] = {
    "model": None,
    "metadata": {},
}


def load_model_artifacts():
    """
    Loads the trained scikit-learn pipeline and metadata from disk.
    """
    if MODEL_FILE.exists():
        logger.info(f"Loading trained model pipeline from {MODEL_FILE}")
        app_state["model"] = joblib.load(MODEL_FILE)
    else:
        logger.warning(
            f"Model file not found at {MODEL_FILE}. Make sure to run the training notebook first."
        )
        app_state["model"] = None

    if PIPELINE_METADATA_FILE.exists():
        with open(PIPELINE_METADATA_FILE, "r") as f:
            app_state["metadata"] = json.load(f)
    else:
        app_state["metadata"] = {
            "model_version": "1.0.0",
            "model_type": "DecisionTreeClassifier",
            "features_count": 19,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    load_model_artifacts()
    yield
    # Shutdown: clean up resources if needed
    app_state.clear()


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "Production REST API for real-time and batch customer churn prediction "
        "using an end-to-end scikit-learn machine learning pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Telco Customer Churn Prediction API is active.",
        "documentation": "/docs",
        "health_check": "/health",
        "predict_endpoint": "/predict",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check & Model Metadata",
    tags=["System"],
)
def health_check():
    """
    Returns the service health status, model metadata, and whether the pipeline is loaded.
    """
    model_loaded = app_state["model"] is not None
    meta = app_state.get("metadata", {})
    return HealthResponse(
        status="healthy" if model_loaded else "degraded (model not loaded)",
        model_version=meta.get("model_version", "1.0.0"),
        model_type=meta.get("model_type", "DecisionTreeClassifier"),
        features_count=meta.get("features_count", 19),
        model_loaded=model_loaded,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict customer churn",
    tags=["Inference"],
)
def predict_churn(customer: CustomerData):
    """
    Accepts customer demographic, service, and billing information in JSON format,
    applies the end-to-end preprocessing pipeline, and returns the churn prediction
    and probability.
    """
    if app_state["model"] is None:
        # Attempt lazy reload in case model was trained recently
        load_model_artifacts()
        if app_state["model"] is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained model artifact is not available on the server. Please train the model first.",
            )

    try:
        # Convert Pydantic model to DataFrame matching training format
        input_data = customer.model_dump()
        input_df = pd.DataFrame([input_data])

        pipeline = app_state["model"]
        raw_pred = pipeline.predict(input_df)[0]

        # Get probabilities
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(input_df)[0]
            # Identify index of 'Yes' or positive class
            classes = list(pipeline.classes_)
            if "Yes" in classes:
                churn_idx = classes.index("Yes")
            elif 1 in classes:
                churn_idx = classes.index(1)
            else:
                churn_idx = 1
            churn_prob = float(probabilities[churn_idx])
        else:
            churn_prob = 1.0 if raw_pred in ["Yes", 1] else 0.0

        prediction_label = "Yes" if raw_pred in ["Yes", 1] else "No"

        return PredictionResponse(
            prediction=prediction_label,
            churn_probability=round(churn_prob, 4),
        )

    except Exception as exc:
        logger.error(f"Inference error: {str(exc)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failure: {str(exc)}",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch customer churn prediction",
    tags=["Inference"],
)
def predict_churn_batch(batch_request: BatchPredictionRequest):
    """
    Accepts a list of customer records and returns batch churn predictions with probabilities.
    """
    if app_state["model"] is None:
        load_model_artifacts()
        if app_state["model"] is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained model artifact is not available on the server.",
            )

    if not batch_request.customers:
        return BatchPredictionResponse(predictions=[], total_records=0)

    try:
        input_dicts = [cust.model_dump() for cust in batch_request.customers]
        input_df = pd.DataFrame(input_dicts)

        pipeline = app_state["model"]
        raw_preds = pipeline.predict(input_df)

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(input_df)
            classes = list(pipeline.classes_)
            churn_idx = classes.index("Yes") if "Yes" in classes else (classes.index(1) if 1 in classes else 1)
            churn_probs = probabilities[:, churn_idx]
        else:
            churn_probs = [1.0 if p in ["Yes", 1] else 0.0 for p in raw_preds]

        results = []
        for pred, prob in zip(raw_preds, churn_probs):
            pred_label = "Yes" if pred in ["Yes", 1] else "No"
            results.append(
                PredictionResponse(
                    prediction=pred_label,
                    churn_probability=round(float(prob), 4),
                )
            )

        return BatchPredictionResponse(predictions=results, total_records=len(results))

    except Exception as exc:
        logger.error(f"Batch inference error: {str(exc)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference pipeline failure: {str(exc)}",
        ) from exc

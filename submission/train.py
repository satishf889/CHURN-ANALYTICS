"""
Training and Evaluation Script for Telco Customer Churn Prediction.
Executes the full pipeline:
1. Data Ingestion & Preprocessing
2. Train/Test Split (70:30, random_state=42, stratified)
3. Model Training & Comparison across Decision Tree configurations
4. Hyperparameter tuning via GridSearchCV
5. Evaluation across Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC-AUC
6. Model Interpretation (Feature Importances)
7. Pipeline Serialization to model/churn_model.pkl and model/pipeline_metadata.json
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ID_COLUMN,
    MODEL_DIR,
    MODEL_FILE,
    PIPELINE_METADATA_FILE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_FILE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_pipeline import create_full_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline():
    logger.info(f"Loading raw data from {RAW_DATA_FILE}")
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_FILE}")

    df = pd.read_csv(RAW_DATA_FILE)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    # Drop ID column from modeling features
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN]

    logger.info(
        f"Target distribution:\n{y.value_counts(normalize=True).mul(100).round(2).to_dict()}"
    )

    # 70:30 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Save processed splits
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(
        PROCESSED_DATA_DIR / "train.csv", index=False
    )
    pd.concat([X_test, y_test], axis=1).to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    logger.info(f"Saved processed train and test splits to {PROCESSED_DATA_DIR}")

    # Experiment across multiple configurations
    configs = {
        "Config_1_Baseline": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Config_2_Pruned_Depth4": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=15, random_state=RANDOM_STATE
        ),
        "Config_3_Balanced_CostSensitive": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    logger.info("=== Evaluating Initial Decision Tree Configurations ===")
    for name, clf in configs.items():
        pipe = create_full_pipeline(clf)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        logger.info(
            f"[{name}] Acc: {accuracy_score(y_test, preds):.4f} | "
            f"Prec: {precision_score(y_test, preds, pos_label='Yes'):.4f} | "
            f"Rec: {recall_score(y_test, preds, pos_label='Yes'):.4f} | "
            f"F1: {f1_score(y_test, preds, pos_label='Yes'):.4f} | "
            f"AUC: {roc_auc_score(y_test, prob):.4f}"
        )

    # Hyperparameter Grid Search
    logger.info("Running GridSearchCV for optimal Decision Tree configuration...")
    base_pipe = create_full_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE))

    param_grid = {
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__min_samples_split": [10, 20, 40],
        "classifier__min_samples_leaf": [5, 10, 20],
        "classifier__class_weight": [None, "balanced"],
        "classifier__criterion": ["gini", "entropy"],
    }

    grid_search = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    logger.info(f"Best Parameters: {grid_search.best_params_}")

    # Final Evaluation on 30% test set
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Yes")
    rec = recall_score(y_test, y_pred, pos_label="Yes")
    f1 = f1_score(y_test, y_pred, pos_label="Yes")
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred, labels=["No", "Yes"]).tolist()

    logger.info("================ FINAL EVALUATION ================")
    logger.info(f"Accuracy:         {acc:.4f} ({acc * 100:.2f}%)")
    logger.info(f"Precision:        {prec:.4f}")
    logger.info(f"Recall:           {rec:.4f}")
    logger.info(f"F1 Score:         {f1:.4f}")
    logger.info(f"ROC-AUC:          {roc_auc:.4f}")
    logger.info(f"Confusion Matrix: {cm}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    # Serialize Model & Metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_FILE)
    logger.info(f"Saved trained pipeline to {MODEL_FILE}")

    metadata = {
        "model_version": "1.0.0",
        "model_type": "DecisionTreeClassifier",
        "best_params": {k: str(v) for k, v in grid_search.best_params_.items()},
        "evaluation_metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "confusion_matrix": cm,
        },
        "features_count": len(X.columns),
        "feature_names": list(X.columns),
    }

    with open(PIPELINE_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {PIPELINE_METADATA_FILE}")


if __name__ == "__main__":
    run_training_pipeline()

# Customer Churn Prediction — Requirements Specification

## 1. Project Overview & Business Problem
A telecommunications company aims to predict customer churn in advance so the customer retention team can proactively engage at-risk customers with targeted retention campaigns and tailored incentives.

- **Objective**: Build an end-to-end Machine Learning and API solution that ingests raw customer data, predicts churn likelihood, and serves real-time inference via a production-ready Python backend.
- **Dataset**: IBM Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Target Variable**: `Churn` (`Yes` / `No`)
- **Backend Language**: Strictly Python (Python 3.10+)

---

## 2. Architecture & Modular Split

The project is architected into two primary modules:

```
┌─────────────────────────────────────────────────────────┐
│        MODULE 1: ML Modeling & Analysis (Notebook)      │
│  • Data Prep & Cleaning      • Feature Engineering      │
│  • Exploratory Data Analysis • Decision Tree Training   │
│  • Model Evaluation          • Tree Interpretation      │
│  • Leak-Free Preprocessing Pipeline & Model Export      │
└────────────────────────────┬────────────────────────────┘
                             │ Exports `churn_model.pkl` & Pipeline
                             ▼
┌─────────────────────────────────────────────────────────┐
│      MODULE 2: Python Backend / REST API (FastAPI)      │
│  • Request Schema Validation (Pydantic)                 │
│  • Preprocessing & Inference Execution Engine          │
│  • Endpoints: POST /predict, GET /health                │
│  • Error Handling & Structured JSON Responses           │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Module 1: ML Modeling & Analysis (`notebook/churn_analysis.ipynb`)

### 3.1 Data Understanding & Preparation
- **Structure & Types**: Inspect column types, dimensions, and identify numerical vs categorical features. Convert `TotalCharges` from object/string to numeric (imputing or handling blank/whitespace values properly).
- **Missing & Duplicate Values**: Conduct thorough missing value and duplicate analysis.
- **Target Analysis**: Analyze class distribution of `Churn` (`Yes` vs `No`) and quantify class imbalance.
- **Train/Test Split**:
  - Split ratio: **70% Training / 30% Testing**.
  - Seed: **`random_state = 42`** for strict reproducibility.
  - Stratification: Stratify by target variable `Churn` to preserve class proportions.
- **Data Leakage Prevention**: Fit encoders, scalers, and imputers **strictly on the training dataset**, then transform the test dataset and future production inputs.
- **Categorical Encoding**: Encode categorical variables using consistent, serializable transformers (e.g., One-Hot Encoding for nominal variables, Ordinal/Binary for binary flags).

### 3.2 Exploratory Data Analysis (EDA)
Produce at least **5 meaningful visualizations** with accompanied business insights:
1. **Churn Distribution**: Target distribution visualization highlighting class proportions.
2. **Customer & Service Characteristics**: Analysis of internet service type, contract type, tech support, online security, and payment method vs churn.
3. **Churn vs Customer Demographics & Tenancy**: Relationship between `tenure`, `SeniorCitizen`, `Partner`, `Dependents`, and churn rate.
4. **Numerical Distributions & Bivariate Relationships**: Distribution of `MonthlyCharges`, `TotalCharges`, and `tenure` stratified by churn status (KDE plots/boxplots).
5. **Contract & Billing Dynamics**: Impact of month-to-month contracts vs 1-year/2-year contracts and paperless billing on customer retention.

### 3.3 Feature Engineering
Engineer at least **2 meaningful features** with clear domain justification:
- **Feature 1: `tenure_cohort` / `tenure_group`**: Grouping tenure into lifecycle stages (e.g., 0–12 mos, 13–24 mos, 25–48 mos, 49–72 mos) to capture early onboarding drop-offs.
- **Feature 2: `monthly_to_total_ratio` / `avg_charge_per_tenure`**: Interaction metric `TotalCharges / (tenure + 1)` or `MonthlyCharges / (TotalCharges + 1)` capturing price sensitivity and charge velocity.
- **Feature 3 (Bonus): `total_services_subscribed`**: Count of active add-on services (OnlineSecurity, Backup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies) measuring customer stickiness.

### 3.4 Model Development
- **Algorithm**: Decision Tree Classifier (mandatory baseline & tuned configurations).
- **Experiments**:
  - Experiment with at least two distinct Decision Tree configurations (e.g., default/unconstrained vs tuned depth/min_samples_split/min_samples_leaf/class_weight).
  - Compare model performance across configurations using Cross-Validation and validation sets.
  - Select and justify the optimal final configuration.
- **Bonus Activities**:
  - Hyperparameter optimization via `GridSearchCV` / `RandomizedSearchCV`.
  - Handling class imbalance (e.g., `class_weight='balanced'`, threshold tuning).
  - Comparing Decision Tree against alternative classifiers (e.g., Logistic Regression, Random Forest, XGBoost).

### 3.5 Model Evaluation
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix & ROC-AUC Curve
- **Business Perspective & Trade-off Analysis**:
  - Explicit written explanation answering: *For a telecom company trying to identify customers who may churn, would you prioritize Precision or Recall? Why?*
  - Justification: Highlighting why **Recall** is critical (minimizing False Negatives to prevent loss of high-LTV customers) balanced against **Precision** (avoiding wasteful retention budget on non-churners).

### 3.6 Model Interpretation
- **Feature Importance**: Calculate and plot Gini feature importances or permutation importances to identify top drivers of churn.
- **Tree Visualization**: Export decision tree diagram (`plot_tree` or text export) showing top decision rules.
- **Key Findings Summary**: Executive summary of top factors driving telecom churn (e.g., contract type, tenure, tech support availability, fiber optic pricing).

### 3.7 Pipeline Serialization
- Package the end-to-end preprocessing pipeline and trained model into a single scikit-learn pipeline or artifact bundle (`model/churn_model.pkl`).
- Save metadata including expected input feature names, types, and schema definition (`model/pipeline_metadata.json`).

---

## 4. Module 2: Python Backend / REST API (`src/app.py`)

### 4.1 Framework & Stack
- **Framework**: FastAPI (high-performance Python web framework with native async support and OpenAPI / Swagger generation).
- **Validation**: Pydantic v2 schemas for robust request validation and automated serialization.
- **Server**: Uvicorn ASGI server.

### 4.2 API Endpoints

#### `POST /predict` (Required)
Accepts customer attributes in JSON format, processes features through the saved pipeline, and returns predictions.

- **Request Payload (`application/json`)**:
```json
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
  "TotalCharges": 29.85
}
```

- **Successful Response (`200 OK`)**:
```json
{
  "prediction": "Yes",
  "churn_probability": 0.82
}
```

- **Error Responses**:
  - `422 Unprocessable Entity`: Triggered for missing fields, invalid types (e.g., negative tenure, invalid categorical value).
  - `500 Internal Server Error`: Safe fallback with error message if model inference encounters an unhandled exception.

#### `GET /health` (System Health & Metadata)
- **Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "model_type": "DecisionTreeClassifier",
  "features_count": 19
}
```

#### `POST /predict/batch` (Bonus Utility)
- Accepts a list of customer records and returns batch churn predictions with probabilities.

---

## 5. Submission & Deliverables Checklist

| Deliverable | Location | Description |
|-------------|----------|-------------|
| **Jupyter Notebook** | `notebook/churn_analysis.ipynb` | Complete end-to-end data analysis, EDA, feature engineering, modeling, evaluation, and pipeline export. |
| **Backend REST API** | `src/app.py` | FastAPI application exposing `POST /predict` and `GET /health`. |
| **Saved Model / Pipeline** | `model/churn_model.pkl` | Serialized scikit-learn pipeline capable of raw-to-prediction inference. |
| **Project Dependencies** | `requirements.txt` | Explicitly pinned Python dependencies. |
| **Setup & Run Instructions** | `README.md` | Comprehensive documentation on environment setup, notebook execution, API serving, and testing. |
| **Sample API Request** | `sample_request.json` | Sample JSON payload for testing `POST /predict`. |
| **Requirements Document** | `requirements.md` | Complete functional and non-functional requirements specification. |
| **Automated Tests** | `tests/test_api.py`, `tests/test_pipeline.py` | Unit and integration tests for preprocessing pipeline and API endpoints. |

---

## 6. Target Directory Structure

```
CHURN-ANALYTICS/
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
│
├── notebook/
│   └── churn_analysis.ipynb
│
├── model/
│   ├── churn_model.pkl
│   └── pipeline_metadata.json
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_pipeline.py
│   ├── schemas.py
│   └── app.py
│
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── sample_request.json
├── requirements.txt
├── requirements.md
└── README.md
```

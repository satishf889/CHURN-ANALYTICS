# Development Workflow & Step-by-Step Guide

This guide outlines the development phases, technical steps, and best practices for building the **Customer Churn Prediction & API Service** project.

---

## Architecture Overview

The project is divided into two distinct modules:
- **Module 1 (ML Modeling & Analysis)**: Data exploration, cleaning, feature engineering, Decision Tree model development, evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix), model interpretation, and pipeline serialization in `notebook/churn_analysis.ipynb`.
- **Module 2 (Backend REST API)**: Production-ready Python FastAPI application (`src/app.py`) that consumes the serialized model pipeline to provide real-time churn predictions via `POST /predict`.

---

## Phase 0: Repository & Environment Setup

1. **Clone and Navigate to Repository**:
   ```bash
   cd CHURN-ANALYTICS
   ```

2. **Set Up Python Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Folder Structure**:
   ```
   CHURN-ANALYTICS/
   ├── data/
   │   ├── raw/            # Place raw CSV dataset here
   │   └── processed/      # Train / Test split artifacts
   ├── notebook/           # Module 1: Jupyter notebooks
   ├── model/              # Serialized pipeline & metadata
   ├── src/                # Module 2: Python backend source code
   ├── tests/              # Automated test suite
   ├── requirement/        # Requirement docs & PDF
   ├── DEVELOPMENT_STEPS.md
   ├── README.md
   └── sample_request.json
   ```

---

## Phase 1: Data Ingestion (User Action)

1. **Place Raw Data**:
   - Place the IBM Telco Customer Churn CSV dataset in `data/raw/Telco-Customer-Churn.csv`.
2. **Verify Schema**:
   - Confirm expected columns exist: `customerID`, demographic fields (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), service fields (`PhoneService`, `InternetService`, `TechSupport`, etc.), account/contract fields (`Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`), and target `Churn`.

---

## Phase 2: Module 1 — ML Modeling & Analysis (`notebook/churn_analysis.ipynb`)

### Step 2.1: Data Understanding & Cleaning
- **Load Data**: Read `data/raw/Telco-Customer-Churn.csv` using pandas.
- **Type Inspection**:
  - Identify and fix data type issues (specifically convert `TotalCharges` from object/string to numeric; handle whitespace/blank entries).
- **Missing & Duplicate Analysis**:
  - Verify zero or handled missing values.
  - Check for and drop unnecessary identifiers (e.g., `customerID`) before modeling.
- **Target Analysis**:
  - Check distribution of `Churn` (`Yes` vs `No`) to quantify class imbalance (~73% No, ~27% Yes).
- **Train/Test Split**:
  - Split data into **70% Training** and **30% Testing** sets.
  - Set `random_state=42` and `stratify=y` to preserve target proportions across splits.
  - Save processed splits to `data/processed/train.csv` and `data/processed/test.csv`.

### Step 2.2: Exploratory Data Analysis (EDA)
Create at least **5 meaningful visualizations** with actionable business insights:
1. **Target Distribution**: Bar chart / donut chart showing Churn vs Non-Churn proportion.
2. **Contract & Tenure Analysis**: Churn rates across Month-to-month, One year, and Two year contracts vs customer tenure.
3. **Service Offerings Impact**: Churn rates by Internet Service type (DSL vs Fiber Optic vs None) and Tech Support presence.
4. **Billing & Charges Analysis**: Boxplots and KDE distributions of `MonthlyCharges` and `TotalCharges` stratified by Churn.
5. **Demographics & Payment Methods**: Churn distribution across Senior Citizens and payment methods (e.g., Electronic check vs automated payments).

### Step 2.3: Feature Engineering
Engineer at least **2 meaningful features** and document domain rationale:
- **Feature 1 (`tenure_cohort`)**: Bin tenure into stages (`0-12m`, `13-24m`, `25-48m`, `49-72m`) representing customer lifecycle phases.
- **Feature 2 (`avg_monthly_charges_ratio`)**: Interaction ratio `TotalCharges / (tenure + 1)` to detect sudden price spikes or billing instability.
- **Feature 3 (`total_services_count`)**: Aggregated count of active add-on services (OnlineSecurity, Backup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies) measuring product stickiness.

### Step 2.4: Model Development (Decision Tree Classifier)
- **Baseline Model**: Fit default `DecisionTreeClassifier(random_state=42)`.
- **Configuration Tuning**:
  - *Configuration 1*: Shallow tree (`max_depth=3` or `4`) for high interpretability.
  - *Configuration 2*: Tuned tree with `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion='gini'`.
  - *Configuration 3*: Cost-sensitive tree using `class_weight='balanced'` to better handle class imbalance.
- **Bonus Modeling**: Run hyperparameter search (`GridSearchCV`) and benchmark against Logistic Regression and Random Forest.
- **Model Selection**: Select the final Decision Tree model with justified rationale.

### Step 2.5: Model Evaluation
Evaluate the final model on the 30% held-out test set using all required metrics:
- **Accuracy**: Overall prediction correctness.
- **Precision**: Proportion of predicted churners who actually churned ($TP / (TP + FP)$).
- **Recall**: Proportion of actual churners correctly captured ($TP / (TP + FN)$).
- **F1 Score**: Harmonic mean balancing Precision and Recall.
- **Confusion Matrix**: Detailed matrix showing True Positives, True Negatives, False Positives, and False Negatives.
- **ROC-AUC**: Discriminative power across probability thresholds.

#### Business Trade-off Explanation (Precision vs. Recall):
- **Why Recall is Prioritized**: In telecommunications, losing an existing customer incurs high customer acquisition cost (CAC) and lost recurring lifetime value (LTV). Maximizing Recall ensures the retention team catches as many at-risk customers as possible.
- **Balancing Precision**: False Positives result in offering unnecessary retention incentives (e.g., discounts) to customers who would have stayed anyway, so F1-score optimization is maintained to keep retention campaigns cost-effective.

### Step 2.6: Model Interpretation
- **Feature Importance**: Calculate Gini importance scores and plot the top 10 features driving churn.
- **Decision Tree Visualization**: Plot tree structure using `sklearn.tree.plot_tree` to expose exact decision rules.
- **Key Business Takeaways**: Summarize findings (e.g., short tenure + month-to-month contracts + fiber optic without tech support = highest risk segment).

### Step 2.7: Preprocessing Pipeline & Model Serialization
- Bundle imputation, one-hot encoding, custom feature engineering, and the trained Decision Tree into a single scikit-learn `Pipeline`.
- Serialize pipeline to `model/churn_model.pkl` using `joblib`.
- Save schema metadata and expected feature names to `model/pipeline_metadata.json`.

---

## Phase 3: Module 2 — Python Backend / REST API (`src/`)

### Step 3.1: Data Schema & Validation (`src/schemas.py`)
- Define `CustomerData` Pydantic model enforcing type validation and categorical constraints matching the IBM Telco dataset.
- Define `PredictionResponse` model returning `{"prediction": "Yes"|"No", "churn_probability": float}`.
- Define `HealthResponse` model returning service health, model version, and status.

### Step 3.2: Pipeline Adapter (`src/data_pipeline.py`)
- Provide custom transformer classes (e.g., `FeatureEngineer`) ensuring identical transformations during real-time API inference as in notebook training.

### Step 3.3: FastAPI Server (`src/app.py`)
- Initialize FastAPI app with CORS middleware.
- Load `model/churn_model.pkl` at startup.
- Implement endpoints:
  - `POST /predict`: Ingests single customer JSON, returns churn prediction and probability.
  - `GET /health`: Health check and model metadata.
  - `POST /predict/batch`: Ingests a list of customer JSONs, returns batch predictions.
- Add exception handlers returning informative HTTP 422/400 errors for invalid inputs.

---

## Phase 4: Automated Testing & Verification

### Step 4.1: Unit & Integration Tests (`tests/`)
- `tests/test_pipeline.py`:
  - Test raw data transformation with missing values.
  - Verify feature engineering outputs (`tenure_cohort`, `avg_monthly_charges_ratio`).
  - Verify pipeline `.predict()` and `.predict_proba()` output shapes and value bounds.
- `tests/test_api.py`:
  - Test `GET /health` returns status `healthy`.
  - Test `POST /predict` with valid payload returns HTTP 200 with `prediction` and `churn_probability`.
  - Test `POST /predict` with invalid types/missing fields returns HTTP 422.
  - Test `POST /predict/batch` with multiple customer records.

### Step 4.2: Execute Test Suite
```bash
pytest tests/ -v
```

---

## Phase 5: Local Serving & Demonstration

1. **Launch Backend API**:
   ```bash
   uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Interactive API Documentation**:
   - Access Swagger UI: `http://localhost:8000/docs`
   - Access ReDoc: `http://localhost:8000/redoc`

3. **Test with `sample_request.json`**:
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict" \
        -H "Content-Type: application/json" \
        -d @sample_request.json
   ```

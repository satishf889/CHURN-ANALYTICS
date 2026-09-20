# Telco Customer Churn Prediction & Analytics REST API

An end-to-end Machine Learning solution designed to identify telecommunications customers who are at risk of churning, enabling retention teams to proactively engage them with targeted incentives.

The solution is divided into two core modules:
- **Module 1 (Data Science & ML Modeling)**: End-to-end analysis, EDA, feature engineering, Decision Tree modeling, evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrix), and pipeline serialization in `notebook/churn_analysis.ipynb`.
- **Module 2 (Python Backend REST API)**: A high-performance FastAPI service in `src/app.py` serving real-time predictions via `POST /predict`.

---

## Repository Structure

```
CHURN-ANALYTICS/
│
├── data/
│   ├── raw/                              # Place raw dataset here (e.g. Telco-Customer-Churn.csv)
│   └── processed/                        # Processed train/test split data
│
├── notebook/
│   └── churn_analysis.ipynb              # Module 1: Complete analysis & modeling notebook
│
├── model/
│   ├── churn_model.pkl                   # Serialized scikit-learn pipeline & model
│   └── pipeline_metadata.json            # Model schema & feature metadata
│
├── src/
│   ├── __init__.py
│   ├── config.py                         # Application configuration & paths
│   ├── data_pipeline.py                  # Custom feature transformers & pipeline utilities
│   ├── schemas.py                        # Pydantic request/response data models
│   └── app.py                            # Module 2: FastAPI REST API service
│
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py                  # Data processing & model unit tests
│   └── test_api.py                       # API endpoint & schema validation tests
│
├── requirement/
│   ├── Data Science Assignment.pdf       # Original assignment specification
│   └── requirements.md                   # Full requirements breakdown
│
├── DEVELOPMENT_STEPS.md                  # Detailed phased development workflow
├── sample_request.json                   # Sample JSON payload for API testing
├── requirements.txt                      # Project dependencies
└── README.md                             # Setup and execution guide
```

---

## Prerequisites

- **Python**: Version 3.10 or higher
- **Virtual Environment Tool**: `venv` or `conda`

---

## Local Setup Instructions

### 1. Clone the Repository
```bash
git clone <repo-url>
cd CHURN-ANALYTICS
```

### 2. Create & Activate a Virtual Environment
```bash
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the Dataset
Place the raw IBM Telco Customer Churn dataset into the `data/raw/` directory:
```bash
# Ensure the file is at:
data/raw/Telco-Customer-Churn.csv
```

---

## Running Module 1: ML Analysis & Modeling (Jupyter Notebook)

1. **Launch Jupyter Lab / Notebook**:
   ```bash
   jupyter notebook
   ```
2. **Open the Analysis Notebook**:
   Navigate to `notebook/churn_analysis.ipynb`.
3. **Execute All Cells**:
   The notebook performs the full pipeline:
   - **Data Cleaning**: Handling whitespace in `TotalCharges`, duplicate removal, type casting.
   - **Exploratory Data Analysis (EDA)**: 5+ visualizations covering churn distribution, contract types, service subscriptions, and billing trends with business takeaways.
   - **Feature Engineering**: Creating `tenure_cohort`, `avg_monthly_charges_ratio`, and `total_services_count`.
   - **Model Training**: Decision Tree Classifier with multiple configurations (baseline, depth-limited, cost-sensitive/balanced).
   - **Model Evaluation**:
     - Accuracy
     - Precision
     - Recall
     - F1 Score
     - Confusion Matrix & ROC-AUC
   - **Model Interpretation**: Gini feature importance ranking and Decision Tree structure visualization.
   - **Pipeline Export**: Packages and serializes the complete preprocessing + model pipeline to `model/churn_model.pkl`.

---

## Running Module 2: Python Backend REST API (FastAPI)

### 1. Start the API Server
Ensure `model/churn_model.pkl` has been generated from Module 1, then run:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`.

### 2. Interactive API Documentation
Open your browser and navigate to:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## API Endpoints

### `POST /predict` (Required)
Accepts customer demographic, service, and billing information in JSON format and returns real-time churn prediction with probability.

#### Example Request (`curl`):
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d @sample_request.json
```

#### Sample Request Body (`sample_request.json`):
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

#### Sample Response (`200 OK`):
```json
{
  "prediction": "Yes",
  "churn_probability": 0.82
}
```

---

### `GET /health` (Health Check)
Returns API health status, loaded model type, and metadata.

#### Sample Response:
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "model_type": "DecisionTreeClassifier",
  "features_count": 19
}
```

---

### `POST /predict/batch` (Batch Inference)
Accepts an array of customer records and returns an array of predictions and probabilities.

---

## Model Evaluation & Business Perspective

### Evaluation Metrics
The final Decision Tree model is evaluated on the 30% test set across key classification metrics:
- **Accuracy**: Overall fraction of correct predictions.
- **Precision**: Accuracy of positive churn predictions ($TP / (TP + FP)$).
- **Recall**: Proportion of actual churners identified ($TP / (TP + FN)$).
- **F1 Score**: Harmonic mean of Precision and Recall.
- **Confusion Matrix**: Quantitative breakdown of True Positives, False Positives, True Negatives, and False Negatives.

### Business Trade-Off: Precision vs. Recall in Telecom Churn
> **Why Recall is Prioritized**: In telecommunications, acquiring a new customer is significantly more expensive than retaining an existing one (high Customer Acquisition Cost vs. low retention outreach cost). Failing to identify a churner (**False Negative**) results in direct loss of monthly recurring revenue (MRR) and customer lifetime value (LTV). Therefore, **Recall** is prioritized to catch as many potential churners as possible.
> 
> **Balancing Precision**: A low Precision means offering discounts or retention promotions to non-churning customers (**False Positives**). While acceptable within reason, tuning for an optimal **F1-Score** ensures marketing retention budgets remain cost-effective without eroding profit margins.

---

## Running Automated Tests

Run the complete test suite using `pytest`:

```bash
# Run all unit and API tests
pytest tests/ -v

# Run with test coverage report
pytest tests/ -v --cov=src
```

---

## Requirements & Environment File

Refer to [requirements.txt](file:///Users/satishfulwani/Documents/Github/CHURN-ANALYTICS/requirements.txt) for exact package dependencies and [requirements.md](file:///Users/satishfulwani/Documents/Github/CHURN-ANALYTICS/requirement/requirements.md) for full project requirements.

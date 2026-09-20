# Telco Customer Churn Prediction & Analytics REST API

[![GitHub Repository](https://img.shields.io/badge/GitHub-satishf889%2FCHURN--ANALYTICS-blue?logo=github)](https://github.com/satishf889/CHURN-ANALYTICS.git)
**GitHub Repository**: [https://github.com/satishf889/CHURN-ANALYTICS.git](https://github.com/satishf889/CHURN-ANALYTICS.git)

An end-to-end Machine Learning solution designed to identify telecommunications customers who are at risk of churning, uncover root-cause churn drivers through exploratory data analysis, and enable proactive customer retention via a high-performance REST API.

---

## 📖 The Customer Churn Analytics Story & Business Insights

This end-to-end project uncovers the behavioral, contractual, and service patterns driving customer churn across 7,043 telecommunications accounts, translating machine learning insights into actionable customer retention strategies.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │              TELCO CHURN ANALYTICS JOURNEY              │
                  └──────────────────────────┬──────────────────────────────┘
                                             │
      ┌──────────────────────────────────────┼──────────────────────────────────────┐
      ▼                                      ▼                                      ▼
[ 1. Macro Churn Baseline ]         [ 2. Behavioral Friction ]             [ 3. Machine Learning ]
 • 26.5% Customer Loss Rate          • Month-to-Month Contract Volatility   • Pruned Decision Tree Classifier
 • ~$3M+ ARR Leakage Risk            • 0–12 Month Tenure Churn Cliff        • 0.8200 ROC-AUC & 78.6% Accuracy
 • Urgent Retention Need             • Fiber Optic Pricing/Quality Friction • Explainable Decision Paths
                                     • Electronic Check Payment Failure
```

---

### Chapter 1: The Churn Baseline & Revenue Leakage

![Overall Customer Churn Distribution](UI%20Screenshots/IMAGE_1.png)

* **Key Observation**: Out of **7,043 customers**, **1,869 churned (26.5%)** while **5,174 (73.5%)** were retained.
* **Business Impact**: In subscription-based telecom businesses, losing over a quarter of the customer base represents severe Annual Recurring Revenue (ARR) leakage and inflates Customer Acquisition Costs (CAC).
* **Strategic Takeaway**: Standard retention campaigns are insufficient. The business requires automated, real-time risk scoring at early lifecycle stages to deploy targeted retention offers before cancellation decisions finalize.

---

### Chapter 2: Contract Friction & The First-Year Churn Cliff

![Churn Rate by Contract Type and Tenure Distribution](UI%20Screenshots/IMAGE_2.png)

* **Key Observation**:
  * **Contract Type**: Customers on **Month-to-month contracts experience an alarming churn rate (>40%)**, compared to **<12% for One-Year** and **<3% for Two-Year** commitments.
  * **Tenure Distribution**: Churn is heavily front-loaded—the probability density peaks dramatically within the **first 0–12 months of tenure** before tapering off sharply as customer loyalty establishes.
* **Business Impact**: Short-term flexibility attracts price-sensitive or dissatisfied customers who defect readily when encountering onboarding friction.
* **Strategic Takeaway**: Implement a 90-day structured onboarding nurture sequence and offer discounted multi-month or annual upgrade incentives to transition month-to-month customers into committed contracts.

---

### Chapter 3: Premium Fiber Optic Risk & Tech Support Stickiness

![Churn Rate by Internet Service Type and Technical Support Status](UI%20Screenshots/IMAGE_3.png)

* **Key Observation**:
  * **Internet Service**: **Fiber Optic subscribers exhibit the highest churn rate (~42%)**, even though they pay premium monthly rates.
  * **Support Ecosystem**: Customers with **Tech Support have less than half the churn rate (~15%)** of those without technical assistance (~41%).
* **Business Impact**: High Fiber Optic churn indicates possible service instability, expectation mismatches, or price friction at high price tiers. Conversely, value-added support acts as a retention anchor.
* **Strategic Takeaway**: Bundle complimentary Technical Support and proactive network quality monitoring with all Fiber Optic tiers to increase customer satisfaction and service stickiness.

---

### Chapter 4: Payment Friction & Demographic Vulnerability

![Churn Rate by Payment Method and Senior Citizen Status](UI%20Screenshots/IMAGE_4.png)

* **Key Observation**:
  * **Payment Channel**: Customers paying via **Electronic Check churn at an astonishing 45% rate**, compared to **~16% for automated methods** (Bank Transfer and Credit Card auto-pay).
  * **Demographics**: **Senior Citizens churn at ~41%**, noticeably higher than non-senior customers (~23%).
* **Business Impact**: Manual payment channels introduce recurring friction, billing surprises, and involuntary churn due to lapsed payments. Senior citizens may encounter usability barriers or fixed-income pricing sensitivity.
* **Strategic Takeaway**: Incentivize enrollment into automated recurring billing (e.g., $5 monthly bill credit for Auto-Pay) and introduce dedicated senior customer care lines with simplified billing summaries.

---

### Chapter 5: Model Experimentation & Hyperparameter Tuning

![Model Comparison Across Configurations](UI%20Screenshots/IMAGE_5.png)

During model development, three Decision Tree configurations were trained and systematically benchmarked against the held-out validation set:

| Configuration | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Key Characteristics |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Config 1: Unconstrained Baseline** | 73.02% | 0.4919 | 0.4848 | 0.4883 | 0.6521 | Severe overfitting to training noise, poor generalization |
| **Config 2: Pruned (`max_depth=4`)** | **79.18%** | **0.6580** | 0.4492 | 0.5339 | 0.8265 | High precision, compact explainable rules, strong AUC |
| **Config 3: Balanced + Pruned** | 71.08% | 0.4735 | **0.7968** | **0.5940** | **0.8289** | High recall for aggressive churn intervention strategies |

---

### Chapter 6: Final Model Performance & Classification Metrics

![Final Model Evaluation Metrics](UI%20Screenshots/IMAGE_6.png)

The production Decision Tree model balances precision, recall, and operational interpretability on the unseen test partition:

* **Accuracy**: **78.61%**
* **ROC-AUC Score**: **0.8200**
* **Precision (Churn = Yes)**: **0.6105** (61% of flagged customers are true churners)
* **Recall (Churn = Yes)**: **0.5365** (Detects over 53% of all churn events)
* **F1-Score (Churn = Yes)**: **0.5712**

```
=== Classification Report ===
              precision    recall  f1-score   support
          No       0.84      0.88      0.86      1552
         Yes       0.61      0.54      0.57       561

    accuracy                           0.79      2113
   macro avg       0.73      0.71      0.71      2113
weighted avg       0.78      0.79      0.78      2113
```

---

### Chapter 7: Key Predictive Drivers (Feature Importance)

![Top Feature Importances Driving Churn Predictions](UI%20Screenshots/IMAGE_7.png)

Gini feature importance analysis reveals the dominant variables dictating customer retention vs. defection:

1. **`Contract_Month-to-month` (~48.0%)**: The single most decisive factor; absence of long-term commitment allows immediate churn.
2. **`tenure` (~17.5%)**: Early lifecycle stage is heavily correlated with churn vulnerability.
3. **`InternetService_Fiber optic` (~14.5%)**: High price and service sensitivity in premium fiber tiers.
4. **`MonthlyCharges` (~4.2%)**: Price-sensitivity thresholding.
5. **`PaymentMethod_Electronic check` (~3.0%)**: Manual billing channel friction.
6. **`TotalCharges` & `TechSupport_No`**: Cumulative commitment and presence of support safety net.

---

### Chapter 8: Decision Tree Architecture & Rule Explainability

![Decision Tree Hierarchy Top 3 Levels](UI%20Screenshots/IMAGE_8.png)

The serialized Decision Tree provides transparent, auditable decision boundaries suitable for business stakeholders and regulatory compliance:
* **Root Split**: Checks whether `Contract_Month-to-month <= 0.5`.
  * **Long-Term Contracts (Left Subtree)**: Customers branch into low-risk profiles with minimal churn probability.
  * **Month-to-Month Contracts (Right Subtree)**: Evaluates `InternetService_Fiber optic` and customer `tenure`, immediately isolating the highest risk segments for targeted retention workflows.

---

## 🏛️ System Architecture

```
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│   Module 1: DS & ML       │      │   Pipeline Artifacts      │      │   Module 2: REST API      │
│  churn_analysis.ipynb     │ ───► │  • churn_model.pkl        │ ───► │  FastAPI Server (app.py)  │
│  - Data Cleaning & EDA    │      │  • pipeline_metadata.json │      │  - POST /predict          │
│  - Feature Engineering    │      └───────────────────────────┘      │  - POST /predict/batch    │
│  - Decision Tree Pipeline │                                         │  - GET /health            │
└───────────────────────────┘                                         └───────────────────────────┘
```

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
├── UI Screenshots/                       # Visualizations, EDA charts & model evaluation plots
│   ├── IMAGE_1.png                       # Overall Customer Churn Distribution
│   ├── IMAGE_2.png                       # Contract Type & Tenure Distribution
│   ├── IMAGE_3.png                       # Internet Service & Tech Support Impact
│   ├── IMAGE_4.png                       # Payment Method & Senior Citizen Status
│   ├── IMAGE_5.png                       # Model Comparison Across Configurations
│   ├── IMAGE_6.png                       # Final Model Evaluation Metrics
│   ├── IMAGE_7.png                       # Top Feature Importances Driving Churn
│   └── IMAGE_8.png                       # Decision Tree Hierarchy (Top 3 Levels)
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
└── README.md                             # Setup, visual story & execution guide
```

---

## Prerequisites

- **Python**: Version 3.10 or higher
- **Virtual Environment Tool**: `venv` or `conda`

---

## Local Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/satishf889/CHURN-ANALYTICS.git
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
# Ensure the file is located at:
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
   - **Exploratory Data Analysis (EDA)**: Visualizations covering churn distribution, contract types, service subscriptions, and billing trends with business takeaways.
   - **Feature Engineering**: Creating `tenure_cohort`, `avg_monthly_charges_ratio`, and `total_services_count`.
   - **Model Training**: Decision Tree Classifier with multiple configurations (baseline, depth-limited, cost-sensitive/balanced).
   - **Model Evaluation**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC-AUC.
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

### `POST /predict` (Single Customer Inference)
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

Refer to [requirements.txt](requirements.txt) for exact package dependencies and [requirements.md](requirement/requirements.md) for full project requirements.


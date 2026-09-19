from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "model"
NOTEBOOK_DIR = BASE_DIR / "notebook"

# File Paths
RAW_DATA_FILE = RAW_DATA_DIR / "TelcoCustomerChurn.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"
MODEL_FILE = MODEL_DIR / "churn_model.pkl"
PIPELINE_METADATA_FILE = MODEL_DIR / "pipeline_metadata.json"

# Column Definitions
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Random State
RANDOM_STATE = 42
TEST_SIZE = 0.30

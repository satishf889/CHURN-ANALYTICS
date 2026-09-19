import pytest
from fastapi.testclient import TestClient

from src.app import app, app_state


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    return {
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


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "documentation" in response.json()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_type"] == "DecisionTreeClassifier"


def test_predict_endpoint_happy_path(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert data["prediction"] in ["Yes", "No"]
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_endpoint_with_zero_tenure_and_blank_charges(client, valid_payload):
    payload = valid_payload.copy()
    payload["tenure"] = 0
    payload["TotalCharges"] = " "

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Yes", "No"]
    assert isinstance(data["churn_probability"], float)


def test_predict_endpoint_numeric_string_total_charges(client, valid_payload):
    payload = valid_payload.copy()
    payload["TotalCharges"] = "150.50"

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Yes", "No"]


def test_predict_endpoint_validation_error_invalid_gender(client, valid_payload):
    invalid_payload = valid_payload.copy()
    invalid_payload["gender"] = "Unknown"  # Invalid literal

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_endpoint_validation_error_negative_charges(client, valid_payload):
    invalid_payload = valid_payload.copy()
    invalid_payload["MonthlyCharges"] = -50.0  # Invalid negative float

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_endpoint_validation_error_invalid_total_charges_string(client, valid_payload):
    invalid_payload = valid_payload.copy()
    invalid_payload["TotalCharges"] = "not-a-number"

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_batch_predict_endpoint(client, valid_payload):
    batch_payload = {"customers": [valid_payload, valid_payload]}
    response = client.post("/predict/batch", json=batch_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    for item in data["predictions"]:
        assert item["prediction"] in ["Yes", "No"]
        assert 0.0 <= item["churn_probability"] <= 1.0


def test_batch_predict_empty_list(client):
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 0
    assert data["predictions"] == []

# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app  # make sure main.py defines FastAPI() as 'app'

client = TestClient(app)

# -----------------------------
# Test /health endpoint
# -----------------------------
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# -----------------------------
# Test /predict endpoint
# -----------------------------
def test_predict_valid_input():
    # Example input matching your model features (V1-V28 + Amount)
    test_payload = {
        "V1": 0.5, "V2": -0.1, "V3": 1.2, "V4": 0.0, "V5": 0.2,
        "V6": -0.3, "V7": 0.1, "V8": 0.0, "V9": 0.4, "V10": -0.2,
        "V11": 0.1, "V12": 0.0, "V13": -0.1, "V14": 0.2, "V15": -0.3,
        "V16": 0.0, "V17": 0.1, "V18": -0.2, "V19": 0.0, "V20": 0.2,
        "V21": -0.1, "V22": 0.0, "V23": 0.1, "V24": -0.2, "V25": 0.0,
        "V26": 0.2, "V27": -0.1, "V28": 0.0, "Amount": 100.0
    }
    response = client.post("/predict", json=test_payload)
    assert response.status_code == 200
    # Expect JSON with prediction and optionally probability
    assert "prediction" in response.json()
    assert "probability" in response.json()

# -----------------------------
# Test /predict with missing fields
# -----------------------------
def test_predict_missing_field():
    # Remove V1
    test_payload = {
        "V2": -0.1, "V3": 1.2, "V4": 0.0, "V5": 0.2,
        "V6": -0.3, "V7": 0.1, "V8": 0.0, "V9": 0.4, "V10": -0.2,
        "V11": 0.1, "V12": 0.0, "V13": -0.1, "V14": 0.2, "V15": -0.3,
        "V16": 0.0, "V17": 0.1, "V18": -0.2, "V19": 0.0, "V20": 0.2,
        "V21": -0.1, "V22": 0.0, "V23": 0.1, "V24": -0.2, "V25": 0.0,
        "V26": 0.2, "V27": -0.1, "V28": 0.0, "Amount": 100.0
    }
    response = client.post("/predict", json=test_payload)
    assert response.status_code == 422  # Pydantic validation error

# -----------------------------
# Test /predict with invalid type
# -----------------------------
def test_predict_invalid_type():
    test_payload = {
        "V1": "invalid", "V2": -0.1, "V3": 1.2, "V4": 0.0, "V5": 0.2,
        "V6": -0.3, "V7": 0.1, "V8": 0.0, "V9": 0.4, "V10": -0.2,
        "V11": 0.1, "V12": 0.0, "V13": -0.1, "V14": 0.2, "V15": -0.3,
        "V16": 0.0, "V17": 0.1, "V18": -0.2, "V19": 0.0, "V20": 0.2,
        "V21": -0.1, "V22": 0.0, "V23": 0.1, "V24": -0.2, "V25": 0.0,
        "V26": 0.2, "V27": -0.1, "V28": 0.0, "Amount": 100.0
    }
    response = client.post("/predict", json=test_payload)
    assert response.status_code == 422  # Pydantic will reject string instead of float

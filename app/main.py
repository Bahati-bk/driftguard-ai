from fastapi import FastAPI, UploadFile, File, HTTPException
from app.drift import generate_drift_report
from app.retrain import retrain_candidate_model, get_model_status, promote_candidate_model
from app.storage import save_uploaded_file
import os

app = FastAPI(
    title="DriftGuard AI",
    version="1.0.0",
    description="Model-agnostic drift monitoring and maintenance platform for deployed AI systems."
)

BASELINE_PATH = "data/baseline.csv"
CURRENT_PATH = "data/current.csv"


@app.get("/")
def home():
    return {
        "app": {
            "name": "DriftGuard AI",
            "version": "1.0.0",
            "status": "running",
            "tagline": "Monitor drift. Maintain trust."
        },
        "overview": {
            "description": "DriftGuard AI monitors deployed machine learning systems by comparing baseline data with current production data, detecting drift, and supporting controlled model retraining."
        },
        "quick_start": [
            {
                "step": 1,
                "action": "Upload baseline dataset",
                "endpoint": "/upload-baseline",
                "method": "POST"
            },
            {
                "step": 2,
                "action": "Upload current dataset",
                "endpoint": "/upload-current",
                "method": "POST"
            },
            {
                "step": 3,
                "action": "Run drift detection",
                "endpoint": "/detect-drift",
                "method": "GET"
            },
            {
                "step": 4,
                "action": "Retrain candidate model",
                "endpoint": "/retrain",
                "method": "POST"
            },
            {
                "step": 5,
                "action": "Promote candidate model",
                "endpoint": "/promote-model",
                "method": "POST"
            },
            {
                "step": 6,
                "action": "Check model status",
                "endpoint": "/model-status",
                "method": "GET"
            }
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "DriftGuard AI"
    }


@app.post("/upload-baseline")
def upload_baseline(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Baseline file must be a CSV.")
    save_uploaded_file(file, BASELINE_PATH)
    return {"message": "Baseline dataset uploaded successfully.", "path": BASELINE_PATH}


@app.post("/upload-current")
def upload_current(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Current file must be a CSV.")
    save_uploaded_file(file, CURRENT_PATH)
    return {"message": "Current dataset uploaded successfully.", "path": CURRENT_PATH}


@app.get("/detect-drift")
def detect_drift():
    return generate_drift_report(BASELINE_PATH, CURRENT_PATH)


@app.post("/retrain")
def retrain(target_column: str = "target"):
    return retrain_candidate_model(current_path=CURRENT_PATH, target_column=target_column)


@app.post("/promote-model")
def promote_model():
    return promote_candidate_model()


@app.get("/model-status")
def model_status():
    return get_model_status()
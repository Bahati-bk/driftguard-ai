from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load model and scaler
MODEL_PATH = "models/model_v1.pkl"
SCALER_PATH = "models/scaler_v1.pkl"
LOG_FILE = "data/processed/api_logs.csv"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="DriftGuard AI - Fraud Detection API")

# Pydantic model for request
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict")
def predict(transaction: Transaction):
    import pandas as pd
    
    df = pd.DataFrame([transaction.dict()])
    
    # Scale features
    X_scaled = scaler.transform(df)
    
    proba = model.predict_proba(X_scaled)[0][1]
    prediction = int(proba > 0.5)
    
    # Log features + prediction
    df["is_fraud"] = prediction
    df["fraud_probability"] = proba
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)
    
    return {"is_fraud": prediction, "fraud_probability": float(proba)}

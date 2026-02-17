# driftguard-ai 🚀

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.26-orange)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

**driftguard-ai** is a production-ready Machine Learning system for **real-time fraud detection** with continuous monitoring for data drift and model performance.

## System Overview

The system consists of **four main layers**:

1. **Model Layer**
   - Train fraud detection models on historical transaction data
   - Save model artifacts for deployment
   - Store training feature distributions for drift detection

2. **API Layer**
   - FastAPI endpoint for real-time predictions
   - Input validation and logging of predictions

3. **Monitoring Layer**
   - Detect data drift using **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS) test**
   - Streamlit dashboard displays metrics and alerts

4. **Infrastructure Layer**
   - Dockerized services for API and monitoring
   - CI/CD pipelines via GitHub Actions
   - Cloud-ready deployment

## Goal

Ensure ML models **maintain accuracy, reliability, and fairness** in production, even as the underlying data evolves.

**Use Case:** Credit Card Fraud Detection

## Folder Structure

```
driftguard-ai/
│
├── app/                # FastAPI application
├── training/           # Model training, preprocessing, evaluation
├── monitoring/         # Drift detection and Streamlit dashboard
├── data/               # Raw and processed datasets
├── models/             # Saved model artifacts and feature distributions
├── tests/              # Unit tests for API and model
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Compose file to run API + dashboard
├── requirements.txt    # Python dependencies
└── README.md
```

## Dataset

This project uses the Credit Card Fraud Detection dataset.

Download manually from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

OR use the automated script:

1. Install Kaggle API:
   pip install kaggle

2. Add your Kaggle API credentials:
   - Download kaggle.json from your Kaggle account
   - Place it in:
     C:\Users\YOUR_USERNAME\.kaggle\

3. Run:
   python scripts/download_data.py

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-username/driftguard-ai.git
cd driftguard-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

```bash
python training/train.py
```

### 4. Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

**Endpoint:**
`POST http://localhost:8000/predict`

**Example Request Body:**

```json
{
  "V1": -1.3598,
  "V2": -0.0727,
  "V3": 2.5363,
  "V4": 1.3781,
  "V5": -0.3383,
  "V6": 0.4623,
  "V7": 0.2395,
  "V8": 0.0986,
  "V9": 0.3637,
  "V10": 0.0907,
  "V11": -0.5515,
  "V12": -0.6178,
  "V13": -0.9913,
  "V14": -0.3111,
  "V15": 1.4681,
  "V16": -0.4704,
  "V17": 0.2079,
  "V18": 0.0257,
  "V19": 0.4039,
  "V20": 0.2514,
  "V21": -0.0183,
  "V22": 0.2778,
  "V23": -0.1104,
  "V24": 0.0669,
  "V25": 0.1285,
  "V26": -0.1891,
  "V27": 0.1335,
  "V28": -0.021,
  "Amount": 149.62
}
```

### 5. Run Streamlit Dashboard

```bash
streamlit run monitoring/dashboard.py
```

**Access Dashboard:** [http://localhost:8501](http://localhost:8501)

- Upload `api_logs.csv` to check for **drift alerts**
- Alerts triggered if **PSI > 0.2** or **KS p-value < 0.05**

### 6. Dockerized Version

```bash
docker-compose up --build
```

- **FastAPI API:** [http://localhost:8000](http://localhost:8000)
- **Streamlit Dashboard:** [http://localhost:8501](http://localhost:8501)

## Logging

- API predictions are logged to `data/processed/api_logs.csv`
- Logged features + predictions are used for **drift detection**

## License

MIT License

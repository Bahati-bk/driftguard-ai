# DriftGuard AI 🚀

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.26-orange)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

## Problem Statement

Machine learning models do not stay reliable forever after deployment. As real-world environments change, the data reaching a model can begin to look very different from the data it was trained on. When this happens, the model may continue making predictions with high confidence while its true performance is quietly degrading. This creates serious risk in production systems because teams may not notice the failure early enough.

In many real-world deployments, the challenge is not only building a good model once, but also maintaining that model over time. Data distributions shift, user behavior changes, operational conditions evolve, and models can become stale. Without a monitoring and maintenance workflow, organizations are left reacting to failures after damage has already happened.

## Why This Project Is Useful

DriftGuard AI was built to address this exact post-deployment problem. Instead of focusing only on model training, it focuses on what happens **after** a model is already in use.

This platform is useful because it helps teams:

- detect when incoming production data is drifting away from the baseline
- identify which specific features are changing
- decide when model maintenance may be necessary
- retrain a candidate model using updated data
- safely manage the transition from an old model to a new one

By doing this, DriftGuard AI helps improve trust, reliability, and operational visibility in machine learning systems.

## Why I Built It

I worked on DriftGuard AI because I wanted to move beyond building standalone machine learning models and focus on a more realistic production challenge: **how to keep AI systems reliable after deployment**.

A lot of machine learning projects stop at training and evaluation, but real-world systems require continuous monitoring, maintenance, and adaptation. I built this project to explore that gap between experimentation and production, and to create a practical system that detects drift, supports retraining, and helps maintain model quality over time.

This project also reflects my interest in MLOps, model observability, and building AI systems that are not only accurate, but also maintainable, transparent, and production-ready.

## 🧠 Overview

**DriftGuard AI** is a **model-agnostic monitoring and maintenance platform** for deployed machine learning systems.

It helps you:

- 📊 Detect **data drift** between baseline and production data
- ⚠️ Identify **which features are changing and why**
- 🔄 Retrain models using new incoming data
- 🚀 Safely promote updated models into production

## 🎯 Problem It Solves

Machine learning models degrade over time because real-world data changes.

This leads to:

- Reduced accuracy
- Biased predictions
- System failures in production

**DriftGuard AI ensures your models stay reliable, accurate, and up-to-date.**

## 🏗️ System Architecture

DriftGuard AI consists of three core layers:

### 1. API Layer (FastAPI)

- Upload baseline and current datasets
- Run drift detection
- Retrain candidate models
- Promote models to production
- Track model lifecycle

### 2. Monitoring Layer (Drift Engine)

- Mean shift detection
- Standard deviation shift detection
- Population Stability Index (PSI)
- Kolmogorov-Smirnov (KS) test
- Feature-level drift reporting

### 3. Dashboard Layer (Streamlit)

- Interactive UI for:
  - dataset uploads
  - drift visualization
  - retraining workflow
  - model promotion
- Displays:
  - drift alerts
  - feature drift metrics
  - model performance

## ⚙️ Key Features

- ✅ Model-agnostic (works with any tabular ML model)
- 📈 Feature-level drift analysis
- 📊 Visual drift insights (PSI, mean shift, std shift)
- 🔁 Automated retraining pipeline
- 🚀 Candidate → Active model promotion
- 🧩 Clean modular architecture
- 🌐 API + Dashboard integration

## 📁 Project Structure

```

driftguard-ai/
│
├── app/                # FastAPI backend (API + logic)
│   ├── main.py
│   ├── drift.py
│   ├── retrain.py
│   ├── storage.py
│   └── schemas.py
│
├── monitoring/         # Drift metrics (PSI, KS test)
│   └── metrics.py
│
├── data/               # Uploaded datasets (ignored in Git)
│
├── artifacts/          # Models + registry (ignored in Git)
│
├── dashboard.py        # Streamlit dashboard
├── requirements.txt
├── README.md
└── .gitignore

```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Bahati-bk/driftguard-ai.git
cd driftguard-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI server

```bash
uvicorn app.main:app --reload --port 8010
```

Access API:

- [http://127.0.0.1:8010](http://127.0.0.1:8010)
- [http://127.0.0.1:8010/docs](http://127.0.0.1:8010/docs)

### 4. Start the Streamlit dashboard

```bash
streamlit run dashboard.py
```

Access dashboard:

- [http://localhost:8501](http://localhost:8501)

## 🔄 Workflow (How to Use)

1. **Upload Baseline Dataset**
   - Represents normal/expected data

2. **Upload Current Dataset**
   - Represents recent production data

3. **Run Drift Detection**
   - Identify distribution changes

4. **Inspect Feature Drift**
   - See which features changed and how

5. **Retrain Candidate Model**
   - Train on updated data

6. **Promote Candidate Model**
   - Replace active model if performance is acceptable

## 📊 Example Dataset Format

```csv
age,income,balance,credit_score,target
25,50000,2000,650,0
42,120000,5000,780,1
```

Requirements:

- Same feature columns in baseline and current datasets
- Include a `target` column for retraining

## 📌 Drift Detection Logic

Drift is flagged if any of the following occur:

- Mean shift > 20%
- Standard deviation shift > 20%
- PSI > 0.2
- KS test p-value < 0.05

## 🧪 Example Use Cases

- Fraud detection systems
- Credit scoring models
- Healthcare prediction models
- Recommendation systems
- Any tabular ML model in production

## 🔐 Notes

- Uploaded datasets are stored locally in `/data`
- Trained models are stored in `/artifacts`
- These are excluded from Git using `.gitignore`

## 🚀 Future Improvements

- SHAP explainability integration
- Automated retraining triggers
- Model performance monitoring (accuracy drift)
- Feature distribution visualizations
- Cloud deployment (AWS / GCP)

## 📄 License

MIT License

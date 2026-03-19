import os
import json
import shutil
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


ACTIVE_MODEL_PATH = "artifacts/active_model.pkl"
CANDIDATE_MODEL_PATH = "artifacts/candidate_model.pkl"
REGISTRY_PATH = "artifacts/model_registry.json"


def build_pipeline(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


def retrain_candidate_model(current_path: str, target_column: str = "target"):
    if not os.path.exists(current_path):
        return {"error": "Current dataset not found."}

    df = pd.read_csv(current_path)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in current dataset."}

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if len(df) < 20:
        return {"error": "Current dataset is too small for retraining. Upload more data."}

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    candidate_model = build_pipeline(X)
    candidate_model.fit(X_train, y_train)

    preds = candidate_model.predict(X_test)

    candidate_metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds, average="weighted")), 4)
    }

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(candidate_model, CANDIDATE_MODEL_PATH)

    registry = {
        "active_model_exists": os.path.exists(ACTIVE_MODEL_PATH),
        "candidate_model_exists": True,
        "candidate_metrics": candidate_metrics,
        "promotion_ready": True
    }

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    return {
        "message": "Candidate model retrained successfully.",
        "candidate_metrics": candidate_metrics,
        "promotion_ready": True
    }


def promote_candidate_model():
    if not os.path.exists(CANDIDATE_MODEL_PATH):
        return {"error": "No candidate model found."}

    os.makedirs("artifacts", exist_ok=True)
    shutil.copyfile(CANDIDATE_MODEL_PATH, ACTIVE_MODEL_PATH)

    registry = get_model_status()
    registry["active_model_exists"] = True
    registry["last_action"] = "candidate_promoted_to_active"

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    return {"message": "Candidate model promoted to active model successfully."}


def get_model_status():
    if not os.path.exists(REGISTRY_PATH):
        return {
            "active_model_exists": os.path.exists(ACTIVE_MODEL_PATH),
            "candidate_model_exists": os.path.exists(CANDIDATE_MODEL_PATH),
            "message": "No model registry found yet."
        }

    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)
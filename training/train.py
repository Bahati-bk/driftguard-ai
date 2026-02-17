# training/train.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from training.preprocess import load_data, preprocess, train_test_split_data
from training.evaluate import evaluate_model

DATA_PATH = "data/raw/fraud_data.csv"
MODEL_PATH = "models/model_v1.pkl"
SCALER_PATH = "models/scaler_v1.pkl"

def train():
    # Load and preprocess
    df = load_data(DATA_PATH)
    X, y, scaler = preprocess(df)
    
    # Save training distributions for drift detection
    feature_distributions = {}
    for i, col in enumerate(df.drop(columns=["Class", "Time"]).columns):
        feature_distributions[col] = X[:, i]  # scaled features

    joblib.dump(feature_distributions, "models/feature_distributions.pkl")
    print("Saved training feature distributions to models/feature_distributions.pkl")
    
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    
    # Candidate models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = evaluate_model(model, X_test, y_test)
        print(f"{name} ROC-AUC: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model
    
    # Save best model and scaler
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Best model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()

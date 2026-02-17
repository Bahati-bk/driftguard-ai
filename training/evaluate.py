# training/evaluate.py
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """Return ROC-AUC score and print F1 & confusion matrix"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of fraud
    
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return roc_auc

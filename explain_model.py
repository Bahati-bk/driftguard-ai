import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

MODEL_PATH = "models/model_v1.pkl"
PREPROCESSOR_PATH = "models/preprocessor_v1.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(PREPROCESSOR_PATH)

sample = pd.DataFrame([{
    "step": 1,
    "amount": 5000.0,
    "oldbalanceOrg": 10000.0,
    "newbalanceOrig": 5000.0,
    "oldbalanceDest": 2000.0,
    "newbalanceDest": 7000.0
}])

X_scaled = scaler.transform(sample)

explainer = shap.Explainer(model, X_scaled)
shap_values = explainer(X_scaled)

print("SHAP values:")
print(shap_values.values)

shap.plots.waterfall(shap_values[0], max_display=10)
plt.show()
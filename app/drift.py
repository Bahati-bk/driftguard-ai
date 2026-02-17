import pandas as pd
from monitoring.metrics import psi, ks_test
import joblib

PREPROCESSOR_PATH = "models/preprocessor_v1.pkl"
MODEL_PATH = "models/model_v1.pkl"

def check_drift(new_data_path, feature_name, historical_feature_distributions):
    """
    new_data_path: CSV of new data collected from API logs
    feature_name: which feature to check
    historical_feature_distributions: dictionary of training distributions
    """
    new_data = pd.read_csv(new_data_path)
    new_feature = new_data[feature_name].values
    training_feature = historical_feature_distributions[feature_name]
    
    # PSI
    psi_val = psi(training_feature, new_feature)
    
    # KS Test
    ks_stat, p_value = ks_test(training_feature, new_feature)
    
    drift_alert = psi_val > 0.2 or p_value < 0.05
    return {"psi": psi_val, "ks_stat": ks_stat, "p_value": p_value, "drift_alert": drift_alert}

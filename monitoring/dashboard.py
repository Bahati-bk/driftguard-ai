import streamlit as st
import pandas as pd
import joblib
from monitoring.metrics import psi, ks_test

st.title("DriftGuard AI - Monitoring Dashboard")

# Upload new API log data
new_data_file = st.file_uploader("Upload new API log data (CSV)", type="csv")

# Load training distributions
feature_distributions = joblib.load("models/feature_distributions.pkl")  # {feature: array}


if new_data_file:
    new_data = pd.read_csv(new_data_file)
    results = {}
    for feature, training_feature in feature_distributions.items():
        new_feature = new_data[feature].values
        psi_val = psi(training_feature, new_feature)
        ks_stat, p_value = ks_test(training_feature, new_feature)
        drift_alert = psi_val > 0.2 or p_value < 0.05
        results[feature] = {
            "PSI": psi_val,
            "KS": ks_stat,
            "P-Value": p_value,
            "Drift Alert": drift_alert
        }
    
    st.subheader("Drift Detection Results")
    for f, res in results.items():
        st.write(f"**{f}**")
        st.write(f"PSI: {res['PSI']:.4f}, KS: {res['KS']:.4f}, P-Value: {res['P-Value']:.4f}")
        st.write(f"🚨 Drift Alert: {'YES' if res['Drift Alert'] else 'NO'}")
        
        
def color_alert(val):
    color = 'red' if val else 'green'
    return f'color: {color}'

st.dataframe(pd.DataFrame(results).T.style.applymap(color_alert, subset=['Drift Alert']))

psi_thresh = st.slider("PSI Threshold", 0.0, 1.0, 0.2)
pval_thresh = st.slider("P-Value Threshold", 0.0, 0.1, 0.05)

drift_alert = psi_val > psi_thresh or p_value < pval_thresh
st.write(f"🚨 Drift Alert: {'YES' if drift_alert else 'NO'}")
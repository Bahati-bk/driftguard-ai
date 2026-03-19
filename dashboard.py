import requests
import pandas as pd
import streamlit as st

API_BASE = "http://127.0.0.1:8010"

st.set_page_config(
    page_title="DriftGuard AI Dashboard",
    layout="wide"
)

st.title("DriftGuard AI Dashboard")
st.caption("Model-agnostic drift monitoring and maintenance for deployed AI systems")

# -----------------------------
# Helpers
# -----------------------------
def safe_get(url):
    try:
        response = requests.get(url, timeout=30)
        return response.status_code, response.json()
    except Exception as e:
        return None, {"error": str(e)}

def safe_post(url, files=None, params=None):
    try:
        response = requests.post(url, files=files, params=params, timeout=60)
        return response.status_code, response.json()
    except Exception as e:
        return None, {"error": str(e)}

# -----------------------------
# 1. System Overview
# -----------------------------
st.header("1. System Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("API Health")
    status_code, health_data = safe_get(f"{API_BASE}/health")
    if status_code == 200:
        st.success("API is running")
        st.json(health_data)
    else:
        st.error("Could not connect to the API")
        st.json(health_data)

with col2:
    st.subheader("Model Status")
    status_code, model_status = safe_get(f"{API_BASE}/model-status")
    if status_code == 200:
        st.info("Current model registry status")
        st.json(model_status)
    else:
        st.warning("Could not fetch model status yet")
        st.json(model_status)

st.divider()

# -----------------------------
# 2. Upload Baseline Dataset
# -----------------------------
st.header("2. Upload Baseline Dataset")
st.write("Upload the reference dataset that represents stable or expected behavior.")

baseline_file = st.file_uploader(
    "Choose baseline CSV",
    type=["csv"],
    key="baseline_file"
)

if baseline_file is not None:
    try:
        baseline_preview = pd.read_csv(baseline_file)
        st.write("Baseline preview")
        st.dataframe(baseline_preview.head(), use_container_width=True)
        st.caption(f"Rows: {baseline_preview.shape[0]} | Columns: {baseline_preview.shape[1]}")
        baseline_file.seek(0)
    except Exception as e:
        st.error(f"Could not preview baseline file: {e}")

    if st.button("Upload Baseline Dataset"):
        files = {
            "file": (baseline_file.name, baseline_file.getvalue(), "text/csv")
        }
        status_code, response_data = safe_post(f"{API_BASE}/upload-baseline", files=files)
        if status_code == 200:
            st.success("Baseline dataset uploaded successfully")
        else:
            st.error("Baseline upload failed")
        st.json(response_data)

st.divider()

# -----------------------------
# 3. Upload Current Dataset
# -----------------------------
st.header("3. Upload Current Dataset")
st.write("Upload recent production-like data to compare against the baseline.")

current_file = st.file_uploader(
    "Choose current CSV",
    type=["csv"],
    key="current_file"
)

if current_file is not None:
    try:
        current_preview = pd.read_csv(current_file)
        st.write("Current dataset preview")
        st.dataframe(current_preview.head(), use_container_width=True)
        st.caption(f"Rows: {current_preview.shape[0]} | Columns: {current_preview.shape[1]}")
        current_file.seek(0)
    except Exception as e:
        st.error(f"Could not preview current file: {e}")

    if st.button("Upload Current Dataset"):
        files = {
            "file": (current_file.name, current_file.getvalue(), "text/csv")
        }
        status_code, response_data = safe_post(f"{API_BASE}/upload-current", files=files)
        if status_code == 200:
            st.success("Current dataset uploaded successfully")
        else:
            st.error("Current dataset upload failed")
        st.json(response_data)

st.divider()

# -----------------------------
# 4. Run Drift Detection
# -----------------------------
st.header("4. Drift Detection")
st.write("Run drift analysis to compare the baseline and current datasets.")

if st.button("Run Drift Detection"):
    status_code, drift_result = safe_get(f"{API_BASE}/detect-drift")

    if status_code == 200 and "error" not in drift_result:
        drift_status = drift_result.get("drift_status", "unknown")
        drifting_features = drift_result.get("drifting_features", [])
        total_checked = drift_result.get("total_features_checked", 0)
        maintenance_recommended = drift_result.get("maintenance_recommended", False)

        top1, top2, top3 = st.columns(3)

        with top1:
            if drift_status == "drift_detected":
                st.error(f"Drift Status: {drift_status}")
            else:
                st.success(f"Drift Status: {drift_status}")

        with top2:
            st.metric("Features Checked", total_checked)

        with top3:
            st.metric("Drifting Features", len(drifting_features))

        if maintenance_recommended:
            st.warning("Maintenance is recommended based on the detected drift.")
        else:
            st.info("No significant drift detected. Maintenance is not currently recommended.")

        st.subheader("Drifting Features")
        if drifting_features:
            st.write(drifting_features)
        else:
            st.write("No drifting features detected.")

        st.subheader("Raw Drift Report")
        st.json(drift_result)

        feature_report = drift_result.get("feature_report", {})
        if feature_report:
            st.subheader("Feature Drift Table")
            feature_df = pd.DataFrame(feature_report).T
            feature_df.index.name = "feature"
            st.dataframe(feature_df, use_container_width=True)

            numeric_cols = [c for c in ["mean_shift", "std_shift", "psi", "ks_stat", "p_value"] if c in feature_df.columns]

            if "psi" in feature_df.columns:
                st.subheader("PSI by Feature")
                st.bar_chart(feature_df["psi"])

            if "mean_shift" in feature_df.columns:
                st.subheader("Mean Shift by Feature")
                st.bar_chart(feature_df["mean_shift"])

            if "std_shift" in feature_df.columns:
                st.subheader("Standard Deviation Shift by Feature")
                st.bar_chart(feature_df["std_shift"])
    else:
        st.error("Drift detection failed")
        st.json(drift_result)

st.divider()

# -----------------------------
# 5. Maintenance Decision
# -----------------------------
st.header("5. Maintenance Decision")
st.write(
    "If drift is meaningful and the current dataset contains a valid label column, "
    "you can retrain a candidate model."
)

target_column = st.text_input("Target column name", value="target")

if st.button("Retrain Candidate Model"):
    status_code, retrain_result = safe_post(
        f"{API_BASE}/retrain",
        params={"target_column": target_column}
    )

    if status_code == 200 and "error" not in retrain_result:
        st.success("Candidate model retrained successfully")
    else:
        st.error("Retraining failed")

    st.json(retrain_result)

    candidate_metrics = retrain_result.get("candidate_metrics", {})
    if candidate_metrics:
        st.subheader("Candidate Metrics")
        metric_cols = st.columns(len(candidate_metrics))
        for idx, (metric_name, metric_value) in enumerate(candidate_metrics.items()):
            metric_cols[idx].metric(metric_name.upper(), metric_value)

st.divider()

# -----------------------------
# 6. Promote Candidate Model
# -----------------------------
st.header("6. Promote Candidate Model")
st.write("Promote the retrained candidate model to become the active model when ready.")

if st.button("Promote Candidate to Active"):
    status_code, promote_result = safe_post(f"{API_BASE}/promote-model")

    if status_code == 200 and "error" not in promote_result:
        st.success("Candidate model promoted successfully")
    else:
        st.error("Promotion failed")

    st.json(promote_result)

st.divider()

# -----------------------------
# 7. Refresh Model Status
# -----------------------------
st.header("7. Refresh Model Status")

if st.button("Refresh Status"):
    status_code, refreshed_status = safe_get(f"{API_BASE}/model-status")

    if status_code == 200:
        st.success("Fetched latest model status")
    else:
        st.error("Could not fetch latest model status")

    st.json(refreshed_status)
import os
import pandas as pd
import numpy as np
from monitoring.metrics import psi, ks_test


def generate_drift_report(baseline_path, current_path, ignore_columns=None):
    if ignore_columns is None:
        ignore_columns = []

    if not os.path.exists(baseline_path):
        return {"error": "Baseline dataset not found."}

    if not os.path.exists(current_path):
        return {"error": "Current dataset not found."}

    baseline_df = pd.read_csv(baseline_path)
    current_df = pd.read_csv(current_path)

    baseline_df = baseline_df.drop(columns=ignore_columns, errors="ignore")
    current_df = current_df.drop(columns=ignore_columns, errors="ignore")

    common_columns = [c for c in baseline_df.columns if c in current_df.columns]

    feature_report = {}
    drifting_features = []

    for col in common_columns:
        if pd.api.types.is_numeric_dtype(baseline_df[col]) and pd.api.types.is_numeric_dtype(current_df[col]):
            base = baseline_df[col].dropna().values
            curr = current_df[col].dropna().values

            if len(base) == 0 or len(curr) == 0:
                continue

            baseline_mean = float(np.mean(base))
            current_mean = float(np.mean(curr))
            baseline_std = float(np.std(base))
            current_std = float(np.std(curr))

            mean_shift = abs(current_mean - baseline_mean) / (abs(baseline_mean) + 1e-6)
            std_shift = abs(current_std - baseline_std) / (abs(baseline_std) + 1e-6)

            psi_val = float(psi(base, curr))
            ks_stat, p_value = ks_test(base, curr)

            drift_alert = (
                mean_shift > 0.2 or
                std_shift > 0.2 or
                psi_val > 0.2 or
                float(p_value) < 0.05
            )

            feature_report[col] = {
                "baseline_mean": round(baseline_mean, 4),
                "current_mean": round(current_mean, 4),
                "baseline_std": round(baseline_std, 4),
                "current_std": round(current_std, 4),
                "mean_shift": round(mean_shift, 4),
                "std_shift": round(std_shift, 4),
                "psi": round(psi_val, 4),
                "ks_stat": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 6),
                "drift_alert": drift_alert
            }

            if drift_alert:
                drifting_features.append(col)

    return {
        "drift_status": "drift_detected" if drifting_features else "no_significant_drift",
        "total_features_checked": len(feature_report),
        "drifting_features": drifting_features,
        "feature_report": feature_report,
        "maintenance_recommended": len(drifting_features) > 0
    }
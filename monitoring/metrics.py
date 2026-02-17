import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def psi(expected, actual, bins=10):
    """Population Stability Index"""
    def scale_values(arr, bins):
        return pd.cut(arr, bins=bins, duplicates='drop')
    
    expected_counts = scale_values(expected, bins).value_counts(normalize=True)
    actual_counts = scale_values(actual, bins).value_counts(normalize=True)
    
    df = pd.concat([expected_counts, actual_counts], axis=1).fillna(1e-6)
    df.columns = ['expected', 'actual']
    
    psi_val = np.sum((df['expected'] - df['actual']) * np.log(df['expected'] / df['actual']))
    return psi_val

def ks_test(expected, actual):
    """Kolmogorov-Smirnov test for distribution difference"""
    statistic, p_value = ks_2samp(expected, actual)
    return statistic, p_value

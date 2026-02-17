# training/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path="data/raw/fraud_data.csv"):
    """Load the credit card fraud dataset"""
    df = pd.read_csv(path)
    return df

def preprocess(df):
    """
    Preprocess features:
    - Drop Time column (not needed for ML)
    - Scale Amount + V1-V28 features
    """
    df = df.copy()
    X = df.drop(columns=["Class", "Time"])  # Features
    y = df["Class"]  # Target
    
    # Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """Split dataset into train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

import csv
from datetime import datetime

LOG_FILE = "../data/processed/api_logs.csv"

def log_prediction(tx_dict, prediction):
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), *tx_dict.values(), prediction])

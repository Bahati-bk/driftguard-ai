import os
import subprocess

DATA_PATH = "data/raw"
DATA_FILE = os.path.join(DATA_PATH, "creditcard.csv")

def download_dataset():
    if os.path.exists(DATA_FILE):
        print("Dataset already exists.")
        return

    print("Downloading dataset from Kaggle...")
    os.makedirs(DATA_PATH, exist_ok=True)

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud"],
            check=True
        )

        subprocess.run(
            ["unzip", "creditcardfraud.zip", "-d", DATA_PATH],
            check=True
        )

        os.remove("creditcardfraud.zip")
        print("Download complete.")

    except Exception as e:
        print("Error downloading dataset.")
        print("Make sure you installed Kaggle API and configured it.")
        print(e)

if __name__ == "__main__":
    download_dataset()

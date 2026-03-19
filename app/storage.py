import os
import shutil


def save_uploaded_file(upload_file, destination: str):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
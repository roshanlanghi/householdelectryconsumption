import os
import zipfile
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
ZIP_PATH = os.path.join(DATA_DIR, "individual_household_electric_power_consumption.zip")
TARGET_FILE = os.path.join(DATA_DIR, "household_power_consumption.txt")
DATASET_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TARGET_FILE):
        print(f"Dataset already present at: {TARGET_FILE}")
        return

    print(f"Downloading dataset from {DATASET_URL}...")
    response = requests.get(DATASET_URL, stream=True)
    if response.status_code == 200:
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete. Extracting zip archive...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"Extracted to {DATA_DIR}")
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
    else:
        print(f"Failed to download dataset. HTTP Status Code: {response.status_code}")

if __name__ == "__main__":
    download_data()

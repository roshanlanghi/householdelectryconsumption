import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "raw", "household_power_consumption.txt")
APP_PATH = os.path.join(BASE_DIR, "dashboard", "app.py")

def main():
    if not os.path.exists(DATA_FILE):
        print("Dataset not found. Launching automatic data downloader...")
        subprocess.run([sys.executable, "download_data.py"], check=True)
    
    print("Starting Streamlit Dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", APP_PATH])

if __name__ == "__main__":
    main()

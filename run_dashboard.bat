@echo off
echo ========================================================
echo   Launching PowerANN Streamlit Dashboard...
echo ========================================================

IF NOT EXIST "data\raw\household_power_consumption.txt" (
    echo Dataset not found in data\raw. Running data downloader...
    python download_data.py
)

echo Starting Streamlit app...
python -m streamlit run dashboard\app.py

pause

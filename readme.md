# 🏠 Household Electricity Consumption Prediction Using ANN

## 📌 Project Overview

This project aims to predict **household electricity consumption** using an **Artificial Neural Network (ANN)** implemented with **TensorFlow/Keras**.

The project uses a real-world household electricity consumption dataset containing electrical measurements collected at regular time intervals. Since the dataset contains missing and inconsistent values, data preprocessing and cleaning are performed before training the ANN model.

The project demonstrates the complete machine learning workflow:

```text
Raw Data
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Data Preprocessing
   ↓
ANN Model
   ↓
Dropout / Batch Normalization
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Streamlit Dashboard
```

---

# 🎯 Objectives

The main objectives of this project are:

- Analyze household electricity consumption data.
- Clean and preprocess real-world data.
- Handle missing values.
- Perform exploratory data analysis.
- Extract useful features from date and time.
- Normalize input features.
- Build an Artificial Neural Network using TensorFlow/Keras.
- Apply Dropout and Batch Normalization.
- Predict electricity consumption.
- Evaluate model performance.
- Visualize actual vs predicted consumption.
- Develop a simple interactive Streamlit dashboard.

---

# 📊 Dataset

## UCI Individual Household Electric Power Consumption Dataset

Dataset:

**Individual Household Electric Power Consumption**

Source:

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The dataset contains household electricity measurements collected at approximately **one-minute intervals** over several years.

### Original Features

| Feature | Description |
|---|---|
| Date | Date of measurement |
| Time | Time of measurement |
| Global_active_power | Global household active power |
| Global_reactive_power | Global reactive power |
| Voltage | Household voltage |
| Global_intensity | Household current intensity |
| Sub_metering_1 | Energy sub-metering 1 |
| Sub_metering_2 | Energy sub-metering 2 |
| Sub_metering_3 | Energy sub-metering 3 |

The dataset contains missing values, making it suitable for demonstrating real-world data cleaning and preprocessing.

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
### Interactive & Dynamic Visualization

- Plotly Express & Plotly Objects

### Machine Learning & Deep Learning

- Scikit-learn
- TensorFlow / Keras

### Web Application

- Streamlit

### Testing & Automation

- Python `unittest`
- Makefile automation

### Development Environment

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

# 📁 Project Structure

```text
household-electricity-ann/
│
├── data/
│   ├── raw/
│   │   └── household_power_consumption.txt (Auto-downloaded)
│   │
│   └── processed/
│       └── cleaned_data.csv
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_preprocessing.ipynb
│   ├── 05_ann_model.ipynb
│   └── 06_model_evaluation.ipynb
│
├── models/
│   ├── electricity_ann.keras
│   ├── ann_weights.pkl
│   └── scaler.pkl
│
├── dashboard/
│   └── app.py
│
├── reports/
│   ├── figures/
│   ├── metrics.txt
│   └── metrics.json
│
├── tests/
│   └── test_model.py
│
├── download_data.py
├── Makefile
├── LICENSE
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🔄 Project Workflow

## Phase 1 — Data Collection

Download the dataset automatically from the UCI Machine Learning Repository:

```bash
python download_data.py
```
*Or using Makefile:*
```bash
make download
```

This script will fetch and unpack `household_power_consumption.txt` into `data/raw/` automatically.


---

# 🧹 Phase 2 — Data Cleaning

The raw dataset contains missing values and requires preprocessing.

### Steps

1. Load the dataset.
2. Convert missing values to `NaN`.
3. Convert numerical columns to numeric data types.
4. Combine `Date` and `Time`.
5. Convert the combined column to datetime.
6. Sort data chronologically.
7. Check duplicate records.
8. Handle missing values.
9. Remove unnecessary columns.

### Example

```python
import pandas as pd

df = pd.read_csv(
    "data/raw/household_power_consumption.txt",
    sep=";",
    na_values="?"
)

print(df.head())
print(df.info())
print(df.isnull().sum())
```

---

# 📅 Phase 3 — Date and Time Processing

The original dataset contains separate:

```text
Date
Time
```

These will be combined into a single datetime column.

```python
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True
)
```

Additional features can then be extracted:

```python
df["Hour"] = df["Datetime"].dt.hour
df["Day"] = df["Datetime"].dt.day
df["Month"] = df["Datetime"].dt.month
df["DayOfWeek"] = df["Datetime"].dt.dayofweek
```

---

# 📊 Phase 4 — Exploratory Data Analysis

EDA is performed to understand electricity consumption patterns.

### Analysis includes:

- Distribution of electricity consumption
- Hourly consumption
- Daily consumption
- Monthly consumption
- Voltage distribution
- Global intensity distribution
- Correlation between features
- Detection of unusual values

### Example visualizations

```text
1. Electricity Consumption Distribution

2. Electricity Consumption vs Time

3. Hourly Average Consumption

4. Monthly Average Consumption

5. Correlation Heatmap

6. Boxplot of Electricity Consumption

7. Actual Consumption Trend
```

---

# ⚙️ Phase 5 — Feature Selection

The following features can be used as input variables:

```text
Global_active_power
Global_reactive_power
Voltage
Global_intensity
Sub_metering_1
Sub_metering_2
Sub_metering_3
Hour
Day
Month
DayOfWeek
```

### Target Variable

The target variable is:

```text
Global_active_power
```

The model learns the relationship between the selected input features and electricity consumption.

---

# 🔧 Phase 6 — Data Preprocessing

Before training the ANN:

### 1. Separate features and target

```python
X = df[features]
y = df["Global_active_power"]
```

### 2. Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 3. Feature Scaling

ANN models generally perform better when numerical features are scaled.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The scaler is fitted only on the training data to avoid data leakage.

---

# 🧠 Phase 7 — Artificial Neural Network

The main model is an Artificial Neural Network implemented using TensorFlow/Keras.

### Proposed Architecture

```text
Input Layer
     ↓
Dense Layer - 64 neurons
     ↓
ReLU
     ↓
Batch Normalization
     ↓
Dropout
     ↓
Dense Layer - 32 neurons
     ↓
ReLU
     ↓
Dropout
     ↓
Dense Layer - 16 neurons
     ↓
ReLU
     ↓
Output Layer
     ↓
Predicted Electricity Consumption
```

---

# 🏗️ ANN Implementation

Example architecture:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

model = Sequential()

model.add(
    Dense(
        64,
        activation="relu",
        input_shape=(X_train.shape[1],)
    )
)

model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(Dropout(0.2))

model.add(
    Dense(
        16,
        activation="relu"
    )
)

model.add(
    Dense(
        1,
        activation="linear"
    )
)

model.summary()
```

---

# ⚡ Phase 8 — Model Compilation

For the regression problem, the model can use:

```python
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
```

### Optimizer

**Adam** is used to update the neural network weights efficiently.

### Loss Function

**Mean Squared Error (MSE)** is used because this is a regression problem.

### Metric

**Mean Absolute Error (MAE)** is used to monitor prediction error.

---

# 🚀 Phase 9 — Model Training

```python
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2
)
```

The training and validation loss can be visualized to determine whether the model is learning properly.

---

# 🛡️ Phase 10 — Regularization

To reduce overfitting, the project uses:

## Dropout

Dropout randomly disables some neurons during training.

Example:

```python
Dropout(0.2)
```

This means approximately 20% of neurons are temporarily ignored during each training step.

## Batch Normalization

Batch Normalization helps stabilize and speed up neural network training.

Example:

```python
BatchNormalization()
```

---

# 📈 Phase 11 — Model Evaluation

The trained model is evaluated using:

### MAE

Mean Absolute Error

```text
Lower MAE = Better performance
```

### MSE

Mean Squared Error

```text
Lower MSE = Better performance
```

### RMSE

Root Mean Squared Error

```text
Lower RMSE = Better performance
```

### R² Score

```text
Higher R² = Better performance
```

Example:

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
```

### 🏆 Model Performance Results:

| Metric | Score | Unit | Interpretation |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | **0.0573** | kW | Average prediction error is only ~57 Watts |
| **Mean Squared Error (MSE)** | **0.0061** | kW² | Low variance of squared errors |
| **Root Mean Squared Error (RMSE)** | **0.0783** | kW | Penalizes large deviations; demonstrates high consistency |
| **Coefficient of Determination (R²)** | **0.9943** | - | Explains **99.43%** of power consumption variance |

---

# 📊 Phase 12 — Visualization of Results

The following graphs will be created:

### Training vs Validation Loss

```text
Training Loss
      vs
Validation Loss
```

This helps identify overfitting.

### Actual vs Predicted

```text
Actual Consumption
        vs
Predicted Consumption
```

This shows how closely the ANN follows the real values.

---

# 🌐 Phase 13 — Streamlit Dashboard

# 🌐 Phase 13 — Streamlit Dashboard

An interactive Streamlit dashboard demonstrates the trained model in real-time.

### Dashboard Features

```text
🏠 Home & Overview              - Key metrics & architecture overview
📊 Data Analysis                - Dataset explorer & distribution plots
📈 Consumption Visualization    - Interactive Plotly breakdown & temporal curves
🤖 ANN Prediction Engine        - Real-time single prediction & Indian electricity tariff calculator
📁 Batch CSV Prediction         - Upload telemetry CSV for bulk inference & export results
📉 Model Performance            - MAE, MSE, RMSE, R² scores & training loss curves
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/roshanlanghi/householdelectryconsumption.git
```

Navigate to the project:

```bash
cd householdelectryconsumption
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```
*Or using Makefile:*
```bash
make install
```

---

# 📋 Requirements

The main dependencies in `requirements.txt`:

```text
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
tensorflow>=2.13.0
streamlit>=1.30.0
plotly>=5.15.0
joblib>=1.3.0
requests>=2.28.0
```

---

# ▶️ Running the Project

## 1. Download Dataset Automatically

```bash
python download_data.py
```
*Or:*
```bash
make download
```

---

## 2. Run Jupyter Notebooks

```bash
jupyter notebook
```

Start with:

```text
notebooks/01_data_loading.ipynb
```

and execute the notebooks in sequence.

---

## 3. Run Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```
*Or:*
```bash
make dashboard
```

The dashboard will open automatically in your web browser.

---

## 4. Run Unit Tests

```bash
python -m unittest discover tests/
```
*Or:*
```bash
make test
```


The dashboard will open in your browser.

---

# 📁 Notebook Execution Order

Follow this order:

```text
01_data_loading.ipynb
        ↓
02_data_cleaning.ipynb
        ↓
03_eda.ipynb
        ↓
04_preprocessing.ipynb
        ↓
05_ann_model.ipynb
        ↓
06_model_evaluation.ipynb
```

---

# 📌 Expected Learning Outcomes

After completing this project, the following concepts will be demonstrated:

### Data Science

- Data collection
- Data cleaning
- Missing value handling
- Exploratory Data Analysis
- Feature engineering
- Feature scaling

### Machine Learning

- Train/test split
- Regression
- Model evaluation
- MAE
- MSE
- RMSE
- R² score

### Deep Learning

- Artificial Neural Networks
- Dense layers
- Activation functions
- ReLU
- Adam optimizer
- Loss functions
- Dropout
- Batch Normalization
- Epochs
- Batch size

### Deployment

- Streamlit
- Model serialization
- Interactive prediction

---

# 🔮 Future Scope

The project can be extended in the future with:

- LSTM-based electricity forecasting
- GRU-based forecasting
- XGBoost comparison
- Real-time IoT electricity meter data
- Weather information
- Electricity price information
- Appliance-level consumption prediction
- Anomaly detection
- Cloud deployment
- FastAPI backend
- Docker containerization

These features are not required for the basic ANN implementation.

---

# 📌 Limitations

- Prediction performance depends on the quality of historical data.
- The initial model uses historical electrical measurements rather than real-time IoT data.
- External factors such as weather, occupancy, appliance usage, and electricity tariffs are not initially included.
- The model is intended primarily for educational and experimental purposes.

---

# 👨‍💻 Author

**Roshan Langhi**

B.Tech — Computer Engineering

Sanjivani College of Engineering

---

# 📚 Dataset Reference

**UCI Machine Learning Repository**

Individual Household Electric Power Consumption Dataset

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

---

# ⭐ Project Summary

This project demonstrates an end-to-end implementation of an **Artificial Neural Network for household electricity consumption prediction**.

The project covers:

```text
Real-World Dataset
       ↓
Data Cleaning
       ↓
EDA
       ↓
Feature Engineering
       ↓
Feature Scaling
       ↓
ANN
       ↓
Dropout
       ↓
Batch Normalization
       ↓
Training
       ↓
Evaluation
       ↓
Prediction
       ↓
Streamlit Dashboard
```

**Final Goal:**

> Build a simple, practical, and explainable ANN-based system capable of predicting household electricity consumption from historical electricity data.
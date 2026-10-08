import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Streamlit Page Configuration
st.set_page_config(
    page_title="PowerANN | Household Electricity Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "electricity_ann.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

class NumpyANN:
    """Ultra-fast, zero-overhead neural network forward pass for cloud deployment."""
    def __init__(self, weights):
        self.w = weights

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        w = self.w
        z1 = np.maximum(0, np.dot(X, w['w0']) + w['b0'])
        z1_bn = w['gamma'] * (z1 - w['mean']) / np.sqrt(w['var'] + 1e-3) + w['beta']
        z2 = np.maximum(0, np.dot(z1_bn, w['w1']) + w['b1'])
        z3 = np.maximum(0, np.dot(z2, w['w2']) + w['b2'])
        out = np.dot(z3, w['w3']) + w['b3']
        return out

@st.cache_resource
def load_model_and_scaler():
    try:
        scaler = joblib.load(SCALER_PATH)
        # Attempt TensorFlow first if available
        try:
            import tensorflow as tf
            if os.path.exists(MODEL_PATH):
                model = tf.keras.models.load_model(MODEL_PATH)
                return model, scaler
        except Exception:
            pass

        # Resilient cloud-optimized loader
        weights_path = os.path.join(BASE_DIR, "models", "ann_weights.pkl")
        if os.path.exists(weights_path):
            weights = joblib.load(weights_path)
            return NumpyANN(weights), scaler
    except Exception as e:
        pass
    return None, None

@st.cache_data
def load_sample_data():
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df
    return None

# Sidebar Navigation & Settings
with st.sidebar:
    st.markdown("## ⚡ **PowerANN System**")
    st.markdown("Artificial Neural Network for Electric Power Consumption Prediction")
    st.markdown("---")
    
    # Theme Toggle: Light (White) by default!
    dark_theme = st.toggle("🌙 Dark Mode", value=False)
    
    page = st.radio(
        "Navigation Menu",
        [
            "🏠 Home & Overview",
            "📊 Data Analysis",
            "📈 Consumption Visualizations",
            "🤖 ANN Prediction Engine",
            "📁 Batch CSV Prediction",
            "📉 Model Performance"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🇮🇳 Electricity Tariff")
    tariff_per_unit = st.number_input(
        "Tariff (₹ / kWh Unit)",
        min_value=1.00,
        max_value=30.00,
        value=7.50,
        step=0.50,
        help="Typical Indian residential electricity tariff slab is ₹6.00 - ₹9.00 per unit (kWh)."
    )
    st.caption("Default: ₹7.50 per unit")

    st.markdown("---")
    st.markdown("""
    **Project Info**
    - **Dataset:** UCI Household Power
    - **Model:** Deep ANN (TensorFlow/Keras)
    - **Batch Norm & Dropout:** Applied
    - **Target:** Global Active Power (kW)
    """)
    

# Dynamic High-Contrast Styling based on Theme Toggle
if dark_theme:
    bg_color = "#0F172A"
    text_primary = "#F8FAFC"
    sub_title_color = "#94A3B8"
    card_bg = "linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02))"
    card_border = "rgba(255, 255, 255, 0.12)"
    metric_num_color = "#38BDF8"
    metric_label_color = "#94A3B8"
    pred_box_bg = "linear-gradient(135deg, #1E1B4B 0%, #1E293B 100%)"
    pred_box_border = "#6366F1"
    pred_header_color = "#A5B4FC"
    pred_val_color = "#38BDF8"
    pred_sub_color = "#CBD5E1"
    plotly_template = "plotly_dark"
else:
    bg_color = "#FFFFFF"
    text_primary = "#0F172A"
    sub_title_color = "#334155"
    card_bg = "linear-gradient(145deg, #F8FAFC, #EDF2F7)"
    card_border = "#CBD5E1"
    metric_num_color = "#1E40AF"
    metric_label_color = "#1E293B"
    pred_box_bg = "linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%)"
    pred_box_border = "#3B82F6"
    pred_header_color = "#1E3A8A"
    pred_val_color = "#1D4ED8"
    pred_sub_color = "#1E293B"
    plotly_template = "plotly_white"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
    }}

    .stApp {{
        background-color: {bg_color};
        color: {text_primary};
    }}

    .main-title {{
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1D4ED8 0%, #6D28D9 50%, #DB2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }}

    .sub-title {{
        color: {sub_title_color};
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }}

    .metric-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        border-color: #6D28D9;
    }}

    .metric-num {{
        font-size: 1.8rem;
        font-weight: 800;
        color: {metric_num_color};
        font-family: 'JetBrains Mono', monospace;
    }}

    .metric-label {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {metric_label_color};
        font-weight: 700;
        margin-top: 0.2rem;
    }}

    .prediction-box {{
        background: {pred_box_bg};
        border: 2px solid {pred_box_border};
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
    }}

    .pred-header {{
        font-size: 1.1rem;
        color: {pred_header_color};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }}

    .pred-val {{
        font-size: 3.5rem;
        font-weight: 800;
        color: {pred_val_color};
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -1px;
    }}

    .pred-sub {{
        margin-top: 1rem;
        color: {pred_sub_color};
        font-size: 0.95rem;
        font-weight: 600;
    }}

    .stButton>button {{
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s;
    }}
</style>
""", unsafe_allow_html=True)

# ----------------- PAGE 1: HOME & OVERVIEW -----------------
if page == "🏠 Home & Overview":
    st.markdown('<div class="main-title">Household Electricity Consumption Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deep Learning Artificial Neural Network (ANN) trained on real-world UCI power telemetry data.</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-num">2.07M+</div>
            <div class="metric-label">Raw Telemetry Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-num">0.057 kW</div>
            <div class="metric-label">Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-num">99.43%</div>
            <div class="metric-label">Model Accuracy (R² Score)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-num">64-32-16</div>
            <div class="metric-label">ANN Dense Architecture</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    col_left, col_right = st.columns([1.2, 0.8])
    with col_left:
        st.subheader("🎯 Project Purpose & Objectives")
        st.markdown("""
        Modern smart grids and households benefit significantly from accurate short-term electricity demand prediction.
        
        This project demonstrates the complete engineering lifecycle:
        1. **Telemetry Ingestion:** Parsing over 2 million 1-minute electrical measurements (2006–2010).
        2. **Cleaning & Feature Extraction:** Processing missing values, temporal decomposition (Hour, Day, Month, DayOfWeek).
        3. **Standard Scaling:** Preventing data leakage and conditioning inputs for deep learning.
        4. **Deep Neural Network:** Sequential MLP with **Batch Normalization** and **Dropout (0.2)** layers for generalization.
        5. **Interactive Dashboard:** Real-time inference dashboard allowing users to experiment with electrical loads.
        """)
        
    with col_right:
        st.subheader("🏗️ Neural Network Architecture")
        st.markdown("""
        ```text
        [ Input Layer: 10 Scaled Features ]
                     ↓
        [ Dense: 64 Neurons | ReLU ]
                     ↓
        [ Batch Normalization + Dropout 0.2 ]
                     ↓
        [ Dense: 32 Neurons | ReLU ]
                     ↓
        [ Dropout 0.2 ]
                     ↓
        [ Dense: 16 Neurons | ReLU ]
                     ↓
        [ Output Layer: 1 Neuron | Linear ]
        ```
        """)

# ----------------- PAGE 2: DATA ANALYSIS -----------------
elif page == "📊 Data Analysis":
    st.markdown('<div class="main-title">Data Exploration & Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect the preprocessed dataset and feature distributions.</div>', unsafe_allow_html=True)

    df = load_sample_data()
    if df is not None:
        st.subheader("📋 Cleaned Data Sample (Top 50 Records)")
        st.dataframe(df.head(50), use_container_width=True)

        st.subheader("📊 Descriptive Statistics")
        numeric_df = df.select_dtypes(include=[np.number])
        st.dataframe(numeric_df.describe().T, use_container_width=True)

        st.subheader("🔍 Feature Distribution Explorer")
        feature_choice = st.selectbox(
            "Select Feature to Visualize Distribution:",
            ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        )
        
        fig = px.histogram(
            df, x=feature_choice, marginal="rug",
            title=f"Distribution of {feature_choice}",
            template=plotly_template,
            color_discrete_sequence=['#2563EB']
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Cleaned data file not found. Please ensure dataset pipeline is initialized.")

# ----------------- PAGE 3: CONSUMPTION VISUALIZATIONS -----------------
elif page == "📈 Consumption Visualizations":
    st.markdown('<div class="main-title">Consumption Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed graphical insights into household power dynamics with interactive exploration.</div>', unsafe_allow_html=True)

    df_sample = load_sample_data()

    tab1, tab2, tab3 = st.tabs(["🕒 Temporal Patterns", "🔥 Correlation Analysis", "🔌 Sub-Metering Breakdown"])

    with tab1:
        st.subheader("Hourly Energy Consumption Curve")
        if df_sample is not None and 'Hour' in df_sample.columns and 'Global_active_power' in df_sample.columns:
            hourly_avg = df_sample.groupby('Hour')['Global_active_power'].mean().reset_index()
            fig = px.line(
                hourly_avg, x='Hour', y='Global_active_power',
                title="Average Global Active Power by Hour of Day",
                labels={'Hour': 'Hour of Day (0-23)', 'Global_active_power': 'Mean Active Power (kW)'},
                markers=True,
                template=plotly_template
            )
            fig.update_traces(line_color='#2563EB', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig_hourly = os.path.join(FIGURES_DIR, "hourly_consumption.png")
            if os.path.exists(fig_hourly):
                st.image(fig_hourly, use_container_width=True)
        st.info("💡 **Observation:** Consumption peaks during early evening (19:00 - 22:00) corresponding to meal preparation, lighting, and home appliances.")

    with tab2:
        st.subheader("Feature Correlation Matrix")
        if df_sample is not None:
            numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
            corr = df_sample[numeric_cols].corr()
            fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="Blues" if not dark_theme else "Viridis",
                title="Feature Correlation Heatmap",
                template=plotly_template
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig_heatmap = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
            if os.path.exists(fig_heatmap):
                st.image(fig_heatmap, use_container_width=True)
        st.info("💡 **Observation:** Near-perfect collinearity between `Global_intensity` and `Global_active_power` (~0.99) reflects Ohm's and Joule's electrical laws (P ≈ V × I).")

    with tab3:
        st.subheader("Sub-Metering Breakdown by Appliance Category")
        if df_sample is not None and all(c in df_sample.columns for c in ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']):
            sub_df = df_sample[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean().reset_index()
            sub_df.columns = ['Sub_metering', 'Average_Wh']
            sub_df['Category'] = ['Kitchen (Sub 1)', 'Laundry (Sub 2)', 'Water Heater / AC (Sub 3)']
            fig = px.bar(
                sub_df, x='Category', y='Average_Wh',
                color='Category',
                title="Average Watt-Hours (Wh) by Sub-Metering Appliance Category",
                template=plotly_template,
                text_auto=".2f"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig_submetering = os.path.join(FIGURES_DIR, "sub_metering_distribution.png")
            if os.path.exists(fig_submetering):
                st.image(fig_submetering, use_container_width=True)
        st.markdown("""
        - **Sub-Metering 1 (Kitchen):** Dishwasher, microwave, oven
        - **Sub-Metering 2 (Laundry):** Washing machine, tumble-dryer, refrigerator
        - **Sub-Metering 3 (Climate Control):** Electric water heater and AC (Dominant component)
        """)

# ----------------- PAGE 4: ANN PREDICTION ENGINE -----------------
elif page == "🤖 ANN Prediction Engine":
    st.markdown('<div class="main-title">ANN Real-Time Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Configure electrical and temporal parameters to predict active electricity consumption (kW).</div>', unsafe_allow_html=True)

    model, scaler = load_model_and_scaler()

    if model is None or scaler is None:
        st.error("Model or Scaler not loaded. Please ensure models are trained and present in `models/`.")
    else:
        # Quick Presets
        st.markdown("#### ⚡ Quick Presets")
        preset_cols = st.columns(4)
        
        # Session state defaults
        if 'preset' not in st.session_state:
            st.session_state.preset = {
                'reactive': 0.12,
                'voltage': 240.5,
                'intensity': 4.6,
                'sub1': 0.0,
                'sub2': 0.0,
                'sub3': 1.0,
                'hour': 14,
                'day': 15,
                'month': 6,
                'dayofweek': 2
            }

        if preset_cols[0].button("🌙 Night Standby"):
            st.session_state.preset = {'reactive': 0.08, 'voltage': 242.0, 'intensity': 1.4, 'sub1': 0.0, 'sub2': 0.0, 'sub3': 0.0, 'hour': 3, 'day': 10, 'month': 4, 'dayofweek': 1}
        if preset_cols[1].button("🍳 Morning Routine"):
            st.session_state.preset = {'reactive': 0.15, 'voltage': 239.0, 'intensity': 6.2, 'sub1': 1.0, 'sub2': 2.0, 'sub3': 1.0, 'hour': 8, 'day': 12, 'month': 5, 'dayofweek': 0}
        if preset_cols[2].button("🌆 Evening Peak Load"):
            st.session_state.preset = {'reactive': 0.32, 'voltage': 235.0, 'intensity': 18.5, 'sub1': 2.0, 'sub2': 1.0, 'sub3': 18.0, 'hour': 20, 'day': 22, 'month': 11, 'dayofweek': 4}
        if preset_cols[3].button("🧺 Weekend Laundry & AC"):
            st.session_state.preset = {'reactive': 0.22, 'voltage': 238.5, 'intensity': 14.8, 'sub1': 0.0, 'sub2': 28.0, 'sub3': 17.0, 'hour': 15, 'day': 7, 'month': 8, 'dayofweek': 6}

        st.markdown("---")
        
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("### 🔌 Electrical Telemetry")
                reactive = st.number_input(
                    "Global Reactive Power (kVAR)",
                    min_value=0.0, max_value=2.0,
                    value=float(st.session_state.preset['reactive']),
                    step=0.01
                )
                voltage = st.number_input(
                    "Household Voltage (V)",
                    min_value=200.0, max_value=260.0,
                    value=float(st.session_state.preset['voltage']),
                    step=0.5
                )
                intensity = st.number_input(
                    "Global Current Intensity (Amperes)",
                    min_value=0.1, max_value=50.0,
                    value=float(st.session_state.preset['intensity']),
                    step=0.1
                )
                
                st.markdown("### 🍳 Sub-Metering Loads (Wh)")
                sub1 = st.slider(
                    "Sub-Metering 1 (Kitchen Wh)",
                    min_value=0.0, max_value=80.0,
                    value=float(st.session_state.preset['sub1']),
                    step=1.0
                )
                sub2 = st.slider(
                    "Sub-Metering 2 (Laundry Wh)",
                    min_value=0.0, max_value=80.0,
                    value=float(st.session_state.preset['sub2']),
                    step=1.0
                )
                sub3 = st.slider(
                    "Sub-Metering 3 (Water Heater / AC Wh)",
                    min_value=0.0, max_value=80.0,
                    value=float(st.session_state.preset['sub3']),
                    step=1.0
                )

            with col_b:
                st.markdown("### 📅 Temporal Parameters")
                hour = st.slider("Hour of Day (0 = Midnight, 23 = 11 PM)", min_value=0, max_value=23, value=int(st.session_state.preset['hour']))
                day = st.slider("Day of the Month", min_value=1, max_value=31, value=int(st.session_state.preset['day']))
                month = st.slider("Month (1 = Jan, 12 = Dec)", min_value=1, max_value=12, value=int(st.session_state.preset['month']))
                dayofweek = st.selectbox(
                    "Day of Week",
                    options=[0, 1, 2, 3, 4, 5, 6],
                    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
                    index=int(st.session_state.preset['dayofweek'])
                )
                
                st.markdown("<br><br>", unsafe_allow_html=True)
                submit_button = st.form_submit_button("⚡ Predict Electricity Consumption", use_container_width=True)

        if submit_button:
            # Prepare feature vector
            feature_names = [
                'Global_reactive_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
                'Hour', 'Day', 'Month', 'DayOfWeek'
            ]
            input_data = pd.DataFrame([[
                reactive, voltage, intensity,
                sub1, sub2, sub3,
                hour, day, month, dayofweek
            ]], columns=feature_names)

            # Scale inputs
            scaled_input = scaler.transform(input_data)

            # Predict
            raw_prediction = model.predict(scaled_input)[0][0]
            prediction_kw = max(0.0, float(raw_prediction))

            st.markdown("---")
            st.markdown(f"""
            <div class="prediction-box">
                <div class="pred-header">Predicted Active Electricity Consumption</div>
                <div class="pred-val">{prediction_kw:.3f} <span style="font-size: 1.8rem; font-weight: 600;">kW</span></div>
                <div class="pred-sub">Equivalent to <b>{prediction_kw * 1000:.0f} Watts</b> instantaneous household demand.</div>
            </div>
            """, unsafe_allow_html=True)

            # Dynamic Tier Classification
            tier_col1, tier_col2, tier_col3 = st.columns(3)
            with tier_col1:
                tier = "🟢 Low Load" if prediction_kw < 1.0 else ("🟡 Moderate Load" if prediction_kw < 3.0 else "🔴 High Load")
                st.metric("Consumption Tier", tier)
            with tier_col2:
                est_hourly_cost = prediction_kw * tariff_per_unit
                st.metric(
                    "Estimated Cost / Hour",
                    f"₹{est_hourly_cost:.2f}",
                    help=f"Calculated at ₹{tariff_per_unit:.2f} per unit (kWh)"
                )
            with tier_col3:
                est_daily_cost = prediction_kw * 24 * tariff_per_unit
                st.metric(
                    "Estimated Daily Equivalent",
                    f"{prediction_kw * 24:.2f} kWh",
                    f"₹{est_daily_cost:.2f} / day"
                )

# ----------------- PAGE 5: BATCH CSV PREDICTION -----------------
elif page == "📁 Batch CSV Prediction":
    st.markdown('<div class="main-title">Batch CSV Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a batch CSV file containing household telemetry to generate batch predictions.</div>', unsafe_allow_html=True)

    model, scaler = load_model_and_scaler()

    if model is None or scaler is None:
        st.error("Model or Scaler not loaded. Please ensure models are trained and present in `models/`.")
    else:
        feature_names = [
            'Global_reactive_power', 'Voltage', 'Global_intensity',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
            'Hour', 'Day', 'Month', 'DayOfWeek'
        ]

        st.info(f"📋 **Required CSV Columns:** `{', '.join(feature_names)}` (or a `Datetime` column from which Hour/Day/Month/DayOfWeek can be parsed).")

        uploaded_file = st.file_uploader("Upload CSV File for Batch Prediction", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded CSV with {len(batch_df)} rows.")

                # If Datetime present and missing temporal columns, parse them
                if 'Datetime' in batch_df.columns:
                    dt_col = pd.to_datetime(batch_df['Datetime'])
                    if 'Hour' not in batch_df.columns: batch_df['Hour'] = dt_col.dt.hour
                    if 'Day' not in batch_df.columns: batch_df['Day'] = dt_col.dt.day
                    if 'Month' not in batch_df.columns: batch_df['Month'] = dt_col.dt.month
                    if 'DayOfWeek' not in batch_df.columns: batch_df['DayOfWeek'] = dt_col.dt.dayofweek

                missing_cols = [c for c in feature_names if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"Missing required columns in CSV: {missing_cols}")
                else:
                    scaled_inputs = scaler.transform(batch_df[feature_names])
                    raw_preds = model.predict(scaled_inputs)
                    preds = [max(0.0, float(p[0])) for p in raw_preds]
                    batch_df['Predicted_Global_active_power_kW'] = preds
                    batch_df['Estimated_Cost_INR'] = [p * tariff_per_unit for p in preds]

                    st.success("Batch Prediction Completed Successfully!")
                    st.dataframe(batch_df.head(100), use_container_width=True)

                    total_kwh = sum(preds)
                    st.metric("Total Batch Predicted Consumption", f"{total_kwh:.2f} kWh", f"₹{total_kwh * tariff_per_unit:.2f} Total Cost")

                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_data,
                        file_name="electricity_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")

# ----------------- PAGE 6: MODEL PERFORMANCE -----------------
elif page == "📉 Model Performance":
    st.markdown('<div class="main-title">ANN Performance & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Quantitative and visual metrics computed on the test dataset.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-num">0.0573</div>
            <div class="metric-label">MAE (kW)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-num">0.0061</div>
            <div class="metric-label">MSE (kW²)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-num">0.0783</div>
            <div class="metric-label">RMSE (kW)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-num">0.9943</div>
            <div class="metric-label">R² Score (99.4%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    fig_loss = os.path.join(FIGURES_DIR, "loss_curve.png")
    fig_actual_pred = os.path.join(FIGURES_DIR, "actual_vs_predicted.png")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📉 Training vs. Validation Loss")
        if os.path.exists(fig_loss):
            st.image(fig_loss, use_container_width=True)
        st.caption("Batch Normalization and Dropout ensure convergence without overfitting.")

    with col_r:
        st.subheader("🎯 Actual vs. Predicted Curve")
        if os.path.exists(fig_actual_pred):
            st.image(fig_actual_pred, use_container_width=True)
        st.caption("Demonstrating high tracking fidelity between actual test samples and ANN output.")

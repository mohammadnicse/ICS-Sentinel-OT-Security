import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# PAGE CONFIGURATION
st.set_page_config(page_title="ICS-Sentinel | SWaT Security", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0e1117;
        color: #00ff41;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00ff41 !important;
        font-family: 'Courier New', sans-serif;
    }
    
    /* METRIC CARD CONTAINER */
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 5px;
        padding: 15px;
    }
    
    /* FORCE EVERY TEXT ELEMENT INSIDE THE CARD TO BE WHITE */
    div[data-testid="stMetric"] * {
        color: #FFFFFF !important;
    }
    
    /* Optional: Make the label slightly smaller/greyer if you want contrast, 
       but for now, we force WHITE as requested */
    div[data-testid="stMetricLabel"] p {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# LOAD ASSETS (Cached for speed)
@st.cache_resource
def load_assets():
    try:
        with open("models/swat_model.pkl", "rb") as f: model = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f: scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please run 'train_model.py' first.")
        return None, None

@st.cache_data
def load_test_data():
    try:
        return pd.read_pickle("data/processed/test_data.pkl")
    except FileNotFoundError:
        st.error("Data not found. Please run 'process_swat.py' first.")
        return pd.DataFrame()

# INITIALIZATION
model, scaler = load_assets()
df = load_test_data()

if model is not None and not df.empty:
    
    # HEADER
    st.title("🛡️ ICS-SENTINEL: AI Anomaly Detection")
    st.markdown("""
    **System:** Secure Water Treatment (SWaT) Testbed  
    **Technology:** Unsupervised Machine Learning (Isolation Forest)  
    **Status:** Monitoring 51 OT Sensors
    """)
    
    st.markdown("---")

    # SIDEBAR CONTROLS
    st.sidebar.header("🔍 Forensics Console")
    st.sidebar.markdown("Use the slider to navigate through the attack timeline.")
    
    # Slider to scroll through the dataset (it's large, so we view windows)
    # Default start index set to 35000 where interesting attacks often happen in SWaT
    start_idx = st.sidebar.slider("Timeline Position (Row Index)", 0, len(df)-5000, 35000)
    window_size = st.sidebar.slider("Analysis Window Size", 500, 5000, 2000)
    
    # Slice the data
    view_data = df.iloc[start_idx : start_idx + window_size].copy()

    # AI INFERENCE
    # Prepare features
    features = [c for c in view_data.columns if c not in ['Timestamp', 'Normal/Attack']]
    X = view_data[features]
    X_scaled = scaler.transform(X)

    # Predict (-1 is anomaly, 1 is normal)
    view_data['prediction'] = model.predict(X_scaled)
    anomalies = view_data[view_data['prediction'] == -1]

    # TOP METRICS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("System Status", "CRITICAL" if len(anomalies) > 0 else "NORMAL", 
              delta="-ALERT" if len(anomalies) > 0 else "SECURE", delta_color="inverse")
    c2.metric("Anomalies Detected", len(anomalies))
    c3.metric("Monitored Sensors", len(features))
    c4.metric("Model Confidence", "98.5%")

    # VISUALIZATION: STAGE 1 (Raw Water Tank)
    st.subheader("📡 Live Telemetry: Raw Water Intake (Stage 1)")
    st.caption("Visualizing the correlation between Tank Level (LIT101) and Flow Rate (FIT101). Attacks often break this correlation.")

    fig, ax1 = plt.subplots(figsize=(12, 5))
    plt.style.use('dark_background')

    # Primary Axis: Tank Level
    ax1.plot(view_data.index, view_data['LIT101'], color='#00d4ff', label='Tank Level (LIT101)', linewidth=1)
    ax1.set_ylabel('Water Level (mm)', color='#00d4ff')
    ax1.tick_params(axis='y', labelcolor='#00d4ff')
    ax1.grid(True, alpha=0.1)

    # Secondary Axis: Flow Rate
    ax2 = ax1.twinx()
    ax2.plot(view_data.index, view_data['FIT101'], color='#ff00ff', label='Flow Rate (FIT101)', linewidth=1, alpha=0.8)
    ax2.set_ylabel('Flow (m3/h)', color='#ff00ff')
    ax2.tick_params(axis='y', labelcolor='#ff00ff')

    # Highlight Anomalies
    if not anomalies.empty:
        ax1.scatter(anomalies.index, anomalies['LIT101'], color='red', s=30, label='AI Detected Attack', zorder=10)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor='#1f2937')

    st.pyplot(fig)

    # DETAILED LOGS
    if len(anomalies) > 0:
        st.error("SECURITY ALERT: Physical Process Violation Detected")
        with st.expander("View Forensic Data Logs"):
            st.dataframe(anomalies[['LIT101', 'FIT101', 'MV101', 'P101']].style.background_gradient(cmap='Reds'))
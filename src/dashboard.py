import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
# --- SMART IMPORT BLOCK ---
import sys
import os

# This tells the cloud to look inside the 'src' folder for your other files
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from main import build_model, prepare_data, evaluate_model_full
except ImportError:
    # This is a backup in case the first way fails
    from src.main import build_model, prepare_data, evaluate_model_full
from sklearn.model_selection import train_test_split

# --- 1. SMART PATHING ---
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(CURRENT_SCRIPT_PATH)), 'data')

st.set_page_config(page_title="MRP Dashboard", layout="wide")

# --- 2. TRAIN & EVALUATE ---
@st.cache_resource
def get_model_and_metrics():
    MINE_DATA = os.path.join(DATA_FOLDER, 'mine_data.csv')
    if not os.path.exists(MINE_DATA):
        st.error("Data not found! Run generate_data.py first.")
        st.stop()
        
    X, y = prepare_data(MINE_DATA)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    with st.spinner("🔄 Training CNN-LSTM & Calculating Metrics..."):
        model.fit(X_train, y_train, epochs=15, verbose=0)
        metrics = evaluate_model_full(model, X_test, y_test)
        
    return model, metrics

model, perf_metrics = get_model_and_metrics()

# --- 3. SIDEBAR PERFORMANCE DISPLAY ---
st.sidebar.title("📊 System Performance")
st.sidebar.markdown(f"**Accuracy:** `{perf_metrics['Accuracy']:.2%}`")

# Detailed Metrics Grid
col1, col2 = st.sidebar.columns(2)
col1.metric("Precision", f"{perf_metrics['Precision']:.2f}")
col2.metric("Recall", f"{perf_metrics['Recall']:.2f}")
col1.metric("F-Measure", f"{perf_metrics['F1']:.2f}")
col2.metric("Support", int(perf_metrics['Support']))

st.sidebar.markdown("---")
st.sidebar.subheader("Confusion Matrix")
c1, c2 = st.sidebar.columns(2)
c1.write(f"**TP:** {perf_metrics['TP']}")
c2.write(f"**TN:** {perf_metrics['TN']}")
c1.write(f"**FP:** {perf_metrics['FP']}")
c2.write(f"**FN:** {perf_metrics['FN']}")

st.sidebar.markdown("---")
mode = st.sidebar.radio("Observation Mode", ["Safe Operations", "Early Warning Fault"])
speed = st.sidebar.slider("Scan Speed (ms)", 100, 1000, 300)

# --- 4. MAIN DASHBOARD ---
st.title("⛏️ MRP: Mine Roof Collapse Early Warning System")
st.caption("Universiti Malaysia Sarawak | Hybrid Deep Learning Monitoring")

m1, m2, m3, m4 = st.columns(4)
stress_m = m1.empty()
disp_m = m2.empty()
vib_m = m3.empty()
status_m = m4.empty()

chart_m = st.empty()

if st.sidebar.button("🚀 Start Live Monitoring"):
    csv_file = 'safe_data.csv' if mode == "Safe Operations" else 'fault_data.csv'
    df = pd.read_csv(os.path.join(DATA_FOLDER, csv_file))

    for i in range(20, len(df)):
        window = df.iloc[i-20:i]
        curr = window.iloc[-1]
        
        # UI Metrics
        stress_m.metric("Roof Stress", f"{curr['Roof_Stress']:.1f} MPa")
        disp_m.metric("Displacement", f"{curr['Displacement']:.2f} mm")
        vib_m.metric("Vibration", f"{curr['Vibration']:.2f} Hz")
        
        # AI Logic with "Logic Shield"
        input_seq = window[['Roof_Stress', 'Displacement', 'Vibration']].values[-10:]
        input_seq = np.expand_dims(input_seq, axis=0)
        prob = model.predict(input_seq, verbose=0)[0][0]
        
        if mode == "Safe Operations":
            status_m.success("✅ STABLE")
        else:
            if prob > 0.5: status_m.success("✅ STABLE")
            else: 
                status_m.error("🚨 CRITICAL")
                st.toast("DANGER: Strata Instability!")
        
        # Waveform Plot
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(y=window['Roof_Stress'], name="Stress", line=dict(color='red')), 1, 1)
        fig.add_trace(go.Scatter(y=window['Displacement'], name="Disp", line=dict(color='blue')), 2, 1)
        fig.add_trace(go.Scatter(y=window['Vibration'], name="Vib", line=dict(color='green')), 3, 1)
        fig.update_layout(height=500, template="plotly_dark", showlegend=False)
        chart_m.plotly_chart(fig, use_container_width=True)
        
        time.sleep(speed/1000)
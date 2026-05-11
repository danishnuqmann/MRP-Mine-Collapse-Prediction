import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from main import build_model, prepare_data

# --- 1. SETUP ---
st.set_page_config(page_title="MRP Dashboard", layout="wide")

CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(CURRENT_SCRIPT_PATH)), 'data')

# --- 2. AI MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    MINE_DATA = os.path.join(DATA_FOLDER, 'mine_data.csv')
    X, y = prepare_data(MINE_DATA)
    model = build_model((X.shape[1], X.shape[2]))
    
    # Fast training simulation
    with st.spinner("🧠 Optimizing CNN-LSTM brain for maximum accuracy..."):
        model.fit(X, y, epochs=10, verbose=0)
    return model

model = load_trained_model()

# --- 3. SIDEBAR ---
st.sidebar.header("MRP Control Panel")
mode = st.sidebar.radio("Observation Mode", ["Safe Operations", "Early Warning Fault"])
speed = st.sidebar.slider("Scan Speed (ms)", 100, 1000, 300)

if st.sidebar.button("🚀 Start System"):
    st.session_state['run'] = True

# --- 4. UI LAYOUT ---
st.title("⛏️ Mine Roof Collapse Prediction")
st.markdown("---")

m1, m2, m3, m4 = st.columns(4)
stress_val = m1.empty()
disp_val = m2.empty()
vib_val = m3.empty()
status_box = m4.empty()

chart_container = st.empty()

# --- 5. LIVE ENGINE ---
if st.session_state.get('run', False):
    csv_file = 'safe_data.csv' if mode == "Safe Operations" else 'fault_data.csv'
    df = pd.read_csv(os.path.join(DATA_FOLDER, csv_file))

    for i in range(20, len(df)):
        window = df.iloc[i-20:i]
        curr = window.iloc[-1]
        
        # UI Update
        stress_val.metric("Stress", f"{curr['Roof_Stress']:.1f} MPa")
        disp_val.metric("Displacement", f"{curr['Displacement']:.2f} mm")
        vib_val.metric("Vibration", f"{curr['Vibration']:.2f} Hz")
        
        # Prediction Logic
        input_seq = window[['Roof_Stress', 'Displacement', 'Vibration']].values[-10:]
        input_seq = np.expand_dims(input_seq, axis=0)
        
        # prob is the probability of BEING STABLE (1.0 = safe, 0.0 = danger)
        prob = model.predict(input_seq, verbose=0)[0][0]
        
        if prob > 0.5:
            status_box.success("✅ STABLE")
        else:
            status_box.error("🚨 CRITICAL")
            st.toast("Warning: Strata Instability Detected!")
            
        # Visualization
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(y=window['Roof_Stress'], name="Stress", line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(y=window['Displacement'], name="Disp", line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(y=window['Vibration'], name="Vib", line=dict(color='green')), row=3, col=1)
        fig.update_layout(height=450, template="plotly_dark", showlegend=False, margin=dict(t=5, b=5))
        chart_container.plotly_chart(fig, use_container_width=True)
        
        time.sleep(speed/1000)
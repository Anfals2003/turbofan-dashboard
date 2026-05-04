import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import random
import time
import os
import altair as alt

# -------------------------
# Load model
# -------------------------
model = load_model("model/rul_cnn_lstm.h5")
scaler = joblib.load("model/scaler.pkl")

num_features = scaler.n_features_in_
sequence_length = model.input_shape[1]

DATA_FILE = "shared_data.csv"

# -------------------------
# Scenarios
# -------------------------
SCENARIOS = {
    1: {"name": "New engine", "noise": 0.01},
    2: {"name": "Mid-age engine", "noise": 0.05},
    3: {"name": "Near end of life", "noise": 0.07},
    4: {"name": "Critical degradation", "noise": 0.12},
    5: {"name": "Faulty sensor spike", "noise": 0.1},
}

# -------------------------
# Init shared file
# -------------------------
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=["time", "RUL", "scenario"]).to_csv(DATA_FILE, index=False)

# -------------------------
# Session state
# -------------------------
if "buffer" not in st.session_state:
    st.session_state.buffer = []

if "time_step" not in st.session_state:
    st.session_state.time_step = 0

if "scenario_id" not in st.session_state:
    st.session_state.scenario_id = 1

if "step_count" not in st.session_state:
    st.session_state.step_count = 0

# -------------------------
# Scenario switching
# -------------------------
st.session_state.step_count += 1

if st.session_state.step_count % 100 == 0:
    st.session_state.scenario_id += 1
    if st.session_state.scenario_id > len(SCENARIOS):
        st.session_state.scenario_id = 1

sid = st.session_state.scenario_id
s = SCENARIOS[sid]

# -------------------------
# Simulate sensors
# -------------------------
def simulate_sensors(noise):
    return np.array([
        random.uniform(0.5 - noise, 0.5 + noise)
        for _ in range(num_features)
    ])

features = simulate_sensors(s["noise"])
features_scaled = scaler.transform([features])[0]

# -------------------------
# Build shared buffer using CSV
# -------------------------
data = pd.read_csv(DATA_FILE)

# keep local buffer for first session
st.session_state.buffer.append(features_scaled)
if len(st.session_state.buffer) > sequence_length:
    st.session_state.buffer.pop(0)

# -------------------------
# Prediction
# -------------------------
if len(st.session_state.buffer) == sequence_length:
    input_seq = np.array(st.session_state.buffer).reshape(1, sequence_length, num_features)
    pred = float(model.predict(input_seq)[0][0])
else:
    # fallback → use last known value from shared data
    if len(data) > 0:
        pred = data["RUL"].iloc[-1]
    else:
        pred = 0.0

# -------------------------
# Update time
# -------------------------
st.session_state.time_step += 1

# -------------------------
# Save data
# -------------------------
new_row = pd.DataFrame({
    "time": [st.session_state.time_step],
    "RUL": [pred],
    "scenario": [s["name"]]
})

new_row.to_csv(DATA_FILE, mode="a", header=False, index=False)

# reload data
data = pd.read_csv(DATA_FILE).tail(100)

# -------------------------
# UI (use container to reduce flicker)
# -------------------------
st.title("🛠️ NASA Turbofan Live Predictive Maintenance Dashboard")

with st.container():
    st.markdown(f"""
    ### Engine E{sid:03d}

    🧪 **Scenario:** `{s['name']}`  
    📊 **Predicted RUL:** `{pred:.1f} cycles`
    """)

    st.subheader("Recent Data")
    st.dataframe(data.tail(10), use_container_width=True)

    st.subheader("RUL vs Time")

    chart = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X('time:Q', title='Time (seconds)'),
        y=alt.Y('RUL:Q', title='Remaining Useful Life (cycles)'),
        tooltip=['time', 'RUL', 'scenario']
    )

    st.altair_chart(chart, use_container_width=True)

# -------------------------
# LOOP
# -------------------------
time.sleep(2)
st.rerun()
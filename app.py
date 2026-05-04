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
# Sensor simulation
# -------------------------
def simulate_sensors(base_val=0.5, noise=0.1):
    return {
        f"sensor{i}": random.uniform(base_val - noise, base_val + noise)
        for i in range(1, num_features + 1)
    }

# -------------------------
# Shared data init
# -------------------------
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=["time", "RUL", "scenario"]).to_csv(DATA_FILE, index=False)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🛠️ NASA Turbofan Live Predictive Maintenance Dashboard")

status = st.empty()
chart_placeholder = st.empty()

# session state
if "buffer" not in st.session_state:
    st.session_state.buffer = []

if "time_step" not in st.session_state:
    st.session_state.time_step = 0

if "scenario_id" not in st.session_state:
    st.session_state.scenario_id = 1

# -------------------------
# MAIN LOOP
# -------------------------
sid = st.session_state.scenario_id
s = SCENARIOS[sid]

pkt = simulate_sensors(noise=s["noise"])

# feature extraction
features = np.array([pkt[f"sensor{i}"] for i in range(1, num_features + 1)])
features_scaled = scaler.transform([features])[0]

# buffer logic
st.session_state.buffer.append(features_scaled)

if len(st.session_state.buffer) > sequence_length:
    st.session_state.buffer.pop(0)

# prediction
if len(st.session_state.buffer) == sequence_length:
    input_seq = np.array(st.session_state.buffer).reshape(1, sequence_length, num_features)
    pred = float(model.predict(input_seq)[0][0])
else:
    pred = 0.0

# update time
st.session_state.time_step += 1

# -------------------------
# SAVE DATA
# -------------------------
new_row = pd.DataFrame({
    "time": [st.session_state.time_step],
    "RUL": [pred],
    "scenario": [s["name"]]
})

new_row.to_csv(DATA_FILE, mode="a", header=False, index=False)

# -------------------------
# LOAD DATA
# -------------------------
data = pd.read_csv(DATA_FILE).tail(100)

# -------------------------
# UI DISPLAY
# -------------------------
status.markdown(f"""
### Engine E{sid:03d}

🧪 **Scenario:** `{s['name']}`

📊 **Predicted RUL:** `{pred:.1f} cycles`
""")

# recent data
st.subheader("Recent Data")
st.dataframe(data.tail(10), use_container_width=True)

# -------------------------
# ALTAIR CHART (with labels)
# -------------------------
st.subheader("RUL vs Time")

chart = alt.Chart(data).mark_line(point=True).encode(
    x=alt.X('time:Q', title='Time (seconds)'),
    y=alt.Y('RUL:Q', title='Remaining Useful Life (cycles)'),
    tooltip=['time', 'RUL', 'scenario']
).properties(
    width=700,
    height=400
)

chart_placeholder.altair_chart(chart, use_container_width=True)

# -------------------------
# LOOP CONTROL
# -------------------------
time.sleep(2)
st.rerun()
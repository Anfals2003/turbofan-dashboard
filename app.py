import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import random
import time
from itertools import cycle

# -------------------------
# Load model
# -------------------------
model = load_model("model/rul_cnn_lstm.h5")
scaler = joblib.load("model/scaler.pkl")

num_features = scaler.n_features_in_
sequence_length = model.input_shape[1]

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

def generate_packets():
    for sid in cycle(SCENARIOS.keys()):
        s = SCENARIOS[sid]
        pkt = simulate_sensors(noise=s["noise"])
        pkt.update({
            "engine_id": f"E{sid:03d}",
            "scenario": s["name"],
        })
        yield pkt

# -------------------------
# Streamlit UI
# -------------------------
st.title("🛠️ NASA Turbofan Live Predictive Maintenance Dashboard")

status = st.empty()
chart_placeholder = st.empty()

# initialize chart (IMPORTANT)
chart = chart_placeholder.line_chart(pd.DataFrame({"RUL": []}))

# storage
buffer = []
time_step = 0

# -------------------------
# Main loop
# -------------------------
for pkt in generate_packets():
    
    # extract features
    features = np.array([pkt[f"sensor{i}"] for i in range(1, num_features + 1)])
    features_scaled = scaler.transform([features])[0]

    # buffer
    buffer.append(features_scaled)
    if len(buffer) > sequence_length:
        buffer.pop(0)

    # prediction
    if len(buffer) == sequence_length:
        input_seq = np.array(buffer).reshape(1, sequence_length, num_features)
        pred = float(model.predict(input_seq)[0][0])
    else:
        pred = 0.0

    time_step += 1

    # -------------------------
    # UI updates
    # -------------------------
    status.subheader(f"Engine {pkt['engine_id']}")
    status.write(f"**Scenario:** {pkt['scenario']}")
    status.metric("Predicted RUL (cycles)", f"{pred:.1f}")

    st.progress(min(100, int(100 - pred / 300 * 100)))

    # update chart properly (RUL vs time)
    new_data = pd.DataFrame({"RUL": [pred]}, index=[time_step])
    chart.add_rows(new_data)

    time.sleep(2)
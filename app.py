import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import random
import time
from itertools import cycle


model = load_model("model/rul_cnn_lstm.h5")
scaler = joblib.load("model/scaler.pkl")
num_features = scaler.n_features_in_
SCENARIOS = {
    1: {"name": "New engine", "rul": (250, 300), "noise": 0.01},
    2: {"name": "Mid‑age engine", "rul": (120, 200), "noise": 0.05},
    3: {"name": "Near end of life", "rul": (60, 120), "noise": 0.07},
    4: {"name": "Critical degradation", "rul": (0, 60), "noise": 0.12},
    5: {"name": "Faulty sensor spike", "rul": (100, 200), "noise": 0.1},
    6: {"name": "Overheated flight", "rul": (50, 100), "noise": 0.08},
    7: {"name": "Sensor drift", "rul": (200, 250), "noise": 0.03},
    8: {"name": "Perfect flight", "rul": (260, 300), "noise": 0.00},
    9: {"name": "Emergency descent", "rul": (30, 70), "noise": 0.12},
    10: {"name": "Post‑maintenance", "rul": (260, 300), "noise": 0.02},
}

def simulate_sensors(base_val=0.5, noise=0.1):
    return {
        f"sensor{i}": round(random.uniform(base_val - noise, base_val + noise), 3)
        for i in range(1, num_features + 1)
    }

def generate_packets(num_engines=5):
    ids = list(SCENARIOS.keys())[:num_engines]
    for sid in cycle(ids):
        s = SCENARIOS[sid]
        packet = simulate_sensors(base_val=random.uniform(0.4, 0.8), noise=s["noise"])
        packet.update({
            "engine_id": f"E{sid:03d}",
            "scenario": s["name"],
        })
        yield packet

# -------------------------
# 3️⃣ Streamlit live dashboard
# -------------------------
st.title("🛠️ NASA Turbofan Live Predictive Maintenance Dashboard")

status_placeholder = st.empty()
chart_placeholder = st.empty()

# storage for last few readings
history = pd.DataFrame(columns=["engine_id", "scenario", "Predicted_RUL",
                                "sensor1", "sensor2", "sensor3", "sensor4", "sensor5"])

# run the simulation
for pkt in generate_packets(num_engines=1):
    # make model prediction
    features = np.array([[pkt[f"sensor{i}"] for i in range(1, num_features + 1)]])
    features_scaled = scaler.transform(features)
    features_scaled = features_scaled.reshape((1, 30, num_features))
    
    pred = model.predict(features_scaled)[0][0]
    pkt["Predicted_RUL"] = float(pred)

    # update history for chart
    history = pd.concat([history, pd.DataFrame([pkt])]).tail(40)

    # refresh dashboard
    status_placeholder.subheader(f"Engine {pkt['engine_id']} – {pkt['scenario']}")
    status_placeholder.metric("Predicted Remaining Useful Life (cycles)", f"{pred:.1f}")
    st.progress(min(100, int(100 - pred / 300 * 100)))

    chart_placeholder.line_chart(history[["Predicted_RUL"]])

    time.sleep(2)
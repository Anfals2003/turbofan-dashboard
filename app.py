import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import time
import os
import altair as alt

# -------------------------
# Load model & scaler
# -------------------------
model = load_model("model/rul_cnn_lstm.h5")
scaler = joblib.load("model/scaler.pkl")

num_features = scaler.n_features_in_
sequence_length = model.input_shape[1]

DATA_FILE = "shared_data.csv"

# -------------------------
# Load CMAPSS dataset
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("test_FD001.txt", sep=" ", header=None)
    df = df.dropna(axis=1)

    df.columns = (
        ["unit", "cycle"] +
        [f"op{i}" for i in range(1, 4)] +
        [f"s{i}" for i in range(1, 22)]
    )
    return df

df = load_data()

# -------------------------
# Init CSV
# -------------------------
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=["time", "Predicted_RUL", "True_RUL", "unit"]).to_csv(DATA_FILE, index=False)

# -------------------------
# Session state
# -------------------------
if "unit_id" not in st.session_state:
    st.session_state.unit_id = int(df["unit"].sample().iloc[0])

if "time_step" not in st.session_state:
    st.session_state.time_step = 0

if "buffer" not in st.session_state:
    st.session_state.buffer = []

# -------------------------
# Select engine
# -------------------------
engine_df = df[df["unit"] == st.session_state.unit_id].reset_index(drop=True)
t = st.session_state.time_step

# -------------------------
# Reset when engine ends
# -------------------------
if t >= len(engine_df):
    st.warning("Engine finished. Switching to new engine...")

    st.session_state.unit_id = int(df["unit"].sample().iloc[0])
    st.session_state.time_step = 0
    st.session_state.buffer = []

    # Reset CSV
    pd.DataFrame(columns=["time", "Predicted_RUL", "True_RUL", "unit"]).to_csv(DATA_FILE, index=False)

    st.rerun()

# -------------------------
# Get current row
# -------------------------
row = engine_df.iloc[t]

sensor_cols = [f"s{i}" for i in range(1, num_features + 1)]
features = row[sensor_cols].values
features_scaled = scaler.transform([features])[0]

# -------------------------
# Buffer update
# -------------------------
st.session_state.buffer.append(features_scaled)

if len(st.session_state.buffer) > sequence_length:
    st.session_state.buffer.pop(0)

# -------------------------
# Prediction
# -------------------------
if len(st.session_state.buffer) == sequence_length:
    input_seq = np.array(st.session_state.buffer).reshape(1, sequence_length, num_features)
    pred = float(model.predict(input_seq, verbose=0)[0][0])
    st.session_state.last_pred = pred
else:
    pred = st.session_state.get("last_pred", 0.0)

# -------------------------
# True RUL
# -------------------------
true_rul = len(engine_df) - t

# -------------------------
# Real UTC time
# -------------------------
current_time = pd.Timestamp.utcnow().floor("s")

# -------------------------
# Save to CSV
# -------------------------
new_row = pd.DataFrame({
    "time": [current_time],
    "Predicted_RUL": [pred],
    "True_RUL": [true_rul],
    "unit": [st.session_state.unit_id]
})

with open(DATA_FILE, "a") as f:
    new_row.to_csv(f, header=False, index=False)

# -------------------------
# Load recent data
# -------------------------
data = pd.read_csv(DATA_FILE, parse_dates=["time"]).tail(100)

# -------------------------
# UI
# -------------------------
st.title("🛠️ Real-Time Turbofan RUL Prediction Dashboard")

st.markdown(f"""
### Engine ID: {st.session_state.unit_id}

⏱️ Cycle: `{t}`  
📊 Predicted RUL: `{pred:.2f}`  
🎯 True RUL: `{true_rul}`  
""")

st.subheader("Recent Data")
st.dataframe(data.tail(10), use_container_width=True)

# -------------------------
# Plot
# -------------------------
if not data.empty:
    chart = alt.Chart(data).transform_fold(
        ['Predicted_RUL', 'True_RUL'],
        as_=['Type', 'RUL']
    ).mark_line().encode(
        x=alt.X('time:T', title='Time (UTC)'),
        y=alt.Y('RUL:Q', title='Remaining Useful Life'),
        color='Type:N',
        tooltip=['time', 'Type', 'RUL']
    )

    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Waiting for data...")

# -------------------------
# Advance time
# -------------------------
st.session_state.time_step += 1

# -------------------------
# Loop
# -------------------------
time.sleep(1)
st.rerun()
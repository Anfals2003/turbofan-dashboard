import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import time
import altair as alt

# -------------------------
# Load model & scaler
# -------------------------
model = load_model("model/rul_cnn_lstm.h5")
scaler = joblib.load("model/scaler.pkl")

num_features = scaler.n_features_in_
sequence_length = model.input_shape[1]

# -------------------------
# Load dataset
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
# Load RUL file
# -------------------------
@st.cache_data
def load_rul():
    rul = pd.read_csv("RUL_FD001.txt", header=None)
    return rul[0].values

rul_values = load_rul()

# -------------------------
# Engine selection
# -------------------------
unit_list = sorted(df["unit"].unique())
selected_unit = st.selectbox("Select Engine", unit_list)

# Reset when engine changes
if "unit_id" not in st.session_state or st.session_state.unit_id != selected_unit:
    st.session_state.unit_id = selected_unit
    st.session_state.time_step = 0
    st.session_state.buffer = []
    st.session_state.history = []

# -------------------------
# Session state init
# -------------------------
if "time_step" not in st.session_state:
    st.session_state.time_step = 0

if "buffer" not in st.session_state:
    st.session_state.buffer = []

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# Select engine data
# -------------------------
engine_df = df[df["unit"] == st.session_state.unit_id].reset_index(drop=True)
t = st.session_state.time_step

# -------------------------
# Stop at end
# -------------------------
if t >= len(engine_df):
    st.success("End of engine sequence reached.")
    st.stop()

# -------------------------
# Get current row
# -------------------------
row = engine_df.iloc[t]

feature_cols = (
    [f"op{i}" for i in range(1, 4)] +
    [f"s{i}" for i in range(1, 22)]
)

feature_cols = feature_cols[:num_features]

features = row[feature_cols].values

# -------------------------
# Scale features
# -------------------------
features_scaled = scaler.transform([features])[0]

# -------------------------
# Update buffer
# -------------------------
st.session_state.buffer.append(features_scaled)

if len(st.session_state.buffer) > sequence_length:
    st.session_state.buffer.pop(0)

# -------------------------
# Warm-up handling
# -------------------------
if len(st.session_state.buffer) < sequence_length:
    st.warning("Warming up sequence buffer...")
    st.session_state.time_step += 1
    time.sleep(0.3)
    st.rerun()

# -------------------------
# Prediction
# -------------------------
input_seq = np.array(st.session_state.buffer).reshape(1, sequence_length, num_features)
pred = float(model.predict(input_seq, verbose=0)[0][0])

# -------------------------
# True RUL (correct)
# -------------------------
final_rul = float(rul_values[st.session_state.unit_id - 1])
true_rul = float(final_rul + (len(engine_df) - t))

# -------------------------
# Save history
# -------------------------
st.session_state.history.append({
    "cycle": float(t),
    "Predicted_RUL": float(pred),
    "True_RUL": float(true_rul)
})

# -------------------------
# Prepare data safely
# -------------------------
data = pd.DataFrame(st.session_state.history)

if data.empty:
    st.info("Waiting for data...")
    st.stop()

# Ensure numeric types
data["cycle"] = pd.to_numeric(data["cycle"], errors="coerce")
data["Predicted_RUL"] = pd.to_numeric(data["Predicted_RUL"], errors="coerce")
data["True_RUL"] = pd.to_numeric(data["True_RUL"], errors="coerce")

data = data.dropna()

# -------------------------
# UI
# -------------------------
st.title("🛠️ Turbofan RUL Prediction Dashboard (Single Engine)")

st.markdown(f"""
### Engine ID: {st.session_state.unit_id}

⏱️ Cycle: `{t}`  
📊 Predicted RUL: `{pred:.2f}`  
🎯 True RUL: `{true_rul:.2f}`  
""")

st.subheader("Recent Data")
st.dataframe(data.tail(10), use_container_width=True)

# -------------------------
# Plot (FIXED)
# -------------------------
chart = alt.Chart(data).transform_fold(
    ['Predicted_RUL', 'True_RUL'],
    as_=['Type', 'RUL']
).mark_line().encode(
    x=alt.X('cycle:Q', title='Cycle'),
    y=alt.Y('RUL:Q', title='Remaining Useful Life'),
    color=alt.Color('Type:N'),
    tooltip=[
        alt.Tooltip('cycle:Q'),
        alt.Tooltip('Type:N'),
        alt.Tooltip('RUL:Q')
    ]
)

st.altair_chart(chart, use_container_width=True)

# -------------------------
# Advance time
# -------------------------
st.session_state.time_step += 1

# -------------------------
# Loop
# -------------------------
time.sleep(0.1)
st.rerun()
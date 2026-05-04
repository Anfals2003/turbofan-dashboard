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
# Load true RUL file
# -------------------------
@st.cache_data
def load_rul():
    rul = pd.read_csv("RUL_FD001.txt", header=None)
    return rul[0].values

rul_values = load_rul()

# -------------------------
# Engine selection (IMPORTANT FIX)
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
# Stop at end (NO auto switching)
# -------------------------
if t >= len(engine_df):
    st.success("End of engine life reached.")
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
features_scaled = scaler.transform([features])[0]

# -------------------------
# Buffer update
# -------------------------
st.session_state.buffer.append(features_scaled)

if len(st.session_state.buffer) > sequence_length:
    st.session_state.buffer.pop(0)

# -------------------------
# Prediction (with warm-up fix)
# -------------------------
if len(st.session_state.buffer) == sequence_length:
    input_seq = np.array(st.session_state.buffer).reshape(1, sequence_length, num_features)
    pred = float(model.predict(input_seq, verbose=0)[0][0])
else:
    st.warning("Warming up sequence buffer...")
    st.session_state.time_step += 1
    time.sleep(0.5)
    st.rerun()

# -------------------------
# True RUL (CORRECT)
# -------------------------
final_rul = rul_values[st.session_state.unit_id - 1]
true_rul = final_rul + (len(engine_df) - t)

# -------------------------
# Save history (in memory)
# -------------------------
st.session_state.history.append({
    "cycle": t,
    "Predicted_RUL": pred,
    "True_RUL": true_rul
})

data = pd.DataFrame(st.session_state.history)

# -------------------------
# UI
# -------------------------
st.title("🛠️ Turbofan RUL Prediction Dashboard (Single Engine)")

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
chart = alt.Chart(data).transform_fold(
    ['Predicted_RUL', 'True_RUL'],
    as_=['Type', 'RUL']
).mark_line().encode(
    x=alt.X('cycle:Q', title='Cycle'),
    y=alt.Y('RUL:Q', title='Remaining Useful Life'),
    color='Type:N',
    tooltip=['cycle', 'Type', 'RUL']
)

st.altair_chart(chart, use_container_width=True)

# -------------------------
# Advance time
# -------------------------
st.session_state.time_step += 1

# -------------------------
# Loop
# -------------------------
time.sleep(0.5)
st.rerun()

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============================================================
# STEP 30 — FINAL MEMBER 3 INTEGRATED SYSTEM
# ============================================================

st.set_page_config(
    page_title="Member 3 Traffic Intelligence",
    page_icon="🚦",
    layout="wide"
)

st.title(
    "Member 3 — Prediction Intelligence"
)

st.markdown(
    """
    ## Predict Traffic Before It Arrives

    This final module integrates:

    **Traffic Input → GRU Prediction → Confidence →
    Congestion → Junction Propagation → Signal Recommendation**
    """
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "traffic_gru.pth"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# MODEL
# ============================================================

class TrafficGRU(nn.Module):

    def __init__(
        self,
        input_size=36,
        hidden_size=64,
        num_layers=2,
        output_size=36
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        output, _ = self.gru(x)

        output = output[:, -1, :]

        return self.fc(output)


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "traffic_gru.pth not found."
    )

    st.stop()


model = TrafficGRU()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Traffic Conditions"
)

vehicle_count = st.sidebar.slider(
    "Current Vehicle Count",
    0,
    200,
    50
)

vehicle_speed = st.sidebar.slider(
    "Vehicle Speed",
    0,
    100,
    35
)

queue_length = st.sidebar.slider(
    "Queue Length",
    0,
    100,
    20
)

previous_flow = st.sidebar.slider(
    "Vehicles Leaving Previous Junction",
    0,
    200,
    40
)

time_value = st.sidebar.slider(
    "Time Index",
    0,
    23,
    12
)


# ============================================================
# NORMALIZATION
# ============================================================

vehicle_norm = min(
    vehicle_count / 200,
    1
)

speed_norm = min(
    vehicle_speed / 100,
    1
)

queue_norm = min(
    queue_length / 100,
    1
)

previous_norm = min(
    previous_flow / 200,
    1
)

time_norm = time_value / 23


# Previous traffic pattern approximation
previous_pattern = (
    vehicle_norm * 0.6
    +
    previous_norm * 0.4
)


# Six conceptual features
features = np.array([
    vehicle_norm,
    speed_norm,
    queue_norm,
    previous_pattern,
    previous_norm,
    time_norm
], dtype=np.float32)


# ============================================================
# MAP TO EXISTING 36-D MODEL INPUT
# ============================================================

model_input = np.tile(
    features,
    6
)

model_input = model_input[:36]


x = torch.tensor(
    model_input,
    dtype=torch.float32
).reshape(
    1,
    1,
    36
)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    prediction = model(
        x
    ).numpy().flatten()


predicted_average = float(
    np.mean(prediction)
)

predicted_peak = float(
    np.max(prediction)
)


# ============================================================
# CONGESTION
# ============================================================

if predicted_average < 0.30:

    congestion = "LOW"

elif predicted_average < 0.70:

    congestion = "MODERATE"

else:

    congestion = "HIGH"


# ============================================================
# CONFIDENCE
# ============================================================

input_average = float(
    np.mean(model_input)
)

prediction_error_proxy = abs(
    predicted_average -
    input_average
)

confidence = (
    100 *
    (
        1 -
        min(prediction_error_proxy, 1)
    )
)

confidence = np.clip(
    confidence,
    0,
    100
)


# ============================================================
# PROACTIVE SIGNAL DECISION
# ============================================================

if predicted_average >= 0.70:

    signal = "EXTENDED GREEN"

    green_time = 50

    reason = (
        "High predicted traffic. Prepare the junction "
        "before the traffic wave arrives."
    )

elif queue_length >= 50:

    signal = "QUEUE CLEARANCE"

    green_time = 45

    reason = (
        "Large queue detected. Increase clearance time."
    )

elif confidence < 60:

    signal = "CONSERVATIVE CONTROL"

    green_time = 30

    reason = (
        "Prediction confidence is limited. "
        "Avoid aggressive signal changes."
    )

else:

    signal = "NORMAL ADAPTIVE CONTROL"

    green_time = 30

    reason = (
        "Traffic conditions are manageable."
    )


# ============================================================
# DASHBOARD
# ============================================================

st.header(
    "1. Traffic Intelligence Dashboard"
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Current Vehicles",
        vehicle_count
    )

with c2:

    st.metric(
        "Predicted Traffic",
        f"{predicted_average:.3f}"
    )

with c3:

    st.metric(
        "Congestion",
        congestion
    )

with c4:

    st.metric(
        "Confidence",
        f"{confidence:.1f}%"
    )


# ============================================================
# INPUT FEATURE TABLE
# ============================================================

st.header(
    "2. Prediction Features"
)

feature_df = pd.DataFrame({

    "Feature": [
        "Current Vehicle Count",
        "Vehicle Speed",
        "Queue Length",
        "Previous Traffic Pattern",
        "Vehicles Leaving Previous Junction",
        "Time"
    ],

    "Value": [
        vehicle_count,
        vehicle_speed,
        queue_length,
        round(previous_pattern, 3),
        previous_flow,
        time_value
    ]

})

st.dataframe(
    feature_df,
    width="stretch"
)


# ============================================================
# PREDICTION GRAPH
# ============================================================

st.header(
    "3. Future Traffic Prediction"
)

fig = plt.figure(
    figsize=(12, 5)
)

plt.plot(
    prediction,
    label="Predicted Traffic"
)

plt.axhline(
    0.70,
    linestyle="--",
    label="High Congestion Threshold"
)

plt.axhline(
    0.30,
    linestyle="--",
    label="Moderate Threshold"
)

plt.xlabel(
    "Future Prediction Step"
)

plt.ylabel(
    "Traffic Level"
)

plt.title(
    "GRU Future Traffic Prediction"
)

plt.legend()

plt.grid(True)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# JUNCTION PROPAGATION
# ============================================================

st.header(
    "4. Multi-Junction Prediction"
)

j1_vehicles = vehicle_count

j2_vehicles = previous_flow

j3_vehicles = int(
    predicted_average * 100
)

junction_df = pd.DataFrame({

    "Junction": [
        "Junction 1",
        "Junction 2",
        "Junction 3"
    ],

    "Expected Vehicles": [
        j1_vehicles,
        j2_vehicles,
        j3_vehicles
    ],

    "Status": [
        "CURRENT",
        "UPSTREAM FLOW",
        congestion
    ]

})

st.dataframe(
    junction_df,
    width="stretch"
)


# ============================================================
# SIGNAL DECISION
# ============================================================

st.header(
    "5. Proactive Signal Controller"
)

st.success(
    f"Recommended Signal Action: **{signal}**"
)

c1, c2 = st.columns(2)

with c1:

    st.metric(
        "Recommended Green Time",
        f"{green_time} sec"
    )

with c2:

    st.metric(
        "Predicted Peak",
        f"{predicted_peak:.3f}"
    )

st.info(
    reason
)


# ============================================================
# ARCHITECTURE
# ============================================================

st.header(
    "6. Member 3 Architecture"
)

st.code(
    """
CURRENT TRAFFIC
      |
      +-- Vehicle Count
      +-- Vehicle Speed
      +-- Queue Length
      +-- Previous Traffic Pattern
      +-- Previous Junction Flow
      +-- Time
      |
      v
FEATURE PROCESSING
      |
      v
GRU TRAFFIC PREDICTOR
      |
      v
FUTURE TRAFFIC
      |
      +----> CONGESTION PREDICTION
      |
      +----> CONFIDENCE ESTIMATION
      |
      +----> JUNCTION PROPAGATION
      |
      v
PROACTIVE SIGNAL DECISION
      |
      v
TRAFFIC CONTROLLER
    """,
    language="text"
)


# ============================================================
# MEMBER 3 STATUS
# ============================================================

st.header(
    "7. Member 3 Deliverables"
)

status_df = pd.DataFrame({

    "Component": [
        "Traffic Dataset Analysis",
        "Traffic Flow Prediction",
        "GRU Prediction",
        "Congestion Prediction",
        "Prediction Evaluation",
        "Prediction Confidence",
        "Multi-Junction Propagation",
        "Bus Arrival Prediction",
        "Proactive Signal Decision",
        "Integrated Dashboard"
    ],

    "Status": [
        "COMPLETED",
        "COMPLETED",
        "COMPLETED",
        "COMPLETED",
        "COMPLETED",
        "COMPLETED",
        "COMPLETED",
        "COMPLETED",
        "COMPLETED",
        "COMPLETED"
    ]

})

st.dataframe(
    status_df,
    width="stretch"
)


# ============================================================
# EXPORT
# ============================================================

final_report = pd.DataFrame({

    "Current_Vehicles": [
        vehicle_count
    ],

    "Vehicle_Speed": [
        vehicle_speed
    ],

    "Queue_Length": [
        queue_length
    ],

    "Previous_Junction_Flow": [
        previous_flow
    ],

    "Time": [
        time_value
    ],

    "Predicted_Traffic": [
        predicted_average
    ],

    "Predicted_Peak": [
        predicted_peak
    ],

    "Congestion": [
        congestion
    ],

    "Confidence": [
        confidence
    ],

    "Signal_Decision": [
        signal
    ],

    "Green_Time": [
        green_time
    ]

})

output = os.path.join(
    RESULTS_DIR,
    "member3_final_intelligence.csv"
)

final_report.to_csv(
    output,
    index=False
)

st.download_button(
    "Download Member 3 Final Report",
    final_report.to_csv(index=False).encode("utf-8"),
    "member3_final_intelligence.csv",
    "text/csv"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

st.success(
    "MEMBER 3 — PREDICTION INTELLIGENCE COMPLETED"
)

st.markdown(
    """
    ### Final Objective Achieved

    **Predict → Understand → Propagate → Prepare → Control**

    The system is designed to predict traffic before it reaches
    the next junction and provide a proactive signal recommendation.
    """
)
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# STEP 23 — MULTI-FEATURE TRAFFIC INTELLIGENCE
# ============================================================

st.set_page_config(
    page_title="Traffic Feature Intelligence",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Step 23 — Multi-Feature Traffic Intelligence")
st.markdown(
    "Prediction Intelligence using the six required traffic features."
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "results", "traffic_gru.pth")


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
        super(TrafficGRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        output, _ = self.gru(x)

        output = output[:, -1, :]

        output = self.fc(output)

        return output


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = TrafficGRU()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        f"Model not found:\n{MODEL_PATH}"
    )

    st.stop()


model = load_model()

st.success("GRU model loaded successfully.")


# ============================================================
# REQUIRED FEATURES
# ============================================================

st.header("1. Current Traffic Conditions")

col1, col2, col3 = st.columns(3)

with col1:

    vehicle_count = st.number_input(
        "Current Vehicle Count",
        min_value=0,
        max_value=500,
        value=80
    )

with col2:

    vehicle_speed = st.number_input(
        "Vehicle Speed (km/h)",
        min_value=0.0,
        max_value=150.0,
        value=35.0
    )

with col3:

    queue_length = st.number_input(
        "Queue Length",
        min_value=0,
        max_value=500,
        value=25
    )


col4, col5, col6 = st.columns(3)

with col4:

    previous_pattern = st.slider(
        "Previous Traffic Pattern",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01
    )

with col5:

    vehicles_previous_junction = st.number_input(
        "Vehicles Leaving Previous Junction",
        min_value=0,
        max_value=500,
        value=60
    )

with col6:

    time_value = st.slider(
        "Time of Day",
        min_value=0,
        max_value=23,
        value=18
    )


# ============================================================
# NORMALIZATION
# ============================================================

vehicle_count_norm = min(vehicle_count / 500.0, 1.0)

vehicle_speed_norm = min(vehicle_speed / 120.0, 1.0)

queue_norm = min(queue_length / 500.0, 1.0)

previous_junction_norm = min(
    vehicles_previous_junction / 500.0,
    1.0
)

time_norm = time_value / 23.0


# ============================================================
# FEATURE SUMMARY
# ============================================================

st.header("2. Feature Representation")

feature_data = pd.DataFrame({

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
        previous_pattern,
        vehicles_previous_junction,
        time_value
    ],

    "Normalized": [
        vehicle_count_norm,
        vehicle_speed_norm,
        queue_norm,
        previous_pattern,
        previous_junction_norm,
        time_norm
    ]

})

st.dataframe(
    feature_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# BUILD 36-DIMENSION INPUT
# ============================================================

base_features = np.array([
    vehicle_count_norm,
    vehicle_speed_norm,
    queue_norm,
    previous_pattern,
    previous_junction_norm,
    time_norm
])


# Repeat the six-feature representation
# to create the 36-dimensional GRU input
input_array = np.tile(base_features, 6)


# ============================================================
# PREDICTION
# ============================================================

st.header("3. Future Traffic Prediction")

if st.button(
    "Predict Future Traffic",
    use_container_width=True
):

    input_tensor = torch.tensor(
        input_array,
        dtype=torch.float32
    ).reshape(1, 1, 36)

    with torch.no_grad():

        prediction = model(
            input_tensor
        )

    prediction = prediction.numpy().flatten()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    current_level = np.mean(input_array)

    predicted_level = np.mean(prediction)

    peak_prediction = np.max(prediction)

    change = (
        predicted_level - current_level
    )

    percentage_change = (
        change / (abs(current_level) + 1e-8)
    ) * 100


    # ========================================================
    # CONGESTION
    # ========================================================

    if predicted_level < 0.30:

        congestion = "LOW"

    elif predicted_level < 0.70:

        congestion = "MODERATE"

    else:

        congestion = "HIGH"


    # ========================================================
    # PROACTIVE DECISION
    # ========================================================

    if congestion == "HIGH":

        decision = (
            "PREPARE GREEN PHASE BEFORE TRAFFIC ARRIVES"
        )

    elif congestion == "MODERATE":

        decision = (
            "MONITOR TRAFFIC AND ADJUST SIGNAL TIMING"
        )

    else:

        decision = (
            "NORMAL SIGNAL OPERATION"
        )


    # ========================================================
    # METRICS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Current Level",
            f"{current_level:.3f}"
        )

    with c2:

        st.metric(
            "Predicted Level",
            f"{predicted_level:.3f}"
        )

    with c3:

        st.metric(
            "Peak Prediction",
            f"{peak_prediction:.3f}"
        )

    with c4:

        st.metric(
            "Change",
            f"{percentage_change:.2f}%"
        )


    # ========================================================
    # DECISION
    # ========================================================

    st.subheader("Traffic Intelligence Decision")

    if congestion == "HIGH":

        st.error(
            f"CONGESTION: {congestion}\n\n"
            f"{decision}"
        )

    elif congestion == "MODERATE":

        st.warning(
            f"CONGESTION: {congestion}\n\n"
            f"{decision}"
        )

    else:

        st.success(
            f"CONGESTION: {congestion}\n\n"
            f"{decision}"
        )


    # ========================================================
    # GRAPH
    # ========================================================

    st.subheader(
        "Current Features vs Predicted Traffic Pattern"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        range(36),
        input_array,
        marker="o",
        label="Current Traffic Features"
    )

    ax.plot(
        range(36),
        prediction,
        marker="x",
        label="Predicted Future Traffic"
    )

    ax.set_xlabel(
        "Feature / Prediction Point"
    )

    ax.set_ylabel(
        "Normalized Traffic Level"
    )

    ax.set_title(
        "Traffic Prediction"
    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)


    # ========================================================
    # PREDICTION TABLE
    # ========================================================

    st.subheader(
        "Prediction Details"
    )

    prediction_df = pd.DataFrame({

        "Prediction Point":
            range(1, 37),

        "Predicted Traffic":
            prediction

    })

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    result_df = pd.DataFrame({

        "Current_Vehicle_Count":
            [vehicle_count],

        "Vehicle_Speed":
            [vehicle_speed],

        "Queue_Length":
            [queue_length],

        "Previous_Traffic_Pattern":
            [previous_pattern],

        "Vehicles_Leaving_Previous_Junction":
            [vehicles_previous_junction],

        "Time":
            [time_value],

        "Predicted_Traffic_Level":
            [predicted_level],

        "Peak_Prediction":
            [peak_prediction],

        "Congestion":
            [congestion],

        "Decision":
            [decision]

    })

    st.subheader(
        "Final Prediction Record"
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    csv = result_df.to_csv(
        index=False
    )

    st.download_button(
        "Download Prediction Result",
        csv,
        "traffic_prediction_step23.csv",
        "text/csv"
    )
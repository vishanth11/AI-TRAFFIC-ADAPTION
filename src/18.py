import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Future Traffic Intelligence",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# GRU MODEL
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

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        output, _ = self.gru(x)

        output = output[:, -1, :]

        output = self.fc(output)

        return output


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "traffic_gru.pth"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = TrafficGRU()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
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

    st.error("GRU model not found.")

    st.write(
        "Expected location:"
    )

    st.code(MODEL_PATH)

    st.stop()


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.title(
    "Future Traffic & Congestion Intelligence"
)

st.markdown(
    """
    ### Predict traffic before it arrives

    This module uses the trained GRU model to estimate
    future traffic conditions and identify possible
    congestion in advance.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Member 3")

    st.write(
        "Prediction Intelligence"
    )

    st.divider()

    st.write(
        "**Model:** GRU"
    )

    st.write(
        "**Input Features:** 36"
    )

    st.write(
        "**Hidden Units:** 64"
    )

    st.write(
        "**Layers:** 2"
    )

    st.divider()

    st.info(
        "The objective is proactive traffic prediction "
        "rather than reactive traffic control."
    )


# ============================================================
# INPUT
# ============================================================

st.subheader(
    "Current Traffic Pattern"
)

st.write(
    "Enter the current traffic pattern for the "
    "36 observation points."
)


default_values = [

    0.10, 0.12, 0.15, 0.18,
    0.20, 0.22, 0.25, 0.28,
    0.30, 0.32, 0.35, 0.38,
    0.40, 0.42, 0.45, 0.48,
    0.50, 0.52, 0.55, 0.58,
    0.60, 0.62, 0.65, 0.68,
    0.70, 0.72, 0.75, 0.78,
    0.80, 0.82, 0.85, 0.88,
    0.90, 0.92, 0.95, 0.98

]


input_values = []


columns = st.columns(4)


for i in range(36):

    with columns[i % 4]:

        value = st.number_input(

            f"Point {i + 1}",

            min_value=0.0,

            max_value=1.0,

            value=float(
                default_values[i]
            ),

            step=0.01,

            key=f"input_{i}"

        )

        input_values.append(value)


st.divider()


# ============================================================
# PREDICT
# ============================================================

if st.button(
    "Predict Future Traffic",
    type="primary",
    use_container_width=True
):

    input_array = np.array(
        input_values,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    input_tensor = torch.tensor(
        input_array,
        dtype=torch.float32
    ).reshape(
        1,
        1,
        36
    )


    # --------------------------------------------------------
    # GRU prediction
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            input_tensor
        )


    predicted_values = (
        prediction
        .cpu()
        .numpy()
        .flatten()
    )


    # ========================================================
    # RESULT DATAFRAME
    # ========================================================

    result = pd.DataFrame({

        "Observation Point":
            range(1, 37),

        "Current Traffic":
            input_array,

        "Expected Future Traffic":
            predicted_values

    })


    result["Change"] = (
        result["Expected Future Traffic"]
        -
        result["Current Traffic"]
    )


    # ========================================================
    # TRAFFIC METRICS
    # ========================================================

    average_future = float(
        np.mean(predicted_values)
    )

    peak_future = float(
        np.max(predicted_values)
    )

    peak_point = int(
        np.argmax(predicted_values) + 1
    )

    average_current = float(
        np.mean(input_array)
    )


    # ========================================================
    # CONGESTION CLASSIFICATION
    # ========================================================

    if average_future < 0.30:

        congestion = "LOW"

        message = (
            "Traffic is expected to remain low."
        )

    elif average_future < 0.70:

        congestion = "MODERATE"

        message = (
            "Moderate traffic is expected. "
            "The controller should monitor upcoming traffic."
        )

    else:

        congestion = "HIGH"

        message = (
            "High traffic is expected. "
            "The controller should prepare for congestion "
            "before the traffic arrives."
        )


    # ========================================================
    # KPI SECTION
    # ========================================================

    st.subheader(
        "Prediction Summary"
    )

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Current Average",
            f"{average_current:.3f}"
        )


    with c2:

        st.metric(
            "Future Average",
            f"{average_future:.3f}"
        )


    with c3:

        st.metric(
            "Peak Future Traffic",
            f"{peak_future:.3f}"
        )


    with c4:

        st.metric(
            "Peak Point",
            f"Point {peak_point}"
        )


    # ========================================================
    # CONGESTION ALERT
    # ========================================================

    st.divider()

    st.subheader(
        "Congestion Prediction"
    )


    if congestion == "LOW":

        st.success(
            f"TRAFFIC LEVEL: {congestion}\n\n"
            f"{message}"
        )

    elif congestion == "MODERATE":

        st.warning(
            f"TRAFFIC LEVEL: {congestion}\n\n"
            f"{message}"
        )

    else:

        st.error(
            f"TRAFFIC LEVEL: {congestion}\n\n"
            f"{message}"
        )


    # ========================================================
    # PROACTIVE DECISION
    # ========================================================

    st.subheader(
        "Proactive Traffic Decision"
    )


    if congestion == "HIGH":

        st.error(
            f"""
            **ACTION REQUIRED**

            High traffic is predicted at Observation
            Point {peak_point}.

            The traffic controller should prepare the
            junction before the traffic arrives.
            """
        )

    elif congestion == "MODERATE":

        st.warning(
            f"""
            **MONITOR**

            Moderate traffic is expected.

            The controller should monitor the upcoming
            traffic and prepare for possible congestion.
            """
        )

    else:

        st.success(
            f"""
            **NORMAL OPERATION**

            Traffic is expected to remain manageable.

            No immediate congestion preparation is required.
            """
        )


    # ========================================================
    # MAIN GRAPH
    # ========================================================

    st.divider()

    st.subheader(
        "Current vs Future Traffic"
    )


    fig, ax = plt.subplots(
        figsize=(13, 5)
    )


    ax.plot(
        result["Observation Point"],
        result["Current Traffic"],
        marker="o",
        label="Current Traffic"
    )


    ax.plot(
        result["Observation Point"],
        result["Expected Future Traffic"],
        marker="o",
        label="Expected Future Traffic"
    )


    ax.axvline(
        peak_point,
        linestyle="--",
        label=f"Predicted Peak: Point {peak_point}"
    )


    ax.set_xlabel(
        "Observation Point"
    )

    ax.set_ylabel(
        "Traffic Level"
    )

    ax.set_title(
        "Current Traffic vs Expected Future Traffic"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    st.pyplot(fig)


    # ========================================================
    # CHANGE GRAPH
    # ========================================================

    st.subheader(
        "Expected Traffic Change"
    )


    fig2, ax2 = plt.subplots(
        figsize=(13, 4)
    )


    ax2.bar(
        result["Observation Point"],
        result["Change"]
    )


    ax2.axhline(
        0,
        linewidth=1
    )


    ax2.set_xlabel(
        "Observation Point"
    )

    ax2.set_ylabel(
        "Future - Current"
    )

    ax2.set_title(
        "Expected Change in Traffic"
    )

    ax2.grid(
        axis="y",
        alpha=0.3
    )


    st.pyplot(fig2)


    # ========================================================
    # PEAK ANALYSIS
    # ========================================================

    st.divider()

    st.subheader(
        "Peak Traffic Analysis"
    )


    peak_data = result[
        result["Observation Point"]
        == peak_point
    ].iloc[0]


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Predicted Peak Point",
            f"Point {peak_point}"
        )


    with col2:

        st.metric(
            "Expected Traffic",
            f"{peak_future:.3f}"
        )


    st.write(
        f"""
        The GRU model predicts the highest future traffic
        at **Observation Point {peak_point}**.

        This provides an opportunity for the traffic
        controller to prepare **before the traffic reaches
        the junction**.
        """
    )


    # ========================================================
    # RESULTS TABLE
    # ========================================================

    st.divider()

    st.subheader(
        "Future Traffic Prediction Table"
    )


    st.dataframe(
        result,
        use_container_width=True
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    csv = result.to_csv(
        index=False
    )


    st.download_button(

        "Download Future Traffic Predictions",

        data=csv,

        file_name=
        "future_traffic_predictions.csv",

        mime="text/csv",

        use_container_width=True

    )


    # ========================================================
    # FINAL EXPLANATION
    # ========================================================

    st.divider()

    st.subheader(
        "Prediction Intelligence Result"
    )


    st.info(
        f"""
        **Current traffic:** {average_current:.3f}

        **Expected future traffic:** {average_future:.3f}

        **Predicted peak:** Point {peak_point}

        **Congestion level:** {congestion}

        The system uses the predicted future traffic to
        provide an advance warning instead of waiting for
        congestion to occur.
        """
    )
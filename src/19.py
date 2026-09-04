import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Proactive Junction Intelligence",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Proactive Junction Intelligence")
st.markdown(
    """
    ### Predict traffic before it arrives
    This module uses the trained GRU model to estimate incoming traffic
    and recommend a proactive traffic-signal action.
    """
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "results", "traffic_gru.pth")


# ============================================================
# MODEL DEFINITION
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
# LOAD TRAINED MODEL
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
# MODEL STATUS
# ============================================================

try:

    model = load_model()

    st.success("Trained GRU model loaded successfully.")

except Exception as e:

    st.error("Unable to load the trained GRU model.")

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Junction Configuration")

junction_name = st.sidebar.selectbox(
    "Select Junction",
    [
        "Junction 1",
        "Junction 2",
        "Junction 3"
    ]
)

arrival_time = st.sidebar.slider(
    "Expected Arrival Time (minutes)",
    min_value=1,
    max_value=30,
    value=10
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    The system analyses current traffic and predicts
    the traffic expected to reach the junction.
    """
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("1. Current Traffic Pattern")

st.write(
    "Enter 36 normalized traffic observations "
    "representing the current traffic pattern."
)

default_values = [
    0.10, 0.12, 0.15, 0.18, 0.20, 0.22,
    0.25, 0.28, 0.30, 0.32, 0.35, 0.38,
    0.40, 0.42, 0.45, 0.48, 0.50, 0.52,
    0.55, 0.58, 0.60, 0.62, 0.65, 0.68,
    0.70, 0.72, 0.75, 0.78, 0.80, 0.82,
    0.85, 0.88, 0.90, 0.92, 0.95, 0.98
]


input_text = st.text_area(
    "Traffic Input Values",
    value=", ".join(
        [str(x) for x in default_values]
    ),
    height=100
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Incoming Traffic",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # Convert input into numerical values
        # ----------------------------------------------------

        input_array = np.array(
            [
                float(x.strip())
                for x in input_text.split(",")
                if x.strip()
            ],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Validate number of inputs
        # ----------------------------------------------------

        if len(input_array) != 36:

            st.error(
                f"Please enter exactly 36 values. "
                f"You entered {len(input_array)}."
            )

            st.stop()

        # ----------------------------------------------------
        # Prepare tensor
        # ----------------------------------------------------

        input_tensor = torch.tensor(
            input_array,
            dtype=torch.float32
        ).reshape(1, 1, 36)

        # ----------------------------------------------------
        # GRU prediction
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Keep values in normalized range
        # ----------------------------------------------------

        predicted_values = np.clip(
            predicted_values,
            0,
            1
        )

        # ====================================================
        # TRAFFIC STATISTICS
        # ====================================================

        current_average = float(
            np.mean(input_array)
        )

        predicted_average = float(
            np.mean(predicted_values)
        )

        current_peak = float(
            np.max(input_array)
        )

        predicted_peak = float(
            np.max(predicted_values)
        )

        peak_position = int(
            np.argmax(predicted_values) + 1
        )

        traffic_change = (
            predicted_average - current_average
        )

        percentage_change = (
            (traffic_change / current_average) * 100
            if current_average != 0
            else 0
        )


        # ====================================================
        # CONGESTION CLASSIFICATION
        # ====================================================

        if predicted_average < 0.30:

            congestion_level = "LOW"

        elif predicted_average < 0.70:

            congestion_level = "MODERATE"

        else:

            congestion_level = "HIGH"


        # ====================================================
        # PROACTIVE SIGNAL DECISION
        # ====================================================

        if congestion_level == "HIGH":

            signal_action = (
                "PREPARE GREEN PHASE"
            )

            action_description = (
                "High incoming traffic is expected. "
                "Prepare the signal to provide additional "
                "green time before the traffic arrives."
            )

            alert_type = "warning"

        elif congestion_level == "MODERATE":

            signal_action = (
                "MONITOR & ADJUST"
            )

            action_description = (
                "Moderate incoming traffic is expected. "
                "Monitor the junction and dynamically "
                "adjust signal timing if required."
            )

            alert_type = "info"

        else:

            signal_action = (
                "NORMAL SIGNAL OPERATION"
            )

            action_description = (
                "Incoming traffic is expected to remain low. "
                "Continue normal signal operation."
            )

            alert_type = "success"


        # ====================================================
        # MAIN RESULT
        # ====================================================

        st.header("2. Proactive Prediction Result")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Current Traffic",
                f"{current_average:.3f}"
            )

        with col2:

            st.metric(
                "Predicted Traffic",
                f"{predicted_average:.3f}",
                f"{traffic_change:+.3f}"
            )

        with col3:

            st.metric(
                "Predicted Peak",
                f"{predicted_peak:.3f}"
            )

        with col4:

            st.metric(
                "Arrival Window",
                f"{arrival_time} min"
            )


        # ====================================================
        # CONGESTION STATUS
        # ====================================================

        st.subheader("3. Congestion Intelligence")

        if alert_type == "warning":

            st.warning(
                f"⚠️ HIGH CONGESTION EXPECTED AT "
                f"{junction_name}"
            )

        elif alert_type == "info":

            st.info(
                f"Traffic is expected to be MODERATE at "
                f"{junction_name}"
            )

        else:

            st.success(
                f"Traffic is expected to remain LOW at "
                f"{junction_name}"
            )


        # ====================================================
        # PROACTIVE ACTION
        # ====================================================

        st.subheader("4. Recommended Signal Action")

        st.success(
            f"### {signal_action}"
        )

        st.write(
            action_description
        )

        st.write(
            f"**Junction:** {junction_name}"
        )

        st.write(
            f"**Expected traffic arrival:** "
            f"{arrival_time} minutes"
        )


        # ====================================================
        # TRAFFIC FLOW VISUALIZATION
        # ====================================================

        st.header("5. Traffic Flow Prediction")

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        time_points = np.arange(
            1,
            37
        )

        ax.plot(
            time_points,
            input_array,
            marker="o",
            label="Current Traffic"
        )

        ax.plot(
            time_points,
            predicted_values,
            marker="o",
            label="Predicted Incoming Traffic"
        )

        ax.set_title(
            f"{junction_name} - Current vs Predicted Traffic"
        )

        ax.set_xlabel(
            "Traffic Observation"
        )

        ax.set_ylabel(
            "Normalized Traffic Level"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3
        )

        st.pyplot(fig)


        # ====================================================
        # CHANGE VISUALIZATION
        # ====================================================

        st.subheader(
            "6. Traffic Change Analysis"
        )

        difference = (
            predicted_values - input_array
        )

        fig2, ax2 = plt.subplots(
            figsize=(12, 4)
        )

        ax2.bar(
            time_points,
            difference
        )

        ax2.axhline(
            0,
            linewidth=1
        )

        ax2.set_title(
            "Predicted Traffic Change"
        )

        ax2.set_xlabel(
            "Traffic Observation"
        )

        ax2.set_ylabel(
            "Predicted Change"
        )

        ax2.grid(
            True,
            axis="y",
            alpha=0.3
        )

        st.pyplot(fig2)


        # ====================================================
        # JUNCTION INTELLIGENCE TABLE
        # ====================================================

        st.header(
            "7. Junction Intelligence"
        )

        result_df = pd.DataFrame(
            {
                "Observation": time_points,

                "Current Traffic": np.round(
                    input_array,
                    4
                ),

                "Predicted Traffic": np.round(
                    predicted_values,
                    4
                ),

                "Traffic Change": np.round(
                    difference,
                    4
                )
            }
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )


        # ====================================================
        # PEAK ARRIVAL ANALYSIS
        # ====================================================

        st.header(
            "8. Peak Traffic Analysis"
        )

        peak_col1, peak_col2, peak_col3 = st.columns(3)

        with peak_col1:

            st.metric(
                "Peak Traffic Level",
                f"{predicted_peak:.3f}"
            )

        with peak_col2:

            st.metric(
                "Peak Observation",
                str(peak_position)
            )

        with peak_col3:

            st.metric(
                "Expected Arrival",
                f"{arrival_time} min"
            )


        st.info(
            f"""
            The model predicts the highest incoming traffic
            around observation {peak_position}.

            The traffic controller should consider preparing
            the junction approximately {arrival_time} minutes
            before the expected traffic reaches the junction.
            """
        )


        # ====================================================
        # SYSTEM DECISION
        # ====================================================

        st.header(
            "9. Automated Traffic Decision"
        )

        decision_data = pd.DataFrame(
            {
                "Parameter": [
                    "Junction",
                    "Current Traffic",
                    "Predicted Traffic",
                    "Traffic Change",
                    "Predicted Peak",
                    "Congestion Level",
                    "Arrival Time",
                    "Recommended Action"
                ],

                "Value": [
                    junction_name,
                    f"{current_average:.3f}",
                    f"{predicted_average:.3f}",
                    f"{percentage_change:+.2f}%",
                    f"{predicted_peak:.3f}",
                    congestion_level,
                    f"{arrival_time} minutes",
                    signal_action
                ]
            }
        )

        st.table(
            decision_data
        )


        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        st.header(
            "10. Export Prediction"
        )

        csv_data = result_df.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download Prediction CSV",
            data=csv_data,
            file_name="proactive_junction_prediction.csv",
            mime="text/csv",
            use_container_width=True
        )


        # ====================================================
        # FINAL SYSTEM MESSAGE
        # ====================================================

        st.markdown("---")

        st.success(
            f"""
            **Proactive Traffic Intelligence Complete**

            {junction_name} is expected to experience
            **{congestion_level}** incoming traffic.

            Recommended action:

            **{signal_action}**

            This allows the traffic controller to respond
            before the predicted traffic reaches the junction.
            """
        )


    except ValueError:

        st.error(
            "Invalid input detected. Please enter only numerical values separated by commas."
        )

    except Exception as e:

        st.error(
            "An error occurred during prediction."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Member 3 — Prediction Intelligence | Traffic Prediction System"
)
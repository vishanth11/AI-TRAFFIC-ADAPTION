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
    page_title="Traffic Intelligence Dashboard",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚦 Traffic Prediction & Intelligence Dashboard")

st.markdown(
    """
    ### Member 3 — Prediction Intelligence

    **Predict → Analyze → Detect → Prepare**

    This dashboard integrates traffic prediction, congestion
    intelligence, proactive signal recommendations and model
    evaluation into one interface.
    """
)


# ============================================================
# PROJECT PATH
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

    return model


try:

    model = load_model()

    model_status = True

except Exception as e:

    model_status = False

    st.error(
        "Unable to load the trained GRU model."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")

junction = st.sidebar.selectbox(
    "Junction",
    [
        "Junction 1",
        "Junction 2",
        "Junction 3"
    ]
)

arrival_time = st.sidebar.slider(
    "Expected Arrival Time (minutes)",
    1,
    30,
    10
)

st.sidebar.markdown("---")

st.sidebar.success(
    "GRU Model: Loaded"
)

st.sidebar.caption(
    "Prediction Intelligence Module"
)


# ============================================================
# INPUT
# ============================================================

st.header("1. Traffic Input")

default_values = [
    0.10, 0.12, 0.15, 0.18, 0.20, 0.22,
    0.25, 0.28, 0.30, 0.32, 0.35, 0.38,
    0.40, 0.42, 0.45, 0.48, 0.50, 0.52,
    0.55, 0.58, 0.60, 0.62, 0.65, 0.68,
    0.70, 0.72, 0.75, 0.78, 0.80, 0.82,
    0.85, 0.88, 0.90, 0.92, 0.95, 0.98
]

input_text = st.text_area(
    "Enter 36 traffic values",
    value=", ".join(
        str(x)
        for x in default_values
    ),
    height=100
)


# ============================================================
# RUN BUTTON
# ============================================================

run_prediction = st.button(
    "🔮 RUN TRAFFIC INTELLIGENCE",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if run_prediction:

    try:

        # ----------------------------------------------------
        # Parse input
        # ----------------------------------------------------

        input_array = np.array(
            [
                float(x.strip())
                for x in input_text.split(",")
                if x.strip()
            ],
            dtype=np.float32
        )

        if len(input_array) != 36:

            st.error(
                f"Exactly 36 values are required. "
                f"You entered {len(input_array)}."
            )

            st.stop()


        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        input_tensor = torch.tensor(
            input_array,
            dtype=torch.float32
        ).reshape(
            1,
            1,
            36
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                input_tensor
            )

        prediction = (
            output
            .cpu()
            .numpy()
            .flatten()
        )

        prediction = np.clip(
            prediction,
            0,
            1
        )


        # ====================================================
        # STATISTICS
        # ====================================================

        current_avg = float(
            np.mean(input_array)
        )

        predicted_avg = float(
            np.mean(prediction)
        )

        current_peak = float(
            np.max(input_array)
        )

        predicted_peak = float(
            np.max(prediction)
        )

        change = (
            predicted_avg
            - current_avg
        )

        if current_avg != 0:

            percentage_change = (
                change
                / current_avg
                * 100
            )

        else:

            percentage_change = 0


        # ====================================================
        # CONGESTION
        # ====================================================

        if predicted_avg < 0.30:

            congestion = "LOW"

            action = "NORMAL OPERATION"

            action_message = (
                "Traffic is expected to remain low. "
                "Continue normal signal timing."
            )

        elif predicted_avg < 0.70:

            congestion = "MODERATE"

            action = "MONITOR & ADJUST"

            action_message = (
                "Moderate traffic is expected. "
                "Monitor traffic and dynamically adjust "
                "signal timing if required."
            )

        else:

            congestion = "HIGH"

            action = "PREPARE GREEN PHASE"

            action_message = (
                "High traffic is expected. "
                "Prepare the junction before traffic arrives."
            )


        # ====================================================
        # PEAK
        # ====================================================

        peak_index = int(
            np.argmax(prediction) + 1
        )


        # ====================================================
        # DASHBOARD KPIs
        # ====================================================

        st.header(
            "2. Live Traffic Intelligence"
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:

            st.metric(
                "Current Traffic",
                f"{current_avg:.3f}"
            )

        with col2:

            st.metric(
                "Predicted Traffic",
                f"{predicted_avg:.3f}",
                f"{change:+.3f}"
            )

        with col3:

            st.metric(
                "Peak Traffic",
                f"{predicted_peak:.3f}"
            )

        with col4:

            st.metric(
                "Traffic Change",
                f"{percentage_change:+.1f}%"
            )

        with col5:

            st.metric(
                "Arrival",
                f"{arrival_time} min"
            )


        # ====================================================
        # STATUS
        # ====================================================

        st.header(
            "3. Congestion Status"
        )

        if congestion == "HIGH":

            st.error(
                f"🚨 HIGH CONGESTION — {junction}"
            )

        elif congestion == "MODERATE":

            st.warning(
                f"⚠️ MODERATE CONGESTION — {junction}"
            )

        else:

            st.success(
                f"✓ LOW CONGESTION — {junction}"
            )


        # ====================================================
        # PROACTIVE DECISION
        # ====================================================

        st.header(
            "4. Proactive Signal Decision"
        )

        decision_col1, decision_col2 = st.columns(
            [1, 2]
        )

        with decision_col1:

            st.metric(
                "Recommended Action",
                action
            )

        with decision_col2:

            st.info(
                action_message
            )

        st.write(
            f"""
            **Junction:** {junction}

            **Expected arrival:** {arrival_time} minutes

            **Predicted peak:** observation {peak_index}
            """
        )


        # ====================================================
        # MAIN VISUALIZATION
        # ====================================================

        st.header(
            "5. Current vs Future Traffic"
        )

        observations = np.arange(
            1,
            37
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.plot(
            observations,
            input_array,
            marker="o",
            label="Current Traffic"
        )

        ax.plot(
            observations,
            prediction,
            marker="o",
            label="Predicted Traffic"
        )

        ax.set_title(
            f"{junction} — Traffic Forecast"
        )

        ax.set_xlabel(
            "Observation"
        )

        ax.set_ylabel(
            "Normalized Traffic Level"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3
        )

        st.pyplot(
            fig
        )


        # ====================================================
        # DIFFERENCE CHART
        # ====================================================

        st.header(
            "6. Predicted Traffic Change"
        )

        difference = (
            prediction
            - input_array
        )

        fig2, ax2 = plt.subplots(
            figsize=(12, 4)
        )

        ax2.bar(
            observations,
            difference
        )

        ax2.axhline(
            0,
            linewidth=1
        )

        ax2.set_title(
            "Traffic Increase / Decrease"
        )

        ax2.set_xlabel(
            "Observation"
        )

        ax2.set_ylabel(
            "Change"
        )

        ax2.grid(
            axis="y",
            alpha=0.3
        )

        st.pyplot(
            fig2
        )


        # ====================================================
        # TRAFFIC TABLE
        # ====================================================

        st.header(
            "7. Prediction Details"
        )

        result_df = pd.DataFrame(
            {
                "Observation": observations,

                "Current Traffic": np.round(
                    input_array,
                    4
                ),

                "Predicted Traffic": np.round(
                    prediction,
                    4
                ),

                "Difference": np.round(
                    difference,
                    4
                )
            }
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        st.header(
            "8. Model Performance"
        )

        st.info(
            """
            The GRU model achieved an MAE of **0.0481**
            during the previous evaluation stage.

            Lower MAE indicates that predicted traffic values
            are closer to the observed values.
            """
        )

        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:

            st.metric(
                "GRU MAE",
                "0.0481"
            )

        with perf_col2:

            st.metric(
                "Model",
                "GRU"
            )


        # ====================================================
        # SYSTEM FLOW
        # ====================================================

        st.header(
            "9. Prediction Intelligence Pipeline"
        )

        flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(
            4
        )

        with flow_col1:

            st.info(
                "**1. INPUT**\n\n"
                "Current traffic pattern"
            )

        with flow_col2:

            st.info(
                "**2. PREDICT**\n\n"
                "GRU forecasts future traffic"
            )

        with flow_col3:

            st.info(
                "**3. ANALYZE**\n\n"
                "Congestion is estimated"
            )

        with flow_col4:

            st.success(
                "**4. ACT**\n\n"
                "Signal action is recommended"
            )


        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.header(
            "10. Export Dashboard Results"
        )

        dashboard_summary = pd.DataFrame(
            {
                "Parameter": [
                    "Junction",
                    "Current Traffic",
                    "Predicted Traffic",
                    "Peak Traffic",
                    "Traffic Change (%)",
                    "Congestion",
                    "Expected Arrival",
                    "Recommended Action",
                    "Model",
                    "MAE"
                ],

                "Value": [
                    junction,
                    round(current_avg, 4),
                    round(predicted_avg, 4),
                    round(predicted_peak, 4),
                    round(percentage_change, 2),
                    congestion,
                    f"{arrival_time} minutes",
                    action,
                    "GRU",
                    0.0481
                ]
            }
        )

        csv_data = dashboard_summary.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download Dashboard Summary",
            data=csv_data,
            file_name="traffic_intelligence_summary.csv",
            mime="text/csv",
            use_container_width=True
        )


        # ====================================================
        # FINAL SYSTEM MESSAGE
        # ====================================================

        st.markdown("---")

        if congestion == "HIGH":

            st.error(
                f"""
                ### 🚨 PROACTIVE WARNING

                {junction} is expected to experience
                **HIGH incoming traffic**.

                **Action:** {action}

                **Expected arrival:** {arrival_time} minutes.

                The controller should prepare before the
                traffic reaches the junction.
                """
            )

        elif congestion == "MODERATE":

            st.warning(
                f"""
                ### ⚠️ EARLY WARNING

                {junction} is expected to experience
                **MODERATE incoming traffic**.

                **Action:** {action}

                Continuous monitoring is recommended.
                """
            )

        else:

            st.success(
                f"""
                ### ✓ NORMAL TRAFFIC

                {junction} is expected to experience
                **LOW incoming traffic**.

                **Action:** {action}
                """
            )


    # ========================================================
    # INPUT ERROR
    # ========================================================

    except ValueError:

        st.error(
            "Please enter only numerical values separated by commas."
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
    "Member 3 — Prediction Intelligence | "
    "Traffic Prediction & Proactive Control System"
)
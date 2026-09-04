import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# STEP 17 - TRAFFIC PREDICTION VISUALIZATION
# ============================================================

st.set_page_config(
    page_title="Traffic Prediction Analysis",
    layout="wide"
)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "traffic_gru.pth"
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
        map_location="cpu"
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:

            model.load_state_dict(checkpoint)

    else:

        model = checkpoint

    model.eval()

    return model


# ============================================================
# TITLE
# ============================================================

st.title("Traffic Prediction Analysis")

st.write(
    "Visualization and statistical analysis of traffic "
    "predictions generated using the trained GRU model."
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

    st.success("GRU model loaded successfully.")

except Exception as error:

    st.error("Unable to load GRU model.")

    st.code(str(error))

    st.stop()


# ============================================================
# INPUT
# ============================================================

st.subheader("Enter Traffic Values")

values = []

columns = st.columns(3)

for i in range(36):

    with columns[i % 3]:

        value = st.number_input(
            f"Traffic {i + 1}",
            value=0.0,
            key=f"traffic_{i}"
        )

        values.append(value)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Generate Prediction & Analysis",
    type="primary",
    use_container_width=True
):

    input_array = np.array(
        values,
        dtype=np.float32
    )

    input_array = input_array.reshape(
        1,
        1,
        36
    )

    input_tensor = torch.tensor(
        input_array,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            input_tensor
        )

    prediction_values = (
        prediction
        .cpu()
        .numpy()
        .flatten()
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.success(
        "Traffic prediction generated successfully."
    )


    # ========================================================
    # DATAFRAME
    # ========================================================

    result_df = pd.DataFrame({

        "Traffic Point": range(1, 37),

        "Input Traffic": values,

        "Predicted Traffic": prediction_values

    })

    result_df["Difference"] = (
        result_df["Predicted Traffic"]
        -
        result_df["Input Traffic"]
    )


    # ========================================================
    # METRICS
    # ========================================================

    st.subheader("Prediction Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Input Average",
            f"{np.mean(values):.4f}"
        )

    with col2:

        st.metric(
            "Prediction Average",
            f"{np.mean(prediction_values):.4f}"
        )

    with col3:

        st.metric(
            "Maximum Prediction",
            f"{np.max(prediction_values):.4f}"
        )

    with col4:

        st.metric(
            "Minimum Prediction",
            f"{np.min(prediction_values):.4f}"
        )


    # ========================================================
    # COMPARISON GRAPH
    # ========================================================

    st.divider()

    st.subheader(
        "Input vs Predicted Traffic"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        range(1, 37),
        values,
        marker="o",
        label="Input Traffic"
    )

    ax.plot(
        range(1, 37),
        prediction_values,
        marker="x",
        label="Predicted Traffic"
    )

    ax.set_xlabel(
        "Traffic Point"
    )

    ax.set_ylabel(
        "Traffic Value"
    )

    ax.set_title(
        "Traffic Prediction Comparison"
    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)


    # ========================================================
    # DIFFERENCE GRAPH
    # ========================================================

    st.subheader(
        "Prediction Difference"
    )

    fig2, ax2 = plt.subplots(
        figsize=(12, 4)
    )

    ax2.bar(
        range(1, 37),
        result_df["Difference"]
    )

    ax2.axhline(
        y=0,
        linewidth=1
    )

    ax2.set_xlabel(
        "Traffic Point"
    )

    ax2.set_ylabel(
        "Prediction Difference"
    )

    ax2.set_title(
        "Difference Between Input and Predicted Traffic"
    )

    st.pyplot(fig2)


    # ========================================================
    # RESULT TABLE
    # ========================================================

    st.divider()

    st.subheader(
        "Prediction Results"
    )

    st.dataframe(
        result_df,
        use_container_width=True
    )


    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    csv_data = result_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Prediction Results",
        data=csv_data,
        file_name="traffic_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.divider()

    st.subheader(
        "Prediction Interpretation"
    )

    average_prediction = np.mean(
        prediction_values
    )

    maximum_prediction = np.max(
        prediction_values
    )

    if average_prediction < 0.3:

        st.info(
            "The predicted traffic level is relatively low."
        )

    elif average_prediction < 0.7:

        st.warning(
            "The predicted traffic level is moderate."
        )

    else:

        st.error(
            "The predicted traffic level is relatively high."
        )


    st.write(
        f"The average predicted traffic value is "
        f"**{average_prediction:.4f}**, while the maximum "
        f"predicted value is **{maximum_prediction:.4f}**."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Step 17 — Traffic Prediction Visualization and Analysis"
)
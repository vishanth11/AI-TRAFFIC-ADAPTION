import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============================================================
# STEP 26 — PREDICTION PERFORMANCE & ERROR ANALYSIS
# ============================================================

st.set_page_config(
    page_title="Prediction Performance Analysis",
    page_icon="🚦",
    layout="wide"
)

st.title("Prediction Performance & Error Analysis")

st.markdown(
    """
    Evaluate the traffic prediction model by comparing
    **actual traffic patterns with GRU predictions**.
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
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "traffic_gru.pth was not found in the results folder."
    )

    st.stop()


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


st.success(
    "GRU model loaded successfully."
)


# ============================================================
# INPUT DATA
# ============================================================

st.header("1. Traffic Evaluation Input")

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


input_text = st.text_area(
    "Enter 36 actual traffic values",
    value=", ".join(
        str(x)
        for x in default_values
    ),
    height=120
)


try:

    actual = np.array(
        [
            float(x.strip())
            for x in input_text.split(",")
            if x.strip()
        ],
        dtype=np.float32
    )

except ValueError:

    st.error(
        "Please enter numeric values separated by commas."
    )

    st.stop()


if len(actual) != 36:

    st.warning(
        f"Exactly 36 values are required. "
        f"Currently entered: {len(actual)}"
    )

    st.stop()


# ============================================================
# GENERATE PREDICTION
# ============================================================

input_tensor = torch.tensor(
    actual,
    dtype=torch.float32
).reshape(
    1,
    1,
    36
)


with torch.no_grad():

    prediction = model(
        input_tensor
    ).numpy().flatten()


# ============================================================
# PERFORMANCE METRICS
# ============================================================

errors = actual - prediction

absolute_errors = np.abs(
    errors
)

squared_errors = (
    errors ** 2
)


mae = np.mean(
    absolute_errors
)

rmse = np.sqrt(
    np.mean(
        squared_errors
    )
)


non_zero_actual = np.where(
    np.abs(actual) > 1e-6,
    actual,
    1e-6
)

mape = np.mean(
    np.abs(
        errors /
        non_zero_actual
    )
) * 100


# ============================================================
# METRICS
# ============================================================

st.header("2. Prediction Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "MAE",
        f"{mae:.4f}"
    )

with col2:

    st.metric(
        "RMSE",
        f"{rmse:.4f}"
    )

with col3:

    st.metric(
        "MAPE",
        f"{mape:.2f}%"
    )

with col4:

    st.metric(
        "Max Error",
        f"{np.max(absolute_errors):.4f}"
    )


# ============================================================
# MODEL QUALITY
# ============================================================

st.subheader("Prediction Quality")

if mae < 0.05:

    st.success(
        "Excellent prediction accuracy based on MAE."
    )

elif mae < 0.10:

    st.success(
        "Good prediction accuracy based on MAE."
    )

elif mae < 0.20:

    st.warning(
        "Moderate prediction error. Further tuning may improve performance."
    )

else:

    st.error(
        "High prediction error. Model retraining or feature engineering is recommended."
    )


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.header("3. Actual vs Predicted Traffic")

fig = plt.figure(
    figsize=(12, 5)
)

plt.plot(
    actual,
    label="Actual Traffic"
)

plt.plot(
    prediction,
    label="GRU Prediction"
)

plt.title(
    "Actual vs GRU Predicted Traffic"
)

plt.xlabel(
    "Prediction Step"
)

plt.ylabel(
    "Traffic Value"
)

plt.legend()

plt.grid(True)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# ERROR GRAPH
# ============================================================

st.header("4. Prediction Error")

fig = plt.figure(
    figsize=(12, 5)
)

plt.plot(
    errors,
    label="Prediction Error"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title(
    "Prediction Residuals"
)

plt.xlabel(
    "Prediction Step"
)

plt.ylabel(
    "Actual - Predicted"
)

plt.legend()

plt.grid(True)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

st.header("5. Error Distribution")

fig = plt.figure(
    figsize=(9, 5)
)

plt.hist(
    errors,
    bins=10
)

plt.title(
    "Distribution of Prediction Errors"
)

plt.xlabel(
    "Prediction Error"
)

plt.ylabel(
    "Frequency"
)

plt.grid(True)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# PREDICTION TABLE
# ============================================================

st.header("6. Detailed Prediction Results")

evaluation_df = pd.DataFrame({

    "Step": np.arange(
        1,
        37
    ),

    "Actual": actual,

    "Predicted": prediction,

    "Error": errors,

    "Absolute_Error": absolute_errors

})

st.dataframe(
    evaluation_df.round(4),
    width="stretch"
)


# ============================================================
# BEST / WORST PREDICTIONS
# ============================================================

best_index = np.argmin(
    absolute_errors
)

worst_index = np.argmax(
    absolute_errors
)

col1, col2 = st.columns(2)

with col1:

    st.info(
        f"""
        **Best Prediction**

        Prediction Step: {best_index + 1}

        Actual: {actual[best_index]:.4f}

        Predicted: {prediction[best_index]:.4f}

        Error: {absolute_errors[best_index]:.4f}
        """
    )


with col2:

    st.warning(
        f"""
        **Largest Prediction Error**

        Prediction Step: {worst_index + 1}

        Actual: {actual[worst_index]:.4f}

        Predicted: {prediction[worst_index]:.4f}

        Error: {absolute_errors[worst_index]:.4f}
        """
    )


# ============================================================
# KNOWN TRAINING / REFERENCE METRIC
# ============================================================

st.header("7. Reference Model Performance")

st.write(
    """
    The previously established GRU benchmark for this project
    recorded:
    """
)

st.metric(
    "Reference GRU MAE",
    "0.0481"
)

st.caption(
    "This reference value is retained from the earlier model evaluation. "
    "The MAE calculated above is based on the current evaluation input."
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    RESULTS_DIR,
    "prediction_performance_analysis.csv"
)

evaluation_df.to_csv(
    output_file,
    index=False
)


csv_data = evaluation_df.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    label="Download Prediction Evaluation CSV",
    data=csv_data,
    file_name="prediction_performance_analysis.csv",
    mime="text/csv"
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

st.header("8. Interpretation")

st.markdown(
    f"""
    ### Prediction Intelligence Result

    The GRU model generated **36 future traffic predictions**.

    **MAE:** {mae:.4f}

    **RMSE:** {rmse:.4f}

    **MAPE:** {mape:.2f}%

    The error analysis shows how closely the predicted traffic
    pattern follows the supplied traffic pattern.

    This evaluation supports the Member 3 objective:

    **Historical / Current Traffic**
    ↓

    **GRU Prediction**
    ↓

    **Future Traffic Estimate**
    ↓

    **Prediction Error Analysis**
    ↓

    **Confidence for Proactive Traffic Control**
    """
)


st.success(
    "Step 26 completed: Prediction Performance & Error Analysis"
)

st.caption(
    "Member 3 — Prediction Intelligence"
)
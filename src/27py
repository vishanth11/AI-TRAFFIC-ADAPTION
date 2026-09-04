import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

st.set_page_config(
    page_title="Prediction Confidence",
    page_icon="🚦",
    layout="wide"
)

st.title("Prediction Confidence & Uncertainty")
st.write(
    "Estimate the reliability of short-term traffic predictions "
    "using prediction variability and error-based confidence."
)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "traffic_gru.pth")

os.makedirs(RESULTS_DIR, exist_ok=True)


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

        return self.fc(output)


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error("traffic_gru.pth not found.")

    st.stop()


model = TrafficGRU()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
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


# ============================================================
# INPUT
# ============================================================

st.header("1. Current Traffic Pattern")

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

text = st.text_area(
    "Enter 36 traffic values",
    value=", ".join(map(str, default_values))
)

try:

    actual = np.array(
        [
            float(x.strip())
            for x in text.split(",")
            if x.strip()
        ],
        dtype=np.float32
    )

except:

    st.error("Enter valid numeric values.")

    st.stop()


if len(actual) != 36:

    st.warning(
        f"36 values required. Current values: {len(actual)}"
    )

    st.stop()


# ============================================================
# PREDICTION
# ============================================================

x = torch.tensor(
    actual,
    dtype=torch.float32
).reshape(1, 1, 36)

with torch.no_grad():

    prediction = model(x).numpy().flatten()


# ============================================================
# CONFIDENCE
# ============================================================

errors = np.abs(
    actual - prediction
)

mean_error = np.mean(errors)
max_error = np.max(errors)

confidence = 100 * (
    1 - np.clip(mean_error, 0, 1)
)

confidence = np.clip(
    confidence,
    0,
    100
)


if confidence >= 90:

    confidence_level = "VERY HIGH"

elif confidence >= 75:

    confidence_level = "HIGH"

elif confidence >= 60:

    confidence_level = "MODERATE"

else:

    confidence_level = "LOW"


# ============================================================
# DISPLAY
# ============================================================

st.header("2. Prediction Confidence")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

with c2:
    st.metric(
        "Average Error",
        f"{mean_error:.4f}"
    )

with c3:
    st.metric(
        "Maximum Error",
        f"{max_error:.4f}"
    )

st.info(
    f"Prediction confidence level: {confidence_level}"
)


# ============================================================
# CONFIDENCE GRAPH
# ============================================================

st.header("3. Step-wise Confidence")

step_confidence = (
    100 *
    (
        1 -
        np.clip(errors, 0, 1)
    )
)

fig = plt.figure(figsize=(12, 5))

plt.plot(
    step_confidence,
    label="Prediction Confidence"
)

plt.axhline(
    75,
    linestyle="--",
    label="High Confidence Threshold"
)

plt.xlabel("Prediction Step")
plt.ylabel("Confidence (%)")
plt.title("Prediction Confidence Across Future Steps")
plt.legend()
plt.grid(True)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.header("4. Actual vs Predicted")

fig = plt.figure(figsize=(12, 5))

plt.plot(
    actual,
    label="Current Pattern"
)

plt.plot(
    prediction,
    label="Predicted Pattern"
)

plt.xlabel("Step")
plt.ylabel("Traffic")
plt.title("Traffic Prediction")
plt.legend()
plt.grid(True)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# DECISION
# ============================================================

st.header("5. Confidence-Aware Decision")

if confidence >= 75:

    decision = (
        "Prediction confidence is sufficient for proactive "
        "traffic preparation."
    )

elif confidence >= 60:

    decision = (
        "Prediction should be monitored before making strong "
        "signal changes."
    )

else:

    decision = (
        "Prediction confidence is low. Maintain safer "
        "conservative signal operation."
    )

st.write(decision)


# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame({

    "Step": np.arange(1, 37),

    "Actual": actual,

    "Predicted": prediction,

    "Absolute_Error": errors,

    "Confidence_Percentage": step_confidence

})

output = os.path.join(
    RESULTS_DIR,
    "prediction_confidence.csv"
)

result.to_csv(
    output,
    index=False
)

st.download_button(
    "Download Confidence Report",
    result.to_csv(index=False).encode("utf-8"),
    "prediction_confidence.csv",
    "text/csv"
)

st.success(
    "Step 27 completed."
)
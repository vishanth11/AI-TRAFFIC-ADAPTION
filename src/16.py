import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os


# ============================================================
# STEP 16 - TRAFFIC PREDICTION MODEL DEPLOYMENT
# ============================================================

st.set_page_config(
    page_title="Traffic Prediction System",
    page_icon="🚦",
    layout="centered"
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

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

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
# APPLICATION
# ============================================================

st.title("Traffic Prediction System")

st.write(
    "GRU-based traffic prediction using the trained "
    "PyTorch model."
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

    st.success(
        "Traffic GRU model loaded successfully."
    )

except Exception as error:

    st.error(
        "Unable to load the trained GRU model."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# INPUT
# ============================================================

st.subheader("Traffic Input")

st.write(
    "Enter 36 traffic features for the prediction."
)


# Create 36 input fields

values = []

columns = st.columns(3)

for i in range(36):

    with columns[i % 3]:

        value = st.number_input(
            f"Feature {i + 1}",
            value=0.0,
            key=f"feature_{i}"
        )

        values.append(value)


# ============================================================
# CREATE MODEL INPUT
# ============================================================

input_array = np.array(
    values,
    dtype=np.float32
)

# Shape:
# (batch, sequence, features)
#
# 1 sample
# 1 timestep
# 36 features

input_array = input_array.reshape(
    1,
    1,
    36
)

input_tensor = torch.tensor(
    input_array,
    dtype=torch.float32
)


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "Predict Traffic",
    type="primary",
    use_container_width=True
):

    try:

        with torch.no_grad():

            prediction = model(
                input_tensor
            )

        prediction = prediction.cpu().numpy()

        prediction_values = prediction[0]

        st.success(
            "Traffic prediction generated successfully."
        )


        # ----------------------------------------------------
        # DISPLAY PREDICTIONS
        # ----------------------------------------------------

        st.subheader(
            "Predicted Traffic Values"
        )

        for i, value in enumerate(
            prediction_values
        ):

            st.write(
                f"Prediction {i + 1}: "
                f"**{value:.4f}**"
            )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Minimum",
                f"{prediction_values.min():.4f}"
            )

        with col2:

            st.metric(
                "Maximum",
                f"{prediction_values.max():.4f}"
            )

        with col3:

            st.metric(
                "Average",
                f"{prediction_values.mean():.4f}"
            )


    except Exception as error:

        st.error(
            "Prediction failed."
        )

        st.code(str(error))


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("Model")
    st.write("GRU")

with col2:
    st.write("Framework")
    st.write("PyTorch")

with col3:
    st.write("Input Size")
    st.write("36")

with col4:
    st.write("Output Size")
    st.write("36")


st.caption(
    "Step 16 — Traffic Prediction Model Deployment"
)
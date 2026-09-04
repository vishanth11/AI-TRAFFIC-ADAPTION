import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Proactive Signal Decision",
    page_icon="🚦",
    layout="wide"
)

st.title("Proactive Signal Decision Engine")

st.write(
    """
    Convert predicted traffic conditions into proactive
    signal-control recommendations.
    """
)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# INPUTS
# ============================================================

st.header("1. Predicted Traffic Conditions")

c1, c2 = st.columns(2)

with c1:

    predicted_traffic = st.slider(
        "Predicted Traffic Level",
        0.0,
        1.0,
        0.75,
        0.01
    )

with c2:

    confidence = st.slider(
        "Prediction Confidence (%)",
        0,
        100,
        85
    )


queue = st.slider(
    "Current Queue Length",
    0,
    100,
    30
)

speed = st.slider(
    "Current Average Speed",
    0,
    100,
    35
)

emergency = st.checkbox(
    "Emergency Vehicle Detected"
)

pedestrian = st.checkbox(
    "High Pedestrian Demand"
)


# ============================================================
# CONGESTION
# ============================================================

if predicted_traffic < 0.30:

    congestion = "LOW"

elif predicted_traffic < 0.70:

    congestion = "MODERATE"

else:

    congestion = "HIGH"


# ============================================================
# SIGNAL DECISION
# ============================================================

if emergency:

    signal = "EMERGENCY GREEN CORRIDOR"

    green_time = 60

    reason = (
        "Emergency vehicle priority activated."
    )

elif confidence < 60:

    signal = "CONSERVATIVE ADAPTIVE CONTROL"

    green_time = 30

    reason = (
        "Prediction confidence is low."
    )

elif predicted_traffic >= 0.70:

    signal = "EXTENDED GREEN"

    green_time = 50

    reason = (
        "High predicted traffic requires additional green time."
    )

elif queue >= 50:

    signal = "QUEUE CLEARANCE"

    green_time = 45

    reason = (
        "Large queue detected."
    )

elif pedestrian:

    signal = "PEDESTRIAN PRIORITY"

    green_time = 25

    reason = (
        "High pedestrian demand detected."
    )

else:

    signal = "NORMAL ADAPTIVE CONTROL"

    green_time = 30

    reason = (
        "Traffic conditions remain manageable."
    )


# ============================================================
# OUTPUT
# ============================================================

st.header("2. Controller Decision")

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Congestion",
        congestion
    )

with c2:

    st.metric(
        "Recommended Green",
        f"{green_time} sec"
    )

with c3:

    st.metric(
        "Confidence",
        f"{confidence}%"
    )


st.success(
    f"Recommended Action: {signal}"
)

st.info(
    reason
)


# ============================================================
# DECISION FACTORS
# ============================================================

st.header("3. Decision Factors")

decision_df = pd.DataFrame({

    "Factor": [
        "Predicted Traffic",
        "Confidence",
        "Queue Length",
        "Average Speed",
        "Emergency Vehicle",
        "Pedestrian Demand"
    ],

    "Value": [
        predicted_traffic,
        f"{confidence}%",
        queue,
        speed,
        "YES" if emergency else "NO",
        "HIGH" if pedestrian else "NORMAL"
    ]

})

st.dataframe(
    decision_df,
    width="stretch"
)


# ============================================================
# VISUALIZATION
# ============================================================

st.header("4. Traffic Control Visualization")

labels = [
    "Traffic",
    "Queue",
    "Speed"
]

values = [
    predicted_traffic * 100,
    min(queue, 100),
    min(speed, 100)
]

fig = plt.figure(figsize=(9, 5))

plt.bar(
    labels,
    values
)

plt.title(
    "Traffic Control Decision Factors"
)

plt.ylabel(
    "Normalized Value"
)

plt.grid(
    axis="y"
)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# CONTROL LOG
# ============================================================

control_log = pd.DataFrame({

    "Predicted_Traffic": [
        predicted_traffic
    ],

    "Congestion": [
        congestion
    ],

    "Confidence": [
        confidence
    ],

    "Queue_Length": [
        queue
    ],

    "Speed": [
        speed
    ],

    "Emergency": [
        emergency
    ],

    "Pedestrian_Demand": [
        pedestrian
    ],

    "Decision": [
        signal
    ],

    "Green_Time_Seconds": [
        green_time
    ],

    "Reason": [
        reason
    ]

})


output = os.path.join(
    RESULTS_DIR,
    "proactive_signal_decision.csv"
)

control_log.to_csv(
    output,
    index=False
)


st.download_button(
    "Download Signal Decision",
    control_log.to_csv(index=False).encode("utf-8"),
    "proactive_signal_decision.csv",
    "text/csv"
)


st.success(
    "Step 29 completed."
)
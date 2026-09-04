import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ============================================================
# STEP 24 — BUS ARRIVAL & DELAY PREDICTION
# ============================================================

st.set_page_config(
    page_title="Bus Arrival & Delay Prediction",
    page_icon="Bus",
    layout="wide"
)

st.title("Bus Arrival & Delay Prediction")

st.write(
    "Short-term prediction of bus arrival time and expected delay "
    "based on traffic and route conditions."
)

# ============================================================
# 1. BUS INFORMATION
# ============================================================

st.header("1. Bus Information")

col1, col2, col3 = st.columns(3)

with col1:
    bus_number = st.text_input(
        "Bus Number",
        value="B101"
    )

with col2:
    route_name = st.text_input(
        "Route",
        value="Junction 1 - Junction 3"
    )

with col3:
    current_location = st.selectbox(
        "Current Location",
        [
            "Previous Junction",
            "Junction 1",
            "Junction 2",
            "Junction 3"
        ]
    )

# ============================================================
# 2. TRAFFIC CONDITIONS
# ============================================================

st.header("2. Traffic Conditions")

col4, col5, col6 = st.columns(3)

with col4:
    distance = st.number_input(
        "Distance to Stop (km)",
        min_value=0.1,
        max_value=50.0,
        value=3.0,
        step=0.1
    )

with col5:
    average_speed = st.number_input(
        "Average Bus Speed (km/h)",
        min_value=1.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

with col6:
    traffic_level = st.slider(
        "Traffic Level",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05
    )

# ============================================================
# 3. ADDITIONAL CONDITIONS
# ============================================================

st.header("3. Additional Conditions")

col7, col8, col9 = st.columns(3)

with col7:
    scheduled_minutes = st.number_input(
        "Scheduled Arrival (minutes from now)",
        min_value=1,
        max_value=120,
        value=10
    )

with col8:
    stops_remaining = st.number_input(
        "Stops Remaining",
        min_value=0,
        max_value=30,
        value=3
    )

with col9:
    passenger_delay = st.number_input(
        "Passenger Boarding Delay (minutes)",
        min_value=0.0,
        max_value=30.0,
        value=1.0,
        step=0.5
    )

# ============================================================
# 4. PREDICTION
# ============================================================

if st.button(
    "Predict Bus Arrival",
    width="stretch"
):

    # --------------------------------------------------------
    # BASE TRAVEL TIME
    # --------------------------------------------------------

    base_travel_time = (
        distance / average_speed
    ) * 60

    # --------------------------------------------------------
    # TRAFFIC DELAY
    # --------------------------------------------------------

    traffic_delay = (
        traffic_level
        * base_travel_time
        * 0.60
    )

    # --------------------------------------------------------
    # STOP DELAY
    # --------------------------------------------------------

    stop_delay = (
        stops_remaining * 0.5
    )

    # --------------------------------------------------------
    # TOTAL PREDICTED TRAVEL TIME
    # --------------------------------------------------------

    predicted_minutes = (
        base_travel_time
        + traffic_delay
        + stop_delay
        + passenger_delay
    )

    # --------------------------------------------------------
    # EXPECTED DELAY
    # --------------------------------------------------------

    expected_delay = (
        predicted_minutes
        - scheduled_minutes
    )

    expected_delay = max(
        expected_delay,
        0
    )

    # --------------------------------------------------------
    # CURRENT TIME
    # --------------------------------------------------------

    current_time = datetime.now()

    scheduled_arrival = (
        current_time
        + timedelta(
            minutes=scheduled_minutes
        )
    )

    predicted_arrival = (
        current_time
        + timedelta(
            minutes=predicted_minutes
        )
    )

    # ========================================================
    # 5. DELAY CLASSIFICATION
    # ========================================================

    if expected_delay <= 2:

        delay_status = "ON TIME"

    elif expected_delay <= 5:

        delay_status = "SLIGHT DELAY"

    elif expected_delay <= 10:

        delay_status = "MODERATE DELAY"

    else:

        delay_status = "HIGH DELAY"

    # ========================================================
    # 6. PREDICTION RESULTS
    # ========================================================

    st.header("4. Prediction Result")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Predicted Travel Time",
            f"{predicted_minutes:.1f} min"
        )

    with c2:

        st.metric(
            "Expected Delay",
            f"{expected_delay:.1f} min"
        )

    with c3:

        st.metric(
            "Scheduled Arrival",
            scheduled_arrival.strftime("%H:%M")
        )

    with c4:

        st.metric(
            "Predicted Arrival",
            predicted_arrival.strftime("%H:%M")
        )

    # ========================================================
    # 7. BUS STATUS
    # ========================================================

    st.subheader("Bus Status")

    if delay_status == "ON TIME":

        st.success(
            f"{bus_number} — {delay_status}"
        )

    elif delay_status == "SLIGHT DELAY":

        st.info(
            f"{bus_number} — {delay_status}"
        )

    elif delay_status == "MODERATE DELAY":

        st.warning(
            f"{bus_number} — {delay_status}"
        )

    else:

        st.error(
            f"{bus_number} — {delay_status}"
        )

    # ========================================================
    # 8. TRAFFIC IMPACT ANALYSIS
    # ========================================================

    st.subheader(
        "Traffic Impact Analysis"
    )

    factors = [
        "Base Travel Time",
        "Traffic Delay",
        "Stop Delay",
        "Passenger Delay"
    ]

    values = [
        float(base_travel_time),
        float(traffic_delay),
        float(stop_delay),
        float(passenger_delay)
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        factors,
        values
    )

    ax.set_xlabel(
        "Delay Factor"
    )

    ax.set_ylabel(
        "Minutes"
    )

    ax.set_title(
        "Bus Arrival Time Components"
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    st.pyplot(fig)

    # ========================================================
    # 9. PREDICTION DETAILS
    # ========================================================

    st.subheader(
        "Prediction Details"
    )

    # IMPORTANT:
    # Every Value is converted to STRING.
    # This prevents PyArrow/Streamlit ArrowTypeError.

    result = pd.DataFrame({

        "Parameter": [
            "Bus Number",
            "Route",
            "Current Location",
            "Distance",
            "Average Speed",
            "Traffic Level",
            "Stops Remaining",
            "Passenger Delay",
            "Predicted Travel Time",
            "Scheduled Arrival",
            "Predicted Arrival",
            "Expected Delay",
            "Status"
        ],

        "Value": [
            str(bus_number),
            str(route_name),
            str(current_location),
            f"{distance:.1f} km",
            f"{average_speed:.1f} km/h",
            f"{traffic_level:.2f}",
            str(stops_remaining),
            f"{passenger_delay:.1f} min",
            f"{predicted_minutes:.1f} min",
            scheduled_arrival.strftime("%H:%M"),
            predicted_arrival.strftime("%H:%M"),
            f"{expected_delay:.1f} min",
            str(delay_status)
        ]
    })

    # Force entire column to string
    result["Parameter"] = (
        result["Parameter"]
        .astype(str)
    )

    result["Value"] = (
        result["Value"]
        .astype(str)
    )

    st.dataframe(
        result,
        width="stretch",
        hide_index=True
    )

    # ========================================================
    # 10. PROACTIVE TRAFFIC RECOMMENDATION
    # ========================================================

    st.subheader(
        "Proactive Traffic Recommendation"
    )

    if expected_delay > 10:

        recommendation = (
            "High bus delay detected. "
            "Consider prioritizing the bus at upcoming signals "
            "and preparing downstream junctions."
        )

    elif expected_delay > 5:

        recommendation = (
            "Moderate bus delay detected. "
            "Monitor upcoming junction traffic and adjust "
            "signal timing if required."
        )

    elif expected_delay > 2:

        recommendation = (
            "Slight bus delay detected. "
            "Continue monitoring traffic conditions."
        )

    else:

        recommendation = (
            "Bus is expected to arrive normally. "
            "No special signal intervention required."
        )

    st.info(
        recommendation
    )

    # ========================================================
    # 11. PREDICTION SUMMARY
    # ========================================================

    st.subheader(
        "Prediction Summary"
    )

    summary_data = pd.DataFrame({

        "Metric": [
            "Bus",
            "Route",
            "Traffic Level",
            "Distance",
            "Predicted Arrival",
            "Expected Delay",
            "Status"
        ],

        "Result": [
            str(bus_number),
            str(route_name),
            f"{traffic_level:.2f}",
            f"{distance:.1f} km",
            predicted_arrival.strftime("%H:%M"),
            f"{expected_delay:.1f} minutes",
            str(delay_status)
        ]

    })

    summary_data["Metric"] = (
        summary_data["Metric"]
        .astype(str)
    )

    summary_data["Result"] = (
        summary_data["Result"]
        .astype(str)
    )

    st.dataframe(
        summary_data,
        width="stretch",
        hide_index=True
    )

    # ========================================================
    # 12. EXPORT RESULT
    # ========================================================

    export_data = pd.DataFrame({

        "Bus_Number": [
            str(bus_number)
        ],

        "Route": [
            str(route_name)
        ],

        "Current_Location": [
            str(current_location)
        ],

        "Distance_km": [
            float(distance)
        ],

        "Average_Speed_kmph": [
            float(average_speed)
        ],

        "Traffic_Level": [
            float(traffic_level)
        ],

        "Stops_Remaining": [
            int(stops_remaining)
        ],

        "Passenger_Delay_min": [
            float(passenger_delay)
        ],

        "Predicted_Travel_Time_min": [
            float(predicted_minutes)
        ],

        "Scheduled_Arrival": [
            scheduled_arrival.strftime("%H:%M")
        ],

        "Predicted_Arrival": [
            predicted_arrival.strftime("%H:%M")
        ],

        "Expected_Delay_min": [
            float(expected_delay)
        ],

        "Status": [
            str(delay_status)
        ]

    })

    csv = export_data.to_csv(
        index=False
    )

    st.download_button(
        "Download Bus Prediction",
        csv,
        "bus_arrival_prediction.csv",
        "text/csv",
        width="stretch"
    )

    # ========================================================
    # 13. FINAL MESSAGE
    # ========================================================

    st.success(
        "Bus arrival prediction completed successfully."
    )

    st.caption(
        "Note: This is a standalone prediction/simulation "
        "component. It is not a live GTFS/GTFS-RT feed."
    )
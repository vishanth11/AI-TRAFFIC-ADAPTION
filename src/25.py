import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# STEP 25 — NGSIM / TRAFFIC DATASET INTELLIGENCE
# ============================================================

st.set_page_config(
    page_title="NGSIM Traffic Dataset Intelligence",
    page_icon="🚦",
    layout="wide"
)

st.title("NGSIM / Traffic Dataset Intelligence")
st.markdown(
    """
    Analyze traffic datasets and extract important intelligence for
    **traffic flow prediction, speed analysis, queue analysis and
    congestion prediction**.
    """
)

# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# FIND CSV FILES
# ============================================================

csv_files = []

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.lower().endswith(".csv"):
            csv_files.append(os.path.join(root, file))


# ============================================================
# DATASET LOADING
# ============================================================

st.header("1. Dataset Selection")

if csv_files:

    display_names = [
        os.path.relpath(file, BASE_DIR)
        for file in csv_files
    ]

    selected_name = st.selectbox(
        "Select Traffic Dataset",
        display_names
    )

    selected_path = os.path.join(BASE_DIR, selected_name)

    try:
        df = pd.read_csv(selected_path)

        st.success(
            f"Dataset loaded successfully: {selected_name}"
        )

    except Exception as e:
        st.error(f"Unable to read dataset: {e}")
        st.stop()

else:

    st.warning(
        "No CSV dataset was found. A demonstration traffic dataset "
        "will be generated."
    )

    np.random.seed(42)

    n = 500

    time_values = np.arange(n)

    vehicle_count = (
        40
        + 25 * np.sin(time_values / 35)
        + np.random.normal(0, 5, n)
    )

    vehicle_count = np.maximum(vehicle_count, 5)

    speed = (
        55
        - 0.25 * vehicle_count
        + np.random.normal(0, 3, n)
    )

    speed = np.clip(speed, 10, 70)

    queue_length = (
        0.15 * vehicle_count
        + np.random.normal(0, 2, n)
    )

    queue_length = np.maximum(queue_length, 0)

    df = pd.DataFrame({
        "time": time_values,
        "vehicle_count": vehicle_count,
        "speed": speed,
        "queue_length": queue_length
    })

    st.info("Demonstration traffic dataset generated.")


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

st.header("2. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Rows",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

with col3:
    st.metric(
        "Missing Values",
        f"{int(df.isnull().sum().sum()):,}"
    )

with col4:
    numeric_count = len(df.select_dtypes(include=np.number).columns)

    st.metric(
        "Numeric Features",
        f"{numeric_count}"
    )


st.subheader("Dataset Preview")

preview_df = df.head(10).copy()

# Convert object/mixed columns to safe display strings
for column in preview_df.columns:
    if preview_df[column].dtype == "object":
        preview_df[column] = preview_df[column].astype(str)

st.dataframe(
    preview_df,
    width="stretch"
)


# ============================================================
# NUMERIC FEATURES
# ============================================================

numeric_columns = list(
    df.select_dtypes(include=np.number).columns
)

if not numeric_columns:

    st.error(
        "The dataset does not contain numeric columns required "
        "for traffic analysis."
    )

    st.stop()


# ============================================================
# FEATURE SELECTION
# ============================================================

st.header("3. Traffic Feature Selection")

col1, col2, col3 = st.columns(3)

with col1:

    vehicle_candidates = [
        c for c in numeric_columns
        if any(
            word in c.lower()
            for word in [
                "vehicle",
                "traffic",
                "count",
                "flow"
            ]
        )
    ]

    if not vehicle_candidates:
        vehicle_candidates = numeric_columns

    vehicle_column = st.selectbox(
        "Vehicle / Traffic Count",
        vehicle_candidates
    )


with col2:

    speed_candidates = [
        c for c in numeric_columns
        if "speed" in c.lower()
    ]

    if speed_candidates:
        speed_column = st.selectbox(
            "Vehicle Speed",
            speed_candidates
        )
    else:
        speed_column = st.selectbox(
            "Vehicle Speed",
            numeric_columns
        )


with col3:

    queue_candidates = [
        c for c in numeric_columns
        if any(
            word in c.lower()
            for word in [
                "queue",
                "waiting",
                "length"
            ]
        )
    ]

    if queue_candidates:
        queue_column = st.selectbox(
            "Queue Length",
            queue_candidates
        )
    else:
        queue_column = st.selectbox(
            "Queue Length",
            numeric_columns
        )


# ============================================================
# CLEAN SELECTED DATA
# ============================================================

analysis_df = pd.DataFrame({
    "Vehicle_Count": pd.to_numeric(
        df[vehicle_column],
        errors="coerce"
    ),
    "Speed": pd.to_numeric(
        df[speed_column],
        errors="coerce"
    ),
    "Queue_Length": pd.to_numeric(
        df[queue_column],
        errors="coerce"
    )
})

analysis_df = analysis_df.dropna().reset_index(drop=True)


# ============================================================
# TRAFFIC STATISTICS
# ============================================================

st.header("4. Traffic Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average Vehicles",
        f"{analysis_df['Vehicle_Count'].mean():.2f}"
    )

with col2:
    st.metric(
        "Peak Vehicles",
        f"{analysis_df['Vehicle_Count'].max():.2f}"
    )

with col3:
    st.metric(
        "Average Speed",
        f"{analysis_df['Speed'].mean():.2f}"
    )

with col4:
    st.metric(
        "Average Queue",
        f"{analysis_df['Queue_Length'].mean():.2f}"
    )


# ============================================================
# TRAFFIC FLOW ANALYSIS
# ============================================================

st.header("5. Traffic Flow Analysis")

fig = plt.figure(figsize=(12, 5))

plt.plot(
    analysis_df["Vehicle_Count"].values
)

plt.title("Traffic / Vehicle Count Over Time")
plt.xlabel("Observation")
plt.ylabel("Vehicle Count")
plt.grid(True)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# MOVING AVERAGE
# ============================================================

st.subheader("Moving Average Traffic Trend")

window = st.slider(
    "Moving Average Window",
    min_value=3,
    max_value=50,
    value=10
)

analysis_df["Moving_Average"] = (
    analysis_df["Vehicle_Count"]
    .rolling(window=window)
    .mean()
)

fig = plt.figure(figsize=(12, 5))

plt.plot(
    analysis_df["Vehicle_Count"].values,
    label="Actual Traffic"
)

plt.plot(
    analysis_df["Moving_Average"].values,
    label=f"{window}-Point Moving Average"
)

plt.title("Traffic Trend with Moving Average")
plt.xlabel("Observation")
plt.ylabel("Vehicle Count")
plt.legend()
plt.grid(True)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# SPEED ANALYSIS
# ============================================================

st.header("6. Vehicle Speed Analysis")

fig = plt.figure(figsize=(12, 5))

plt.plot(
    analysis_df["Speed"].values
)

plt.title("Vehicle Speed Over Time")
plt.xlabel("Observation")
plt.ylabel("Speed")
plt.grid(True)

st.pyplot(fig)

plt.close(fig)


speed_mean = analysis_df["Speed"].mean()

if speed_mean < 20:

    speed_status = "LOW SPEED — Possible congestion"

elif speed_mean < 40:

    speed_status = "MODERATE SPEED — Traffic slowing"

else:

    speed_status = "NORMAL SPEED — Traffic moving well"


st.info(
    f"Average speed: {speed_mean:.2f} | {speed_status}"
)


# ============================================================
# QUEUE ANALYSIS
# ============================================================

st.header("7. Queue Length Analysis")

fig = plt.figure(figsize=(12, 5))

plt.plot(
    analysis_df["Queue_Length"].values
)

plt.title("Queue Length Over Time")
plt.xlabel("Observation")
plt.ylabel("Queue Length")
plt.grid(True)

st.pyplot(fig)

plt.close(fig)


queue_mean = analysis_df["Queue_Length"].mean()
queue_peak = analysis_df["Queue_Length"].max()


if queue_mean < 10:

    queue_status = "LOW QUEUE"

elif queue_mean < 25:

    queue_status = "MODERATE QUEUE"

else:

    queue_status = "HIGH QUEUE"


st.info(
    f"Average queue: {queue_mean:.2f} | "
    f"Peak queue: {queue_peak:.2f} | "
    f"{queue_status}"
)


# ============================================================
# PEAK TRAFFIC DETECTION
# ============================================================

st.header("8. Peak Traffic Detection")

traffic_mean = analysis_df["Vehicle_Count"].mean()

peak_threshold = traffic_mean * 1.20

peak_points = analysis_df[
    analysis_df["Vehicle_Count"] >= peak_threshold
].copy()


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Average Traffic",
        f"{traffic_mean:.2f}"
    )

with col2:

    st.metric(
        "Peak Threshold",
        f"{peak_threshold:.2f}"
    )

with col3:

    st.metric(
        "Peak Observations",
        f"{len(peak_points)}"
    )


fig = plt.figure(figsize=(12, 5))

plt.plot(
    analysis_df["Vehicle_Count"].values,
    label="Traffic"
)

plt.axhline(
    peak_threshold,
    linestyle="--",
    label="Peak Threshold"
)

plt.title("Peak Traffic Detection")
plt.xlabel("Observation")
plt.ylabel("Vehicle Count")
plt.legend()
plt.grid(True)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

st.header("9. Traffic Feature Correlation")

correlation_df = analysis_df[
    [
        "Vehicle_Count",
        "Speed",
        "Queue_Length"
    ]
].corr()

st.dataframe(
    correlation_df.round(3),
    width="stretch"
)


fig = plt.figure(figsize=(7, 5))

plt.imshow(
    correlation_df.values,
    aspect="auto"
)

plt.xticks(
    range(len(correlation_df.columns)),
    correlation_df.columns,
    rotation=30
)

plt.yticks(
    range(len(correlation_df.columns)),
    correlation_df.columns
)

plt.title("Traffic Feature Correlation")

plt.colorbar()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# PREPARED DATASET
# ============================================================

st.header("10. Prepared Traffic Dataset")

prepared_df = analysis_df.copy()

prepared_df["Traffic_Level"] = pd.cut(
    prepared_df["Vehicle_Count"],
    bins=[
        -np.inf,
        traffic_mean * 0.70,
        traffic_mean * 1.20,
        np.inf
    ],
    labels=[
        "LOW",
        "MODERATE",
        "HIGH"
    ]
)


prepared_df["Speed_Status"] = pd.cut(
    prepared_df["Speed"],
    bins=[
        -np.inf,
        20,
        40,
        np.inf
    ],
    labels=[
        "LOW",
        "MODERATE",
        "NORMAL"
    ]
)


prepared_df["Queue_Status"] = pd.cut(
    prepared_df["Queue_Length"],
    bins=[
        -np.inf,
        10,
        25,
        np.inf
    ],
    labels=[
        "LOW",
        "MODERATE",
        "HIGH"
    ]
)


display_prepared = prepared_df.head(20).copy()

# Ensure Streamlit Arrow serialization works
for column in display_prepared.columns:

    if (
        str(display_prepared[column].dtype)
        == "category"
    ):
        display_prepared[column] = (
            display_prepared[column]
            .astype(str)
        )


st.dataframe(
    display_prepared,
    width="stretch"
)


# ============================================================
# TRAFFIC LEVEL DISTRIBUTION
# ============================================================

st.header("11. Traffic Level Distribution")

traffic_distribution = (
    prepared_df["Traffic_Level"]
    .value_counts()
    .reindex(
        ["LOW", "MODERATE", "HIGH"],
        fill_value=0
    )
)


fig = plt.figure(figsize=(8, 5))

plt.bar(
    traffic_distribution.index.astype(str),
    traffic_distribution.values
)

plt.title("Traffic Level Distribution")
plt.xlabel("Traffic Level")
plt.ylabel("Number of Observations")

st.pyplot(fig)

plt.close(fig)


# ============================================================
# CONGESTION INTELLIGENCE
# ============================================================

st.header("12. Congestion Intelligence")

high_traffic_percentage = (
    (
        prepared_df["Traffic_Level"]
        == "HIGH"
    ).mean()
    * 100
)

high_queue_percentage = (
    (
        prepared_df["Queue_Status"]
        == "HIGH"
    ).mean()
    * 100
)


col1, col2 = st.columns(2)

with col1:

    st.metric(
        "High Traffic %",
        f"{high_traffic_percentage:.2f}%"
    )

with col2:

    st.metric(
        "High Queue %",
        f"{high_queue_percentage:.2f}%"
    )


if high_traffic_percentage > 30:

    st.error(
        "HIGH CONGESTION RISK: "
        "A significant portion of observations show heavy traffic."
    )

elif high_traffic_percentage > 10:

    st.warning(
        "MODERATE CONGESTION RISK: "
        "Traffic peaks should be monitored."
    )

else:

    st.success(
        "LOW CONGESTION RISK: "
        "Most observations remain below the high-traffic range."
    )


# ============================================================
# EXPORT DATASET
# ============================================================

st.header("13. Export Analysis")

output_file = os.path.join(
    RESULTS_DIR,
    "ngsim_traffic_analysis.csv"
)

export_df = prepared_df.copy()

for column in export_df.columns:

    if (
        str(export_df[column].dtype)
        == "category"
    ):
        export_df[column] = (
            export_df[column]
            .astype(str)
        )


export_df.to_csv(
    output_file,
    index=False
)


csv_data = export_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Traffic Analysis CSV",
    data=csv_data,
    file_name="ngsim_traffic_analysis.csv",
    mime="text/csv"
)

st.success(
    f"Analysis saved to: {output_file}"
)


# ============================================================
# FINAL CONCLUSION
# ============================================================

st.header("14. Step 25 Conclusion")

st.markdown(
    """
    ### Dataset Intelligence Generated

    The traffic dataset has been analyzed for:

    - Vehicle / traffic volume
    - Vehicle speed
    - Queue length
    - Moving-average traffic trend
    - Peak traffic periods
    - Traffic-level classification
    - Speed conditions
    - Queue conditions
    - Feature correlations
    - Congestion risk

    ### Connection with Member 3

    This dataset intelligence provides the foundation for:

    **Historical Traffic Data**
    ↓

    **Feature Extraction**
    ↓

    **Traffic Pattern Analysis**
    ↓

    **GRU Traffic Prediction**
    ↓

    **Future Congestion Prediction**
    ↓

    **Proactive Signal Control**

    Therefore, Member 3 moves from simply observing current traffic
    to **predicting future traffic before it arrives**.
    """
)


st.caption(
    "Step 25 — NGSIM / Traffic Dataset Intelligence | Member 3"
)
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Traffic Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Traffic Prediction Model Comparison")

st.markdown(
    """
    ### Final Evaluation of Traffic Prediction Models

    This module compares the prediction performance of the
    developed traffic prediction approaches using MAE and RMSE.
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

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Evaluation Settings")

st.sidebar.info(
    """
    The comparison uses the evaluation results generated
    during the previous model-development stages.

    Lower MAE and RMSE indicate better prediction performance.
    """
)


# ============================================================
# INTRODUCTION
# ============================================================

st.header("1. Models Evaluated")

model_names = [
    "Moving Average",
    "Linear Regression",
    "Random Forest",
    "GRU"
]

model_description = [
    "Simple statistical baseline",
    "Linear machine-learning model",
    "Non-linear ensemble model",
    "Deep-learning sequence model"
]

model_info = pd.DataFrame(
    {
        "Model": model_names,
        "Description": model_description
    }
)

st.dataframe(
    model_info,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FIND EXISTING RESULT FILES
# ============================================================

def find_result_file(keywords):

    if not os.path.exists(RESULTS_DIR):
        return None

    files = os.listdir(RESULTS_DIR)

    for file in files:

        lower_name = file.lower()

        if all(
            keyword.lower() in lower_name
            for keyword in keywords
        ):

            return os.path.join(
                RESULTS_DIR,
                file
            )

    return None


# ============================================================
# LOAD POSSIBLE EVALUATION FILE
# ============================================================

evaluation_file = None

possible_files = [
    "model_comparison.csv",
    "model_comparison_results.csv",
    "evaluation_results.csv",
    "model_evaluation.csv",
    "metrics.csv"
]

for filename in possible_files:

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    if os.path.exists(path):

        evaluation_file = path
        break


# ============================================================
# MANUAL METRIC INPUT
# ============================================================

st.header("2. Model Evaluation Metrics")

st.write(
    """
    Enter the MAE and RMSE values obtained from your earlier
    model evaluation steps.
    """
)

col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# MOVING AVERAGE
# ------------------------------------------------------------

with col1:

    st.subheader("Moving Average")

    ma_mae = st.number_input(
        "MAE",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.4f",
        key="ma_mae"
    )

    ma_rmse = st.number_input(
        "RMSE",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.4f",
        key="ma_rmse"
    )


# ------------------------------------------------------------
# LINEAR REGRESSION
# ------------------------------------------------------------

with col2:

    st.subheader("Linear Regression")

    lr_mae = st.number_input(
        "MAE",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.4f",
        key="lr_mae"
    )

    lr_rmse = st.number_input(
        "RMSE",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.4f",
        key="lr_rmse"
    )


# ------------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------------

with col3:

    st.subheader("Random Forest")

    rf_mae = st.number_input(
        "MAE",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.4f",
        key="rf_mae"
    )

    rf_rmse = st.number_input(
        "RMSE",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.4f",
        key="rf_rmse"
    )


# ------------------------------------------------------------
# GRU
# ------------------------------------------------------------

with col4:

    st.subheader("GRU")

    gru_mae = st.number_input(
        "MAE",
        min_value=0.0,
        value=0.0481,
        step=0.0001,
        format="%.4f",
        key="gru_mae"
    )

    gru_rmse = st.number_input(
        "RMSE",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.4f",
        key="gru_rmse"
    )


# ============================================================
# COMPARE BUTTON
# ============================================================

st.markdown("---")

compare_button = st.button(
    "📊 Compare Models",
    type="primary",
    use_container_width=True
)


# ============================================================
# COMPARISON
# ============================================================

if compare_button:

    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Moving Average",
                "Linear Regression",
                "Random Forest",
                "GRU"
            ],

            "MAE": [
                ma_mae,
                lr_mae,
                rf_mae,
                gru_mae
            ],

            "RMSE": [
                ma_rmse,
                lr_rmse,
                rf_rmse,
                gru_rmse
            ]
        }
    )


    # --------------------------------------------------------
    # CHECK ENTERED VALUES
    # --------------------------------------------------------

    if (
        comparison_df["MAE"].sum() == 0
        or comparison_df["RMSE"].sum() == 0
    ):

        st.warning(
            """
            Please enter the MAE and RMSE values obtained
            from your previous model evaluation steps.
            """
        )

        st.stop()


    # ========================================================
    # BEST MODEL
    # ========================================================

    best_mae_index = comparison_df[
        "MAE"
    ].idxmin()

    best_rmse_index = comparison_df[
        "RMSE"
    ].idxmin()

    best_mae_model = comparison_df.loc[
        best_mae_index,
        "Model"
    ]

    best_rmse_model = comparison_df.loc[
        best_rmse_index,
        "Model"
    ]


    # ========================================================
    # RESULTS
    # ========================================================

    st.header(
        "3. Model Performance"
    )

    st.dataframe(
        comparison_df.style.format(
            {
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # METRIC CARDS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Best MAE Model",
            best_mae_model
        )

    with col2:

        st.metric(
            "Best RMSE Model",
            best_rmse_model
        )

    with col3:

        st.metric(
            "Number of Models",
            "4"
        )


    # ========================================================
    # MAE CHART
    # ========================================================

    st.header(
        "4. MAE Comparison"
    )

    fig1, ax1 = plt.subplots(
        figsize=(10, 5)
    )

    ax1.bar(
        comparison_df["Model"],
        comparison_df["MAE"]
    )

    ax1.set_title(
        "Mean Absolute Error (MAE)"
    )

    ax1.set_xlabel(
        "Model"
    )

    ax1.set_ylabel(
        "MAE"
    )

    ax1.grid(
        axis="y",
        alpha=0.3
    )

    plt.xticks(
        rotation=20
    )

    st.pyplot(
        fig1
    )


    # ========================================================
    # RMSE CHART
    # ========================================================

    st.header(
        "5. RMSE Comparison"
    )

    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )

    ax2.bar(
        comparison_df["Model"],
        comparison_df["RMSE"]
    )

    ax2.set_title(
        "Root Mean Squared Error (RMSE)"
    )

    ax2.set_xlabel(
        "Model"
    )

    ax2.set_ylabel(
        "RMSE"
    )

    ax2.grid(
        axis="y",
        alpha=0.3
    )

    plt.xticks(
        rotation=20
    )

    st.pyplot(
        fig2
    )


    # ========================================================
    # COMBINED COMPARISON
    # ========================================================

    st.header(
        "6. Overall Model Comparison"
    )

    x = np.arange(
        len(comparison_df)
    )

    width = 0.35

    fig3, ax3 = plt.subplots(
        figsize=(11, 5)
    )

    ax3.bar(
        x - width / 2,
        comparison_df["MAE"],
        width,
        label="MAE"
    )

    ax3.bar(
        x + width / 2,
        comparison_df["RMSE"],
        width,
        label="RMSE"
    )

    ax3.set_title(
        "Traffic Prediction Model Performance"
    )

    ax3.set_xlabel(
        "Model"
    )

    ax3.set_ylabel(
        "Error"
    )

    ax3.set_xticks(
        x
    )

    ax3.set_xticklabels(
        comparison_df["Model"],
        rotation=20
    )

    ax3.legend()

    ax3.grid(
        axis="y",
        alpha=0.3
    )

    st.pyplot(
        fig3
    )


    # ========================================================
    # MODEL RANKING
    # ========================================================

    st.header(
        "7. Model Ranking"
    )

    ranking_df = comparison_df.copy()

    ranking_df["MAE Rank"] = (
        ranking_df["MAE"]
        .rank(
            method="min"
        )
        .astype(int)
    )

    ranking_df["RMSE Rank"] = (
        ranking_df["RMSE"]
        .rank(
            method="min"
        )
        .astype(int)
    )

    ranking_df["Overall Rank"] = (
        ranking_df["MAE Rank"]
        + ranking_df["RMSE Rank"]
    )

    ranking_df = ranking_df.sort_values(
        "Overall Rank"
    )

    ranking_df.insert(
        0,
        "Position",
        range(
            1,
            len(ranking_df) + 1
        )
    )

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    final_best_model = ranking_df.iloc[
        0
    ]["Model"]

    final_best_mae = ranking_df.iloc[
        0
    ]["MAE"]

    final_best_rmse = ranking_df.iloc[
        0
    ]["RMSE"]


    st.header(
        "8. Best Performing Model"
    )

    st.success(
        f"""
        ### 🏆 {final_best_model}

        **MAE:** {final_best_mae:.4f}

        **RMSE:** {final_best_rmse:.4f}

        The model with the lowest combined ranking is
        selected as the best-performing traffic prediction model.
        """
    )


    # ========================================================
    # PROJECT INTERPRETATION
    # ========================================================

    st.header(
        "9. Project Interpretation"
    )

    st.markdown(
        f"""
        ### Final Finding

        Among the evaluated models, **{final_best_model}**
        provides the strongest prediction performance based on
        the combined MAE and RMSE ranking.

        Lower error indicates that the predicted traffic values
        are closer to the observed traffic values.

        The selected model can therefore be used as the main
        prediction component of the traffic intelligence system.

        The prediction output can subsequently support:

        - Proactive traffic signal control
        - Early congestion detection
        - Junction preparation
        - Traffic-flow optimization
        - Emergency vehicle corridor planning
        """
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.header(
        "10. Export Evaluation Results"
    )

    csv_data = ranking_df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Model Comparison CSV",
        data=csv_data,
        file_name="traffic_model_comparison.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    st.markdown("---")

    st.info(
        f"""
        **Final Model Selection**

        Best Model: **{final_best_model}**

        Best MAE: **{final_best_mae:.4f}**

        Best RMSE: **{final_best_rmse:.4f}**

        This model should be considered the primary prediction
        model for the Member 3 Prediction Intelligence module.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Member 3 — Prediction Intelligence | Final Model Evaluation"
)
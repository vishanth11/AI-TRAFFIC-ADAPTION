import os
import sys
import importlib.util
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Traffic Prediction - Final Validation",
    page_icon="✅",
    layout="wide"
)

st.title("✅ Traffic Prediction System")
st.subheader("Step 22 — Final Testing & Validation")

st.markdown(
    """
    This module performs the final validation of the
    **Member 3 — Prediction Intelligence** system.

    It checks the project structure, trained model,
    prediction modules and required resources before submission.
    """
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# TEST STORAGE
# ============================================================

tests = []


def add_test(
    name,
    status,
    details
):

    tests.append(
        {
            "Test": name,
            "Status": status,
            "Details": details
        }
    )


# ============================================================
# TEST 1 — PROJECT ROOT
# ============================================================

if os.path.exists(BASE_DIR):

    add_test(
        "Project Root",
        "PASS",
        BASE_DIR
    )

else:

    add_test(
        "Project Root",
        "FAIL",
        "Project directory not found"
    )


# ============================================================
# TEST 2 — SOURCE DIRECTORY
# ============================================================

if os.path.exists(SRC_DIR):

    add_test(
        "Source Directory",
        "PASS",
        "src folder found"
    )

else:

    add_test(
        "Source Directory",
        "FAIL",
        "src folder not found"
    )


# ============================================================
# TEST 3 — RESULTS DIRECTORY
# ============================================================

if os.path.exists(RESULTS_DIR):

    add_test(
        "Results Directory",
        "PASS",
        "results folder found"
    )

else:

    add_test(
        "Results Directory",
        "FAIL",
        "results folder not found"
    )


# ============================================================
# TEST 4 — GRU MODEL
# ============================================================

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "traffic_gru.pth"
)

if os.path.exists(MODEL_PATH):

    model_size = os.path.getsize(
        MODEL_PATH
    )

    add_test(
        "Trained GRU Model",
        "PASS",
        f"traffic_gru.pth found ({model_size / 1024:.2f} KB)"
    )

else:

    add_test(
        "Trained GRU Model",
        "FAIL",
        "traffic_gru.pth not found"
    )


# ============================================================
# TEST 5 — PREVIOUS STREAMLIT MODULES
# ============================================================

required_modules = [
    "16.py",
    "17.py",
    "18.py",
    "19.py",
    "20.py",
    "21.py"
]

for module in required_modules:

    path = os.path.join(
        SRC_DIR,
        module
    )

    if os.path.exists(path):

        add_test(
            f"Module {module}",
            "PASS",
            "File found"
        )

    else:

        add_test(
            f"Module {module}",
            "FAIL",
            "File not found"
        )


# ============================================================
# TEST 6 — FINAL MODULE
# ============================================================

current_file = os.path.join(
    SRC_DIR,
    "22.py"
)

if os.path.exists(current_file):

    add_test(
        "Final Validation Module",
        "PASS",
        "22.py is available"
    )

else:

    add_test(
        "Final Validation Module",
        "FAIL",
        "22.py not found"
    )


# ============================================================
# TEST 7 — RESULTS FILES
# ============================================================

if os.path.exists(RESULTS_DIR):

    result_files = os.listdir(
        RESULTS_DIR
    )

    if len(result_files) > 0:

        add_test(
            "Result Files",
            "PASS",
            f"{len(result_files)} result file(s) found"
        )

    else:

        add_test(
            "Result Files",
            "WARNING",
            "Results folder is empty"
        )


# ============================================================
# TEST 8 — PYTHON VERSION
# ============================================================

python_version = (
    f"{sys.version_info.major}."
    f"{sys.version_info.minor}."
    f"{sys.version_info.micro}"
)

add_test(
    "Python Environment",
    "PASS",
    f"Python {python_version}"
)


# ============================================================
# DISPLAY TEST RESULTS
# ============================================================

st.header("1. Project Validation")

validation_df = pd.DataFrame(
    tests
)

st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SUMMARY
# ============================================================

total_tests = len(tests)

passed_tests = sum(
    1
    for test in tests
    if test["Status"] == "PASS"
)

failed_tests = sum(
    1
    for test in tests
    if test["Status"] == "FAIL"
)

warning_tests = sum(
    1
    for test in tests
    if test["Status"] == "WARNING"
)


# ============================================================
# METRICS
# ============================================================

st.header("2. Validation Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Tests",
        total_tests
    )

with col2:

    st.metric(
        "Passed",
        passed_tests
    )

with col3:

    st.metric(
        "Failed",
        failed_tests
    )

with col4:

    st.metric(
        "Warnings",
        warning_tests
    )


# ============================================================
# FINAL STATUS
# ============================================================

st.header("3. Final Project Status")

if failed_tests == 0:

    st.success(
        """
        ## ✅ PROJECT VALIDATION PASSED

        All essential project checks have passed.

        The Member 3 Prediction Intelligence module is
        ready for final demonstration and documentation.
        """
    )

else:

    st.error(
        f"""
        ## ❌ VALIDATION REQUIRES ATTENTION

        {failed_tests} test(s) failed.

        Check the validation table above and correct the
        missing files before submission.
        """
    )


# ============================================================
# PROJECT ARCHITECTURE
# ============================================================

st.header("4. Final Member 3 Architecture")

architecture = pd.DataFrame(
    {
        "Stage": [
            "Data Preparation",
            "Feature Engineering",
            "Baseline Models",
            "Machine Learning Models",
            "GRU Training",
            "Model Deployment",
            "Prediction Visualization",
            "Future Traffic Prediction",
            "Proactive Junction Intelligence",
            "Model Comparison",
            "Integrated Dashboard",
            "Final Validation"
        ],

        "Status": [
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed"
        ]
    }
)

st.dataframe(
    architecture,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PROJECT FLOW
# ============================================================

st.header("5. Final Prediction Intelligence Flow")

flow = """
CURRENT TRAFFIC
       ↓
DATA PROCESSING
       ↓
FEATURE REPRESENTATION
       ↓
GRU TRAFFIC MODEL
       ↓
FUTURE TRAFFIC PREDICTION
       ↓
CONGESTION ANALYSIS
       ↓
PROACTIVE JUNCTION DECISION
       ↓
SIGNAL RECOMMENDATION
       ↓
TRAFFIC CONTROL SUPPORT
"""

st.code(
    flow,
    language="text"
)


# ============================================================
# MEMBER 3 RESPONSIBILITIES
# ============================================================

st.header("6. Member 3 Deliverables")

deliverables = [
    "Traffic flow prediction",
    "Short-term future traffic prediction",
    "GRU-based prediction model",
    "Prediction visualization",
    "Congestion intelligence",
    "Proactive junction intelligence",
    "Model performance comparison",
    "Integrated prediction dashboard",
    "Final system validation"
]

for item in deliverables:

    st.write(
        f"✅ {item}"
    )


# ============================================================
# FINAL REPORT SUMMARY
# ============================================================

st.header("7. Final Project Conclusion")

st.markdown(
    """
    ### Conclusion

    The Prediction Intelligence module provides a machine-learning
    based approach for forecasting future traffic conditions.

    The trained GRU model processes the available traffic pattern
    and produces future traffic predictions. These predictions are
    subsequently analysed to estimate congestion and generate
    proactive junction recommendations.

    Therefore, the system moves beyond simple traffic prediction
    toward **predictive traffic management**, where expected traffic
    can be considered before it reaches the junction.

    The module can support:

    - Early congestion identification
    - Proactive signal preparation
    - Traffic-flow optimization
    - Junction-level decision support
    - Future integration with emergency vehicle routing
    """
)


# ============================================================
# DOWNLOAD VALIDATION REPORT
# ============================================================

st.header("8. Export Validation Report")

report_df = pd.DataFrame(
    tests
)

report_csv = report_df.to_csv(
    index=False
)

st.download_button(
    label="📥 Download Validation Report",
    data=report_csv,
    file_name="final_project_validation.csv",
    mime="text/csv",
    use_container_width=True
)


# ============================================================
# FINAL MESSAGE
# ============================================================

st.markdown("---")

if failed_tests == 0:

    st.success(
        """
        🎉 **FINAL VALIDATION COMPLETE**

        Your Member 3 Prediction Intelligence module is
        ready for the next stage: **final project packaging,
        report preparation and ZIP submission**.
        """
    )

else:

    st.warning(
        """
        Fix the failed checks shown above and run the
        validation again.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Traffic Prediction System | Member 3 — Prediction Intelligence"
)
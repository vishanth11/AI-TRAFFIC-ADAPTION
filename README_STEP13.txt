TRAFFIC PREDICTION + INTELLIGENT SIGNAL CONTROL
STEP 13 ONWARD PACKAGE

1. Keep the existing project structure.
2. Install the small continuation dependency set:
   pip install -r requirements_step13.txt

3. Run the complete continuation:
   python src/15_run_all.py

The continuation expects:
results/baseline_predictions.npy

The project uses the existing 36-sensor / 3-junction mapping from Step 12.

Main additions:
- Step 13: connected 3-junction simulation
- Step 13A: emergency green corridor
- Step 13B: confidence-aware signal control
- Step 13C: visualization
- Step 14: control evaluation
- Step 15: one-command runner

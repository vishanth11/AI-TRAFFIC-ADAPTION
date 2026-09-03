"""Member 4 Phase 5 — Accident Detection.

Scope:
- Object-level accident detection on the CCTV accident dataset.
- Uses torchvision Faster R-CNN with a MobileNetV3-320 backbone for CPU training.
- Preserves original Parquet files as canonical data source.

Explicitly excluded in this phase:
- Near-miss model modification.
- Risk fusion / risk engine.
- Confidence calibration (detector confidence is not a calibrated probability).
- Member 5 integration.
- Traffic signal control.
- Ambulance logic.
- SUMO integration.
- Final API.
"""

RANDOM_SEED = 42

CLASS_ID_ACCIDENT = 0
CLASS_ID_NON_ACCIDENT = 1

CLASS_NAMES = {
    CLASS_ID_ACCIDENT: "accident",
    CLASS_ID_NON_ACCIDENT: "non_accident",
}

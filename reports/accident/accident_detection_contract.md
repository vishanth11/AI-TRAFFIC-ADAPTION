# Accident Detection Output Contract

This document defines the inference output of the Phase 5 accident detector.

## Output Schema

```json
{
    "detector_version": "accident_detector_v1",
    "image_size": [640, 640],
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],
            "category": "accident",
            "confidence": 0.91
        }
    ],
    "accident_detection_confidence": 0.91
}
```

## Field Definitions

- `detector_version`: static version string identifying the detector artifact.
- `image_size`: width and height of the input image in pixels.
- `detections`: list of detected objects.
  - `bbox`: bounding box in `[x_min, y_min, x_max, y_max]` absolute pixel coordinates.
  - `category`: `"accident"` or `"non_accident"`.
  - `confidence`: detector confidence score in `[0, 1]`.
- `accident_detection_confidence`: aggregated detector confidence for the accident signal.

## Important Notes

- `confidence` is a **detector confidence score**, not a calibrated accident probability.
- Do not label this field as `calibrated_accident_probability`.
- No `risk_level` is produced in this phase.
- No Member 5 integration or traffic-signal control is performed.

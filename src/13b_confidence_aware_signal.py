import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

MIN_GREEN = 15
MAX_GREEN = 60
CYCLE = 90

def confidence_from_spread(values):
    values = np.asarray(values, dtype=float)
    spread = float(np.std(values))
    # A simple, explainable confidence proxy for the simulation.
    return float(np.clip(1.0 - 2.0 * spread, 0.0, 1.0))

def decide(traffic, confidence):
    base = MIN_GREEN + traffic * (MAX_GREEN - MIN_GREEN)
    if confidence < 0.50:
        # Fall back toward a safer mid-range timing when uncertainty is high.
        base = 0.70 * base + 0.30 * ((MIN_GREEN + MAX_GREEN) / 2)
        mode = "CAUTIOUS"
    else:
        mode = "PREDICTION-DRIVEN"

    green = int(round(np.clip(base, MIN_GREEN, MAX_GREEN)))
    return green, CYCLE - green, mode

def main():
    predictions = np.load(os.path.join(RESULTS_DIR, "baseline_predictions.npy"))
    p = predictions[0]

    print("=" * 68)
    print("STEP 13B - CONFIDENCE-AWARE SIGNAL CONTROL")
    print("=" * 68)

    for j in range(3):
        values = p[j*12:(j+1)*12]
        traffic = float(np.mean(values))
        confidence = confidence_from_spread(values)
        green, red, mode = decide(traffic, confidence)

        print(f"\nJunction {j+1}")
        print(f"Traffic level : {traffic:.4f}")
        print(f"Confidence    : {confidence:.4f}")
        print(f"Controller    : {mode}")
        print(f"Green / Red   : {green}s / {red}s")

if __name__ == "__main__":
    main()

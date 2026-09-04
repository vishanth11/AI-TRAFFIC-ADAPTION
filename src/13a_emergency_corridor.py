import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CYCLE_TIME = 90
MIN_CROSS_GREEN = 15

def build_corridor(predictions, route=(0, 1, 2)):
    selected = np.asarray(predictions)[0]
    result = []

    for j in range(3):
        start, end = j * 12, (j + 1) * 12
        traffic = float(np.mean(selected[start:end]))

        if j in route:
            # Reserve only the minimum cross-direction interval.
            green = CYCLE_TIME - MIN_CROSS_GREEN
            mode = "EMERGENCY"
        else:
            green = MIN_CROSS_GREEN
            mode = "NORMAL"

        result.append({
            "junction": j + 1,
            "traffic": traffic,
            "green": int(green),
            "red": int(CYCLE_TIME - green),
            "mode": mode
        })
    return result

def main():
    path = os.path.join(RESULTS_DIR, "baseline_predictions.npy")
    predictions = np.load(path)

    print("=" * 68)
    print("STEP 13A - EMERGENCY GREEN CORRIDOR")
    print("=" * 68)

    for item in build_corridor(predictions):
        print(
            f"Junction {item['junction']}: "
            f"traffic={item['traffic']:.4f}, "
            f"green={item['green']}s, red={item['red']}s, "
            f"{item['mode']}"
        )

    print("\nPolicy: the simulated emergency route receives priority at all "
          "three connected junctions.")

if __name__ == "__main__":
    main()

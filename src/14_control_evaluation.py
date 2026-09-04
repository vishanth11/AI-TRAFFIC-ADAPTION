import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CYCLE = 90
MIN_GREEN = 15
MAX_GREEN = 60

def green_time(traffic):
    return int(round(np.clip(MIN_GREEN + traffic * (MAX_GREEN - MIN_GREEN),
                          MIN_GREEN, MAX_GREEN)))

def evaluate(predictions):
    p = predictions[0]
    rows = []
    for j in range(3):
        level = float(np.mean(p[j*12:(j+1)*12]))
        green = green_time(level)
        utilization = green / CYCLE
        rows.append((j + 1, level, green, CYCLE-green, utilization))
    return rows

def main():
    predictions = np.load(os.path.join(RESULTS_DIR, "baseline_predictions.npy"))
    rows = evaluate(predictions)

    print("=" * 68)
    print("STEP 14 - SIGNAL CONTROL EVALUATION")
    print("=" * 68)
    print("Junction | Traffic | Green(s) | Red(s) | Green utilization")
    for row in rows:
        print(f"{row[0]:8d} | {row[1]:7.4f} | {row[2]:8d} | "
              f"{row[3]:6d} | {row[4]:.3f}")

    avg = np.mean([r[4] for r in rows])
    print(f"\nAverage green utilization: {avg:.3f}")
    print("This is a simulation metric, not a field traffic outcome.")

if __name__ == "__main__":
    main()

import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CYCLE_TIME = 90
MIN_GREEN = 15
MAX_GREEN = 60
JUNCTION_SIZE = 12

def clamp(value, low, high):
    return max(low, min(high, value))

def junction_levels(prediction):
    prediction = np.asarray(prediction, dtype=float)
    if prediction.size != 36:
        raise ValueError("Expected 36 sensor predictions.")
    return {
        "Junction 1": float(np.mean(prediction[0:12])),
        "Junction 2": float(np.mean(prediction[12:24])),
        "Junction 3": float(np.mean(prediction[24:36])),
    }

def normal_signal(traffic):
    # More predicted traffic receives more green time.
    green = MIN_GREEN + traffic * (MAX_GREEN - MIN_GREEN)
    green = int(round(clamp(green, MIN_GREEN, MAX_GREEN)))
    return green, CYCLE_TIME - green

def emergency_corridor(plan, corridor=("Junction 1", "Junction 2", "Junction 3")):
    # Simulation policy: clear the emergency route while keeping a minimum
    # cross-direction safety interval.
    result = {}
    for junction, (green, red) in plan.items():
        if junction in corridor:
            result[junction] = {"green": CYCLE_TIME - MIN_GREEN,
                                "red": MIN_GREEN,
                                "mode": "EMERGENCY PRIORITY"}
        else:
            result[junction] = {"green": green, "red": red,
                                "mode": "NORMAL"}
    return result

def main():
    prediction_path = os.path.join(RESULTS_DIR, "baseline_predictions.npy")
    predictions = np.load(prediction_path)
    selected = predictions[0]
    levels = junction_levels(selected)

    plan = {j: normal_signal(level) for j, level in levels.items()}
    emergency_plan = emergency_corridor(plan)

    print("=" * 72)
    print("STEP 13 - CONNECTED 3-JUNCTION TRAFFIC SIMULATION")
    print("=" * 72)

    for junction in levels:
        green, red = plan[junction]
        print(f"\n{junction}")
        print(f"Predicted traffic : {levels[junction]:.4f}")
        print(f"Normal plan       : GREEN {green}s | RED {red}s")
        print(f"Emergency plan    : GREEN {emergency_plan[junction]['green']}s | "
              f"RED {emergency_plan[junction]['red']}s")

    print("\nEmergency corridor:", " -> ".join(emergency_plan.keys()))
    print("Simulation completed.")

if __name__ == "__main__":
    main()

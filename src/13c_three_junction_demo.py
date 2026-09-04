import os
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def main():
    predictions = np.load(os.path.join(RESULTS_DIR, "baseline_predictions.npy"))
    p = predictions[0]

    traffic = [
        float(np.mean(p[0:12])),
        float(np.mean(p[12:24])),
        float(np.mean(p[24:36]))
    ]

    junctions = ["Junction 1", "Junction 2", "Junction 3"]

    # Visual demo of the predicted load and normal green allocation.
    green = [round(15 + x * 45) for x in traffic]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(junctions, traffic)
    plt.ylabel("Predicted normalized traffic")
    plt.title("Step 13C - Connected Junction Traffic")
    for bar, value in zip(bars, traffic):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "step13_junction_traffic.png"), dpi=300)
    plt.close()

    print("Step 13C completed.")
    for j, t, g in zip(junctions, traffic, green):
        print(f"{j}: traffic={t:.4f}, suggested green={g}s")
    print("Saved: results/step13_junction_traffic.png")

if __name__ == "__main__":
    main()

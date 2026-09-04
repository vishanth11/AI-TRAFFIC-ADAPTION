import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

steps = [
    "13_connected_junction_simulation.py",
    "13a_emergency_corridor.py",
    "13b_confidence_aware_signal.py",
    "13c_three_junction_demo.py",
    "14_control_evaluation.py",
]

for name in steps:
    print("\n" + "=" * 72)
    print("RUNNING", name)
    print("=" * 72)
    subprocess.run([sys.executable, os.path.join(SRC, name)], check=True)

print("\nAll Step 13+ modules completed successfully.")

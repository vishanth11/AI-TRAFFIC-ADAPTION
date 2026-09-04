
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_PATH = r"D:\traffic_prediction\results\baseline_predictions.npy"

MIN_GREEN = 15
MAX_GREEN = 60
CYCLE_TIME = 90


# ============================================================
# LOAD RANDOM FOREST PREDICTIONS
# ============================================================

predictions = np.load(PREDICTION_PATH)

print("=" * 65)
print("RANDOM FOREST → ADAPTIVE TRAFFIC SIGNAL")
print("=" * 65)

print("\nPrediction shape:", predictions.shape)

# Expected:
# (840, 36)


# ============================================================
# SELECT ONE FUTURE TIME STEP
# ============================================================

time_step = 0

traffic = predictions[time_step]

print("\nSelected prediction time step:", time_step)
print("Traffic predictions for 36 sensors:")
print(traffic)


# ============================================================
# CONVERT SENSOR TRAFFIC INTO JUNCTION TRAFFIC
# ============================================================

# 36 sensors → 3 junctions
# 12 sensors per junction

junction_1 = traffic[0:12]
junction_2 = traffic[12:24]
junction_3 = traffic[24:36]


# ============================================================
# CALCULATE TRAFFIC LEVEL
# ============================================================

def calculate_junction_traffic(sensor_values):

    traffic_level = np.mean(sensor_values)

    return traffic_level


j1_traffic = calculate_junction_traffic(junction_1)
j2_traffic = calculate_junction_traffic(junction_2)
j3_traffic = calculate_junction_traffic(junction_3)


# ============================================================
# ADAPTIVE GREEN TIME
# ============================================================

def calculate_green_time(traffic_level):

    green_time = MIN_GREEN + (
        traffic_level * (MAX_GREEN - MIN_GREEN)
    )

    return round(green_time)


# ============================================================
# SIGNAL DECISION
# ============================================================

def signal_decision(traffic_level):

    green_time = calculate_green_time(traffic_level)

    red_time = CYCLE_TIME - green_time

    return green_time, red_time


# ============================================================
# DISPLAY RESULTS
# ============================================================

junctions = {
    "Junction 1": j1_traffic,
    "Junction 2": j2_traffic,
    "Junction 3": j3_traffic
}


print("\n" + "=" * 65)
print("ADAPTIVE SIGNAL PLAN")
print("=" * 65)


for junction, traffic_level in junctions.items():

    green, red = signal_decision(traffic_level)

    print("\n" + junction)
    print("-" * 40)

    print(f"Predicted Traffic Level : {traffic_level:.4f}")
    print(f"Green Time              : {green} seconds")
    print(f"Red Time                : {red} seconds")

    if traffic_level >= 0.70:
        status = "HIGH TRAFFIC"

    elif traffic_level >= 0.40:
        status = "MEDIUM TRAFFIC"

    else:
        status = "LOW TRAFFIC"

    print(f"Traffic Status          : {status}")


print("\n" + "=" * 65)
print("PREDICTION-BASED SIGNAL CONTROL COMPLETED")
print("=" * 65)
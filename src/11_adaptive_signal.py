import numpy as np


# ============================================================
# ADAPTIVE TRAFFIC SIGNAL CONTROLLER
# ============================================================

MIN_GREEN = 15
MAX_GREEN = 60
CYCLE_TIME = 90


def calculate_green_time(traffic_level):
    """
    Convert predicted traffic level (0-1)
    into adaptive green time.
    """

    green_time = MIN_GREEN + (
        traffic_level * (MAX_GREEN - MIN_GREEN)
    )

    return round(green_time)


def calculate_signal_plan(ns_traffic, ew_traffic):
    """
    Decide which direction receives more green time.
    """

    total = ns_traffic + ew_traffic

    if total == 0:
        ns_ratio = 0.5
    else:
        ns_ratio = ns_traffic / total

    ns_green = CYCLE_TIME * ns_ratio
    ew_green = CYCLE_TIME - ns_green

    # Apply safety limits
    ns_green = max(MIN_GREEN, min(MAX_GREEN, ns_green))
    ew_green = max(MIN_GREEN, min(MAX_GREEN, ew_green))

    return round(ns_green), round(ew_green)


# ============================================================
# SIMULATED PREDICTED TRAFFIC
# ============================================================

predicted_traffic = {
    "Junction 1": {
        "North-South": 0.82,
        "East-West": 0.31
    },

    "Junction 2": {
        "North-South": 0.65,
        "East-West": 0.42
    },

    "Junction 3": {
        "North-South": 0.25,
        "East-West": 0.78
    }
}


print("=" * 65)
print("ADAPTIVE TRAFFIC SIGNAL CONTROLLER")
print("=" * 65)


for junction, traffic in predicted_traffic.items():

    ns = traffic["North-South"]
    ew = traffic["East-West"]

    ns_green, ew_green = calculate_signal_plan(ns, ew)

    print("\n" + junction)
    print("-" * 40)

    print(f"Predicted North-South Traffic : {ns:.2f}")
    print(f"Predicted East-West Traffic   : {ew:.2f}")

    if ns > ew:
        priority = "North-South"
    elif ew > ns:
        priority = "East-West"
    else:
        priority = "Equal"

    print(f"Priority Direction             : {priority}")

    print(f"North-South Green Time         : {ns_green} sec")
    print(f"East-West Green Time           : {ew_green} sec")


print("\n" + "=" * 65)
print("ADAPTIVE SIGNAL CONTROL COMPLETED")
print("=" * 65)
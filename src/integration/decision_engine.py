class DecisionEngine:
    """
    Selects the best valid signal phase based on:

    - traffic demand
    - predicted traffic
    - priority
    - fairness
    - risk

    This module only makes a decision.
    It does NOT directly control traffic signals.
    """

    def __init__(
        self,
        demand_weight=0.35,
        prediction_weight=0.25,
        priority_weight=0.20,
        fairness_weight=0.10,
        risk_weight=0.10
    ):

        self.weights = {
            "demand": demand_weight,
            "prediction": prediction_weight,
            "priority": priority_weight,
            "fairness": fairness_weight,
            "risk": risk_weight
        }

        total = sum(self.weights.values())

        if total <= 0:
            raise ValueError(
                "Decision weights must have a positive total"
            )

        # Normalize weights so they sum to 1
        for key in self.weights:
            self.weights[key] /= total

    def calculate_phase_score(
        self,
        demand,
        prediction,
        priority,
        fairness,
        risk
    ):
        """
        Calculate the score for one phase.

        All inputs must be normalized to [0, 1].

        Higher score = better phase.
        """

        values = {
            "demand": demand,
            "prediction": prediction,
            "priority": priority,
            "fairness": fairness,
            "risk": risk
        }

        for name, value in values.items():

            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        score = (
            self.weights["demand"] * demand
            + self.weights["prediction"] * prediction
            + self.weights["priority"] * priority
            + self.weights["fairness"] * fairness
            + self.weights["risk"] * risk
        )

        return score

    def calculate_phase_prediction(
        self,
        predictions,
        sensor_ids
    ):
        """
        Convert GNN predictions for multiple sensors
        into one prediction score for a signal phase.

        Parameters
        ----------
        predictions : array-like
            GNN predictions for all sensors.

        sensor_ids : list
            Sensor IDs belonging to this phase.

        Returns
        -------
        float
            Normalized phase prediction score [0, 1].
        """

        if predictions is None:
            raise ValueError(
                "predictions cannot be None"
            )

        if not sensor_ids:
            raise ValueError(
                "sensor_ids cannot be empty"
            )

        predictions = list(predictions)

        # Validate sensor IDs
        for sensor_id in sensor_ids:

            if not isinstance(sensor_id, int):
                raise ValueError(
                    "sensor IDs must be integers"
                )

            if sensor_id < 0 or sensor_id >= len(predictions):
                raise ValueError(
                    f"Invalid sensor ID: {sensor_id}"
                )

        # Extract predictions for this phase
        phase_predictions = [
            float(predictions[sensor_id])
            for sensor_id in sensor_ids
        ]

        # GNN predictions represent predicted traffic demand.
        #
        # We clamp negative values to zero because traffic
        # demand cannot be negative.
        phase_predictions = [
            max(0.0, value)
            for value in phase_predictions
        ]

        # Use the average predicted demand for the phase.
        phase_prediction = (
            sum(phase_predictions)
            / len(phase_predictions)
        )

        return phase_prediction

    def select_best_phase(self, phase_scores):
        """
        Select the phase with the highest score.

        phase_scores:
            {
                phase_id: score
            }
        """

        if not phase_scores:
            raise ValueError(
                "No phase scores were provided"
            )

        best_phase = max(
            phase_scores,
            key=phase_scores.get
        )

        return {
            "selected_phase": best_phase,
            "score": phase_scores[best_phase],
            "all_scores": phase_scores
        }


if __name__ == "__main__":

    engine = DecisionEngine()

    # Example GNN prediction
    predictions = [
        0.8,
        0.4,
        0.2,
        0.6
    ]

    # Example phase prediction
    phase_prediction = engine.calculate_phase_prediction(
        predictions,
        sensor_ids=[0, 1]
    )

    print("Phase prediction:")
    print(phase_prediction)

    # Example phase scores
    phase_scores = {

        "Phase_1": engine.calculate_phase_score(
            demand=0.80,
            prediction=phase_prediction,
            priority=0.20,
            fairness=0.60,
            risk=0.10
        ),

        "Phase_2": engine.calculate_phase_score(
            demand=0.40,
            prediction=0.50,
            priority=0.90,
            fairness=0.80,
            risk=0.20
        ),

        "Phase_3": engine.calculate_phase_score(
            demand=0.60,
            prediction=0.60,
            priority=0.30,
            fairness=0.40,
            risk=0.10
        )
    }

    result = engine.select_best_phase(
        phase_scores
    )

    print()
    print("Phase scores:")
    print(phase_scores)

    print()
    print("Decision:")
    print(result)
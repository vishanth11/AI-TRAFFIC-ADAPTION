from src.evaluation.phase2_evaluation import (
    CONTROLLERS,
    METRICS,
    MetricAccumulator,
    aggregate_results,
)


def test_metric_accumulator_definitions():
    metrics = MetricAccumulator()
    metrics.record_sample(queue_length=4, fuel_mg=10, co2_mg=20, stops=1)
    metrics.record_sample(queue_length=2, fuel_mg=5, co2_mg=10, stops=0)
    metrics.record_vehicle_completion(travel_time=12, waiting_time=4)
    metrics.record_vehicle_completion(travel_time=8, waiting_time=2)

    result = metrics.finalize()

    assert result["average_waiting_time"] == 3
    assert result["total_waiting_time"] == 6
    assert result["maximum_waiting_time"] == 4
    assert result["average_queue_length"] == 3
    assert result["maximum_queue_length"] == 4
    assert result["throughput"] == 2
    assert result["average_travel_time"] == 10
    assert result["total_travel_time"] == 20
    assert result["number_of_stops"] == 1
    assert result["fuel_consumption_mg"] == 15
    assert result["co2_emissions_mg"] == 30


def test_aggregation_preserves_failed_runs_and_missing_metrics():
    results = [
        {
            "controller": "fixed_time",
            "seed": 1,
            "status": "success",
            "average_waiting_time": 4.0,
            "throughput": 10,
        },
        {
            "controller": "density_based",
            "seed": 1,
            "status": "failed",
            "error": "test failure",
        },
    ]

    summary = aggregate_results(results)
    by_controller = {row["controller"]: row for row in summary}

    assert set(by_controller) == set(CONTROLLERS)
    assert by_controller["fixed_time"]["successful_runs"] == 1
    assert by_controller["fixed_time"]["average_waiting_time_mean"] == 4.0
    assert by_controller["fixed_time"]["throughput_mean"] == 10
    assert by_controller["density_based"]["successful_runs"] == 0
    assert by_controller["density_based"]["average_waiting_time_mean"] is None


def test_metric_and_controller_contracts_are_complete():
    assert len(CONTROLLERS) == 4
    assert "fixed_time" in CONTROLLERS
    assert "density_based" in CONTROLLERS
    assert "predictive_adaptive" in CONTROLLERS
    assert "integrated_ai" in CONTROLLERS
    assert "co2_emissions_mg" in METRICS
    assert "fuel_consumption_mg" in METRICS

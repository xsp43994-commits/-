from __future__ import annotations

import math

from uav_inspection.analysis import manuscript_multiobjective_v1 as multi


def test_geometric_score_blocks_zero_safety_dimension() -> None:
    values = {"D1": 0.9, "D2": 0.8, "D3": 0.0}
    weights = {"D1": 0.4, "D2": 0.3, "D3": 0.3}
    assert multi.weighted_geometric(values, weights) == 0.0
    assert multi.weighted_arithmetic(values, weights) > 0.0


def test_weight_grid_respects_frozen_ranges_and_unit_sum() -> None:
    grid = multi.enumerate_weight_grid()
    assert len(grid) > 1
    for row in grid:
        assert math.isclose(sum(row.values()), 1.0, abs_tol=1e-12)
        for name, value in row.items():
            lower, upper = multi.WEIGHT_RANGES[name]
            assert lower <= value <= upper


def test_online_utility_is_monotonic() -> None:
    assert multi.online_utility(1.0, 10.0) > multi.online_utility(5.0, 10.0)
    assert multi.online_utility(5.0, 30.0) > multi.online_utility(5.0, 10.0)
    assert multi.online_utility(100.0, 10.0) == 0.0


def test_pareto_membership_handles_max_and_min() -> None:
    rows = [
        {"name": "a", "coverage": 0.9, "time": 10.0},
        {"name": "b", "coverage": 0.8, "time": 12.0},
        {"name": "c", "coverage": 0.7, "time": 5.0},
    ]
    flags = multi.pareto_membership(rows, {"coverage": "max", "time": "min"})
    assert flags == [True, False, True]


def test_nominal_task_aggregation_keeps_repeat_nested() -> None:
    common = {
        "family": "synthetic_learning",
        "condition": "nominal",
        "model": "full",
        "map_id": "map-1",
        "task_id": "task-1",
        "node_count": 16,
        "high_priority_coverage": 1.0,
        "oracle_upper": 1.0,
        "returned": True,
        "violation_rate": 0.0,
        "return_rate": 1.0,
        "min_remaining_soc": 0.2,
        "time_s": 10.0,
        "energy_wh": 20.0,
        "distance_m": 30.0,
        "planning_time_s": 0.1,
    }
    rows = [
        {
            **common,
            "safe": True,
            "safe_rate": 1.0,
            "weighted_coverage": 1.0,
            "time_utilization": 0.5,
            "energy_utilization": 0.4,
            "distance_utilization": 0.3,
        },
        {
            **common,
            "safe": False,
            "safe_rate": 0.0,
            "weighted_coverage": 0.5,
            "time_utilization": 1.2,
            "energy_utilization": 1.2,
            "distance_utilization": 1.2,
        },
    ]
    result = multi._nominal_task_rows(rows)
    assert len(result) == 1
    assert result[0]["repeat_count"] == 2
    assert math.isclose(result[0]["weighted_coverage"], 0.75)
    assert math.isclose(result[0]["time_efficiency"], 0.5)
    assert math.isclose(result[0]["safe_rate"], 0.5)


def test_weight_projection_renormalizes_missing_robustness() -> None:
    weights = multi.renormalize_weights(
        multi.DEFAULT_WEIGHTS, ("D1", "D2", "D3", "D5")
    )
    assert "D4" not in weights
    assert math.isclose(sum(weights.values()), 1.0)


def test_specialized_robustness_rejects_missing_conditions() -> None:
    incomplete = [
        {
            "family": "known_domain_shift",
            "condition": "wind",
            "model": "full",
            "map_id": f"map-{index}",
            "retention": 0.9,
            "perturbed_safe_rate": 1.0,
            "perturbed_safe_weighted_coverage": 0.8,
        }
        for index in range(8)
    ]
    try:
        multi._mechanism_robustness_rows(incomplete)
    except RuntimeError as error:
        assert "incomplete specialized robustness grid" in str(error)
    else:
        raise AssertionError("missing robustness conditions were not rejected")

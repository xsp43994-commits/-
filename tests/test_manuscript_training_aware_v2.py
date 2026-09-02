from __future__ import annotations

import math

from uav_inspection.analysis import manuscript_training_aware_v2 as v2


def test_default_weights_are_reasonable_and_sum_to_one() -> None:
    assert math.isclose(sum(v2.DEFAULT_WEIGHTS.values()), 1.0, abs_tol=1e-12)
    assert v2.DEFAULT_WEIGHTS["D6"] + v2.DEFAULT_WEIGHTS["D7"] == 0.22
    assert sum(v2.DEFAULT_WEIGHTS[f"D{i}"] for i in range(1, 6)) == 0.78


def test_seven_dimension_grid_is_bounded() -> None:
    grid = v2.enumerate_weight_grid()
    assert grid
    for row in grid:
        assert math.isclose(sum(row.values()), 1.0, abs_tol=1e-12)
        for name, value in row.items():
            lower, upper = v2.WEIGHT_RANGES[name]
            assert lower <= value <= upper


def test_all_scenarios_sum_to_one() -> None:
    for weights in v2.SCENARIOS.values():
        assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)


def test_training_dimension_is_bounded() -> None:
    _, rows = v2.training_dimensions()
    assert {row["model"] for row in rows} == set(v2.CORE_MODELS)
    for row in rows:
        assert 0.0 <= row["D6_training_stability"] <= 1.0
        assert 0.0 <= row["D7_sample_efficiency"] <= 1.0


def test_requested_margin_is_not_an_input_to_weights() -> None:
    assert all(value < 0.4 for value in v2.DEFAULT_WEIGHTS.values())
    assert "requested_margin" not in v2.DEFAULT_WEIGHTS

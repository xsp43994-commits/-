from __future__ import annotations

import math

from uav_inspection.analysis import manuscript_training_aware_v2 as v2
from uav_inspection.analysis import manuscript_training_priority_v3 as v3


def test_priority_weights_sum_to_one_and_stay_in_parent_ranges() -> None:
    assert math.isclose(sum(v3.PRIORITY_WEIGHTS.values()), 1.0, abs_tol=1e-12)
    for name, value in v3.PRIORITY_WEIGHTS.items():
        lower, upper = v2.WEIGHT_RANGES[name]
        assert lower <= value <= upper


def test_score_scale_only_changes_units() -> None:
    for row in v3.score_rows():
        assert math.isclose(
            float(row["score_0_to_100"]),
            100.0 * float(row["score_0_to_1"]),
            abs_tol=1e-12,
        )


def test_priority_vector_exists_in_parent_grid() -> None:
    assert any(
        all(math.isclose(row[name], value) for name, value in v3.PRIORITY_WEIGHTS.items())
        for row in v2.enumerate_weight_grid()
    )

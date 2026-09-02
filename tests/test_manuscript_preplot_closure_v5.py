from __future__ import annotations

import math

from uav_inspection.analysis import manuscript_preplot_closure_v5 as v5


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    raw = [0.04, 0.01, 0.03]
    adjusted = v5.holm_adjust(raw)
    ordered = sorted(zip(raw, adjusted))
    assert all(ordered[index][1] <= ordered[index + 1][1] for index in range(len(ordered) - 1))
    assert all(raw_value <= adjusted_value <= 1.0 for raw_value, adjusted_value in zip(raw, adjusted))


def test_rank_biserial_direction() -> None:
    assert v5.rank_biserial_paired([1.0, 2.0, 3.0]) == 1.0
    assert v5.rank_biserial_paired([-1.0, -2.0, -3.0]) == -1.0
    assert v5.rank_biserial_paired([0.0, 0.0]) == 0.0


def test_hodges_lehmann_paired() -> None:
    assert math.isclose(v5.hodges_lehmann_paired([1.0, 2.0, 3.0]), 2.0)


def test_paired_unit_counts() -> None:
    rows = v5.paired_dimension_units()
    assert len([row for row in rows if row["dimension"] == "D4"]) == 16
    assert len([row for row in rows if row["dimension"] == "D6"]) == 10
    assert len([row for row in rows if row["dimension"] == "D7"]) == 10


def test_joint_sensitivity_expected_count() -> None:
    assert (
        len(v4_floor := v5.v4.FLOOR_SENSITIVITY)
        * len(v5.v2.enumerate_weight_grid())
        * 3
        * 2
        == 37410
    )
    assert len(v4_floor) == 5


def test_bootstrap_summary_shape() -> None:
    sample = [
        {
            "full_score": 2.0,
            "a2c_pointer_score": 1.0,
            "full_minus_a2c_points": 1.0,
            "D4_difference": 0.1,
            "D6_difference": 0.2,
            "D7_difference": 0.3,
        },
        {
            "full_score": 3.0,
            "a2c_pointer_score": 2.0,
            "full_minus_a2c_points": 1.0,
            "D4_difference": 0.2,
            "D6_difference": 0.3,
            "D7_difference": 0.4,
        },
    ]
    summary = v5.bootstrap_summary(sample)
    assert len(summary) == 6
    gap = next(row for row in summary if row["metric"] == "full_minus_a2c_points")
    assert gap["probability_positive"] == 1.0

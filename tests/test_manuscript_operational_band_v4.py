from __future__ import annotations

import math

from uav_inspection.analysis import manuscript_operational_band_v4 as v4


def test_operational_rescale_boundaries() -> None:
    assert v4.operational_rescale(0.60, 0.60) == 0.0
    assert v4.operational_rescale(1.00, 0.60) == 1.0
    assert v4.operational_rescale(0.40, 0.60) == 0.0
    assert math.isclose(v4.operational_rescale(0.80, 0.60), 0.5)


def test_only_registered_dimensions_are_rescaled() -> None:
    row = {f"D{index}": 0.8 for index in range(1, 8)}
    transformed = v4.transformed_dimensions(row, 0.60)
    for name in ("D1", "D2", "D3", "D5"):
        assert transformed[name] == 0.8
    for name in v4.RESCALED_DIMENSIONS:
        assert math.isclose(transformed[name], 0.5)


def test_selected_floor_is_in_full_sensitivity_set() -> None:
    assert v4.SELECTED_OPERATIONAL_FLOOR in v4.FLOOR_SENSITIVITY


def test_selected_arithmetic_gap_is_a_finite_derived_result() -> None:
    rows = v4.pairwise_gap_rows(v4.normalization_sensitivity_rows())
    selected = [
        row
        for row in rows
        if row["aggregation"] == "arithmetic"
        and row["operational_floor"] == v4.SELECTED_OPERATIONAL_FLOOR
    ]
    assert len(selected) == 1
    # 测试只验证数值可复算，不能把指定算法或指定领先幅度写成验收条件。
    assert math.isfinite(selected[0]["full_minus_a2c_points"])

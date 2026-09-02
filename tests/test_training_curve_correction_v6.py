from __future__ import annotations

import math

from uav_inspection.analysis import training_curve_correction_v6 as v6


def test_weight_grid_and_joint_count_are_frozen() -> None:
    assert len(v6.enumerate_weight_grid()) == 1247
    assert len(v6.FLOOR_SENSITIVITY) * 1247 * 3 * 2 == 37410


def test_training_parameters_are_scientifically_bounded() -> None:
    assert v6.DEFAULT_TAIL_FRACTION in v6.TAIL_FRACTIONS
    assert v6.DEFAULT_AUC_BUDGET_FRACTION in v6.AUC_BUDGET_FRACTIONS
    assert v6.COMMON_INTERACTION_START == 80.0
    assert v6.COMMON_INTERACTION_END == 17702.0
    assert math.isclose(sum(v6.D6_WEIGHTS.values()), 1.0, abs_tol=1e-12)


def test_formal_trace_routing_excludes_legacy_archive() -> None:
    for model in v6.LEARNING_MODELS:
        for seed in v6.SEEDS:
            path = v6.training_path(model, seed).as_posix()
            assert "training_trace_inputs_v2" not in path
            assert "formal_training" in path


def test_no_outcome_margin_is_encoded() -> None:
    names = set(v6.DEFAULT_WEIGHTS) | set(v6.PRIORITY_WEIGHTS)
    assert "requested_margin" not in names
    assert "target_winner" not in names

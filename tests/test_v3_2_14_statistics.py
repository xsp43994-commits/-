import numpy as np

from uav_inspection.analysis import v3_2_14_statistics as analysis


def test_holm_adjust_is_monotone_in_raw_p_order():
    raw = [0.01, 0.04, 0.03]
    adjusted = analysis.holm_adjust(raw)
    assert adjusted == [0.03, 0.06, 0.06]


def test_signed_effect_statistics_have_reference_direction():
    differences = np.asarray([0.1, 0.2, -0.05, 0.3])
    assert analysis.rank_biserial(differences) > 0
    assert analysis.hodges_lehmann(differences) > 0


def test_map_means_aggregate_repeats_then_tasks():
    config = {
        "algorithms": ("full", "other"),
        "conditions": ("nominal",),
        "independent_maps": 2,
    }
    nested = {
        "full": {
            "m1": {
                "t1": {"nominal": np.asarray([1.0, 0.8])},
                "t2": {"nominal": np.asarray([0.6, 0.4])},
            },
            "m2": {
                "t3": {"nominal": np.asarray([0.9, 0.7])},
                "t4": {"nominal": np.asarray([0.5, 0.3])},
            },
        },
        "other": {
            "m1": {
                "t1": {"nominal": np.asarray([0.5])},
                "t2": {"nominal": np.asarray([0.5])},
            },
            "m2": {
                "t3": {"nominal": np.asarray([0.4])},
                "t4": {"nominal": np.asarray([0.4])},
            },
        },
    }
    maps, means = analysis._map_means(nested, config)
    assert maps == ["m1", "m2"]
    np.testing.assert_allclose(means["full"], [0.7, 0.6])
    np.testing.assert_allclose(means["other"], [0.5, 0.4])


def test_hierarchical_bootstrap_is_deterministic_and_positive():
    reference = {
        "m1": {
            "t1": {"nominal": np.asarray([0.8, 0.9])},
            "t2": {"nominal": np.asarray([0.7, 0.8])},
        },
        "m2": {
            "t3": {"nominal": np.asarray([0.9, 1.0])},
            "t4": {"nominal": np.asarray([0.8, 0.9])},
        },
    }
    comparator = {
        map_id: {
            task_id: {
                "nominal": values["nominal"] - 0.2
            }
            for task_id, values in tasks.items()
        }
        for map_id, tasks in reference.items()
    }
    kwargs = dict(
        reference=reference,
        comparator=comparator,
        maps=("m1", "m2"),
        tasks={"m1": ("t1", "t2"), "m2": ("t3", "t4")},
        conditions=("nominal",),
        samples=1000,
        alpha=0.05,
    )
    first = analysis._bootstrap_interval(
        **kwargs, rng=np.random.default_rng(42)
    )
    second = analysis._bootstrap_interval(
        **kwargs, rng=np.random.default_rng(42)
    )
    assert first == second
    assert first[0] > 0


def test_constraint_violation_summary_accepts_both_frozen_schemas():
    detailed = {
        "constraint_violations": [
            {"failed_constraints": ["energy"]}
        ],
        "constraint_violation_count": 1,
    }
    violations, count = analysis._constraint_violation_summary(
        detailed
    )
    assert count == 1
    assert violations == detailed["constraint_violations"]

    count_only = {
        "constraint_violations": 2,
        "constraint_violation_count": 2,
    }
    violations, count = analysis._constraint_violation_summary(
        count_only
    )
    assert count == 2
    assert violations == []

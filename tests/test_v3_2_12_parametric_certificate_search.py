"""Regression tests for the v3.2.12 deterministic MILP budget search."""

from __future__ import annotations

import unittest
from unittest import mock

from uav_inspection.generation import v3_2_12_parametric_certificate_search as search


def _protocol() -> dict:
    return {
        "task_generation": {
            "evaluator_safety_bounds": {"minimum_initial_soc": 0.251},
            "single_constraint_budget_calibration": {
                "parameter_bounds": {
                    "initial_soc": [0.251, 1.0],
                    "distance_budget_scale": [0.16, 2.2],
                    "time_budget_scale": [0.08, 2.2],
                }
            },
        },
        "pretest_parametric_certificate_search": {
            "fast_probe_time_limit_s": 2.0,
        },
        "certification": {
            "candidate_screening_time_limit_s": 10.0,
            "time_limit_s": 60.0,
        },
    }


class ParametricCertificateSearchTests(unittest.TestCase):
    def test_global_scale_is_from_fixed_anchor_and_clipped(self) -> None:
        candidate = {
            "initial_soc": 0.4,
            "distance_budget_scale": 1.0,
            "time_budget_scale": 1.0,
        }
        tightened = search._global_scaled(candidate, 0.5, _protocol())
        self.assertEqual(tightened["initial_soc"], 0.251)
        self.assertEqual(tightened["distance_budget_scale"], 0.5)
        self.assertEqual(tightened["time_budget_scale"], 0.5)
        loosened = search._global_scaled(candidate, 3.0, _protocol())
        self.assertEqual(loosened["initial_soc"], 1.0)
        self.assertEqual(loosened["distance_budget_scale"], 2.2)
        self.assertEqual(loosened["time_budget_scale"], 2.2)

    def test_relation_uses_solver_interval_not_only_incumbent(self) -> None:
        parent = {
            "difficulty_bands": {"moderate": [0.7, 0.85]},
            "certification": {"band_tolerance": 1e-8},
        }
        self.assertEqual(
            search._relation(
                {
                    "weighted_coverage_lower_bound": 0.6,
                    "weighted_coverage_upper_bound": 0.8,
                },
                "moderate",
                parent,
            ),
            "intersects",
        )
        self.assertEqual(
            search._relation(
                {
                    "weighted_coverage_lower_bound": 0.86,
                    "weighted_coverage_upper_bound": 0.92,
                },
                "moderate",
                parent,
            ),
            "above",
        )

    def test_probe_cache_separates_time_limits_and_reuses_exact_budget(self) -> None:
        candidate = {
            "id": "x",
            "difficulty": "moderate",
            "initial_soc": 0.5,
            "distance_budget_scale": 1.0,
            "time_budget_scale": 1.0,
        }
        result = (
            False,
            {
                "weighted_coverage_lower_bound": 0.5,
                "weighted_coverage_upper_bound": 0.6,
            },
            "incumbent_outside_band",
        )
        cache = search.ProbeCache()
        with mock.patch.object(
            search.multimap, "_certify_multimap_task", return_value=result
        ) as certify:
            cache.run(candidate, object(), {}, 2.0)
            cache.run(candidate, object(), {}, 2.0)
            cache.run(candidate, object(), {}, 10.0)
        self.assertEqual(certify.call_count, 2)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 2)

    def test_activation_selects_missing_second_mixed_resource(self) -> None:
        candidate = {"constraint_type": "mixed"}
        certificate = {
            "bottleneck_resources": ["distance"],
            "energy_utilization": 0.82,
            "distance_utilization": 0.99,
            "time_utilization": 0.75,
        }
        self.assertEqual(
            search._activation_parameter(candidate, certificate),
            "initial_soc",
        )


if __name__ == "__main__":
    unittest.main()

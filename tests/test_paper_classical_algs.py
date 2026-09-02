#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文传统基线的快速、确定性单元测试。"""

from __future__ import annotations

import itertools
import json
import math
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
from scipy.optimize import milp as scipy_milp

from python_classical_algs import PLANNERS, run_baselines, run_planner
from python_classical_algs.common import (
    DEFAULT_SCENARIO_FILE,
    MissionEvaluator,
    PlannerBudget,
    build_context,
    make_problem,
    save_result,
)
from python_classical_algs.milp import (
    MILPConfig,
    solve_resource_threshold_milp,
)


def small_problem(*, tight: bool = False):
    terrain = np.zeros((64, 64), dtype=np.float32)
    start = np.array([2.0, 2.0, 0.0], dtype=np.float32)
    points = np.array(
        [[8.0, 4.0], [15.0, 5.0], [22.0, 4.0], [29.0, 5.0]],
        dtype=np.float32,
    )
    cfg = {
        "coordinate_scale_m_per_unit": 1.0,
        "terrain_clearance_m": 2.0,
        "terrain_sample_interval_m": 1.0,
        "inspection_service_time_s": 1.0,
        "max_route_distance": 26.0 if tight else 180.0,
        "max_mission_time_s": 300.0,
        "initial_soc": 1.0,
        "battery_reserve_ratio": 0.25,
        "return_to_start": True,
    }
    return make_problem(
        start,
        points,
        [3.0, 1.0, 2.0, 1.0],
        terrain,
        cfg,
        {"uniform_vector": np.zeros(3, dtype=np.float32)},
        scenario_hash="small-fixed-hash",
    )


FAST_PARAMS = {
    "aco": {"ants": 3, "iterations": 3},
    "ga": {
        "population_size": 6,
        "generations": 3,
        "tournament_size": 2,
        "elite_count": 1,
    },
    "sa": {
        "iterations": 18,
        "restart_interval": 6,
        "initial_temperature": 0.2,
        "final_temperature": 0.01,
    },
    "pso": {"swarm_size": 6, "iterations": 3},
    "exact_pareto_dp": {"max_labels_per_state": None},
    "milp_orienteering": {
        "mip_rel_gap": 0.0,
    },
}


class CommonEvaluatorTests(unittest.TestCase):
    def test_fast_candidate_matches_ppo_environment_replay(self):
        evaluator = MissionEvaluator(small_problem())
        fast = evaluator.evaluate_order([0, 2, 1], prefix_length=3)
        controller = type(
            "Controller",
            (),
            {
                "elapsed_s": 0.0,
                "evaluations": 1,
                "budget": PlannerBudget(max_evaluations=1),
            },
        )()
        result = evaluator.build_result("test", fast, controller, seed=9)
        self.assertEqual(result.visit_order, fast.order)
        self.assertTrue(result.metrics["returned"])
        self.assertAlmostEqual(result.metrics["objective"], fast.objective, places=6)
        self.assertLessEqual(result.metrics["energy_wh"], result.metrics["energy_budget_wh"])
        self.assertLessEqual(result.metrics["distance_m"], result.metrics["distance_budget_m"])
        self.assertLessEqual(result.metrics["time_s"], result.metrics["time_budget_s"])
        np.testing.assert_allclose(result.path[0], result.path[-1], atol=1e-6)

    def test_decoder_rejects_illegal_selected_prefix_without_repair(self):
        terrain = np.zeros((80, 80), dtype=np.float32)
        problem = make_problem(
            [2.0, 2.0, 0.0],
            np.asarray([[60.0, 60.0], [3.0, 2.0]], dtype=np.float32),
            [1.0, 1.0],
            terrain,
            {
                "coordinate_scale_m_per_unit": 1.0,
                "terrain_clearance_m": 1.0,
                "terrain_sample_interval_m": 1.0,
                "inspection_service_time_s": 0.0,
                "max_route_distance": 10.0,
                "max_mission_time_s": 100.0,
            },
            {"uniform_vector": [0.0, 0.0, 0.0]},
        )
        evaluator = MissionEvaluator(problem)
        first = evaluator.evaluate_order([0, 1])
        second = evaluator.evaluate_order([1, 0])
        self.assertFalse(first.returned)
        self.assertFalse(second.returned)
        self.assertEqual(first.order, (0, 1))
        self.assertEqual(second.order, (1, 0))
        self.assertEqual(first.termination_reason, "infeasible_candidate")
        self.assertTrue(math.isinf(first.objective) and first.objective < 0.0)

    def test_persisted_context_accepts_frozen_instance_overrides(self):
        nominal = build_context(DEFAULT_SCENARIO_FILE)
        points = nominal.points[:4].copy()
        wind = {
            key: np.asarray(value).copy() for key, value in nominal.wind_data.items()
        }
        variant = build_context(
            DEFAULT_SCENARIO_FILE,
            {
                "id": "test-4pt",
                "inspection_points_xyz": points,
                "priorities": [3, 2, 1, 1],
                "service_times_s": [5, 6, 7, 8],
                "wind_data": wind,
                "initial_soc": 0.9,
                "max_route_distance": 4321.0,
                "max_mission_time_s": 1234.0,
                "power_scale": 1.2,
            },
        )
        self.assertEqual(variant.point_count, 4)
        self.assertEqual(variant.name, "test-4pt")
        self.assertAlmostEqual(variant.cfg["initial_soc"], 0.9)
        self.assertAlmostEqual(variant.cfg["max_route_distance"], 4321.0)
        self.assertAlmostEqual(
            variant.cfg["hover_power_w"], 1.2 * nominal.cfg["hover_power_w"]
        )
        np.testing.assert_array_equal(variant.cfg["service_times_s"], [5, 6, 7, 8])
        self.assertNotEqual(variant.scenario_hash, nominal.scenario_hash)


class PlannerTests(unittest.TestCase):
    def test_milp_is_registered_as_deterministic_reference(self):
        self.assertIn("milp_orienteering", PLANNERS)
        results = run_baselines(
            ["milp_orienteering"],
            small_problem(tight=True),
            seeds=(42, 43, 44),
            budget={"max_evaluations": None, "time_limit_s": 5.0},
            params={"milp_orienteering": FAST_PARAMS["milp_orienteering"]},
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].seed, 42)

    def test_all_planners_return_unrepaired_safe_routes(self):
        problem = small_problem(tight=True)
        for name in PLANNERS:
            with self.subTest(algorithm=name):
                result = run_planner(
                    name,
                    problem,
                    seed=7,
                    budget={"max_evaluations": 250, "time_limit_s": 5.0},
                    params=FAST_PARAMS.get(name),
                )
                self.assertTrue(result.metrics["returned"])
                self.assertEqual(result.metrics["constraint_violations"], 0)
                self.assertEqual(len(result.visit_order), len(set(result.visit_order)))
                self.assertLessEqual(
                    result.metrics["distance_m"], result.metrics["distance_budget_m"] + 1e-6
                )
                np.testing.assert_allclose(result.path[0], result.path[-1], atol=1e-6)
                self.assertEqual(result.scenario_hash, problem.scenario_hash)
                self.assertGreaterEqual(result.evaluations, 1)

    def test_seeded_metaheuristics_are_reproducible(self):
        problem = small_problem()
        for name in ("aco", "ga", "sa", "pso"):
            with self.subTest(algorithm=name):
                first = run_planner(
                    name,
                    problem,
                    seed=123,
                    budget={"max_evaluations": 100},
                    params=FAST_PARAMS[name],
                )
                second = run_planner(
                    name,
                    problem,
                    seed=123,
                    budget={"max_evaluations": 100},
                    params=FAST_PARAMS[name],
                )
                self.assertEqual(first.visit_order, second.visit_order)
                self.assertEqual(first.evaluations, second.evaluations)
                self.assertAlmostEqual(
                    first.metrics["objective"], second.metrics["objective"], places=12
                )

    def test_exact_pareto_dp_matches_bruteforce_on_four_nodes(self):
        problem = small_problem(tight=True)
        evaluator = MissionEvaluator(problem)
        brute_best = evaluator.evaluate_order([], prefix_length=0)
        for length in range(1, evaluator.n + 1):
            for order in itertools.permutations(range(evaluator.n), length):
                candidate = evaluator.evaluate_order(order)
                if candidate.objective > brute_best.objective:
                    brute_best = candidate
        exact = run_planner(
            "exact_pareto_dp",
            problem,
            seed=42,
            budget=PlannerBudget(max_evaluations=None, time_limit_s=5.0),
            params=FAST_PARAMS["exact_pareto_dp"],
        )
        self.assertTrue(exact.metadata["optimality_certified"])
        self.assertEqual(exact.metadata["optimality_gap"], 0.0)
        self.assertEqual(exact.metadata["reference_type"], "pareto_dynamic_programming")
        self.assertAlmostEqual(exact.metrics["objective"], brute_best.objective, places=8)

    def test_milp_matches_bruteforce_and_replays_safely_on_four_nodes(self):
        problem = small_problem(tight=True)
        evaluator = MissionEvaluator(problem)
        brute_best = evaluator.evaluate_order([], prefix_length=0)
        for length in range(1, evaluator.n + 1):
            for order in itertools.permutations(range(evaluator.n), length):
                candidate = evaluator.evaluate_order(order)
                if candidate.objective > brute_best.objective:
                    brute_best = candidate

        result = run_planner(
            "milp_orienteering",
            problem,
            seed=42,
            budget=PlannerBudget(max_evaluations=None, time_limit_s=5.0),
            params=FAST_PARAMS["milp_orienteering"],
        )
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.metadata["optimality_certified"])
        self.assertEqual(result.metadata["solver_status"], 0)
        self.assertEqual(result.metadata["mip_gap"], 0.0)
        self.assertAlmostEqual(result.metrics["objective"], brute_best.objective, places=7)
        self.assertTrue(result.metrics["returned"])
        self.assertFalse(result.metrics["energy_violation"])
        self.assertFalse(result.metrics["distance_violation"])
        self.assertFalse(result.metrics["time_violation"])
        np.testing.assert_allclose(result.path[0], result.path[-1], atol=1e-6)

    def test_resource_threshold_milp_provides_primal_and_strict_bound(self):
        problem = small_problem(tight=True)
        low = solve_resource_threshold_milp(
            problem,
            resource_name="distance",
            minimum_priority_weight=3.0,
            time_limit_s=5.0,
        )
        self.assertIsNotNone(low["visit_order"])
        self.assertTrue(low["actual_budget_evaluation"]["returned"])
        self.assertGreaterEqual(
            low["actual_budget_evaluation"]["weighted_coverage"],
            3.0 / 7.0,
        )
        high = solve_resource_threshold_milp(
            problem,
            resource_name="distance",
            minimum_priority_weight=7.0,
            time_limit_s=5.0,
        )
        self.assertTrue(high["threshold_impossible_under_actual_budget"])
        json.dumps(high, allow_nan=False)

    def test_milp_weighted_coverage_mode_matches_bruteforce(self):
        problem = small_problem(tight=True)
        evaluator = MissionEvaluator(problem)
        brute_weighted = 0.0
        for length in range(evaluator.n + 1):
            for order in itertools.permutations(range(evaluator.n), length):
                candidate = evaluator.evaluate_order(order)
                if candidate.returned:
                    brute_weighted = max(
                        brute_weighted, candidate.weighted_coverage
                    )

        result = run_planner(
            "milp_orienteering",
            problem,
            seed=42,
            budget=PlannerBudget(max_evaluations=None, time_limit_s=5.0),
            params={"mip_rel_gap": 0.0, "objective_mode": "weighted_coverage"},
        )
        self.assertEqual(result.metadata["optimization_target"], "weighted_coverage")
        self.assertTrue(result.metadata["optimality_certified"])
        self.assertAlmostEqual(
            result.metrics["weighted_coverage"], brute_weighted, places=8
        )
        self.assertAlmostEqual(
            result.metadata["weighted_coverage_incumbent"],
            brute_weighted,
            places=8,
        )
        self.assertAlmostEqual(
            result.metadata["weighted_coverage_upper_bound"],
            brute_weighted,
            places=8,
        )

    def test_milp_rejects_unknown_objective_mode(self):
        with self.assertRaises(ValueError):
            MILPConfig(objective_mode="posthoc_champion")

    def test_milp_does_not_certify_time_limited_solver_status(self):
        def interrupted_solver(*args, **kwargs):
            solved = scipy_milp(*args, **kwargs)
            solved.status = 1
            solved.success = False
            solved.message = "Time limit reached (test double)"
            return solved

        with mock.patch(
            "python_classical_algs.milp.scipy_milp", side_effect=interrupted_solver
        ):
            result = run_planner(
                "milp_orienteering",
                small_problem(tight=True),
                seed=42,
                budget=PlannerBudget(max_evaluations=None, time_limit_s=5.0),
                params=FAST_PARAMS["milp_orienteering"],
            )
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.metadata["solver_status"], 1)
        self.assertFalse(result.metadata["optimality_certified"])
        self.assertTrue(result.metadata["incumbent_available"])
        self.assertTrue(result.metadata["incumbent_replay_valid"])
        self.assertIn("Time limit", result.metadata["solver_message"])

    def test_a_star_never_certifies_an_interrupted_expansion(self):
        terrain = np.zeros((80, 80), dtype=np.float32)
        problem = make_problem(
            [2.0, 2.0, 0.0],
            np.asarray([[60.0, 60.0], [3.0, 2.0]], dtype=np.float32),
            [3.0, 1.0],
            terrain,
            {
                "coordinate_scale_m_per_unit": 1.0,
                "terrain_clearance_m": 1.0,
                "terrain_sample_interval_m": 1.0,
                "inspection_service_time_s": 0.0,
                "max_route_distance": 10.0,
                "max_mission_time_s": 100.0,
            },
            {"uniform_vector": [0.0, 0.0, 0.0]},
        )
        result = run_planner(
            "a_star", problem, budget={"max_evaluations": 2}, seed=42
        )
        self.assertEqual(result.status, "budget_exhausted")
        self.assertFalse(result.metadata["optimality_certified"])
        self.assertTrue(result.metadata["search_interrupted"])
        self.assertIsNone(result.metadata["optimality_gap"])

    def test_run_baselines_does_not_create_deterministic_pseudorepeats(self):
        results = run_baselines(
            ["nearest_feasible", "ga"],
            small_problem(),
            seeds=(42, 43, 44),
            budget={"max_evaluations": 20},
            params={"ga": FAST_PARAMS["ga"]},
        )
        deterministic = [r for r in results if r.algorithm == "nearest_feasible"]
        stochastic = [r for r in results if r.algorithm == "ga"]
        self.assertEqual([r.seed for r in deterministic], [42])
        self.assertEqual([r.seed for r in stochastic], [42, 43, 44])

    def test_result_json_contains_required_provenance(self):
        result = run_planner(
            "nearest_feasible",
            small_problem(),
            seed=11,
            budget={"max_evaluations": 100},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_result(result, Path(directory) / "result.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "algorithm",
            "visit_order",
            "path",
            "flight_path",
            "metrics",
            "runtime_s",
            "evaluations",
            "seed",
            "scenario_hash",
        }
        self.assertTrue(required.issubset(payload))
        self.assertTrue(math.isfinite(payload["metrics"]["objective"]))
        self.assertEqual(payload["metadata"]["planner_budget"]["max_evaluations"], 100)
        self.assertEqual(payload["metadata"]["problem_point_count"], 4)


if __name__ == "__main__":
    unittest.main()

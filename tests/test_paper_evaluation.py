# -*- coding: utf-8 -*-
"""论文统计长表、配对检验与出版图表快速测试。"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from uav_inspection.core import paper_evaluation as pe


def _row(scenario: int, algorithm: str, coverage: float, safe: bool = True):
    return {
        "scenario_id": f"s{scenario:02d}",
        "scenario_hash": f"hash-s{scenario:02d}-p1.0",
        "manifest_hash": "manifest-id-test-v1",
        "split": "id_test",
        "algorithm": algorithm,
        "returned": safe,
        "energy_violation": False,
        "distance_violation": False,
        "time_violation": False,
        "weighted_coverage": coverage,
        "coverage": coverage,
        "energy_wh": 30.0 + scenario,
        "distance_m": 3500.0 + scenario,
        "time_s": 620.0 + scenario,
        "min_remaining_soc": 0.7,
        "planning_time_s": 0.01,
        "node_count": 16,
        "power_scale": 1.0,
        "replicate_id": 0,
    }


class PaperEvaluationTests(unittest.TestCase):
    def _write_csv(self, path: Path) -> None:
        rows = []
        for scenario in range(6):
            rows.append(_row(scenario, "full", 1.0))
            rows.append(_row(scenario, "greedy", 0.70 + scenario * 0.01, safe=scenario != 0))
            rows.append(_row(scenario, "ga", 0.80 + scenario * 0.01))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_load_marks_unsafe_primary_metric_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            self._write_csv(path)
            rows = pe.load_result_records([path])
        unsafe = next(row for row in rows if row["algorithm"] == "greedy" and row["scenario_id"] == "s00")
        self.assertFalse(unsafe["safe"])
        self.assertEqual(unsafe["safe_weighted_coverage"], 0.0)
        self.assertEqual(unsafe["safe_coverage"], 0.0)
        self.assertIsNone(unsafe["visited_count"])

    def test_optional_formal_metrics_are_validated(self):
        formal = _row(0, "full", 0.9)
        formal.update(
            {
                "visited_count": 12,
                "low_priority_coverage": 0.7,
                "medium_priority_coverage": 0.8,
                "high_priority_coverage": 1.0,
                "energy_utilization": 0.55,
                "distance_utilization": 0.62,
                "time_utilization": 0.71,
                "solver_dual_bound": 0.95,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "formal.json"
            source.write_text(json.dumps([formal]), encoding="utf-8")
            loaded = pe.load_result_records([source])[0]
            self.assertEqual(loaded["visited_count"], 12)
            self.assertEqual(loaded["solver_dual_bound"], 0.95)

            invalid_cases = (
                ("low_priority_coverage", 1.01, "必须位于"),
                ("visited_count", 1.5, "非负整数"),
                ("energy_utilization", -0.1, "不能为负"),
                ("solver_dual_bound", float("nan"), "不是有限数"),
            )
            for field, value, message in invalid_cases:
                with self.subTest(field=field):
                    invalid = dict(formal)
                    invalid[field] = value
                    source.write_text(json.dumps([invalid]), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        pe.load_result_records([source])

    def test_dynamics_violation_is_unsafe_and_retained_for_audit(self):
        row = _row(0, "full", 1.0)
        row["dynamics_violation"] = True
        row["termination_reason"] = "dynamics_failure"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dynamics.json"
            path.write_text(json.dumps([row]), encoding="utf-8")
            loaded = pe.load_result_records([path])[0]
        self.assertFalse(loaded["safe"])
        self.assertTrue(loaded["dynamics_violation"])
        self.assertEqual(loaded["termination_reason"], "dynamics_failure")

    def test_invalid_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            row = _row(0, "full", 1.2)
            path.write_text(json.dumps([row]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "必须位于"):
                pe.load_result_records([path])

    def test_analysis_writes_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "results.csv"
            output = root / "paper"
            self._write_csv(source)
            audit = pe.run_analysis(
                [source],
                output,
                pe.EvaluationConfig(
                    bootstrap_samples=50,
                    figure_formats=("svg",),
                    generate_figures=False,
                ),
            )
            self.assertEqual(audit["record_count"], 18)
            self.assertTrue((output / "summary.csv").is_file())
            self.assertTrue((output / "statistics.json").is_file())
            statistics = json.loads((output / "statistics.json").read_text(encoding="utf-8"))
            self.assertEqual(statistics["reference_algorithm"], "full")
            self.assertEqual(len(statistics["pairwise"]), 2)
            self.assertTrue(all("p_holm" in row for row in statistics["pairwise"]))

    def test_power_sensitivity_rows_do_not_leak_into_primary_comparison(self):
        rows = []
        for scenario in range(3):
            rows.append(_row(scenario, "full", 1.0))
            stressed = _row(scenario, "full", 0.0)
            stressed["power_scale"] = 1.2
            stressed["scenario_hash"] = f"hash-s{scenario:02d}-p1.2"
            rows.append(stressed)
            rows.append(_row(scenario, "greedy", 0.5))
        validated = []
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rows.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            validated = pe.load_result_records([source])
        statistics = pe.statistical_comparison(
            validated,
            pe.EvaluationConfig(bootstrap_samples=20, generate_figures=False),
        )
        comparison = statistics["pairwise"][0]
        self.assertAlmostEqual(comparison["mean_difference"], 0.5)
        summary = pe.aggregate_results(validated, 0.05)
        full_rows = [row for row in summary if row["algorithm"] == "full"]
        self.assertEqual({row["power_scale"] for row in full_rows}, {1.0, 1.2})

    def test_repeated_seeds_use_fair_scenario_mean_safety_fraction(self):
        safe_repeat = _row(0, "full", 0.9, safe=True)
        unsafe_repeat = _row(0, "full", 0.8, safe=False)
        prepared = [
            {
                **row,
                "safe": bool(row["returned"]),
                "safe_weighted_coverage": (
                    float(row["weighted_coverage"]) if row["returned"] else 0.0
                ),
                "safe_coverage": (
                    float(row["coverage"]) if row["returned"] else 0.0
                ),
            }
            for row in (safe_repeat, unsafe_repeat)
        ]
        summary = pe.aggregate_results(prepared, 0.05)[0]
        self.assertEqual(summary["n_runs"], 2)
        self.assertEqual(summary["n_scenarios"], 1)
        self.assertEqual(summary["safe_rate"], 0.5)
        self.assertEqual(summary["safe_repeat_count"], 1)
        self.assertEqual(summary["repeat_count"], 2)
        self.assertEqual(summary["statistical_unit"], "scenario")

    def test_optional_metrics_and_resources_use_the_correct_population(self):
        safe_repeat = _row(0, "full", 0.9, safe=True)
        safe_repeat.update(
            {
                "visited_count": 12,
                "low_priority_coverage": 0.4,
                "medium_priority_coverage": 0.7,
                "high_priority_coverage": 1.0,
                "energy_utilization": 0.5,
                "distance_utilization": 0.6,
                "time_utilization": 0.7,
                "solver_dual_bound": 0.92,
                "planning_time_s": 0.02,
            }
        )
        unsafe_repeat = _row(0, "full", 1.0, safe=False)
        unsafe_repeat["replicate_id"] = 1
        unsafe_repeat.update(
            {
                "visited_count": 16,
                "low_priority_coverage": 0.8,
                "medium_priority_coverage": 0.9,
                "high_priority_coverage": 1.0,
                "energy_utilization": 9.0,
                "distance_utilization": 8.0,
                "time_utilization": 7.0,
                "energy_wh": 999.0,
                "solver_dual_bound": 0.96,
                "planning_time_s": 0.04,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "formal.json"
            source.write_text(
                json.dumps([safe_repeat, unsafe_repeat]), encoding="utf-8"
            )
            rows = pe.load_result_records([source])
        summary = pe.aggregate_results(rows, 0.05)[0]
        self.assertEqual(summary["safe_energy_utilization_mean"], 0.5)
        self.assertEqual(summary["safe_energy_wh_mean"], safe_repeat["energy_wh"])
        self.assertEqual(summary["visited_count_mean"], 14.0)
        self.assertAlmostEqual(summary["low_priority_coverage_mean"], 0.6)
        self.assertEqual(summary["solver_dual_bound_mean"], 0.94)
        self.assertEqual(summary["planning_time_s_median"], 0.03)
        self.assertEqual(summary["planning_time_s_iqr"], 0.0)

    def test_loader_rejects_duplicate_identity_and_hash_mismatch(self):
        duplicate = _row(0, "full", 1.0)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "duplicate.json"
            source.write_text(json.dumps([duplicate, duplicate]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重复运行身份"):
                pe.load_result_records([source])

            first = _row(0, "full", 1.0)
            second = _row(0, "ga", 0.8)
            second["scenario_hash"] = "different-hash"
            source.write_text(json.dumps([first, second]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scenario_hash"):
                pe.load_result_records([source])

    def test_primary_comparison_rejects_incomplete_scenario_grid(self):
        rows = [
            _row(0, "full", 1.0),
            _row(1, "full", 1.0),
            _row(0, "ga", 0.8),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "incomplete.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            validated = pe.load_result_records([source])
        with self.assertRaisesRegex(ValueError, "网格不完整"):
            pe.statistical_comparison(
                validated,
                pe.EvaluationConfig(bootstrap_samples=20, generate_figures=False),
            )

    def test_analysis_verifies_expected_frozen_hashes(self):
        rows = [_row(0, "full", 1.0), _row(0, "ga", 0.8)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rows.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest_hash"):
                pe.run_analysis(
                    [source],
                    root / "out",
                    pe.EvaluationConfig(
                        bootstrap_samples=20,
                        generate_figures=False,
                        expected_manifest_hash="wrong-manifest",
                    ),
                )
            with self.assertRaisesRegex(ValueError, "scenario_hash"):
                pe.run_analysis(
                    [source],
                    root / "out2",
                    pe.EvaluationConfig(
                        bootstrap_samples=20,
                        generate_figures=False,
                        expected_manifest_hash="manifest-id-test-v1",
                        expected_scenario_hash="wrong-scenario",
                    ),
                )

    def test_missing_power_scale_matches_nominal_only(self):
        nominal = _row(0, "full", 1.0)
        nominal.pop("power_scale")
        stressed = _row(0, "full", 0.2)
        stressed["power_scale"] = 1.2
        stressed["scenario_hash"] = "hash-s00-p1.2"
        prepared = []
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "power.json"
            source.write_text(json.dumps([nominal, stressed]), encoding="utf-8")
            prepared = pe.load_result_records([source])
        at_nominal = pe._scenario_metric(
            prepared, "id_test", "safe_weighted_coverage", 1.0
        )
        at_stress = pe._scenario_metric(
            prepared, "id_test", "safe_weighted_coverage", 1.2
        )
        self.assertEqual(at_nominal["full"]["s00"], 1.0)
        self.assertEqual(at_stress["full"]["s00"], 0.2)

    def test_exact_reference_certification_is_summarized_without_false_claims(self):
        certified = _row(0, "exact_pareto_dp", 0.9)
        certified["optimality_certified"] = True
        certified["optimality_gap"] = 0.0
        interrupted = _row(1, "exact_pareto_dp", 0.8)
        interrupted["optimality_certified"] = False
        interrupted["optimality_gap"] = None
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "exact.json"
            source.write_text(
                json.dumps([certified, interrupted]), encoding="utf-8"
            )
            rows = pe.load_result_records([source])
        summary = pe.aggregate_results(rows, 0.05)[0]
        self.assertEqual(summary["optimality_certification_eligible_runs"], 2)
        self.assertEqual(summary["optimality_certified_runs"], 1)
        self.assertEqual(summary["optimality_certification_rate"], 0.5)
        self.assertEqual(summary["optimality_gap_mean"], 0.0)

    def test_hierarchical_bootstrap_and_constant_friedman_are_reported(self):
        rows = []
        levels = {"full": 1.0, "ga": 0.5, "greedy": 0.0}
        for scenario in range(3):
            for algorithm, coverage in levels.items():
                for replicate in range(2):
                    row = _row(scenario, algorithm, coverage)
                    row["replicate_id"] = replicate
                    rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "hierarchical.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            validated = pe.load_result_records([source])
        statistics = pe.statistical_comparison(
            validated,
            pe.EvaluationConfig(bootstrap_samples=30, generate_figures=False),
        )
        self.assertEqual(statistics["bootstrap_unit"], "scenario_outer_repeat_inner")
        self.assertIsNotNone(statistics["omnibus"]["statistic"])
        self.assertEqual(statistics["repeat_counts_by_algorithm"]["full"]["total"], 6)
        self.assertTrue(
            all(
                row["bootstrap_method"].startswith("paired_scenario")
                for row in statistics["pairwise"]
            )
        )

    def test_preregistered_algorithm_family_is_filtered_before_holm(self):
        rows = []
        for scenario in range(3):
            rows.extend(
                (
                    _row(scenario, "full", 1.0),
                    _row(scenario, "ga", 0.8),
                    _row(scenario, "pso", 0.7),
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "families.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            validated = pe.load_result_records([source])

        statistics = pe.statistical_comparison(
            validated,
            pe.EvaluationConfig(
                included_algorithms=("full", "ga"),
                bootstrap_samples=20,
                generate_figures=False,
            ),
        )
        self.assertEqual(statistics["included_algorithms"], ["full", "ga"])
        self.assertEqual(len(statistics["pairwise"]), 1)
        self.assertEqual(statistics["pairwise"][0]["comparator"], "ga")
        self.assertEqual(
            statistics["pairwise"][0]["p_holm"],
            statistics["pairwise"][0]["p_value"],
        )
        with self.assertRaisesRegex(ValueError, "不存在"):
            pe.statistical_comparison(
                validated,
                pe.EvaluationConfig(
                    included_algorithms=("full", "missing"),
                    bootstrap_samples=20,
                    generate_figures=False,
                ),
            )

    def test_secondary_analysis_is_descriptive_without_new_pairwise_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "secondary.json"
            rows = []
            for node_count, split in ((8, "scale_8"), (12, "scale_12")):
                for scenario in range(2):
                    for algorithm, coverage in (("full", 0.9), ("greedy", 0.7)):
                        row = _row(scenario + node_count, algorithm, coverage)
                        row["split"] = split
                        row["node_count"] = node_count
                        rows.append(row)
            source.write_text(json.dumps(rows), encoding="utf-8")
            audit = pe.run_analysis(
                [source],
                root / "analysis",
                pe.EvaluationConfig(
                    primary_split="scale",
                    statistics_enabled=False,
                    analysis_role="secondary_descriptive",
                    included_algorithms=("full", "greedy"),
                    generate_figures=False,
                ),
            )
            statistics = json.loads(
                (root / "analysis" / "statistics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(statistics["status"], "secondary_descriptive_only")
            self.assertEqual(statistics["pairwise"], [])
            self.assertEqual(audit["confirmatory_metrics"], [])

    @unittest.skipUnless(os.environ.get("PAPER_PLOT_TEST") == "1", "仅在绘图环境中运行")
    def test_generate_publication_figures_and_qa_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "results.json"
            output = root / "paper"
            rows = []
            for scenario in range(6):
                for algorithm, coverage in (
                    ("full", 1.0),
                    ("greedy", 0.7 + scenario * 0.01),
                    ("ga", 0.8 + scenario * 0.01),
                ):
                    row = _row(scenario, algorithm, coverage)
                    row.update(
                        {
                            "visited_count": 10 + scenario,
                            "low_priority_coverage": min(1.0, coverage * 0.7),
                            "medium_priority_coverage": min(1.0, coverage * 0.85),
                            "high_priority_coverage": coverage,
                            "energy_utilization": 0.5 + scenario * 0.01,
                            "distance_utilization": 0.6 + scenario * 0.01,
                            "time_utilization": 0.7 + scenario * 0.01,
                        }
                    )
                    if algorithm == "ga" and scenario == 0:
                        row["time_violation"] = True
                    rows.append(row)
            source.write_text(json.dumps(rows), encoding="utf-8")
            pe.run_analysis(
                [source],
                output,
                pe.EvaluationConfig(
                    bootstrap_samples=20,
                    figure_dpi=72,
                    figure_formats=("svg", "pdf", "tiff"),
                    included_algorithms=("full", "ga"),
                ),
            )
            figures = output / "figures"
            required_stems = (
                "primary_safe_weighted_coverage",
                "priority_coverage_profile",
                "constraint_violation_rates",
                "planning_time_median_iqr",
                "safe_resource_profile",
                "termination_reason_distribution",
            )
            for stem in required_stems:
                for suffix in ("svg", "pdf", "tiff"):
                    with self.subTest(stem=stem, suffix=suffix):
                        self.assertTrue((figures / f"{stem}.{suffix}").is_file())
            svg = (figures / "priority_coverage_profile.svg").read_text(
                encoding="utf-8"
            )
            self.assertIn("<text", svg)
            manifest = json.loads(
                (figures / "figure_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["figure_contract"]["confirmatory_metrics"],
                ["safe_weighted_coverage"],
            )
            self.assertEqual(
                manifest["figure_contract"]["resource_policy"],
                "safe_routes_only",
            )
            self.assertTrue(manifest["qa"]["all_files_present_and_nonempty"])
            self.assertFalse(manifest["qa"]["radar_or_composite_score_generated"])
            self.assertEqual(
                set(manifest["method_visual_identity"]), {"full", "ga"}
            )

    @unittest.skipUnless(os.environ.get("PAPER_PLOT_TEST") == "1", "仅在绘图环境中运行")
    def test_representative_routes_follow_frozen_scenario_and_seed_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            learning = root / "learning"
            baseline = root / "baseline"
            for path in (learning, baseline):
                (path / "routes").mkdir(parents=True)
            (learning / "run_config.json").write_text(
                json.dumps(
                    {
                        "kind": "learning_evaluation",
                        "immutable": {"variant": "full", "training_seed": 42},
                    }
                ),
                encoding="utf-8",
            )
            (baseline / "run_config.json").write_text(
                json.dumps(
                    {
                        "kind": "traditional_baselines",
                        "immutable": {
                            "algorithms": [
                                "priority_resource_greedy",
                                "aco",
                                "milp_orienteering",
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            record = {
                "inspection_points_xyz": [[1, 0, 1], [2, 1, 2], [3, 0, 1]],
                "priorities": [1, 2, 3],
            }
            displays = [
                {"algorithm": "full", "training_seed": 42},
                {"algorithm": "priority_resource_greedy", "planner_seed": 42},
                {"algorithm": "aco", "planner_seed": 42},
                {"algorithm": "milp_orienteering", "planner_seed": 42},
            ]
            for display in displays:
                algorithm = display["algorithm"]
                seed_field = "training_seed" if "training_seed" in display else "planner_seed"
                seed = display[seed_field]
                row = {
                    "scenario_id": "id_test_034",
                    "algorithm": algorithm,
                    seed_field: seed,
                    "safe_weighted_coverage": 0.75,
                }
                payload = {
                    "row": row,
                    "record": record,
                    ("detail" if algorithm == "full" else "result"): {
                        "flight_path": [[0, 0, 0], [1, 0, 1], [0, 0, 0]]
                    },
                }
                if algorithm == "full":
                    filename = "id_test_034__power1.json"
                    target = learning
                else:
                    filename = f"id_test_034__{algorithm}__seed42__power1.json"
                    target = baseline
                (target / "routes" / filename).write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            output = root / "analysis"
            stem = pe.generate_representative_route_figure(
                [learning, baseline],
                output,
                {"scenario_id": "id_test_034", "display_algorithms": displays},
                pe.EvaluationConfig(figure_dpi=72, figure_formats=("svg",)),
                all_algorithms=[item["algorithm"] for item in displays],
            )
            self.assertEqual(stem, "representative_routes")
            self.assertTrue((output / "figures" / "representative_routes.svg").is_file())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""困难约束纠偏协议、场景生成和门禁规则的快速测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uav_inspection.core import final_python_ppo_pointer as ppo
from uav_inspection.experiments import paper_difficulty_experiments as difficulty
from uav_inspection.experiments import paper_experiments as legacy


class DifficultyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = difficulty.load_protocol()
        cls.scenario_file = (
            difficulty.ROOT / cls.protocol["base_scenario_file"]
        )
        cls.scenario = legacy._load_scenario(cls.scenario_file)

    def test_protocol_identity_and_deferred_formal_test(self):
        self.assertEqual(self.protocol["protocol_name"], "difficulty_test_v2_1")
        self.assertEqual(len(self.protocol["protocol_hash"]), 64)
        self.assertTrue(
            self.protocol["split_design"]["formal_test"][
                "generation_deferred_until_training_freeze"
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "正式test"):
                difficulty.generate_split(
                    difficulty.DEFAULT_PROTOCOL,
                    Path(directory),
                    "formal_test",
                    dry_run=True,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_candidate_generation_is_deterministic_and_stratified(self):
        keyword = dict(
            split="validation",
            node_count=24,
            difficulty="hard",
            constraint_type="energy",
            priority_layout="dispersed",
            replicate=0,
            attempt=3,
            master_seed=2026072602,
        )
        first = difficulty._candidate_record(
            self.scenario, self.protocol, **keyword
        )
        second = difficulty._candidate_record(
            self.scenario, self.protocol, **keyword
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["inspection_points_xyz"]), 24)
        self.assertEqual(len(first["priorities"]), 24)
        self.assertEqual(set(first["point_arm_ids"]), {0, 1, 2, 3})
        self.assertEqual(
            np.bincount(np.asarray(first["point_arm_ids"], dtype=int)).tolist(),
            [6, 6, 6, 6],
        )
        self.assertTrue(0.44 <= float(first["initial_soc"]) <= 0.52)
        self.assertLess(max(first["point_along_arm_distances_m"]), 1600.0)

    def test_priority_counts_preserve_high_medium_low_groups(self):
        for node_count in self.protocol["node_counts"]:
            counts = difficulty._priority_counts(int(node_count))
            self.assertEqual(sum(counts), int(node_count))
            self.assertTrue(all(value > 0 for value in counts))

    def test_qualification_is_globally_all_keep_or_all_retrain(self):
        rows = []
        for variant in self.protocol["qualification"]["all_variants"]:
            for seed in self.protocol["qualification"]["training_seeds"]:
                rows.append(
                    {
                        "variant": variant,
                        "training_seed": seed,
                        "complete": True,
                        "finite": True,
                        "safe_rate": 0.99,
                        "zero_visit_rate": 0.01,
                        "partial_return_rate": 0.9,
                        "median_visited_count": 5.0,
                        "median_safe_weighted_coverage": 0.6,
                    }
                )
        keep = difficulty.qualification_decision(rows, self.protocol)
        self.assertEqual(keep["decision"], "keep_all_35")
        self.assertFalse(keep["partial_retraining_allowed"])

        failing = [dict(row) for row in rows]
        failing[0]["safe_rate"] = 0.5
        retrain = difficulty.qualification_decision(failing, self.protocol)
        self.assertEqual(retrain["decision"], "retrain_all_35")

    def test_dry_run_does_not_write_files(self):
        with tempfile.TemporaryDirectory() as directory:
            result = difficulty.generate_split(
                difficulty.DEFAULT_PROTOCOL,
                Path(directory),
                "validation",
                dry_run=True,
            )
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["expected_count"], 108)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_variable_node_training_pool_and_monitor_boundaries(self):
        terrain = np.zeros((64, 64), dtype=np.float32)
        start = np.asarray([2.0, 2.0, 0.0], dtype=np.float32)

        def record(identifier, points, priorities):
            return {
                "id": identifier,
                "split": "unit",
                "node_count": len(points),
                "inspection_points_xyz": points,
                "priorities": priorities,
                "service_times_s": [1.0] * len(points),
                "initial_soc": 1.0,
                "distance_budget_scale": 1.0,
                "time_budget_scale": 1.0,
                "wind_scale": 1.0,
                "wind_rotation_deg": 0.0,
                "wind_vertical_bias_mps": 0.0,
                "power_scale": 1.0,
            }

        two = record(
            "unit_n2",
            [[8.0, 2.0, 0.0], [14.0, 2.0, 0.0]],
            [3.0, 1.0],
        )
        three = record(
            "unit_n3",
            [[8.0, 2.0, 0.0], [14.0, 2.0, 0.0], [20.0, 2.0, 0.0]],
            [3.0, 2.0, 1.0],
        )
        cfg = {
            "d_model": 16,
            "n_heads": 4,
            "coordinate_scale_m_per_unit": 1.0,
            "terrain_clearance_m": 1.0,
            "terrain_sample_interval_m": 1.0,
            "max_route_distance": 500.0,
            "max_mission_time_s": 500.0,
            "max_episodes": 4,
            "episodes_per_update": 3,
            "ppo_epochs": 1,
            "minibatch_size": 16,
            "validation_interval_updates": 99,
            "monitor_episodes": [1, 3, 4],
            "persist_monitor_checkpoints": True,
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as directory:
            cfg["checkpoint_dir"] = directory
            model, returns = ppo.train_policy_improved(
                start,
                np.asarray(two["inspection_points_xyz"], dtype=np.float32),
                np.asarray(two["priorities"], dtype=np.float32),
                terrain,
                cfg,
                {"uniform_vector": np.zeros(3, dtype=np.float32)},
                target_device="cpu",
                training_instances=[two, three],
                validation_instances=[two, three],
            )
            summary = model.training_summary
            self.assertEqual(len(returns), 4)
            self.assertEqual(model.N, 3)
            self.assertEqual(
                [int(item["episodes_seen"]) for item in summary["history"]],
                [1, 3, 4],
            )
            self.assertEqual(
                [int(item["training_node_count"]) for item in summary["history"]],
                [2, 3, 2],
            )
            _, payload = ppo.load_checkpoint(
                Path(directory) / "latest.pt", map_location="cpu"
            )
            self.assertEqual(
                payload["cfg"]["training_mode"], "external_variable_pool_v2"
            )
            self.assertEqual(payload["cfg"]["training_instance_count"], 2)
            self.assertEqual(
                payload["cfg"]["validation_mode"], "external_variable_v2"
            )
            self.assertTrue((Path(directory) / "monitor_ep0001.pt").exists())
            self.assertTrue((Path(directory) / "monitor_ep0003.pt").exists())
            self.assertTrue((Path(directory) / "monitor_ep0004.pt").exists())

    def test_confirmatory_statistics_helpers_are_directional_and_reproducible(self):
        adjusted = difficulty._holm_adjust({"b": 0.04, "a": 0.01})
        self.assertAlmostEqual(adjusted["a"], 0.02)
        self.assertAlmostEqual(adjusted["b"], 0.04)

        differences = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
        self.assertEqual(difficulty._rank_biserial(differences), 1.0)
        self.assertAlmostEqual(
            difficulty._hodges_lehmann_paired(differences), 0.2
        )

        left = np.full((8, 5), 0.8, dtype=np.float64)
        right = np.full((8, 5), 0.6, dtype=np.float64)
        first = difficulty._hierarchical_difference_ci(
            left, right, seed=42, replicates=100
        )
        second = difficulty._hierarchical_difference_ci(
            left, right, seed=42, replicates=100
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 0.2)
        self.assertAlmostEqual(first[1], 0.2)

    def test_confirmatory_statistics_helpers_handle_ties_and_invalid_shape(self):
        zeros = np.zeros(4, dtype=np.float64)
        self.assertEqual(difficulty._rank_biserial(zeros), 0.0)
        self.assertEqual(difficulty._hodges_lehmann_paired(zeros), 0.0)
        with self.assertRaisesRegex(ValueError, "场景,种子"):
            difficulty._hierarchical_difference_ci(
                np.zeros(3), np.zeros(3), seed=42, replicates=10
            )


if __name__ == "__main__":
    unittest.main()

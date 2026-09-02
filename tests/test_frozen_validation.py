#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结论文验证清单与核心选模适配器测试。"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from uav_inspection.core import final_python_ppo_pointer as ppo


class FrozenValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.terrain = np.zeros((64, 64), dtype=np.float32)
        self.start = np.array([2.0, 2.0, 0.0], dtype=np.float32)
        self.points = np.array(
            [[10.0, 2.0, 0.0], [20.0, 2.0, 0.0]], dtype=np.float32
        )
        self.priorities = np.array([1.0, 3.0], dtype=np.float32)
        self.wind = {
            "uniform_vector": np.array([2.0, 0.0, 1.0], dtype=np.float32)
        }
        self.cfg = ppo.resolve_config(
            {
                "d_model": 32,
                "n_heads": 4,
                "coordinate_scale_m_per_unit": 1.0,
                "max_route_distance": 1000.0,
                "max_mission_time_s": 1000.0,
                "service_times_s": [20.0, 20.0],
                "validation_scenarios": 2,
                "seed": 7,
            }
        )

    def _instance(self, identifier: str, soc: float) -> dict:
        return {
            "id": identifier,
            "initial_soc": soc,
            "distance_budget_scale": 0.9,
            "time_budget_scale": 0.85,
            "wind_scale": 1.1,
            "wind_rotation_deg": 10.0,
            "wind_vertical_bias_mps": -0.2,
            "power_scale": 1.0,
            "node_count": 2,
            "service_times_s": [20.0, 20.0],
            "inspection_points_xyz": self.points.tolist(),
            "priorities": self.priorities.tolist(),
        }

    def test_normalization_is_order_invariant_and_checks_scene(self) -> None:
        first = self._instance("validation_001", 0.9)
        second = self._instance("validation_000", 0.8)
        normalized_a, hash_a = ppo.normalize_validation_instances(
            [first, second], self.points, self.priorities, self.cfg
        )
        normalized_b, hash_b = ppo.normalize_validation_instances(
            [second, first], self.points, self.priorities, self.cfg
        )
        self.assertEqual(normalized_a, normalized_b)
        self.assertEqual(hash_a, hash_b)
        self.assertEqual([item["id"] for item in normalized_a], ["validation_000", "validation_001"])

        broken = copy.deepcopy(first)
        broken["priorities"] = [3.0, 1.0]
        with self.assertRaisesRegex(ValueError, "优先级"):
            ppo.normalize_validation_instances(
                [broken], self.points, self.priorities, self.cfg
            )

        low_soc = self._instance(
            "validation_low_soc", float(self.cfg["battery_reserve_ratio"])
        )
        with self.assertRaisesRegex(ValueError, "battery_reserve_ratio"):
            ppo.normalize_validation_instances(
                [low_soc], self.points, self.priorities, self.cfg
            )

    def test_frozen_wind_transform_matches_core_order(self) -> None:
        instance = self._instance("validation_000", 0.9)
        instance.update(
            {"wind_scale": 2.0, "wind_rotation_deg": 90.0, "wind_vertical_bias_mps": 0.5}
        )
        transformed = ppo.transform_wind_for_domain_instance(self.wind, instance)
        np.testing.assert_allclose(
            transformed["uniform_vector"], [0.0, 4.0, 2.5], atol=1e-6
        )

    def test_frozen_meteorological_wind_uses_model_axes_and_transform(self) -> None:
        instance = self._instance("validation_000", 0.9)
        instance.update(
            {"wind_scale": 2.0, "wind_rotation_deg": 90.0, "wind_vertical_bias_mps": 0.5}
        )
        # 0°表示从正北吹来，在Y向南的模型坐标中原始矢量为[0, 2, 1]。
        transformed = ppo.transform_wind_for_domain_instance(
            {"speed": 2.0, "direction": 0.0, "vertical_speed": 1.0},
            instance,
        )
        np.testing.assert_allclose(
            transformed["uniform_vector"], [-4.0, 0.0, 2.5], atol=1e-6
        )
        field = ppo.WindField.from_data(transformed, self.cfg, randomize=False)
        np.testing.assert_allclose(
            field.vector_at(self.start), [-4.0, 0.0, 2.5], atol=1e-6
        )

        # 未提供基础风时，冻结实例的垂直偏置仍必须生效。
        calm = ppo.transform_wind_for_domain_instance({}, instance)
        np.testing.assert_allclose(
            calm["uniform_vector"], [0.0, 0.0, 0.5], atol=1e-6
        )
        calm_missing = ppo.transform_wind_for_domain_instance(None, instance)
        np.testing.assert_allclose(
            calm_missing["uniform_vector"], [0.0, 0.0, 0.5], atol=1e-6
        )

    def test_all_variants_receive_the_same_explicit_conditions(self) -> None:
        instances, digest = ppo.normalize_validation_instances(
            [self._instance("validation_001", 0.9), self._instance("validation_000", 0.8)],
            self.points,
            self.priorities,
            self.cfg,
        )
        captures = []

        def fake_rollout(*args, **kwargs):
            scenario_cfg = args[5]
            scenario_wind = args[6]
            captures.append(
                (
                    float(scenario_cfg["initial_soc"]),
                    float(scenario_cfg["max_route_distance"]),
                    np.asarray(scenario_wind["uniform_vector"]).copy(),
                    bool(kwargs["randomize"]),
                )
            )
            metrics = {
                "returned": True,
                "constraint_violation_count": 0,
                "weighted_coverage": 0.5,
                "coverage": 0.5,
                "energy_utilization": 0.2,
                "distance_utilization": 0.2,
                "time_utilization": 0.2,
            }
            return (None, None, None, None, None, [{"episode_metrics": metrics}])

        summaries = []
        with mock.patch.object(ppo, "rollout_episode_improved", side_effect=fake_rollout):
            for variant in ("full", "no_domain_randomization"):
                variant_cfg = dict(self.cfg)
                for field in (
                    "policy_architecture",
                    "training_algorithm",
                    "domain_randomization",
                    "resource_shaping",
                    "return_reserve_mask",
                    "simulation_only",
                ):
                    variant_cfg.pop(field, None)
                cfg = ppo.resolve_config(
                    {**variant_cfg, "experiment_variant": variant, "validation_instances_hash": digest}
                )
                summaries.append(
                    ppo._validation_summary(
                        object(), self.start, self.points, self.priorities, self.terrain,
                        cfg, self.wind, instances
                    )
                )

        for left, right in zip(captures[:2], captures[2:]):
            self.assertEqual(left[:2], right[:2])
            np.testing.assert_allclose(left[2], right[2], atol=1e-7)
            self.assertEqual(left[3], right[3])
        self.assertTrue(all(not item[3] for item in captures))
        self.assertEqual([item[0] for item in captures[:2]], [0.8, 0.9])
        self.assertTrue(all(summary["validation_mode"] == "external_fixed_v1" for summary in summaries))
        self.assertTrue(all(summary["validation_instances_hash"] == digest for summary in summaries))

    def test_training_persists_validation_identity_and_rejects_wrong_hash(self) -> None:
        instance = self._instance("validation_000", 0.9)
        cfg = {
            **self.cfg,
            "max_episodes": 2,
            "episodes_per_update": 2,
            "ppo_epochs": 1,
            "minibatch_size": 8,
            "validation_interval_updates": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            cfg["checkpoint_dir"] = temporary
            model, _ = ppo.train_policy_improved(
                self.start,
                self.points,
                self.priorities,
                self.terrain,
                cfg,
                self.wind,
                target_device="cpu",
                validation_instances=[instance],
            )
            summary = model.training_summary
            self.assertEqual(summary["validation_mode"], "external_fixed_v1")
            self.assertEqual(summary["validation_instance_count"], 1)
            _, payload = ppo.load_checkpoint(
                Path(temporary) / "latest.pt", map_location="cpu"
            )
            self.assertEqual(
                payload["cfg"]["validation_instances_hash"],
                summary["validation_instances_hash"],
            )
            changed_instance = copy.deepcopy(instance)
            changed_instance["initial_soc"] = 0.85
            resume_cfg = {**cfg, "max_episodes": 4}
            with self.assertRaisesRegex(ValueError, "不一致项"):
                ppo.train_policy_improved(
                    self.start,
                    self.points,
                    self.priorities,
                    self.terrain,
                    resume_cfg,
                    self.wind,
                    resume_from=Path(temporary) / "latest.pt",
                    target_device="cpu",
                    validation_instances=[changed_instance],
                )

        wrong_cfg = {**cfg, "checkpoint_dir": None, "validation_instances_hash": "0" * 64}
        with self.assertRaisesRegex(ValueError, "hash"):
            ppo.train_policy_improved(
                self.start,
                self.points,
                self.priorities,
                self.terrain,
                wrong_cfg,
                self.wind,
                target_device="cpu",
                validation_instances=[instance],
            )

    def test_intentionally_unsafe_ablation_is_not_mislabeled_best_safe(self) -> None:
        unsafe_validation = {
            "constraint_failures": 1.0,
            "return_rate": 0.0,
            "weighted_coverage": 0.0,
            "coverage": 0.0,
            "resource_utilization": 3.0,
            "validation_mode": "legacy_seeded",
            "validation_instances_hash": "",
            "validation_instance_count": 1,
        }
        cfg = {
            "experiment_variant": "no_return_reserve",
            "d_model": 32,
            "n_heads": 4,
            "coordinate_scale_m_per_unit": 1.0,
            "max_route_distance": 1000.0,
            "max_mission_time_s": 1000.0,
            "service_times_s": [20.0, 20.0],
            "max_episodes": 2,
            "episodes_per_update": 2,
            "ppo_epochs": 1,
            "minibatch_size": 8,
            "validation_scenarios": 1,
            "validation_interval_updates": 1,
            "seed": 11,
        }
        with tempfile.TemporaryDirectory() as temporary:
            cfg["checkpoint_dir"] = temporary
            with mock.patch.object(
                ppo, "_validation_summary", return_value=unsafe_validation
            ):
                model, returns = ppo.train_policy_improved(
                    self.start,
                    self.points,
                    self.priorities,
                    self.terrain,
                    cfg,
                    self.wind,
                    target_device="cpu",
                )
            self.assertEqual(len(returns), 2)
            self.assertEqual(
                model.training_summary["selection_kind"], "best_candidate_unsafe"
            )
            self.assertTrue((Path(temporary) / "best_candidate.pt").is_file())
            self.assertFalse((Path(temporary) / "best_safe.pt").exists())

    def test_unsafe_ablation_still_obeys_validation_interval(self) -> None:
        """无安全模型时也只做首次、间隔和终态验证。"""

        unsafe_validation = {
            "constraint_failures": 1.0,
            "return_rate": 0.0,
            "weighted_coverage": 0.0,
            "coverage": 0.0,
            "resource_utilization": 3.0,
            "validation_mode": "legacy_seeded",
            "validation_instances_hash": "",
            "validation_instance_count": 1,
        }
        cfg = {
            "experiment_variant": "no_return_reserve",
            "d_model": 32,
            "n_heads": 4,
            "coordinate_scale_m_per_unit": 1.0,
            "max_route_distance": 1000.0,
            "max_mission_time_s": 1000.0,
            "service_times_s": [20.0, 20.0],
            "max_episodes": 6,
            "episodes_per_update": 2,
            "ppo_epochs": 1,
            "minibatch_size": 8,
            "validation_scenarios": 1,
            "validation_interval_updates": 3,
            "seed": 17,
        }
        with mock.patch.object(
            ppo, "_validation_summary", return_value=unsafe_validation
        ) as validation_mock:
            model, returns = ppo.train_policy_improved(
                self.start,
                self.points,
                self.priorities,
                self.terrain,
                cfg,
                self.wind,
                target_device="cpu",
            )

        # 第1轮建立候选，第3轮同时是设定间隔和训练终态；第2轮不验证。
        self.assertEqual(validation_mock.call_count, 2)
        self.assertEqual(len(returns), 6)
        self.assertEqual(model.training_summary["selection_kind"], "best_candidate_unsafe")
        self.assertIsNone(model.training_summary["history"][1]["validation"])
        self.assertIsNotNone(model.training_summary["history"][0]["validation"])
        self.assertIsNotNone(model.training_summary["history"][2]["validation"])


if __name__ == "__main__":
    unittest.main()

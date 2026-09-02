#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPO v2 断点续训、指标回调和原子检查点测试。"""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT

import numpy as np
import torch

ROOT = WORKSPACE_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_inspection.core import final_python_ppo_pointer as ppo


class PPOResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.terrain = np.zeros((48, 48), dtype=np.float32)
        self.start = np.array([2.0, 2.0, 0.0], dtype=np.float32)
        self.points = np.array(
            [[8.0, 2.0, 0.0], [14.0, 3.0, 0.0], [20.0, 4.0, 0.0]],
            dtype=np.float32,
        )
        self.priorities = np.array([1.0, 3.0, 2.0], dtype=np.float32)
        self.wind = {"uniform_vector": np.zeros(3, dtype=np.float32)}

    def _config(self, checkpoint_dir=None):
        return ppo.resolve_config(
            {
                "d_model": 16,
                "n_heads": 4,
                "max_route_distance": 1000.0,
                "max_mission_time_s": 1000.0,
                "coordinate_scale_m_per_unit": 1.0,
                "max_episodes": 4,
                "episodes_per_update": 2,
                "ppo_epochs": 2,
                "minibatch_size": 8,
                "validation_scenarios": 1,
                "validation_interval_updates": 1,
                "checkpoint_dir": checkpoint_dir,
                "seed": 23,
            }
        )

    @staticmethod
    def _numpy_rng_equal(left, right) -> bool:
        return (
            left[0] == right[0]
            and np.array_equal(left[1], right[1])
            and left[2:] == right[2:]
        )

    def test_interrupted_resume_matches_uninterrupted_training(self) -> None:
        baseline_model, baseline_returns = ppo.train_policy_improved(
            self.start,
            self.points,
            self.priorities,
            self.terrain,
            self._config(),
            self.wind,
            target_device="cpu",
        )
        baseline_python_rng = random.getstate()
        baseline_numpy_rng = np.random.get_state()
        baseline_torch_rng = torch.get_rng_state().clone()

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)

            def interrupt_after_first_update(record):
                self.assertEqual(record["episodes_seen"], 2.0)
                self.assertTrue(Path(record["latest_checkpoint"]).is_file())
                raise RuntimeError("模拟训练进程中断")

            with self.assertRaisesRegex(RuntimeError, "模拟训练进程中断"):
                ppo.train_policy_improved(
                    self.start,
                    self.points,
                    self.priorities,
                    self.terrain,
                    config,
                    self.wind,
                    metrics_callback=interrupt_after_first_update,
                    target_device="cpu",
                )

            latest = Path(directory) / "latest.pt"
            self.assertTrue(latest.is_file())
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])
            _, interrupted_payload = ppo.load_checkpoint(latest, map_location="cpu")
            self.assertEqual(interrupted_payload["training_state"]["episodes_seen"], 2)
            self.assertEqual(interrupted_payload["training_state"]["update_index"], 1)
            self.assertEqual(len(interrupted_payload["training_state"]["history"]), 1)
            self.assertIn("entropy_progress", interrupted_payload["training_state"])
            self.assertIsNotNone(
                interrupted_payload["training_state"]["best_model_state_dict"]
            )
            self.assertIsNotNone(
                interrupted_payload["training_state"]["best_optimizer_state_dict"]
            )
            self.assertIsNotNone(interrupted_payload["optimizer_state_dict"])

            shortened_cfg = dict(config)
            shortened_cfg["max_episodes"] = 3
            with self.assertRaisesRegex(ValueError, "不能缩短"):
                ppo.train_policy_improved(
                    self.start,
                    self.points,
                    self.priorities,
                    self.terrain,
                    shortened_cfg,
                    self.wind,
                    resume_from=latest,
                    target_device="cpu",
                )

            callback_records = []
            resumed_model, resumed_returns = ppo.train_policy_improved(
                self.start,
                self.points,
                self.priorities,
                self.terrain,
                config,
                self.wind,
                resume_from=latest,
                metrics_callback=callback_records.append,
                target_device=torch.device("cpu"),
            )

            self.assertEqual(len(callback_records), 1)
            self.assertEqual(callback_records[0]["episodes_seen"], 4.0)
            self.assertAlmostEqual(
                callback_records[0]["entropy_coef"],
                float(config["entropy_coef_end"]),
            )
            self.assertEqual(
                sum(callback_records[0]["termination_reason_counts"].values()), 2
            )
            self.assertTrue((Path(directory) / "best_safe.pt").is_file())
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

        self.assertEqual(resumed_returns, baseline_returns)
        for name, expected in baseline_model.state_dict().items():
            torch.testing.assert_close(
                resumed_model.state_dict()[name], expected, rtol=0.0, atol=0.0
            )
        self.assertEqual(random.getstate(), baseline_python_rng)
        self.assertTrue(self._numpy_rng_equal(np.random.get_state(), baseline_numpy_rng))
        torch.testing.assert_close(torch.get_rng_state(), baseline_torch_rng)
        self.assertEqual(
            resumed_model.training_rng_state, baseline_model.training_rng_state
        )
        self.assertEqual(resumed_model.training_summary["episodes_seen"], 4)
        self.assertEqual(resumed_model.training_summary["updates"], 2)
        self.assertEqual(resumed_model.training_summary["entropy_progress"], 1.0)

    def test_resume_rejects_best_safe_and_static_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            initial_cfg = self._config(directory)
            initial_cfg.update(
                {
                    "max_episodes": 2,
                    "experiment_stage": "pilot",
                    "run_id": "resume-contract",
                }
            )
            ppo.train_policy_improved(
                self.start,
                self.points,
                self.priorities,
                self.terrain,
                initial_cfg,
                self.wind,
                target_device="cpu",
            )
            latest = Path(directory) / "latest.pt"
            best_safe = Path(directory) / "best_safe.pt"
            _, latest_payload = ppo.load_checkpoint(latest, map_location="cpu")
            _, best_payload = ppo.load_checkpoint(best_safe, map_location="cpu")
            self.assertEqual(latest_payload["checkpoint_kind"], "latest")
            self.assertEqual(best_payload["checkpoint_kind"], "best_safe")

            extended_cfg = dict(initial_cfg)
            extended_cfg["max_episodes"] = 4
            drifted_cfg = dict(extended_cfg)
            drifted_cfg["gamma"] = 0.90
            with self.assertRaisesRegex(ValueError, "不一致项.*gamma"):
                ppo.train_policy_improved(
                    self.start,
                    self.points,
                    self.priorities,
                    self.terrain,
                    drifted_cfg,
                    self.wind,
                    resume_from=latest,
                    target_device="cpu",
                )

            mixed_stage_cfg = dict(extended_cfg)
            mixed_stage_cfg["experiment_stage"] = "formal"
            with self.assertRaisesRegex(ValueError, "experiment_stage"):
                ppo.train_policy_improved(
                    self.start,
                    self.points,
                    self.priorities,
                    self.terrain,
                    mixed_stage_cfg,
                    self.wind,
                    resume_from=latest,
                    target_device="cpu",
                )

            with self.assertRaisesRegex(ValueError, "best_safe仅用于"):
                ppo.train_policy_improved(
                    self.start,
                    self.points,
                    self.priorities,
                    self.terrain,
                    extended_cfg,
                    self.wind,
                    resume_from=best_safe,
                    target_device="cpu",
                )

            # 只延长累计目标回合数是合法续训，并会保留阶段/运行身份。
            resumed_model, resumed_returns = ppo.train_policy_improved(
                self.start,
                self.points,
                self.priorities,
                self.terrain,
                extended_cfg,
                self.wind,
                resume_from=latest,
                target_device="cpu",
            )
            self.assertEqual(len(resumed_returns), 4)
            self.assertEqual(resumed_model.training_summary["episodes_seen"], 4)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPO+Pointer v2 内核的CPU单元测试。"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = WORKSPACE_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_inspection.core import final_python_ppo_pointer as ppo


class ScriptedPolicy(nn.Module):
    """只用于确认推断严格服从Pointer logits，不读取人工优先级评分。"""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        s_env,
        s_uav,
        m_priority,
        m_visit,
        m_energy,
        m_distance=None,
        m_time=None,
        m_dynamics=None,
        m_remaining=None,
    ):
        del s_uav, m_priority, m_remaining
        legal = m_visit & m_energy
        for mask in (m_distance, m_time, m_dynamics):
            if mask is not None:
                legal = legal & mask
        batch, action_count = legal.shape
        base = torch.zeros((action_count,), dtype=s_env.dtype, device=s_env.device)
        # 第0点概率最高；返航token次高；其他巡检点最低。
        base[-1] = 90.0
        if action_count > 1:
            base[0] = 100.0
        logits = base.unsqueeze(0).expand(batch, -1).clone()
        logits = logits.masked_fill(~legal, -torch.inf) + self.anchor * 0.0
        return torch.softmax(logits, -1), logits, torch.zeros((batch, 1), device=s_env.device)


class PPOPointerV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        ppo.set_global_seed(7)
        self.terrain = np.zeros((64, 64), dtype=np.float32)
        self.start = np.array([2.0, 2.0, 0.0], dtype=np.float32)
        self.points = np.array(
            [[10.0, 2.0, 0.0], [20.0, 2.0, 0.0], [30.0, 2.0, 0.0]],
            dtype=np.float32,
        )
        self.priorities = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.cfg = ppo.resolve_config(
            {
                "d_model": 32,
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
                "seed": 7,
            }
        )
        self.calm_wind = {"uniform_vector": np.zeros(3, dtype=np.float32)}

    def _model(self, node_count: Optional[int] = None) -> ppo.PPO_PtrNet:
        count = int(node_count if node_count is not None else len(self.points))
        return ppo.PPO_PtrNet(
            batch_size=1,
            n_nodes=count,
            d_env=ppo.NODE_FEATURE_DIM,
            d_uav=ppo.UAV_FEATURE_DIM,
            d_model=32,
            n_heads=4,
            lambda_priority=0.5,
        )

    def _synthetic_ppo_batch(
        self, batch_size: int = 16
    ) -> tuple[ppo.PPO_PtrNet, ppo.PPOBatch]:
        """构造一批冻结旧策略概率的合成轨迹，专门检查PPO更新性质。"""

        model = self._model()
        actions = len(self.points) + 1
        s_env = torch.randn(batch_size, actions, ppo.NODE_FEATURE_DIM)
        s_env[:, -1, 5] = 1.0
        s_uav = torch.randn(batch_size, ppo.UAV_FEATURE_DIM)
        priority = torch.zeros(batch_size, 1, actions, actions)
        valid = torch.ones(batch_size, actions, dtype=torch.bool)
        remaining = valid.clone()
        remaining[:, -1] = False
        with torch.no_grad():
            _, logits, value = model(
                s_env, s_uav, priority, valid, valid, valid, valid, valid, remaining
            )
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            old_logp = distribution.log_prob(action)
            old_value = value.squeeze(-1)
        advantages = torch.linspace(-1.0, 1.0, batch_size)
        batch = ppo.PPOBatch(
            s_env=s_env,
            s_uav=s_uav,
            m_priority=priority,
            m_visit=valid,
            m_energy=valid,
            m_distance=valid,
            m_time=valid,
            m_dynamics=valid,
            m_remaining=remaining,
            action=action,
            logp_old=old_logp,
            value_old=old_value,
            returns=old_value + advantages,
            advantages=advantages,
        )
        return model, batch

    def test_model_shapes_for_variable_node_counts(self) -> None:
        for count in (4, 8, 29):
            model = self._model(count)
            actions = count + 1
            s_env = torch.randn(2, actions, ppo.NODE_FEATURE_DIM)
            s_env[:, -1, 5] = 1.0
            s_uav = torch.randn(2, ppo.UAV_FEATURE_DIM)
            priority = torch.zeros(2, 1, actions, actions)
            valid = torch.ones(2, actions, dtype=torch.bool)
            remaining = valid.clone()
            remaining[:, -1] = False
            probability, logits, value = model(
                s_env,
                s_uav,
                priority,
                valid,
                valid,
                valid,
                valid,
                valid,
                remaining,
            )
            self.assertEqual(tuple(probability.shape), (2, actions))
            self.assertEqual(tuple(logits.shape), (2, actions))
            self.assertEqual(tuple(value.shape), (2, 1))
            self.assertTrue(torch.all(torch.isfinite(probability)))
            self.assertTrue(torch.allclose(probability.sum(-1), torch.ones(2), atol=1e-6))

    def test_two_dimensional_start_uses_dem_ground_height(self) -> None:
        elevated_terrain = np.full((32, 32), 100.0, dtype=np.float32)
        state = ppo.build_episode(
            [2.0, 2.0], self.points[:1], elevated_terrain, self.cfg, self.calm_wind
        )
        self.assertAlmostEqual(float(state["start_pos"][2]), 100.0)
        with self.assertRaisesRegex(ValueError, "低于DEM地面高程"):
            ppo.build_episode(
                [2.0, 2.0, 99.0],
                self.points[:1],
                elevated_terrain,
                self.cfg,
                self.calm_wind,
            )

    def test_coordinate_scale_changes_segment_resources(self) -> None:
        cfg = ppo.resolve_config({**self.cfg, "coordinate_scale_m_per_unit": 12.5})
        terrain_model = ppo.TerrainModel(self.terrain, cfg)
        wind = ppo.WindField.from_data(self.calm_wind, cfg)
        estimate = ppo.SegmentEstimator(terrain_model, wind, cfg).estimate(
            [1.0, 1.0, 18.0], [3.0, 1.0, 18.0]
        )
        self.assertTrue(estimate.feasible)
        self.assertAlmostEqual(estimate.distance_m, 25.0, places=5)
        self.assertAlmostEqual(estimate.time_s, 25.0 / 13.0, places=5)
        self.assertGreater(estimate.energy_wh, 0.0)

    @unittest.skipIf(ppo.scipy is None, "需要SciPy验证真实MATLAB结构往返")
    def test_mat_round_trip_preserves_optional_v2_fields(self) -> None:
        elevated_terrain = np.full((16, 16), 100.0, dtype=np.float32)
        payload = {
            "input_data": {
                "start_pos": np.array([2.0, 2.0], dtype=np.float32),
                "inspection_points": np.array(
                    [[4.0, 2.0, 150.0]], dtype=np.float32
                ),
                "inspection_priorities": np.array([3.0], dtype=np.float32),
                "terrain_data": elevated_terrain,
                "config": {
                    "point_z_mode": "flight_altitude",
                    "coordinate_scale_m_per_unit": 12.5,
                },
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            mat_path = Path(directory) / "v2_input.mat"
            ppo.scipy.io.savemat(mat_path, payload)
            loaded = ppo.scipy.io.loadmat(mat_path)
            start, points, priorities, terrain, _wind, cfg = ppo.extract_input(loaded)

        self.assertEqual(len(start), 2)
        self.assertEqual(cfg["point_z_mode"], "flight_altitude")
        self.assertAlmostEqual(cfg["coordinate_scale_m_per_unit"], 12.5)
        state = ppo.build_episode(start, points, terrain, cfg, self.calm_wind)
        self.assertAlmostEqual(float(state["start_pos"][2]), 100.0)
        self.assertAlmostEqual(float(state["target_positions"][0, 2]), 150.0)
        np.testing.assert_allclose(priorities, np.array([3.0], dtype=np.float32))

    def test_visited_action_probability_is_zero(self) -> None:
        state = ppo.build_episode(
            self.start, self.points, self.terrain, self.cfg, self.calm_wind
        )
        state, _, _ = ppo.step_env_improved(
            state, 0, self.points, self.priorities, self.terrain, self.cfg, self.calm_wind
        )
        observation = ppo._build_observation(state, self.priorities)
        tensors = ppo._observation_tensors(observation, torch.device("cpu"))
        probability, _, _ = ppo._model_forward(self._model(), tensors)
        self.assertEqual(float(probability[0, 0].item()), 0.0)

    def test_tight_budgets_leave_only_return_action(self) -> None:
        cfg = ppo.resolve_config(
            {**self.cfg, "max_route_distance": 10.0, "max_mission_time_s": 30.0}
        )
        state = ppo.build_episode(
            self.start, self.points, self.terrain, cfg, self.calm_wind
        )
        observation = ppo._build_observation(state, self.priorities)
        expected = np.zeros((len(self.points) + 1,), dtype=bool)
        expected[-1] = True
        np.testing.assert_array_equal(observation["legal_mask"], expected)

    def test_selected_actions_preserve_return_and_three_budgets(self) -> None:
        state = ppo.build_episode(
            self.start, self.points, self.terrain, self.cfg, self.calm_wind
        )
        observation = ppo._build_observation(state, self.priorities)
        legal_inspections = np.flatnonzero(observation["legal_mask"][:-1])
        self.assertGreater(len(legal_inspections), 0)
        state, _, _ = ppo.step_env_improved(
            state,
            int(legal_inspections[0]),
            self.points,
            self.priorities,
            self.terrain,
            self.cfg,
            self.calm_wind,
        )
        state, _, done = ppo.step_env_improved(
            state,
            len(self.points),
            self.points,
            self.priorities,
            self.terrain,
            self.cfg,
            self.calm_wind,
        )
        self.assertTrue(done)
        np.testing.assert_allclose(state["current"], self.start, atol=1e-6)
        self.assertLessEqual(state["total_energy_consumed"], state["energy_budget_wh"] + 1e-6)
        self.assertLessEqual(state["total_distance"], state["max_route_distance"] + 1e-6)
        self.assertLessEqual(state["total_time_s"], state["max_mission_time_s"] + 1e-6)
        remaining_soc = (
            state["initial_energy_wh"] - state["total_energy_consumed"]
        ) / state["battery_capacity"]
        self.assertGreaterEqual(remaining_soc + 1e-6, self.cfg["battery_reserve_ratio"])

    def test_randomized_masks_preserve_return_invariant(self) -> None:
        points = np.array(
            [
                [8.0, 8.0, 0.0],
                [16.0, 8.0, 0.0],
                [24.0, 8.0, 0.0],
                [8.0, 20.0, 0.0],
                [18.0, 24.0, 0.0],
                [30.0, 20.0, 0.0],
            ],
            dtype=np.float32,
        )
        priorities = np.array([1, 2, 3, 1, 2, 3], dtype=np.float32)
        for seed in range(30):
            state = ppo.build_episode(
                self.start,
                points,
                self.terrain,
                self.cfg,
                self.calm_wind,
                np.random.default_rng(seed),
                randomize=True,
            )
            for _ in range(len(points) + 1):
                observation = ppo._build_observation(state, priorities)
                inspection_actions = np.flatnonzero(observation["legal_mask"][:-1])
                action = int(inspection_actions[-1]) if len(inspection_actions) else len(points)
                state, _, done = ppo.step_env_improved(
                    state,
                    action,
                    points,
                    priorities,
                    self.terrain,
                    self.cfg,
                    self.calm_wind,
                )
                if done:
                    break
            self.assertTrue(state["done"])
            self.assertLessEqual(
                state["total_energy_consumed"], state["energy_budget_wh"] + 1e-6
            )
            self.assertLessEqual(
                state["total_distance"], state["max_route_distance"] + 1e-6
            )
            self.assertLessEqual(
                state["total_time_s"], state["max_mission_time_s"] + 1e-6
            )

    def test_tailwind_calm_headwind_resource_order(self) -> None:
        terrain_model = ppo.TerrainModel(self.terrain, self.cfg)
        start = np.array([2.0, 10.0, 18.0], dtype=np.float32)
        target = np.array([42.0, 10.0, 18.0], dtype=np.float32)

        def estimate(vector):
            wind = ppo.WindField.from_data({"uniform_vector": vector}, self.cfg)
            return ppo.SegmentEstimator(terrain_model, wind, self.cfg).estimate(start, target)

        tail = estimate([5.0, 0.0, 0.0])
        calm = estimate([0.0, 0.0, 0.0])
        head = estimate([-5.0, 0.0, 0.0])
        self.assertTrue(tail.feasible and calm.feasible and head.feasible)
        self.assertLess(tail.time_s, calm.time_s)
        self.assertLess(calm.time_s, head.time_s)
        self.assertLess(tail.energy_wh, calm.energy_wh)
        self.assertLess(calm.energy_wh, head.energy_wh)

    def test_crosswind_can_make_tracking_infeasible(self) -> None:
        cfg = ppo.resolve_config({**self.cfg, "cruise_speed_mps": 10.0})
        terrain_model = ppo.TerrainModel(self.terrain, cfg)
        wind = ppo.WindField.from_data(
            {"uniform_vector": [0.0, 11.0, 0.0]}, cfg
        )
        estimate = ppo.SegmentEstimator(terrain_model, wind, cfg).estimate(
            [2.0, 10.0, 18.0], [42.0, 10.0, 18.0]
        )
        self.assertFalse(estimate.feasible)
        self.assertEqual(estimate.reason, "crosswind_tracking")

    def test_meteorological_wind_uses_east_south_up_axes(self) -> None:
        # 风向表示“从何处吹来”：北风向南(+Y)，东风向西(-X)。
        expected = {
            0.0: [0.0, 5.0, 0.4],
            90.0: [-5.0, 0.0, 0.4],
            180.0: [0.0, -5.0, 0.4],
            270.0: [5.0, 0.0, 0.4],
        }
        for direction, vector in expected.items():
            np.testing.assert_allclose(
                ppo._meteorological_vector(5.0, direction, 0.4),
                np.asarray(vector, dtype=np.float32),
                atol=1e-6,
            )

    def test_spatial_wind_interpolation_and_uniform_fallback(self) -> None:
        field = ppo.WindField(
            positions=np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
            vectors=np.array([[2, 0, 0], [6, 0, 0]], dtype=np.float32),
            uniform_vector=[1, 0, 0],
            xy_scale_m_per_unit=1.0,
        )
        middle = field.vector_at([5, 0, 0])
        outside = field.vector_at([20, 0, 0])
        self.assertAlmostEqual(float(middle[0]), 4.0, places=5)
        self.assertAlmostEqual(float(outside[0]), 1.0, places=5)
        self.assertEqual(field.fallback_count, 1)

    def test_ridge_raises_safe_cruise_altitude(self) -> None:
        terrain = np.zeros((32, 32), dtype=np.float32)
        terrain[:, 15:17] = 100.0
        terrain_model = ppo.TerrainModel(terrain, self.cfg)
        wind = ppo.WindField.from_data(self.calm_wind, self.cfg)
        estimate = ppo.SegmentEstimator(terrain_model, wind, self.cfg).estimate(
            [2.0, 10.0, 18.0], [29.0, 10.0, 18.0]
        )
        self.assertTrue(estimate.feasible)
        self.assertGreaterEqual(estimate.cruise_altitude_m, 118.0)
        self.assertGreater(len(estimate.flight_path), 2)

    def test_gae_terminal_values_match_hand_calculation(self) -> None:
        advantages, returns = ppo.compute_gae(
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.5, 0.25]),
            torch.tensor([False, True]),
            gamma=1.0,
            gae_lambda=1.0,
        )
        torch.testing.assert_close(advantages, torch.tensor([1.5, 0.75]))
        torch.testing.assert_close(returns, torch.tensor([2.0, 1.0]))
        single_advantage, single_return = ppo.compute_gae(
            torch.tensor([0.0]),
            torch.tensor([0.1]),
            torch.tensor([True]),
            gamma=0.99,
            gae_lambda=0.95,
        )
        self.assertTrue(torch.all(torch.isfinite(single_advantage)))
        self.assertTrue(torch.all(torch.isfinite(single_return)))

    def test_multiple_ppo_epochs_move_ratio_away_from_one(self) -> None:
        cfg = ppo.resolve_config(
            {
                **self.cfg,
                "lr": 1e-2,
                "ppo_epochs": 3,
                "minibatch_size": 16,
                # 此测试专门验证第二轮以后概率比会变化，故关闭本例的 KL 早停干扰。
                "target_kl": 100.0,
            }
        )
        model, batch = self._synthetic_ppo_batch()
        frozen_old_logp = batch.logp_old.clone()
        stats = ppo.PPOAgent(model, cfg).update(batch, entropy_coef=0.01)
        self.assertTrue(math.isfinite(stats["approx_kl"]))
        self.assertTrue(math.isfinite(stats["clip_fraction"]))
        self.assertGreater(stats["ratio_deviation"], 1e-6)
        self.assertEqual(stats["epochs_completed"], 3.0)
        torch.testing.assert_close(batch.logp_old, frozen_old_logp)

    def test_post_update_kl_stops_epochs_early(self) -> None:
        cfg = ppo.resolve_config(
            {
                **self.cfg,
                "lr": 1e-2,
                "ppo_epochs": 5,
                "minibatch_size": 16,
                "target_kl": 1e-9,
            }
        )
        model, batch = self._synthetic_ppo_batch()
        stats = ppo.PPOAgent(model, cfg).update(batch, entropy_coef=0.01)
        self.assertGreaterEqual(stats["epochs_completed"], 1.0)
        self.assertLess(stats["epochs_completed"], 5.0)
        self.assertGreater(stats["approx_kl"], 0.0)
        self.assertTrue(math.isfinite(stats["approx_kl"]))

    def test_deterministic_decoding_uses_pointer_logits_only(self) -> None:
        priorities = np.array([1.0, 100.0], dtype=np.float32)
        points = self.points[:2]
        detail = ppo.plan_with_policy_improved(
            ScriptedPolicy(),
            self.start,
            points,
            priorities,
            self.terrain,
            self.cfg,
            self.calm_wind,
            return_details=True,
            decode_mode="deterministic",
        )
        self.assertEqual(detail["visit_order"], [0])
        self.assertTrue(detail["metrics"]["returned"])
        self.assertEqual(len(detail["segments"]), 2)
        self.assertTrue(all(segment["feasible"] for segment in detail["segments"]))
        self.assertTrue(detail["segments"][-1]["is_return"])
        self.assertEqual(detail["termination_reason"], "returned_partial")
        self.assertEqual(detail["energy_wh"], detail["metrics"]["energy_wh"])
        np.testing.assert_allclose(detail["path"][0], detail["path"][-1], atol=1e-6)

    def test_checkpoint_round_trip_preserves_forward_output(self) -> None:
        model = self._model()
        actions = len(self.points) + 1
        s_env = torch.randn(1, actions, ppo.NODE_FEATURE_DIM)
        s_env[:, -1, 5] = 1.0
        s_uav = torch.randn(1, ppo.UAV_FEATURE_DIM)
        priority = torch.zeros(1, 1, actions, actions)
        valid = torch.ones(1, actions, dtype=torch.bool)
        remaining = valid.clone()
        remaining[:, -1] = False
        model.eval()
        with torch.no_grad():
            expected = model(
                s_env, s_uav, priority, valid, valid, valid, valid, valid, remaining
            )[1]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_safe.pt"
            ppo.save_checkpoint(checkpoint, model, self.cfg)
            restored, payload = ppo.load_checkpoint(checkpoint, map_location="cpu")
            with torch.no_grad():
                actual = restored(
                    s_env, s_uav, priority, valid, valid, valid, valid, valid, remaining
                )[1]
        self.assertEqual(payload["schema_version"], ppo.SCHEMA_VERSION)
        self.assertIn("rng_state", payload)
        self.assertIn("python", payload["rng_state"])
        self.assertIn("numpy_global", payload["rng_state"])
        self.assertIn("torch", payload["rng_state"])
        torch.testing.assert_close(actual, expected)

    def test_legacy_checkpoint_is_rejected_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            torch.save({"schema_version": 1}, checkpoint)
            with self.assertRaisesRegex(ValueError, "schema_version=2"):
                ppo.load_checkpoint(checkpoint, map_location="cpu")

    def test_oracle_bounds_accept_rounding_scale_order_reversal(self) -> None:
        instance = {
            "id": "rounding_order",
            "initial_soc": 0.9,
            "distance_budget_scale": 1.0,
            "time_budget_scale": 1.0,
            "wind_scale": 1.0,
            "wind_rotation_deg": 0.0,
            "wind_vertical_bias_mps": 0.0,
            "inspection_points_xyz": [[10.0, 2.0, 0.0]],
            "priorities": [1.0],
            "certificate": {
                "weighted_coverage_lower_bound": 0.4 + 5e-13,
                "weighted_coverage_upper_bound": 0.4,
            },
        }
        normalized, _ = ppo.normalize_variable_instances([instance], self.cfg)
        self.assertLessEqual(
            normalized[0]["oracle_weighted_coverage_lower_bound"],
            normalized[0]["oracle_weighted_coverage_upper_bound"],
        )

    def test_oracle_bounds_reject_material_order_reversal(self) -> None:
        instance = {
            "id": "material_order",
            "initial_soc": 0.9,
            "distance_budget_scale": 1.0,
            "time_budget_scale": 1.0,
            "wind_scale": 1.0,
            "wind_rotation_deg": 0.0,
            "wind_vertical_bias_mps": 0.0,
            "inspection_points_xyz": [[10.0, 2.0, 0.0]],
            "priorities": [1.0],
            "certificate": {
                "weighted_coverage_lower_bound": 0.4 + 2e-12,
                "weighted_coverage_upper_bound": 0.4,
            },
        }
        with self.assertRaises(ValueError):
            ppo.normalize_variable_instances([instance], self.cfg)

    def test_short_cpu_training_smoke_and_return(self) -> None:
        points = np.vstack([self.points, np.array([[10.0, 12.0, 0.0]], dtype=np.float32)])
        priorities = np.array([1.0, 2.0, 3.0, 2.0], dtype=np.float32)
        cfg = ppo.resolve_config(
            {
                **self.cfg,
                "d_model": 16,
                "n_heads": 4,
                "max_episodes": 2,
                "episodes_per_update": 2,
                "ppo_epochs": 2,
                "minibatch_size": 8,
                "validation_scenarios": 1,
            }
        )
        model, returns = ppo.train_policy_improved(
            self.start, points, priorities, self.terrain, cfg, self.calm_wind
        )
        self.assertEqual(len(returns), 2)
        self.assertTrue(all(math.isfinite(value) for value in returns))
        detail = ppo.plan_with_policy_improved(
            model,
            self.start,
            points,
            priorities,
            self.terrain,
            cfg,
            self.calm_wind,
            return_details=True,
        )
        self.assertTrue(detail["metrics"]["returned"])
        np.testing.assert_allclose(detail["path"][0], detail["path"][-1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()

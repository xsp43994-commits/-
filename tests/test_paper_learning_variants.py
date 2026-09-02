import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from uav_inspection.core import final_python_ppo_pointer as ppo


class PaperLearningVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        ppo.set_global_seed(19)
        self.terrain = np.zeros((128, 128), dtype=np.float32)
        self.start = np.array([5.0, 5.0, 0.0], dtype=np.float32)
        self.points = np.array(
            [[20.0, 5.0, 0.0], [35.0, 5.0, 0.0], [50.0, 5.0, 0.0]],
            dtype=np.float32,
        )
        self.priorities = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.wind = {"uniform_vector": np.zeros(3, dtype=np.float32)}
        self.base_cfg = {
            "d_model": 32,
            "n_heads": 4,
            "coordinate_scale_m_per_unit": 1.0,
            "max_route_distance": 2000.0,
            "max_mission_time_s": 1200.0,
            "max_episodes": 2,
            "episodes_per_update": 2,
            "ppo_epochs": 1,
            "minibatch_size": 16,
            "validation_scenarios": 1,
            "validation_interval_updates": 1,
            "seed": 19,
        }

    def _model(self, variant_name: str, n_nodes: int = 3) -> ppo.PPO_PtrNet:
        cfg = ppo.resolve_config(
            {**self.base_cfg, "experiment_variant": variant_name}
        )
        return ppo.PPO_PtrNet(
            n_nodes=n_nodes,
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg["n_heads"]),
            lambda_priority=float(cfg["lambda_priority"]),
            policy_architecture=str(cfg["policy_architecture"]),
            training_algorithm=str(cfg["training_algorithm"]),
            experiment_variant=str(cfg["experiment_variant"]),
        )

    @staticmethod
    def _synthetic_batch(model: ppo.PPO_PtrNet, batch_size: int = 12) -> ppo.PPOBatch:
        action_count = model.N + 1
        s_env = torch.randn(batch_size, action_count, ppo.NODE_FEATURE_DIM)
        s_env[:, -1, 5] = 1.0
        s_uav = torch.randn(batch_size, ppo.UAV_FEATURE_DIM)
        m_priority = torch.zeros(batch_size, 1, action_count, action_count)
        legal = torch.ones(batch_size, action_count, dtype=torch.bool)
        remaining = legal.clone()
        remaining[:, -1] = False
        with torch.no_grad():
            _, logits, values = model(
                s_env,
                s_uav,
                m_priority,
                legal,
                legal,
                legal,
                legal,
                legal,
                remaining,
            )
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample()
            old_logp = distribution.log_prob(actions)
        return ppo.PPOBatch(
            s_env=s_env,
            s_uav=s_uav,
            m_priority=m_priority,
            m_visit=legal,
            m_energy=legal,
            m_distance=legal,
            m_time=legal,
            m_dynamics=legal,
            m_remaining=remaining,
            action=actions,
            logp_old=old_logp,
            value_old=values.squeeze(-1),
            returns=torch.linspace(-0.5, 0.8, batch_size),
            advantages=torch.linspace(-1.0, 1.0, batch_size),
        )

    def test_variant_registry_locks_named_experiments(self) -> None:
        self.assertEqual(
            set(ppo.EXPERIMENT_VARIANTS),
            {
                "full",
                "traditional_ppo",
                "ppo_mlp",
                "a2c_pointer",
                "no_priority_bias",
                "no_domain_randomization",
                "no_resource_shaping",
                "no_return_reserve",
            },
        )
        self.assertEqual(
            ppo.resolve_config({"experiment_variant": "traditional_ppo"})[
                "policy_architecture"
            ],
            "flat_mlp_24",
        )
        self.assertEqual(
            ppo.resolve_config({"experiment_variant": "ppo_mlp"})[
                "policy_architecture"
            ],
            "shared_node_mlp",
        )
        self.assertEqual(
            ppo.resolve_config({"experiment_variant": "a2c_pointer"})[
                "training_algorithm"
            ],
            "a2c",
        )
        self.assertEqual(
            ppo.resolve_config({"experiment_variant": "no_priority_bias"})[
                "lambda_priority"
            ],
            0.0,
        )
        with self.assertRaises(ValueError):
            ppo.resolve_config({"experiment_variant": "unknown"})
        with self.assertRaises(ValueError):
            ppo.resolve_config(
                {
                    "experiment_variant": "no_domain_randomization",
                    "domain_randomization": True,
                }
            )
        with self.assertRaises(ValueError):
            ppo.resolve_config(
                {
                    "experiment_variant": "no_priority_bias",
                    "lambda_priority": 0.5,
                }
            )

    def test_shared_node_mlp_is_variable_n_and_parameter_comparable(self) -> None:
        pointer = self._model("full", n_nodes=4)
        mlp_four = self._model("ppo_mlp", n_nodes=4)
        mlp_eight = self._model("ppo_mlp", n_nodes=8)
        pointer_count = ppo.count_trainable_parameters(pointer)
        mlp_count = ppo.count_trainable_parameters(mlp_four)
        self.assertLess(abs(mlp_count - pointer_count) / pointer_count, 0.10)
        self.assertEqual(mlp_count, ppo.count_trainable_parameters(mlp_eight))

        formal_models = []
        for variant in ("full", "ppo_mlp"):
            cfg = ppo.resolve_config(
                {"experiment_variant": variant, "d_model": 128, "n_heads": 4}
            )
            formal_models.append(
                ppo.PPO_PtrNet(
                    n_nodes=16,
                    d_model=128,
                    n_heads=4,
                    lambda_priority=float(cfg["lambda_priority"]),
                    policy_architecture=str(cfg["policy_architecture"]),
                    training_algorithm=str(cfg["training_algorithm"]),
                    experiment_variant=variant,
                )
            )
        formal_counts = [ppo.count_trainable_parameters(model) for model in formal_models]
        self.assertLess(abs(formal_counts[1] - formal_counts[0]) / formal_counts[0], 0.10)

        for model, node_count in ((mlp_four, 4), (mlp_eight, 8)):
            actions = node_count + 1
            legal = torch.ones(2, actions, dtype=torch.bool)
            legal[:, 1] = False
            probabilities, logits, values = model(
                torch.randn(2, actions, ppo.NODE_FEATURE_DIM),
                torch.randn(2, ppo.UAV_FEATURE_DIM),
                torch.zeros(2, 1, actions, actions),
                legal,
                legal,
                legal,
                legal,
                legal,
                legal,
            )
            self.assertEqual(tuple(probabilities.shape), (2, actions))
            self.assertEqual(tuple(values.shape), (2, 1))
            self.assertTrue(torch.all(probabilities[:, 1] == 0.0))
            self.assertTrue(torch.all(torch.isneginf(logits[:, 1])))

    def test_traditional_ppo_uses_fixed_slots_and_exact_parameter_count(self) -> None:
        models = [
            self._model("traditional_ppo", n_nodes=node_count)
            for node_count in (16, 20, 24)
        ]
        self.assertEqual(
            [ppo.count_trainable_parameters(model) for model in models],
            [350_746, 350_746, 350_746],
        )
        for model, node_count in zip(models, (16, 20, 24)):
            action_count = node_count + 1
            s_env = torch.randn(2, action_count, ppo.NODE_FEATURE_DIM)
            s_env[:, :, 5] = 0.0
            s_env[:, -1, 5] = 1.0
            legal = torch.ones(2, action_count, dtype=torch.bool)
            legal[:, 1] = False
            probabilities, logits, values = model(
                s_env,
                torch.randn(2, ppo.UAV_FEATURE_DIM),
                torch.zeros(2, 1, action_count, action_count),
                legal,
                legal,
                legal,
                legal,
                legal,
                legal,
            )
            self.assertEqual(tuple(probabilities.shape), (2, action_count))
            self.assertEqual(tuple(logits.shape), (2, action_count))
            self.assertEqual(tuple(values.shape), (2, 1))
            self.assertTrue(torch.all(probabilities[:, 1] == 0.0))
            self.assertTrue(torch.all(torch.isneginf(logits[:, 1])))
            fixed_env, valid_slots, fixed_legal, observed_n = (
                ppo.FlatMLPActorCritic._fixed_slots(s_env, legal)
            )
            self.assertEqual(observed_n, node_count)
            self.assertEqual(tuple(fixed_env.shape), (2, 25, 15))
            self.assertTrue(torch.all(fixed_env[:, node_count:24] == 0.0))
            self.assertTrue(torch.all(~valid_slots[:, node_count:24]))
            self.assertTrue(torch.all(~fixed_legal[:, node_count:24]))
            self.assertTrue(torch.all(valid_slots[:, 24]))

        too_large = self._model("traditional_ppo", n_nodes=25)
        action_count = 26
        s_env = torch.zeros(1, action_count, ppo.NODE_FEATURE_DIM)
        s_env[:, -1, 5] = 1.0
        legal = torch.ones(1, action_count, dtype=torch.bool)
        with self.assertRaises(ValueError):
            too_large(
                s_env,
                torch.zeros(1, ppo.UAV_FEATURE_DIM),
                torch.zeros(1, 1, action_count, action_count),
                legal,
                legal,
                legal,
                legal,
                legal,
                legal,
            )

    def test_a2c_update_is_single_pass_and_finite(self) -> None:
        cfg = ppo.resolve_config(
            {**self.base_cfg, "experiment_variant": "a2c_pointer"}
        )
        model = self._model("a2c_pointer")
        batch = self._synthetic_batch(model)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        agent = ppo._make_training_agent(model, cfg)
        self.assertIsInstance(agent, ppo.A2CAgent)
        stats = agent.update(batch, entropy_coef=0.01)
        self.assertEqual(stats["epochs_completed"], 1.0)
        self.assertFalse(stats["kl_early_stopped"])
        for name in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "gradient_norm",
            "gradient_norm_pre_clip",
            "explained_variance",
        ):
            self.assertTrue(np.isfinite(stats[name]), name)
        self.assertLessEqual(stats["gradient_norm"], cfg["max_grad_norm"] + 1e-12)
        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, model.parameters())
            )
        )

    def test_no_domain_randomization_forces_nominal_episode(self) -> None:
        cfg = ppo.resolve_config(
            {**self.base_cfg, "experiment_variant": "no_domain_randomization"}
        )
        state = ppo.build_episode(
            self.start,
            self.points,
            self.terrain,
            cfg,
            self.wind,
            np.random.default_rng(999),
            randomize=True,
        )
        randomization = state["episode_randomization"]
        self.assertEqual(randomization["initial_soc"], cfg["initial_soc"])
        self.assertEqual(randomization["distance_scale"], 1.0)
        self.assertEqual(randomization["time_scale"], 1.0)
        self.assertEqual(randomization["wind_scale"], 1.0)
        self.assertEqual(randomization["wind_rotation_deg"], 0.0)
        self.assertEqual(randomization["wind_vertical_bias_mps"], 0.0)

    def test_no_resource_shaping_removes_only_resource_cost(self) -> None:
        full_cfg = ppo.resolve_config({**self.base_cfg, "experiment_variant": "full"})
        ablated_cfg = ppo.resolve_config(
            {**self.base_cfg, "experiment_variant": "no_resource_shaping"}
        )
        full_state = ppo.build_episode(
            self.start, self.points, self.terrain, full_cfg, self.wind
        )
        ablated_state = ppo.build_episode(
            self.start, self.points, self.terrain, ablated_cfg, self.wind
        )
        _, full_reward, _ = ppo.step_env_improved(
            full_state,
            0,
            self.points,
            self.priorities,
            self.terrain,
            full_cfg,
            self.wind,
        )
        _, ablated_reward, _ = ppo.step_env_improved(
            ablated_state,
            0,
            self.points,
            self.priorities,
            self.terrain,
            ablated_cfg,
            self.wind,
        )
        expected_task_gain = 1.0 / 6.0 + 0.2 / 3.0
        self.assertAlmostEqual(ablated_reward, expected_task_gain, places=6)
        self.assertLess(full_reward, ablated_reward)

    def test_no_return_reserve_records_stranded_without_moving(self) -> None:
        reference_cfg = ppo.resolve_config({**self.base_cfg, "experiment_variant": "full"})
        reference_state = ppo.build_episode(
            self.start,
            self.points[-1:],
            self.terrain,
            reference_cfg,
            self.wind,
        )
        estimates, _ = ppo._compute_action_context(
            reference_state, self.priorities[-1:]
        )
        outgoing = estimates[0].outgoing.distance_m
        round_trip = outgoing + estimates[0].return_segment.distance_m
        distance_budget = (outgoing + round_trip) / 2.0
        cfg = ppo.resolve_config(
            {
                **self.base_cfg,
                "experiment_variant": "no_return_reserve",
                "max_route_distance": distance_budget,
            }
        )
        state = ppo.build_episode(
            self.start, self.points[-1:], self.terrain, cfg, self.wind
        )
        true_masks = ppo._compute_action_context(
            state, self.priorities[-1:], reserve_return=True
        )[1]
        policy_masks = ppo._compute_action_context(
            state, self.priorities[-1:], reserve_return=False
        )[1]
        self.assertFalse(bool(true_masks["legal"][0]))
        self.assertTrue(bool(policy_masks["legal"][0]))

        original_position = state["current"].copy()
        state, reward, done = ppo.step_env_improved(
            state,
            0,
            self.points[-1:],
            self.priorities[-1:],
            self.terrain,
            cfg,
            self.wind,
        )
        self.assertTrue(done)
        self.assertEqual(state["termination_reason"], "stranded")
        self.assertEqual(state["constraint_violation_count"], 1)
        self.assertEqual(len(state["executed_segments"]), 0)
        np.testing.assert_array_equal(state["current"], original_position)
        self.assertEqual(state["total_distance"], 0.0)
        self.assertEqual(reward, cfg["simulation_violation_penalty"])

    def test_dynamics_failure_does_not_fabricate_resource_violations(self) -> None:
        """返程受逆风无法保持地速时，只记动力学失败。"""

        point = self.points[-1:]
        priority = self.priorities[-1:]
        # 机场地面风为0，巡航高度沿+x方向为12 m/s：去程顺风可行，
        # 返程逆风地速低于2 m/s。空间风点覆盖整条航段，避免回退到均匀风。
        wind = {
            "positions": np.array(
                [
                    [5.0, 5.0, 0.0],
                    [5.0, 5.0, 18.0],
                    [15.0, 5.0, 18.0],
                    [25.0, 5.0, 18.0],
                    [35.0, 5.0, 18.0],
                    [45.0, 5.0, 18.0],
                    [50.0, 5.0, 18.0],
                ],
                dtype=np.float32,
            ),
            "vectors": np.array(
                [[0.0, 0.0, 0.0]] + [[12.0, 0.0, 0.0]] * 6,
                dtype=np.float32,
            ),
            "uniform_vector": np.zeros(3, dtype=np.float32),
        }
        cfg = ppo.resolve_config(
            {
                **self.base_cfg,
                "experiment_variant": "no_return_reserve",
                "min_ground_speed_mps": 2.0,
            }
        )
        state = ppo.build_episode(self.start, point, self.terrain, cfg, wind)
        estimates, true_masks = ppo._compute_action_context(
            state, priority, reserve_return=True
        )
        policy_masks = ppo._compute_action_context(
            state, priority, reserve_return=False
        )[1]

        self.assertTrue(estimates[0].outgoing.feasible)
        self.assertFalse(estimates[0].return_segment.feasible)
        self.assertEqual(estimates[0].return_segment.reason, "headwind_no_progress")
        self.assertFalse(bool(true_masks["m_dynamics"][0]))
        self.assertTrue(bool(true_masks["m_energy"][0]))
        self.assertTrue(bool(true_masks["m_distance"][0]))
        self.assertTrue(bool(true_masks["m_time"][0]))
        self.assertTrue(bool(policy_masks["legal"][0]))

        state, _, done = ppo.step_env_improved(
            state, 0, point, priority, self.terrain, cfg, wind
        )
        metrics = ppo._episode_metrics(state, priority)
        self.assertTrue(done)
        self.assertEqual(
            state["constraint_violations"][0]["failed_constraints"],
            ["m_dynamics"],
        )
        self.assertTrue(metrics["dynamics_violation"])
        self.assertFalse(metrics["energy_violation"])
        self.assertFalse(metrics["distance_violation"])
        self.assertFalse(metrics["time_violation"])
        self.assertEqual(metrics["constraint_violation_count"], 1)

    def test_checkpoint_records_variant_metadata_and_loads_legacy_full(self) -> None:
        cfg = ppo.resolve_config(
            {**self.base_cfg, "experiment_variant": "ppo_mlp"}
        )
        model = self._model("ppo_mlp")
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "mlp.pt"
            ppo.save_checkpoint(
                checkpoint,
                model,
                cfg,
                training_summary={
                    "environment_interactions": 23,
                    "interaction_count_complete": True,
                },
                training_state={
                    "environment_interactions": 23,
                    "interaction_count_complete": True,
                },
            )
            restored, payload = ppo.load_checkpoint(checkpoint, map_location="cpu")
            self.assertIsInstance(restored.actor, ppo.SharedNodeMLPActor)
            metadata = payload["experiment_metadata"]
            self.assertEqual(metadata["variant"], "ppo_mlp")
            self.assertEqual(metadata["training_algorithm"], "ppo")
            self.assertEqual(metadata["environment_interactions"], 23)
            self.assertEqual(
                metadata["parameter_count"], ppo.count_trainable_parameters(restored)
            )

            full_cfg = ppo.resolve_config(self.base_cfg)
            full_model = self._model("full")
            current = Path(temp_dir) / "current.pt"
            legacy = Path(temp_dir) / "legacy_v2.pt"
            ppo.save_checkpoint(current, full_model, full_cfg)
            old_payload = torch.load(current, map_location="cpu", weights_only=False)
            for key in (
                "experiment_metadata",
                "experiment_variant",
                "policy_architecture",
                "training_algorithm",
                "parameter_count",
                "environment_interactions",
                "interaction_count_complete",
            ):
                old_payload.pop(key, None)
            for key in (
                "experiment_variant",
                "policy_architecture",
                "training_algorithm",
                "domain_randomization",
                "resource_shaping",
                "return_reserve_mask",
                "simulation_only",
                "simulation_violation_penalty",
            ):
                old_payload["cfg"].pop(key, None)
            torch.save(old_payload, legacy)
            legacy_model, legacy_payload = ppo.load_checkpoint(
                legacy, map_location="cpu"
            )
            self.assertIsInstance(legacy_model.actor, ppo.DecoderActor)
            self.assertEqual(legacy_payload["experiment_variant"], "full")
            self.assertFalse(
                legacy_payload["experiment_metadata"]["interaction_count_complete"]
            )

    def test_traditional_ppo_checkpoint_records_fixed_slot_schema(self) -> None:
        cfg = ppo.resolve_config(
            {**self.base_cfg, "experiment_variant": "traditional_ppo"}
        )
        model = self._model("traditional_ppo", n_nodes=24)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "traditional_ppo.pt"
            ppo.save_checkpoint(checkpoint, model, cfg)
            restored, payload = ppo.load_checkpoint(checkpoint, map_location="cpu")
            self.assertIsInstance(
                restored.flat_actor_critic, ppo.FlatMLPActorCritic
            )
            metadata = payload["experiment_metadata"]
            self.assertEqual(metadata["variant"], "traditional_ppo")
            self.assertEqual(metadata["parameter_count"], 350_746)
            self.assertEqual(metadata["fixed_slot_schema"]["depot_slot"], 24)
            self.assertEqual(
                metadata["fixed_slot_schema"]["max_inspection_nodes"], 24
            )

    def test_tiny_training_reports_true_environment_interactions(self) -> None:
        records = []
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = ppo.resolve_config(
                {
                    **self.base_cfg,
                    "checkpoint_dir": temp_dir,
                    "experiment_variant": "full",
                }
            )
            model, returns = ppo.train_policy_improved(
                self.start,
                self.points,
                self.priorities,
                self.terrain,
                cfg,
                self.wind,
                metrics_callback=records.append,
                target_device="cpu",
            )
            self.assertEqual(len(returns), 2)
            self.assertEqual(len(records), 1)
            self.assertGreater(records[0]["environment_interactions"], 0)
            self.assertEqual(
                records[0]["experiment"]["environment_interactions"],
                records[0]["environment_interactions"],
            )
            _, payload = ppo.load_checkpoint(
                Path(temp_dir) / "latest.pt", map_location="cpu"
            )
            self.assertEqual(
                payload["environment_interactions"],
                records[0]["environment_interactions"],
            )
            self.assertEqual(
                model.training_summary["environment_interactions"],
                records[0]["environment_interactions"],
            )


if __name__ == "__main__":
    unittest.main()

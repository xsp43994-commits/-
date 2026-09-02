#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多地图v3.1协议、地图提供器与训练兼容性的快速测试。"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from uav_inspection.core import final_python_ppo_pointer as ppo
from uav_inspection.experiments import paper_multimap_experiments as multimap


class MultiMapProtocolTests(unittest.TestCase):
    def test_atomic_json_retries_transient_windows_replace_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "status.json"
            real_replace = multimap.os.replace
            calls = 0

            def flaky_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 1:
                    error = PermissionError(13, "access denied")
                    error.winerror = 5
                    raise error
                return real_replace(source, destination)

            with mock.patch.object(
                multimap.os, "replace", side_effect=flaky_replace
            ), mock.patch.object(multimap.time, "sleep") as sleep:
                multimap._atomic_json(target, {"state": "running"})

            self.assertEqual(calls, 2)
            sleep.assert_called_once_with(
                multimap.ATOMIC_REPLACE_INITIAL_BACKOFF_SECONDS
            )
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"state": "running"},
            )

    def test_protocol_identity_and_region_balance(self):
        protocol = multimap.load_protocol()
        self.assertEqual(protocol["protocol_version"], "multimap_generalization_v3_2")
        self.assertEqual(
            protocol["parent_protocol_hash"],
            "8014a94241779ca55745ebcf533784a51682a6ff8cfa1ad41af0ce84760e61ce",
        )
        self.assertEqual(
            protocol["asset_parent_protocol_hash"], protocol["parent_protocol_hash"]
        )
        self.assertEqual(
            protocol["formal_evaluation"]["counts"]["total"], 21_648
        )
        self.assertEqual(
            protocol["supersedes_protocol_hash"],
            "20c246fdc986f4fb6654449ef9f306188c410e7c3637e929e8c88b79ddb79c9b",
        )
        self.assertEqual(protocol["oracle_bound_order_tolerance"], 1e-12)
        self.assertEqual(protocol["node_counts"], [16, 20, 24])
        groups = [region["group"] for region in protocol["regions"]]
        self.assertEqual(groups.count("china"), 4)
        self.assertEqual(groups.count("global"), 4)
        mixed_certificate = protocol["task_generation"][
            "mixed_threshold_certificate"
        ]
        self.assertEqual(
            mixed_certificate["resource_order"],
            ["energy", "distance", "time"],
        )
        self.assertTrue(
            mixed_certificate["standard_incumbent_fast_path"]
        )
        self.assertEqual(
            mixed_certificate["maximum_incumbent_priority_shortfall"], 1
        )
        self.assertEqual(
            set(mixed_certificate["eligible_standard_reasons"]),
            {"incumbent_outside_band", "mixed_bottleneck_not_active"},
        )

    def test_initial_soc_floor_is_strictly_above_frozen_reserve(self):
        protocol = multimap.load_protocol()
        minimum = multimap._minimum_valid_initial_soc(protocol)
        self.assertEqual(minimum, 0.251)
        self.assertEqual(
            protocol["task_generation"]["evaluator_safety_bounds"][
                "battery_reserve_ratio"
            ],
            ppo.DEFAULT_CONFIG["battery_reserve_ratio"],
        )
        self.assertGreater(
            minimum, ppo.DEFAULT_CONFIG["battery_reserve_ratio"]
        )
        for section in (
            "mixed_budget_calibration",
            "geometry_budget_compensation",
            "single_constraint_budget_calibration",
        ):
            self.assertGreaterEqual(
                protocol["task_generation"][section]["parameter_bounds"][
                    "initial_soc"
                ][0],
                minimum,
            )

    def test_copernicus_tile_url_handles_hemispheres(self):
        protocol = multimap.load_protocol()
        north = multimap.copernicus_tile_url(33.4, 108.2, protocol)
        south = multimap.copernicus_tile_url(-43.2, 170.4, protocol)
        self.assertIn("N33_00_E108_00", north)
        self.assertIn("S44_00_E170_00", south)

    def test_candidate_grid_is_deterministic_and_interior(self):
        protocol = multimap.load_protocol()
        region = protocol["regions"][0]
        first = multimap.candidate_centers(region, 5)
        second = multimap.candidate_centers(region, 5)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 25)
        west, south, east, north = region["bbox_wgs84"]
        self.assertTrue(
            all(
                west < item["longitude"] < east
                and south < item["latitude"] < north
                for item in first
            )
        )

    def test_terrain_metrics_distinguish_relief_and_nodata(self):
        flat = np.zeros((20, 20), dtype=np.float32)
        slope = np.tile(
            np.linspace(0.0, 600.0, 20, dtype=np.float32), (20, 1)
        )
        slope[0, 0] = np.nan
        flat_metrics = multimap.terrain_metrics(flat, 30.0)
        slope_metrics = multimap.terrain_metrics(slope, 30.0)
        self.assertEqual(flat_metrics["relief_m"], 0.0)
        self.assertGreater(slope_metrics["relief_m"], 500.0)
        self.assertAlmostEqual(slope_metrics["nodata_fraction"], 1.0 / 400.0)

    def test_procedural_terrain_and_roads_are_reproducible(self):
        first_rng = np.random.default_rng(42)
        second_rng = np.random.default_rng(42)
        first = multimap._spectral_terrain(
            first_rng, (64, 64), hurst=0.7, target_relief_m=800.0
        )
        second = multimap._spectral_terrain(
            second_rng, (64, 64), hurst=0.7, target_relief_m=800.0
        )
        np.testing.assert_array_equal(first, second)
        self.assertGreater(np.percentile(first, 99) - np.percentile(first, 1), 700.0)
        for topology in (
            "multi_arm",
            "t_y",
            "curved_trunk_branches",
            "loop_spur",
        ):
            roads = multimap._procedural_roads(
                np.random.default_rng(7), first.shape, topology
            )
            self.assertGreaterEqual(len(roads), 3)
            self.assertTrue(all(road.shape[1] == 2 for road in roads))

    def test_synthetic_test_generation_requires_training_freeze(self):
        protocol = multimap.load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_registry = root / "real_registry.json"
            real_registry.write_text(
                json.dumps(
                    {
                        "protocol_hash": protocol["protocol_hash"],
                        "registry_hash": "a" * 64,
                        "selection_complete": True,
                        "maps": [
                            {"terrain_metrics": {"relief_m": 500.0 + index}}
                            for index in range(8)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "冻结前"):
                multimap.prepare_procedural_maps(
                    multimap.DEFAULT_PROTOCOL,
                    root / "maps",
                    splits=["synthetic_test"],
                    real_registry_path=real_registry,
                )

    def test_task_shard_cli_preserves_explicit_map_range(self):
        args = multimap._build_parser().parse_args(
            [
                "prepare-tasks",
                "--split",
                "training",
                "--map-registry",
                "registry.json",
                "--map-index-start",
                "4",
                "--map-index-stop",
                "22",
                "--shard-name",
                "shard_00",
            ]
        )
        self.assertEqual(args.map_index_start, 4)
        self.assertEqual(args.map_index_stop, 22)
        self.assertEqual(args.shard_name, "shard_00")
        merge = multimap._build_parser().parse_args(
            [
                "merge-task-shards",
                "--split",
                "training",
                "--map-registry",
                "registry.json",
                "--base-records",
                "base.jsonl",
            ]
        )
        self.assertEqual(merge.command, "merge-task-shards")


class FrozenMapProviderTests(unittest.TestCase):
    @staticmethod
    def _write_map(root: Path, map_id: str, elevation):
        map_dir = root / "procedural" / "training"
        map_dir.mkdir(parents=True, exist_ok=True)
        if np.isscalar(elevation):
            terrain = np.full((64, 64), elevation, dtype=np.float32)
        else:
            terrain = np.asarray(elevation, dtype=np.float32)
        road_points = np.asarray([[2.0, 2.0], [30.0, 2.0]], dtype=np.float32)
        road_offsets = np.asarray([0, 2], dtype=np.int32)
        metadata = {
            "map_id": map_id,
            "coordinate_scale_m_per_unit": 1.0,
        }
        map_hash = multimap._map_hash(
            terrain, road_points, road_offsets, metadata
        )
        path = map_dir / f"{map_id}.npz"
        np.savez_compressed(
            path,
            terrain=terrain,
            road_points=road_points,
            road_offsets=road_offsets,
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
            map_hash=np.asarray(map_hash),
        )
        return {
            "map_id": map_id,
            "map_hash": map_hash,
            "map_file": str(path.relative_to(root)),
            "map_file_sha256": multimap._sha256_file(path),
        }

    def test_provider_loads_distinct_maps_and_checks_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                self._write_map(root, "map_a", 100.0),
                self._write_map(root, "map_b", 900.0),
            ]
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps({"maps": records}), encoding="utf-8"
            )
            provider = multimap.FrozenMapProvider.from_registries(
                root, [registry_path]
            )
            first = provider(
                {
                    "map_id": "map_a",
                    "map_hash": records[0]["map_hash"],
                    "start_xy": [2.0, 2.0],
                }
            )
            second = provider(
                {
                    "map_id": "map_b",
                    "map_hash": records[1]["map_hash"],
                    "start_xy": [2.0, 2.0],
                }
            )
            self.assertEqual(float(first["terrain"][0, 0]), 100.0)
            self.assertEqual(float(second["terrain"][0, 0]), 900.0)
            with self.assertRaisesRegex(RuntimeError, "身份漂移"):
                provider(
                    {
                        "map_id": "map_a",
                        "map_hash": "0" * 64,
                        "start_xy": [2.0, 2.0],
                    }
                )

    def test_provider_uses_bilinear_start_height(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terrain = np.add.outer(
                np.arange(64, dtype=np.float32) * 10.0,
                np.arange(64, dtype=np.float32),
            )
            record = self._write_map(root, "gradient", terrain)
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps({"maps": [record]}), encoding="utf-8"
            )
            provider = multimap.FrozenMapProvider.from_registries(
                root, [registry_path]
            )
            context = provider(
                {
                    "map_id": "gradient",
                    "map_hash": record["map_hash"],
                    "start_xy": [2.25, 3.5],
                }
            )
            self.assertAlmostEqual(
                float(context["start_pos"][2]), 37.251, places=4
            )

    def test_task_design_balances_node_counts(self):
        designs = [multimap._task_design(0, index) for index in range(9)]
        self.assertEqual(
            [item["node_count"] for item in designs],
            [16, 16, 16, 20, 20, 20, 24, 24, 24],
        )
        self.assertEqual(
            [item["difficulty"] for item in designs],
            ["moderate", "hard", "extreme"] * 3,
        )
        self.assertEqual(
            set(item["priority_layout"] for item in designs),
            {"clustered", "dispersed", "far_high_conflict"},
        )

    def test_geometry_only_radius_expands_for_sparse_road(self):
        protocol = multimap.load_protocol()
        bundle = {
            "terrain": np.zeros((267, 267), dtype=np.float32),
            "roads": [
                np.asarray(
                    [[53.0, 133.0], [133.0, 133.0], [213.0, 133.0]],
                    dtype=np.float32,
                )
            ],
            "metadata": {"coordinate_scale_m_per_unit": 30.0},
        }
        record = {"map_id": "sparse_test_map"}
        first = multimap._effective_task_radius_range(
            record,
            bundle,
            protocol,
            node_count=16,
            difficulty="moderate",
        )
        second = multimap._effective_task_radius_range(
            record,
            bundle,
            protocol,
            node_count=16,
            difficulty="moderate",
        )
        self.assertEqual(first, second)
        minimum_feasible, effective_range = first
        self.assertGreater(minimum_feasible, 1400.0)
        self.assertEqual(effective_range[0], minimum_feasible)
        self.assertEqual(effective_range[1] - effective_range[0], 300.0)

    def test_screening_only_forwards_intersecting_bounds(self):
        parent = json.loads(
            multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
        )
        record = {"difficulty": "hard"}
        self.assertTrue(
            multimap._screening_bounds_intersect_band(
                record,
                {
                    "weighted_coverage_lower_bound": 0.45,
                    "weighted_coverage_upper_bound": 0.62,
                    "returned": True,
                    "visited_count": 3,
                },
                parent,
            )
        )
        self.assertFalse(
            multimap._screening_bounds_intersect_band(
                record,
                {
                    "weighted_coverage_lower_bound": 0.2,
                    "weighted_coverage_upper_bound": 0.4,
                    "returned": True,
                    "visited_count": 2,
                },
                parent,
            )
        )

    def test_decisive_screening_avoids_redundant_final_solve(self):
        self.assertTrue(
            multimap._screening_certificate_is_decisive(
                True, {"optimality_certified": False}
            )
        )
        self.assertTrue(
            multimap._screening_certificate_is_decisive(
                False, {"optimality_certified": True}
            )
        )
        self.assertFalse(
            multimap._screening_certificate_is_decisive(
                False, {"optimality_certified": False}
            )
        )

    def test_mixed_budget_calibration_adjusts_only_second_resource(self):
        protocol = multimap.load_protocol()
        parent = json.loads(
            multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
        )
        candidate = {
            "constraint_type": "mixed",
            "difficulty": "moderate",
            "initial_soc": 0.8,
            "distance_budget_scale": 1.2,
            "time_budget_scale": 0.7,
        }
        certificate = {
            "weighted_coverage_lower_bound": 0.8,
            "returned": True,
            "visited_count": 12,
            "energy_utilization": 0.99,
            "distance_utilization": 0.89,
            "time_utilization": 0.5,
            "scenario_hash": "a" * 64,
        }
        calibrated = multimap._calibrate_mixed_candidate(
            candidate,
            certificate,
            protocol,
            parent,
            iteration=1,
        )
        self.assertIsNotNone(calibrated)
        self.assertEqual(calibrated["initial_soc"], candidate["initial_soc"])
        self.assertEqual(
            calibrated["time_budget_scale"], candidate["time_budget_scale"]
        )
        self.assertLess(
            calibrated["distance_budget_scale"],
            candidate["distance_budget_scale"],
        )
        self.assertEqual(
            calibrated["mixed_budget_calibration_trace"][0][
                "adjusted_resource"
            ],
            "distance",
        )
        outside_band = dict(certificate)
        outside_band["weighted_coverage_lower_bound"] = 0.6
        self.assertIsNone(
            multimap._calibrate_mixed_candidate(
                candidate,
                outside_band,
                protocol,
                parent,
                iteration=1,
            )
        )

    def test_single_constraint_calibration_moves_budget_toward_band(self):
        protocol = multimap.load_protocol()
        parent = json.loads(
            multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
        )
        candidate = {
            "constraint_type": "distance",
            "difficulty": "moderate",
            "initial_soc": 1.0,
            "distance_budget_scale": 1.0,
            "time_budget_scale": 1.5,
        }
        certificate = {
            "weighted_coverage_lower_bound": 0.6,
            "weighted_coverage_upper_bound": 0.8,
            "returned": True,
            "visited_count": 10,
            "energy_utilization": 0.5,
            "distance_utilization": 0.95,
            "time_utilization": 0.3,
            "scenario_hash": "b" * 64,
        }
        calibrated = multimap._calibrate_single_constraint_candidate(
            candidate,
            certificate,
            protocol,
            parent,
            iteration=1,
        )
        self.assertIsNotNone(calibrated)
        self.assertGreater(
            calibrated["distance_budget_scale"],
            candidate["distance_budget_scale"],
        )
        self.assertEqual(calibrated["initial_soc"], candidate["initial_soc"])
        self.assertEqual(
            calibrated["time_budget_scale"], candidate["time_budget_scale"]
        )
        self.assertEqual(
            calibrated["single_constraint_budget_calibration_trace"][0][
                "direction"
            ],
            "loosen",
        )

    def test_energy_calibration_never_crosses_evaluator_soc_floor(self):
        protocol = multimap.load_protocol()
        parent = json.loads(
            multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
        )
        candidate = {
            "constraint_type": "energy",
            "difficulty": "moderate",
            "initial_soc": 0.26,
            "distance_budget_scale": 1.2,
            "time_budget_scale": 1.2,
        }
        certificate = {
            "weighted_coverage_lower_bound": 0.95,
            "weighted_coverage_upper_bound": 0.95,
            "returned": True,
            "visited_count": 15,
            "energy_utilization": 0.6,
            "distance_utilization": 0.4,
            "time_utilization": 0.3,
            "scenario_hash": "c" * 64,
        }
        calibrated = multimap._calibrate_single_constraint_candidate(
            candidate,
            certificate,
            protocol,
            parent,
            iteration=1,
        )
        self.assertIsNotNone(calibrated)
        self.assertEqual(calibrated["initial_soc"], 0.251)
        base_cfg = ppo.resolve_config({"reward_schema": "multimap_v3_1"})
        scenario_cfg, _ = ppo.apply_frozen_domain_instance(
            base_cfg,
            {"wind_vectors": np.zeros((1, 3), dtype=np.float32)},
            {
                "initial_soc": calibrated["initial_soc"],
                "distance_budget_scale": 1.0,
                "time_budget_scale": 1.0,
                "power_scale": 1.0,
                "wind_scale": 1.0,
                "wind_rotation_deg": 0.0,
                "wind_vertical_bias_mps": 0.0,
            },
        )
        self.assertEqual(scenario_cfg["initial_soc"], 0.251)

    def test_budget_transform_audit_replays_final_budget(self):
        protocol = multimap.load_protocol()
        parent = json.loads(
            multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
        )
        record = {
            "id": "validation__map_a__task_00",
            "node_count": 16,
            "difficulty": "moderate",
            "constraint_type": "energy",
            "task_radius_m": 1000.0,
            "effective_task_radius_range_m": [900.0, 1200.0],
            "nominal_radius_at_sample_m": 1066.6666666666667,
            "geometry_budget_compensation_factor": 1.0,
            "uncompensated_budget_values": {
                "initial_soc": 0.7,
                "distance_budget_scale": 1.15,
                "time_budget_scale": 1.1,
            },
            "initial_soc": 0.7,
            "distance_budget_scale": 1.15,
            "time_budget_scale": 1.1,
        }
        self.assertEqual(
            multimap._audit_budget_transform_record(record, protocol, parent),
            [],
        )
        record["distance_budget_scale"] = 1.14
        self.assertIn(
            "final_budget_replay_failed=validation__map_a__task_00:"
            "distance_budget_scale",
            multimap._audit_budget_transform_record(record, protocol, parent),
        )

    def test_non_intended_resource_relaxation_is_target_preserving(self):
        protocol = multimap.load_protocol()
        candidate = {
            "constraint_type": "time",
            "initial_soc": 0.94,
            "distance_budget_scale": 1.18,
            "time_budget_scale": 0.64,
        }
        relaxed = multimap._relax_non_intended_resources(
            candidate, protocol
        )
        self.assertIsNotNone(relaxed)
        self.assertEqual(relaxed["initial_soc"], 1.0)
        self.assertEqual(relaxed["distance_budget_scale"], 2.2)
        self.assertEqual(
            relaxed["time_budget_scale"], candidate["time_budget_scale"]
        )
        self.assertEqual(
            relaxed["non_intended_resource_relaxation"][
                "released_parameters"
            ],
            ["initial_soc", "distance_budget_scale"],
        )

    def test_budget_transform_audit_replays_non_intended_relaxation(self):
        protocol = multimap.load_protocol()
        parent = json.loads(
            multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
        )
        record = {
            "id": "validation__map_a__task_00",
            "node_count": 16,
            "difficulty": "moderate",
            "constraint_type": "time",
            "task_radius_m": 1000.0,
            "effective_task_radius_range_m": [900.0, 1200.0],
            "nominal_radius_at_sample_m": 1066.6666666666667,
            "geometry_budget_compensation_factor": 1.0,
            "uncompensated_budget_values": {
                "initial_soc": 0.94,
                "distance_budget_scale": 1.18,
                "time_budget_scale": 0.64,
            },
            "initial_soc": 1.0,
            "distance_budget_scale": 2.2,
            "time_budget_scale": 0.64,
            "non_intended_resource_relaxation": {
                "rule": (
                    "single_constraint_release_non_intended_to_upper_bound_v1"
                ),
                "intended_resource": "time",
                "before": {
                    "initial_soc": 0.94,
                    "distance_budget_scale": 1.18,
                    "time_budget_scale": 0.64,
                },
                "after": {
                    "initial_soc": 1.0,
                    "distance_budget_scale": 2.2,
                    "time_budget_scale": 0.64,
                },
                "released_parameters": [
                    "initial_soc",
                    "distance_budget_scale",
                ],
            },
        }
        self.assertEqual(
            multimap._audit_budget_transform_record(record, protocol, parent),
            [],
        )
        record["non_intended_resource_relaxation"]["after"][
            "distance_budget_scale"
        ] = 2.1
        self.assertIn(
            "non_intended_relaxation_replay_failed="
            "validation__map_a__task_00",
            multimap._audit_budget_transform_record(record, protocol, parent),
        )

    def test_resource_threshold_periodic_trigger_schedule(self):
        fallback = multimap.load_protocol()["task_generation"][
            "resource_threshold_fallback"
        ]
        self.assertFalse(
            multimap._resource_threshold_fallback_is_triggered(19, fallback)
        )
        self.assertTrue(
            multimap._resource_threshold_fallback_is_triggered(20, fallback)
        )
        self.assertFalse(
            multimap._resource_threshold_fallback_is_triggered(199, fallback)
        )
        self.assertTrue(
            multimap._resource_threshold_fallback_is_triggered(200, fallback)
        )
        self.assertTrue(
            multimap._resource_threshold_fallback_is_triggered(700, fallback)
        )
        self.assertFalse(
            multimap._resource_threshold_fallback_is_triggered(701, fallback)
        )
        self.assertTrue(
            multimap._resource_threshold_fallback_is_triggered(1900, fallback)
        )
        self.assertFalse(
            multimap._resource_threshold_fallback_is_triggered(2000, fallback)
        )

    def test_mixed_threshold_rejects_large_priority_shortfall_before_solver(self):
        protocol = multimap.load_protocol()
        parent = json.loads(
            multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
        )
        ok, _record, proof, reason = (
            multimap._certify_mixed_with_lower_threshold_route(
                {
                    "constraint_type": "mixed",
                    "difficulty": "moderate",
                    "priorities": [3.0] * 16,
                },
                {
                    "weighted_coverage_lower_bound": 0.5,
                    "weighted_coverage_upper_bound": 0.8,
                },
                lambda _record: self.fail(
                    "大幅低于难度带时不应构造地图问题或调用求解器"
                ),
                protocol,
                parent,
            )
        )
        self.assertFalse(ok)
        self.assertEqual(
            reason, "mixed_threshold_incumbent_too_far_below_band"
        )
        self.assertGreater(proof["priority_shortfall"], 1)

    def test_multimap_reward_matches_registered_formula(self):
        state = {
            "cfg": {
                "resource_shaping": True,
                "reward_schema": "multimap_v3_1",
            },
            "energy_budget_wh": 100.0,
            "max_route_distance": 100.0,
            "max_mission_time_s": 100.0,
        }
        reward = ppo._resource_cost_reward(
            10.0,
            10.0,
            10.0,
            state,
            np.asarray([3.0, 1.0]),
        )
        self.assertAlmostEqual(reward, -0.00625)

    def test_training_uses_provider_hash_and_preserves_legacy_default(self):
        self.assertEqual(ppo.resolve_config({})["reward_schema"], "legacy_v2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                self._write_map(root, "map_a", 0.0),
                self._write_map(root, "map_b", 10.0),
            ]
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps({"maps": records}), encoding="utf-8"
            )
            provider = multimap.FrozenMapProvider.from_registries(
                root, [registry_path]
            )

            def instance(identifier, record, x_offset):
                return {
                    "id": identifier,
                    "map_id": record["map_id"],
                    "map_hash": record["map_hash"],
                    "start_xy": [2.0, 2.0],
                    "node_count": 2,
                    "inspection_points_xyz": [
                        [8.0 + x_offset, 2.0, 0.0],
                        [14.0 + x_offset, 2.0, 0.0],
                    ],
                    "priorities": [3.0, 1.0],
                    "service_times_s": [1.0, 1.0],
                    "initial_soc": 1.0,
                    "distance_budget_scale": 1.0,
                    "time_budget_scale": 1.0,
                    "wind_scale": 1.0,
                    "wind_rotation_deg": 0.0,
                    "wind_vertical_bias_mps": 0.0,
                    "power_scale": 1.0,
                }

            instances = [
                instance("a", records[0], 0.0),
                instance("b", records[1], 1.0),
            ]
            cfg = {
                "reward_schema": "multimap_v3_1",
                "d_model": 16,
                "n_heads": 4,
                "max_route_distance": 500.0,
                "max_mission_time_s": 500.0,
                "max_episodes": 2,
                "episodes_per_update": 1,
                "ppo_epochs": 1,
                "minibatch_size": 16,
                "validation_interval_updates": 99,
                "seed": 42,
            }
            model, returns = ppo.train_policy_improved(
                [2.0, 2.0, 0.0],
                np.asarray(instances[0]["inspection_points_xyz"]),
                np.asarray(instances[0]["priorities"]),
                np.zeros((64, 64), dtype=np.float32),
                cfg,
                {"uniform_vector": np.zeros(3, dtype=np.float32)},
                target_device="cpu",
                training_instances=instances,
                validation_instances=instances,
                scenario_provider=provider,
            )
            self.assertEqual(len(returns), 2)
            summary = model.training_summary
            self.assertEqual(
                summary["experiment"]["scenario_mode"],
                "frozen_multimap_v3_1",
            )
            self.assertEqual(
                summary["experiment"]["scenario_provider_hash"],
                provider.provider_hash,
            )

    def test_multimap_training_config_freezes_protocol_and_provider(self):
        protocol = multimap.load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._write_map(root, "map_a", 10.0)
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps({"maps": [record]}), encoding="utf-8"
            )
            provider = multimap.FrozenMapProvider.from_registries(
                root, [registry_path]
            )
            context = provider(
                {
                    "map_id": "map_a",
                    "map_hash": record["map_hash"],
                    "start_xy": [2.0, 2.0],
                }
            )
            cfg = multimap._multimap_training_cfg(
                protocol,
                provider,
                context,
                variant="traditional_ppo",
                seed=42,
                episodes=600,
                monitor_episodes=[100, 200, 400, 600],
                run_dir=root / "run",
                training_manifest_hash="a" * 64,
                validation_manifest_hash="b" * 64,
                stage="pilot",
            )
            self.assertEqual(cfg["reward_schema"], "multimap_v3_1")
            self.assertEqual(cfg["experiment_variant"], "traditional_ppo")
            self.assertEqual(cfg["scenario_provider_hash"], provider.provider_hash)
            self.assertEqual(
                cfg["multimap_protocol_hash"], protocol["protocol_hash"]
            )

    def test_formal_group_health_requires_all_five_seeds(self):
        protocol = multimap.load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = 250
            for variant in protocol["formal_training"]["new_training_variants"]:
                for seed in protocol["formal_training"]["seeds"]:
                    path = (
                        root
                        / "formal_training"
                        / f"formal_{variant}_seed{seed}_3000ep"
                        / "health"
                        / f"episode_{episode:04d}.json"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(
                            {
                                "variant": variant,
                                "training_seed": seed,
                                "validation": {
                                    "return_rate": 0.99,
                                    "zero_visit_rate": 0.01,
                                    "median_oracle_attainment_lower": 0.6,
                                    "safe_weighted_coverage": 0.45,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
            report = multimap._refresh_formal_group_health(
                root, protocol, episode
            )
            self.assertIsNotNone(report)
            self.assertEqual(report["core_model_count"], 5)
            self.assertFalse(report["collective_stop_required"])


if __name__ == "__main__":
    unittest.main()

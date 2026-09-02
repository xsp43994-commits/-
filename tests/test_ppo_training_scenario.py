#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPO真实尺度双国道场景的回归测试。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio

from uav_inspection.core import ppo_training_scenario as scenario_api


class TestPPOTrainingScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = scenario_api.build_training_scenario()

    def test_dem_scale_crs_crop_and_intersection(self):
        scene = self.scenario
        self.assertEqual(scene.crs, "EPSG:32651")
        self.assertAlmostEqual(scene.coordinate_scale_m_per_unit, 12.5, places=6)
        self.assertLess(scene.terrain.shape[0], scene.metadata["source_shape"][0])
        self.assertLess(scene.terrain.shape[1], scene.metadata["source_shape"][1])
        np.testing.assert_allclose(
            scene.airport_global_pixel,
            np.array([2837.54, 2364.38]),
            atol=0.10,
        )
        self.assertLess(scene.metadata["intersection_centerline_sample_gap_m"], 2.0)

        # 机场必须同时位于两条道路中心线附近，而不是用手工圆心代替。
        for road in (scene.road_1, scene.road_2):
            nearest_m = float(
                np.min(np.linalg.norm(road[:, :2] - scene.start_pos[:2], axis=1))
                * scene.coordinate_scale_m_per_unit
            )
            self.assertLess(nearest_m, 2.0)

    def test_four_arms_and_spacing_rule(self):
        scene = self.scenario
        self.assertEqual(scene.inspection_points.shape, (16, 3))
        np.testing.assert_array_equal(np.bincount(scene.point_arm_ids), np.full(4, 4))
        for arm_id in range(4):
            mask = scene.point_arm_ids == arm_id
            np.testing.assert_array_equal(scene.point_segment_ids[mask], np.arange(4))
            distances = np.sort(scene.point_along_arm_distances_m[mask])
            self.assertGreaterEqual(float(np.min(np.diff(distances))), 150.0 - 1e-5)

        pairwise = np.linalg.norm(
            (scene.inspection_points[:, None, :2] - scene.inspection_points[None, :, :2])
            * scene.coordinate_scale_m_per_unit,
            axis=2,
        )
        pairwise[pairwise <= 1e-9] = np.inf
        actual_global_min = float(np.min(pairwise))
        self.assertAlmostEqual(
            actual_global_min,
            float(scene.metadata["global_min_euclidean_spacing_m"]),
            places=5,
        )
        self.assertAlmostEqual(
            float(scene.metadata["same_arm_min_along_road_spacing_m"]),
            min(
                float(np.min(np.diff(np.sort(scene.point_along_arm_distances_m[scene.point_arm_ids == arm]))))
                for arm in range(4)
            ),
            places=5,
        )
        self.assertGreaterEqual(actual_global_min, 130.0 - 1e-5)
        self.assertLess(actual_global_min, 150.0)  # 130米是透明折中，不能误报为全局150米
        self.assertIn("global_euclidean_spacing_m>=130", scene.metadata["spacing_rule"])
        self.assertIn("150 m rule is geometrically infeasible", scene.metadata["spacing_geometry_note"])

    def test_points_are_on_corresponding_road_and_ground(self):
        scene = self.scenario
        for index, point in enumerate(scene.inspection_points):
            road = scene.road_1 if int(scene.point_arm_ids[index]) < 2 else scene.road_2
            nearest_m = float(
                np.min(np.linalg.norm(road[:, :2] - point[:2], axis=1))
                * scene.coordinate_scale_m_per_unit
            )
            self.assertLess(nearest_m, 0.10)
            ground = float(scenario_api._bilinear(scene.terrain, point[:2].reshape(1, 2))[0])
            self.assertAlmostEqual(float(point[2]), ground, places=3)

    def test_risk_formula_and_priority_counts(self):
        scene = self.scenario
        weights = np.array([0.35, 0.25, 0.20, 0.20], dtype=np.float32)
        np.testing.assert_allclose(scene.risk_scores, scene.risk_components @ weights, atol=1e-6)
        self.assertEqual(int(np.sum(scene.priorities == 3)), 5)
        self.assertEqual(int(np.sum(scene.priorities == 2)), 6)
        self.assertEqual(int(np.sum(scene.priorities == 1)), 5)
        high = scene.risk_scores[scene.priorities == 3]
        middle = scene.risk_scores[scene.priorities == 2]
        low = scene.risk_scores[scene.priorities == 1]
        self.assertGreaterEqual(float(np.min(high)), float(np.max(middle)) - 1e-7)
        self.assertGreaterEqual(float(np.min(middle)), float(np.max(low)) - 1e-7)

    def test_wind_coordinates_height_and_safety_cap(self):
        scene = self.scenario
        ground = scenario_api._bilinear(scene.terrain, scene.wind_positions[:, :2])
        agl = scene.wind_positions[:, 2] - ground
        np.testing.assert_allclose(agl, 18.0, atol=1e-3)
        self.assertEqual(scene.metadata["wind_display_offset_m"], 0.0)
        self.assertEqual(scene.metadata["wind_axes"], "X=east, Y=south, Z=up")

        config = scenario_api.ScenarioConfig()
        horizontal = np.linalg.norm(scene.wind_vectors[:, :2], axis=1) * config.wind_domain_scale_max
        vertical = np.abs(scene.wind_vectors[:, 2]) * config.wind_domain_scale_max
        worst_speed = np.sqrt(horizontal**2 + (vertical + config.wind_vertical_bias_max_mps) ** 2)
        self.assertLessEqual(float(np.max(worst_speed)), 8.0 + 1e-6)

    def test_pixel_utm_and_wind_axis_round_trips(self):
        scene = self.scenario
        local = np.vstack([scene.start_pos, scene.inspection_points[:3]]).astype(np.float64)
        global_pixel = scene.local_pixel_to_global_pixel(local)
        np.testing.assert_allclose(scene.global_pixel_to_local_pixel(global_pixel), local, atol=1e-9)
        utm = scene.local_pixel_to_utm(local)
        np.testing.assert_allclose(scene.utm_to_local_pixel(utm), local, atol=1e-8)
        np.testing.assert_allclose(utm[0, :2], scene.airport_utm, atol=1e-6)
        affine = rasterio.Affine(*scene.local_affine.tolist())
        expected_east, expected_north = rasterio.transform.xy(
            affine,
            float(scene.start_pos[1]),
            float(scene.start_pos[0]),
            offset="center",
        )
        np.testing.assert_allclose(
            scene.airport_utm,
            np.array([expected_east, expected_north]),
            atol=1e-6,
        )
        self.assertIn("pixel center", scene.metadata["pixel_coordinate_convention"])

        enu = np.array([[2.0, 3.0, 0.5], [-1.0, -4.0, -0.2]], dtype=np.float32)
        model = scenario_api.enu_wind_to_model(enu)
        np.testing.assert_allclose(model[:, 0], enu[:, 0])
        np.testing.assert_allclose(model[:, 1], -enu[:, 1])
        np.testing.assert_allclose(model[:, 2], enu[:, 2])
        np.testing.assert_allclose(scenario_api.model_wind_to_enu(model), enu)

    def test_npz_json_round_trip_and_hash(self):
        scene = self.scenario
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir) / "real_scale_scene"
            npz_path, json_path = scenario_api.save_training_scenario(scene, prefix)
            self.assertTrue(npz_path.exists())
            self.assertTrue(json_path.exists())
            loaded = scenario_api.load_training_scenario(json_path)
            self.assertEqual(loaded.scenario_hash, scene.scenario_hash)
            self.assertEqual(scenario_api.compute_scenario_hash(loaded), scene.scenario_hash)
            np.testing.assert_array_equal(loaded.inspection_points, scene.inspection_points)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["scenario_hash"], scene.scenario_hash)
            self.assertEqual(len(payload["inspection_points"]), 16)
            self.assertIn("risk_components_raw", payload["inspection_points"][0])

    def test_generation_is_deterministic(self):
        repeated = scenario_api.build_training_scenario()
        self.assertEqual(repeated.scenario_hash, self.scenario.scenario_hash)
        np.testing.assert_array_equal(repeated.inspection_points, self.scenario.inspection_points)
        np.testing.assert_array_equal(repeated.wind_vectors, self.scenario.wind_vectors)

        relocated = copy.deepcopy(self.scenario)
        relocated.metadata["dem_source"] = "another/folder/same-dem.tif"
        relocated.metadata["dem_source_absolute"] = "D:/moved/same-dem.tif"
        self.assertEqual(
            scenario_api.compute_scenario_hash(relocated), self.scenario.scenario_hash
        )

    def test_training_mapping_contract(self):
        inputs = self.scenario.as_training_inputs()
        for key in (
            "start",
            "points",
            "priorities",
            "terrain",
            "wind_data",
            "cfg",
            "service_times_s",
            "coordinate_scale_m_per_unit",
        ):
            self.assertIn(key, inputs)
        self.assertEqual(inputs["coordinate_scale_m_per_unit"], 12.5)
        self.assertEqual(inputs["service_times_s"].shape, (16,))

    def test_nominal_witness_and_randomized_preflight(self):
        report = scenario_api.preflight_scenario(self.scenario)
        self.assertTrue(report["passed"])
        self.assertTrue(report["nominal"]["returned_full"])
        self.assertEqual(report["nominal"]["visited_count"], 16)
        self.assertEqual(report["nominal"]["constraint_violations"], 0)
        self.assertEqual(len(report["randomized_initial_feasible_counts"]), 8)
        self.assertGreaterEqual(min(report["randomized_initial_feasible_counts"]), 12)
        self.assertTrue(report["worst_case_has_patrol_action"])
        self.assertGreater(report["worst_corner"]["initial_feasible_patrol_points"], 0)
        self.assertTrue(report["worst_corner"]["return_action_legal"])


if __name__ == "__main__":
    unittest.main()

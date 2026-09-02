from __future__ import annotations

import unittest

from uav_inspection.figures import v3_2_14_multibackend_redraw as redraw


class MultibackendRedrawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = redraw._read_results()

    def test_registry_and_prototype_gate(self) -> None:
        self.assertEqual(len(redraw.FIGURES), 20)
        self.assertEqual(sum(x["tier"] == "main" for x in redraw.FIGURES.values()), 10)
        self.assertEqual(sum(x["tier"] == "supplementary" for x in redraw.FIGURES.values()), 8)
        self.assertEqual(sum(x["tier"] == "showcase" for x in redraw.FIGURES.values()), 2)
        self.assertEqual(sum(x["backend"] == "origin" for x in redraw.FIGURES.values()), 10)
        self.assertEqual(sum(x["backend"] == "python" for x in redraw.FIGURES.values()), 9)
        self.assertEqual(sum(x["backend"] == "matlab" for x in redraw.FIGURES.values()), 1)
        self.assertEqual(redraw.PROTOTYPE_IDS, ("M01", "M02", "M06", "V02"))

    def test_frozen_result_contract(self) -> None:
        self.assertEqual(len(self.results), 21_648)
        self.assertNotIn("ppo_mlp", set(self.results["model"].astype(str)))
        self.assertEqual(redraw._sha256(redraw.FINAL_RESULTS), redraw.EXPECTED_RESULTS_SHA256)
        self.assertEqual(redraw._sha256(redraw.EVALUATION_MATRIX), redraw.EXPECTED_MATRIX_SHA256)

    def test_m01_map_level_source(self) -> None:
        frame = redraw.build_m01(self.results)
        self.assertEqual(len(frame), 192)
        self.assertEqual(frame.loc[frame["domain"].eq("synthetic"), "map_id"].nunique(), 24)
        self.assertEqual(frame.loc[frame["domain"].eq("real"), "map_id"].nunique(), 8)
        self.assertEqual(set(frame["model"]), set(redraw.MAIN_COMPARE))

    def test_m02_robust_effect_source(self) -> None:
        frame = redraw.build_m02(self.results)
        self.assertEqual(len(frame), 12)
        self.assertTrue((frame["n_maps"] == 8).all())
        self.assertEqual(set(frame["metric"]), {"safe", "returned"})
        self.assertTrue(frame[["estimate_pp", "ci_low_pp", "ci_high_pp"]].notna().all().all())

    def test_m06_three_models_share_3000_episode_axis(self) -> None:
        frame = redraw.build_m06()
        raw = frame[frame["record_type"].eq("seed")]
        expected = {"full", "a2c_pointer", "traditional_ppo"}
        self.assertEqual(set(raw["model"]), expected)
        self.assertTrue((raw.groupby("model")["seed"].nunique() == 5).all())
        self.assertTrue((raw.groupby(["model", "seed"])["episodes_seen"].max() == 3000).all())
        self.assertEqual(set(raw["curve_source"]), {"training_batch"})
        self.assertNotIn("validation_mode", frame.columns)
        self.assertIn("同一训练任务分布", redraw.CAPTIONS["M06"])

    def test_v02_fixed_route_assets(self) -> None:
        bundle = redraw.build_v02()
        self.assertEqual(len(bundle["terrain"]), 267 * 267)
        self.assertEqual(len(bundle["points"]), 25)
        self.assertEqual(
            set(bundle["routes"]["model"]),
            {"full", "a2c_pointer", "traditional_ppo", "milp"},
        )
        self.assertEqual(bundle["metadata"].loc[bundle["metadata"]["key"].eq("task_id"), "value"].iloc[0], redraw.REAL_EXAMPLE)


if __name__ == "__main__":
    unittest.main()

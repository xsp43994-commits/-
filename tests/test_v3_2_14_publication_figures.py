from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT

ROOT = WORKSPACE_ROOT
sys.path.insert(0, str(ROOT))

from uav_inspection.figures import v3_2_14_publication_figures as figures  # noqa: E402
from uav_inspection.figures import v3_2_14_split_publication_figures as split_figures  # noqa: E402


class PublicationFigurePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = figures.load_bundle()
        cls.audit = figures.audit_inputs(cls.bundle)
        cls.output = split_figures.DEFAULT_OUTPUT

    def test_frozen_inputs(self) -> None:
        self.assertTrue(self.audit["passed"])
        self.assertEqual(self.audit["row_count"], 21_648)
        self.assertEqual(self.audit["route_count"], 21_648)
        self.assertEqual(self.audit["matrix_sha256"], figures.EXPECTED_MATRIX_SHA256)
        self.assertEqual(self.audit["results_sha256"], figures.EXPECTED_RESULTS_SHA256)
        self.assertNotIn("ppo_mlp", self.audit["active_models"])

    def test_figure_contract_is_complete(self) -> None:
        self.assertEqual(len(figures.FIGURE_ORDER), 16)
        self.assertEqual(len(set(figures.FIGURE_ORDER)), 16)
        for stem in figures.FIGURE_ORDER:
            self.assertIn(stem, figures.FIGURE_REGISTRY)
            self.assertIn(stem, figures.PANEL_INTERFACE)
            self.assertTrue(callable(figures._builder(stem)))

    def test_fixed_examples_match_frozen_rule(self) -> None:
        for task_id in (figures.SYNTHETIC_EXAMPLE, figures.REAL_EXAMPLE):
            task = figures._task_by_id(self.bundle, task_id)
            self.assertEqual(task["node_count"], 24)
            self.assertEqual(task["difficulty"], "extreme")
            self.assertEqual(task["constraint_type"], "mixed")
            self.assertEqual(task["priority_layout"], "far_high_conflict")
            for model in ("full", "a2c_pointer", "traditional_ppo", "milp"):
                self.assertIsNotNone(figures._route_payload(model, task_id, 42), (task_id, model))

    def test_training_history_is_five_seed_complete(self) -> None:
        history = figures.load_training_history(figures.LEARNING_MODELS)
        counts = history.groupby(["model", "training_seed"]).size()
        self.assertEqual(len(counts), 35)
        self.assertTrue((counts == 192).all())
        self.assertTrue((history["environment_interactions"] >= 0).all())

    def test_rendered_delivery_and_registry(self) -> None:
        manifest_path = self.output / "figure_manifest.json"
        qa_path = self.output / "qa_report.json"
        self.assertTrue(manifest_path.exists())
        self.assertTrue(qa_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["panel_count"], 72)
        self.assertEqual(len(manifest["panels"]), 72)
        self.assertTrue(qa["passed"])
        self.assertIn("figV01_3d_taihang_route", manifest["panels"])
        self.assertNotIn("figV01_3d_route_a", manifest["panels"])
        for record in manifest["panels"].values():
            self.assertEqual(set(record["files"]), {"svg", "pdf", "png", "tiff"})
            svg = (self.output / record["files"]["svg"]["path"]).read_text(encoding="utf-8")
            self.assertNotIn("ppo_mlp", svg)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from uav_inspection.figures import v3_2_14_multibackend_full as full
from uav_inspection.figures import v3_2_14_multibackend_redraw as core
from uav_inspection.figures.v3_2_14_literature_audit import audit_summary


class MultibackendFullDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = core.OUTPUT
        cls.source = cls.root / "source_data"

    def read_source(self, figure_id: str) -> pd.DataFrame:
        return pd.read_csv(self.source / f"{figure_id}_source_data.csv")

    def test_frozen_20_figure_backend_contract(self) -> None:
        self.assertEqual(len(core.FIGURES), 20)
        self.assertEqual(full.ALL_IDS, tuple(core.PROTOTYPE_IDS) + full.REMAINING_IDS)
        self.assertEqual(len(full.ORIGIN_IDS) + 1, 10)  # M02加九张全量阶段Origin图
        self.assertEqual(sum(x["backend"] == "python" for x in core.FIGURES.values()), 9)
        self.assertEqual(sum(x["backend"] == "matlab" for x in core.FIGURES.values()), 1)

    def test_origin_template_revisions_are_frozen(self) -> None:
        expected_scatter = {"M03", "M07", "M08", "M10", "S02", "S03", "S07"}
        for figure_id in expected_scatter:
            self.assertEqual(core.FIGURES[figure_id]["template"], "SCATTER.OTP")
        self.assertEqual(core.FIGURES["M04"]["template"], "SCATTERINTERVAL.otp")
        self.assertEqual(core.FIGURES["M05"]["template"], "LINE.OTP")

    def test_source_data_row_contracts(self) -> None:
        expected = {
            "M03": 8, "M04": 18, "M05": 2160, "M07": 18, "M08": 6,
            "M09": 28, "M10": 8, "S01": 4608, "S02": 12, "S03": 9,
            "S04": 39, "S05": 168, "S06": 1050, "S07": 24, "S08": 35,
            "V01": 960,
        }
        for figure_id, row_count in expected.items():
            with self.subTest(figure_id=figure_id):
                self.assertEqual(len(self.read_source(figure_id)), row_count)

    def test_training_curves_end_at_3000_episodes(self) -> None:
        main = self.read_source("M06")
        raw = main[main["record_type"].eq("seed")]
        self.assertEqual(set(raw["model"]), {"full", "a2c_pointer", "traditional_ppo"})
        self.assertTrue((raw.groupby(["model", "seed"])["episodes_seen"].max() == 3000).all())
        supplement = self.read_source("S06")
        self.assertEqual(set(supplement["model"]), set(full.LEARNING_MODELS))
        self.assertTrue((supplement.groupby("model")["episodes_seen"].max() == 3000).all())

    def test_fixed_route_contracts(self) -> None:
        synthetic = self.read_source("V01")
        self.assertEqual(set(synthetic.loc[synthetic["record_type"].eq("route"), "model"]), {"full", "a2c_pointer", "traditional_ppo", "milp"})
        real_dir = self.source / "V02"
        metadata = pd.read_csv(real_dir / "metadata.csv")
        self.assertEqual(metadata.loc[metadata["key"].eq("task_id"), "value"].iloc[0], core.REAL_EXAMPLE)
        self.assertEqual(set(pd.read_csv(real_dir / "routes.csv")["model"]), {"full", "a2c_pointer", "traditional_ppo", "milp"})

    def test_literature_audit_meets_frozen_minimum(self) -> None:
        summary = audit_summary()
        self.assertGreaterEqual(summary["paper_count"], 20)
        self.assertGreaterEqual(summary["figure_count"], 50)
        delivered = pd.read_csv(self.root / "literature_audit" / "literature_style_audit.csv")
        self.assertEqual(len(delivered), summary["paper_count"])
        self.assertEqual(int(delivered["figure_count"].sum()), summary["figure_count"])
        self.assertTrue(delivered["official_url"].str.startswith("https://").all())

    def test_registry_manifests_captions_and_opju_are_complete(self) -> None:
        registry = json.loads((self.root / "manifests" / "figure_registry_manual_v3.json").read_text(encoding="utf-8"))
        self.assertEqual(registry["gate_state"], "completed")
        self.assertEqual(len(registry["figures"]), 20)
        self.assertEqual(registry["literature_audit"], {"paper_count": 20, "figure_count": 60})
        self.assertEqual(sum((self.root / "manifests" / f"{figure_id}.json").is_file() for figure_id in full.ALL_IDS), 20)
        self.assertEqual(sum((self.root / "captions_CN" / f"{figure_id}.md").is_file() for figure_id in full.ALL_IDS), 20)
        self.assertEqual(len(list((self.root / "origin_projects").glob("*.opju"))), 10)

    def test_excluded_old_model_never_enters_delivery_sources(self) -> None:
        for path in sorted(self.source.rglob("*.csv")):
            with self.subTest(path=path.name):
                self.assertNotIn("ppo_mlp", path.read_text(encoding="utf-8-sig", errors="replace"))

    def test_s02_caption_matches_final_encoding(self) -> None:
        caption = (self.root / "captions_CN" / "S02.md").read_text(encoding="utf-8")
        self.assertIn("安全率均为100%", caption)
        self.assertNotIn("点大小表示安全率", caption)
        self.assertTrue((self.read_source("S02")["safe_rate"] == 1.0).all())


if __name__ == "__main__":
    unittest.main()

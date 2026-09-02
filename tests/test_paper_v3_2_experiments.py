#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3.2 正式评价矩阵的行数、排除和层级选择回归测试。"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT

from uav_inspection.experiments import paper_v3_2_experiments as v32


ROOT = WORKSPACE_ROOT
PROTOCOL = ROOT / "paper_runs/protocols/multimap_generalization_v3_2/protocol.json"


def _task(map_id: str, index: int, *, road: int | None = None) -> dict:
    payload = {
        "id": f"test__{map_id}" + (f"__road_{road}" if road is not None else "") + f"__task_{index}",
        "map_id": map_id,
        "task_index": index,
        "node_count": (16, 20, 24)[index // 3],
        "difficulty": ("moderate", "hard", "extreme")[index % 3],
    }
    if road is not None:
        payload["road_index"] = road
    payload["task_hash"] = hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()
    return payload


class V32EvaluationMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.synthetic = [_task(f"synthetic_{map_index:02d}", task_index) for map_index in range(24) for task_index in range(9)]
        self.real = [_task(f"real_{map_index:02d}", task_index, road=road) for map_index in range(8) for road in range(2) for task_index in range(9)]

    def test_matrix_has_exactly_preregistered_rows(self) -> None:
        rows = v32.compile_evaluation_rows(PROTOCOL, self.synthetic, self.real)
        self.assertEqual(len(rows), 21648)
        counts = {family: sum(row["family"] == family for row in rows) for family in {row["family"] for row in rows}}
        self.assertEqual(counts, {
            "synthetic_learning": 7560,
            "synthetic_main_baselines": 3888,
            "synthetic_supplementary": 504,
            "real_learning": 5040,
            "real_baselines": 1152,
            "known_domain_shift": 1008,
            "hidden_model_perception_mismatch": 2496,
        })
        self.assertNotIn("ppo_mlp", {row["model"] for row in rows})

    def test_subsets_keep_prespecified_map_and_road_structure(self) -> None:
        supplementary = v32.select_supplementary_tasks(self.synthetic)
        robust = v32.select_robustness_tasks(self.real)
        self.assertEqual(len(supplementary), 72)
        self.assertEqual(len(robust), 24)
        self.assertEqual(sum(row["road_index"] == 0 for row in robust), 12)
        self.assertEqual(sum(row["road_index"] == 1 for row in robust), 12)
        self.assertEqual({row["node_count"] for row in robust if row["map_id"] == "real_00"}, {16, 20, 24})


if __name__ == "__main__":
    unittest.main()

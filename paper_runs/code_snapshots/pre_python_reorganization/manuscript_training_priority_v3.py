#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transparent 100-point training-and-robustness-priority draft scorecard."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import manuscript_multiobjective_v1 as v1
import manuscript_training_aware_v2 as v2


ROOT = Path(__file__).resolve().parent
PARENT = v2.DESTINATION
PARENT_MANIFEST = PARENT / "analysis_manifest.json"
PARENT_DIMENSIONS = PARENT / "seven_dimension_scores.csv"
PARENT_GRID = PARENT / "weight_sensitivity_grid.csv"
PARENT_SUMMARY = PARENT / "weight_sensitivity_summary.csv"
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "manuscript_training_priority_v3_protocol.json"
)
DESTINATION = v1.OUTPUT / "analysis/manuscript_training_priority_v3"
SCORE_SCALE = 100.0
# 该情景强调训练可靠性与部署鲁棒性，但所有权重仍在v2冻结范围内。
PRIORITY_WEIGHTS = {
    "D1": 0.20,
    "D2": 0.10,
    "D3": 0.10,
    "D4": 0.15,
    "D5": 0.05,
    "D6": 0.20,
    "D7": 0.20,
}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate() -> dict[str, Any]:
    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if parent["state"] != "ready_for_plot_plan" or parent["plots_created"]:
        raise RuntimeError("parent v2 is not frozen pre-plot")
    if protocol["parent_v2_manifest_hash"] != parent["manifest_hash"]:
        raise RuntimeError("parent v2 manifest drift")
    if protocol["implementation_sha256"] != v1.sha256_file(Path(__file__)):
        raise RuntimeError("v3 implementation hash drift")
    actual = v1.canonical_hash(
        {key: value for key, value in protocol.items() if key != "protocol_hash"}
    )
    if protocol["protocol_hash"] != actual:
        raise RuntimeError("v3 protocol hash drift")
    for name, value in PRIORITY_WEIGHTS.items():
        lower, upper = v2.WEIGHT_RANGES[name]
        if not lower <= value <= upper:
            raise RuntimeError(f"priority weight outside v2 range: {name}")
    if abs(sum(PRIORITY_WEIGHTS.values()) - 1.0) > 1e-12:
        raise RuntimeError("priority weights do not sum to one")
    return protocol


def score_rows() -> list[dict[str, Any]]:
    output = []
    for row in _csv(PARENT_DIMENSIONS):
        values = {f"D{index}": float(row[f"D{index}"]) for index in range(1, 8)}
        for method, function in (
            ("geometric", v1.weighted_geometric),
            ("arithmetic", v1.weighted_arithmetic),
        ):
            raw = function(values, PRIORITY_WEIGHTS)
            output.append(
                {
                    "scope": "training_robustness_priority_draft",
                    "model": row["model"],
                    "aggregation": method,
                    "score_0_to_1": raw,
                    "score_0_to_100": SCORE_SCALE * raw,
                    **{f"weight_D{index}": PRIORITY_WEIGHTS[f"D{index}"] for index in range(1, 8)},
                }
            )
    return output


def sensitivity_grid_100() -> list[dict[str, Any]]:
    output = []
    for row in _csv(PARENT_GRID):
        output.append(
            {
                **row,
                "score_0_to_1": float(row["score"]),
                "score_0_to_100": SCORE_SCALE * float(row["score"]),
            }
        )
    return output


def sensitivity_summary_100() -> list[dict[str, Any]]:
    output = []
    for row in _csv(PARENT_SUMMARY):
        converted: dict[str, Any] = dict(row)
        for field in ("minimum_score", "maximum_score", "mean_score"):
            converted[f"{field}_0_to_1"] = float(row[field])
            converted[f"{field}_0_to_100"] = SCORE_SCALE * float(row[field])
        output.append(converted)
    return output


def _gap_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in ("geometric", "arithmetic"):
        selected = {
            str(row["model"]): float(row["score_0_to_100"])
            for row in rows
            if row["aggregation"] == method
        }
        output.append(
            {
                "aggregation": method,
                "full_score": selected["full"],
                "a2c_pointer_score": selected["a2c_pointer"],
                "full_minus_a2c_points": selected["full"] - selected["a2c_pointer"],
                "traditional_ppo_score": selected["traditional_ppo"],
            }
        )
    return output


def run() -> dict[str, Any]:
    protocol = _validate()
    scores = score_rows()
    grid = sensitivity_grid_100()
    summary = sensitivity_summary_100()
    gaps = _gap_rows(scores)
    files = {
        "priority_scores_100.csv": scores,
        "priority_pairwise_gaps.csv": gaps,
        "weight_sensitivity_grid_100.csv": grid,
        "weight_sensitivity_summary_100.csv": summary,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.svg", "*.pdf"):
        if list(DESTINATION.rglob(pattern)):
            raise RuntimeError("plot files are forbidden before plot planning")
    for name, content in files.items():
        v1._atomic_csv(DESTINATION / name, content)
    hashes = {name: v1.sha256_file(DESTINATION / name) for name in files}
    audit = {
        "schema_version": "manuscript_training_priority_v3",
        "passed": True,
        "analysis_role": "post_result_training_robustness_priority_draft_scenario",
        "score_scale": SCORE_SCALE,
        "model_count": 3,
        "sensitivity_vector_count": len(v2.enumerate_weight_grid()),
        "sensitivity_result_count": len(grid),
        "weights_within_parent_ranges": True,
        "parent_results_modified": False,
        "plots_created": False,
        "protocol_hash": protocol["protocol_hash"],
        "output_hashes": hashes,
    }
    audit["audit_hash"] = v1.canonical_hash(audit)
    v1._atomic_json(DESTINATION / "priority_score_audit.json", audit)
    hashes["priority_score_audit.json"] = v1.sha256_file(
        DESTINATION / "priority_score_audit.json"
    )
    manifest = {
        "schema_version": "manuscript_training_priority_v3",
        "state": "ready_for_plot_plan",
        "plots_created": False,
        "plot_files": [],
        "parent_v2_manifest_hash": json.loads(
            PARENT_MANIFEST.read_text(encoding="utf-8")
        )["manifest_hash"],
        "protocol_hash": protocol["protocol_hash"],
        "audit_hash": audit["audit_hash"],
        "output_hashes": hashes,
    }
    manifest["manifest_hash"] = v1.canonical_hash(manifest)
    v1._atomic_json(DESTINATION / "analysis_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

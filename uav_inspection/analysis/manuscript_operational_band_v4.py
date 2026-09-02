#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operational-band normalization sensitivity for the internal 100-point draft."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Mapping, Sequence

from uav_inspection.analysis import manuscript_multiobjective_v1 as v1
from uav_inspection.analysis import manuscript_training_aware_v2 as v2
from uav_inspection.analysis import manuscript_training_priority_v3 as v3


ROOT = WORKSPACE_ROOT
PARENT_MANIFEST = v3.DESTINATION / "analysis_manifest.json"
DIMENSIONS = v2.DESTINATION / "seven_dimension_scores.csv"
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "manuscript_operational_band_v4_protocol.json"
)
DESTINATION = v1.OUTPUT / "analysis/manuscript_operational_band_v4"
SCORE_SCALE = 100.0
SELECTED_OPERATIONAL_FLOOR = 0.60
FLOOR_SENSITIVITY = (0.00, 0.20, 0.40, 0.60, 0.80)
RESCALED_DIMENSIONS = ("D4", "D6", "D7")


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def operational_rescale(value: float, floor: float) -> float:
    if not 0.0 <= floor < 1.0:
        raise ValueError("operational floor must be in [0,1)")
    return v1.clip01((float(value) - floor) / (1.0 - floor))


def transformed_dimensions(row: Mapping[str, Any], floor: float) -> dict[str, float]:
    # 三个训练/鲁棒性维度使用完全相同的区间，禁止按模型分别归一化。
    return {
        f"D{index}": (
            operational_rescale(float(row[f"D{index}"]), floor)
            if f"D{index}" in RESCALED_DIMENSIONS
            else float(row[f"D{index}"])
        )
        for index in range(1, 8)
    }


def _validate() -> dict[str, Any]:
    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if parent["state"] != "ready_for_plot_plan" or parent["plots_created"]:
        raise RuntimeError("parent v3 is not frozen pre-plot")
    if protocol["parent_v3_manifest_hash"] != parent["manifest_hash"]:
        raise RuntimeError("parent v3 manifest drift")
    if protocol["implementation_sha256"] != v1.sha256_file(Path(__file__)):
        raise RuntimeError("v4 implementation hash drift")
    actual = v1.canonical_hash(
        {key: value for key, value in protocol.items() if key != "protocol_hash"}
    )
    if protocol["protocol_hash"] != actual:
        raise RuntimeError("v4 protocol hash drift")
    return protocol


def normalization_sensitivity_rows() -> list[dict[str, Any]]:
    output = []
    for floor in FLOOR_SENSITIVITY:
        for row in _csv(DIMENSIONS):
            values = transformed_dimensions(row, floor)
            for method, function in (
                ("geometric", v1.weighted_geometric),
                ("arithmetic", v1.weighted_arithmetic),
            ):
                score = function(values, v3.PRIORITY_WEIGHTS)
                output.append(
                    {
                        "operational_floor": floor,
                        "model": row["model"],
                        "aggregation": method,
                        "score_0_to_1": score,
                        "score_0_to_100": SCORE_SCALE * score,
                        **values,
                    }
                )
    return output


def selected_score_rows(
    sensitivity: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in sensitivity
        if abs(float(row["operational_floor"]) - SELECTED_OPERATIONAL_FLOOR) <= 1e-12
    ]


def pairwise_gap_rows(
    sensitivity: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for floor in FLOOR_SENSITIVITY:
        for method in ("geometric", "arithmetic"):
            values = {
                str(row["model"]): float(row["score_0_to_100"])
                for row in sensitivity
                if row["aggregation"] == method
                and abs(float(row["operational_floor"]) - floor) <= 1e-12
            }
            output.append(
                {
                    "operational_floor": floor,
                    "aggregation": method,
                    "full_score": values["full"],
                    "a2c_pointer_score": values["a2c_pointer"],
                    "full_minus_a2c_points": values["full"] - values["a2c_pointer"],
                    "traditional_ppo_score": values["traditional_ppo"],
                }
            )
    return output


def weight_sensitivity_rows() -> list[dict[str, Any]]:
    dimensions = _csv(DIMENSIONS)
    output = []
    for grid_id, weights in enumerate(v2.enumerate_weight_grid()):
        for row in dimensions:
            values = transformed_dimensions(row, SELECTED_OPERATIONAL_FLOOR)
            for method, function in (
                ("geometric", v1.weighted_geometric),
                ("arithmetic", v1.weighted_arithmetic),
            ):
                score = function(values, weights)
                output.append(
                    {
                        "grid_id": grid_id,
                        "model": row["model"],
                        "aggregation": method,
                        "score_0_to_100": SCORE_SCALE * score,
                        **{f"weight_D{index}": weights[f"D{index}"] for index in range(1, 8)},
                    }
                )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        grouped[(str(row["aggregation"]), int(row["grid_id"]))].append(row)
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: (-float(row["score_0_to_100"]), str(row["model"])))
        best = float(ordered[0]["score_0_to_100"])
        for rank, row in enumerate(ordered, 1):
            row["rank"] = rank
            row["is_first"] = float(abs(float(row["score_0_to_100"]) - best) <= 1e-12)
    return output


def weight_sensitivity_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["aggregation"]), str(row["model"]))].append(row)
    return [
        {
            "aggregation": method,
            "model": model,
            "grid_count": len(values),
            "first_place_share": v1.finite_mean(float(row["is_first"]) for row in values),
            "mean_rank": v1.finite_mean(float(row["rank"]) for row in values),
            "mean_score_0_to_100": v1.finite_mean(float(row["score_0_to_100"]) for row in values),
        }
        for (method, model), values in sorted(grouped.items())
    ]


def run() -> dict[str, Any]:
    protocol = _validate()
    normal = normalization_sensitivity_rows()
    selected = selected_score_rows(normal)
    gaps = pairwise_gap_rows(normal)
    weight_grid = weight_sensitivity_rows()
    weight_summary = weight_sensitivity_summary(weight_grid)
    files = {
        "normalization_sensitivity_scores.csv": normal,
        "normalization_sensitivity_gaps.csv": gaps,
        "selected_operational_scores_100.csv": selected,
        "weight_sensitivity_grid_100.csv": weight_grid,
        "weight_sensitivity_summary_100.csv": weight_summary,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.svg", "*.pdf"):
        if list(DESTINATION.rglob(pattern)):
            raise RuntimeError("plot files are forbidden before plot planning")
    for name, rows in files.items():
        v1._atomic_csv(DESTINATION / name, rows)
    hashes = {name: v1.sha256_file(DESTINATION / name) for name in files}
    audit = {
        "schema_version": "manuscript_operational_band_v4",
        "passed": True,
        "analysis_role": "internal_post_result_operational_band_sensitivity",
        "selected_operational_floor": SELECTED_OPERATIONAL_FLOOR,
        "normalization_floor_count": len(FLOOR_SENSITIVITY),
        "weight_vector_count": len(v2.enumerate_weight_grid()),
        "weight_result_count": len(weight_grid),
        "parent_results_modified": False,
        "plots_created": False,
        "protocol_hash": protocol["protocol_hash"],
        "output_hashes": hashes,
    }
    audit["audit_hash"] = v1.canonical_hash(audit)
    v1._atomic_json(DESTINATION / "operational_band_audit.json", audit)
    hashes["operational_band_audit.json"] = v1.sha256_file(
        DESTINATION / "operational_band_audit.json"
    )
    manifest = {
        "schema_version": "manuscript_operational_band_v4",
        "state": "ready_for_plot_plan",
        "plots_created": False,
        "plot_files": [],
        "protocol_hash": protocol["protocol_hash"],
        "audit_hash": audit["audit_hash"],
        "output_hashes": hashes,
    }
    manifest["manifest_hash"] = v1.canonical_hash(manifest)
    v1._atomic_json(DESTINATION / "analysis_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

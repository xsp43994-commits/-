#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent audit for the seven-dimension v2 pre-plot package."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from uav_inspection.analysis import manuscript_multiobjective_v1 as v1
from uav_inspection.analysis import manuscript_training_aware_v2 as v2


REPORT = v1.OUTPUT / "analysis/manuscript_training_aware_v2_independent_audit.json"
EXPECTED_ROWS = {
    "training_seed_metrics.csv": 15,
    "training_dimension_scores.csv": 3,
    "seven_dimension_scores.csv": 3,
    "scenario_scores.csv": 30,
    "weight_sensitivity_grid.csv": 7482,
    "weight_sensitivity_summary.csv": 6,
}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run() -> dict:
    destination = v2.DESTINATION
    manifest = json.loads(
        (destination / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (destination / "training_aware_audit.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(v2.PROTOCOL.read_text(encoding="utf-8"))
    parent = json.loads(v2.V1_MANIFEST.read_text(encoding="utf-8"))
    checks = {
        "manifest_hash_valid": manifest["manifest_hash"]
        == v1.canonical_hash(
            {key: value for key, value in manifest.items() if key != "manifest_hash"}
        ),
        "audit_hash_valid": audit["audit_hash"]
        == v1.canonical_hash(
            {key: value for key, value in audit.items() if key != "audit_hash"}
        ),
        "protocol_hash_valid": protocol["protocol_hash"]
        == v1.canonical_hash(
            {key: value for key, value in protocol.items() if key != "protocol_hash"}
        ),
        "implementation_hash_valid": protocol["implementation_sha256"]
        == v1.sha256_file(Path(v2.__file__)),
        "training_source_hashes_valid": protocol["training_source_hashes"]
        == v2.training_source_hashes(),
        "parent_v1_hash_valid": protocol["parent_v1_manifest_hash"]
        == parent["manifest_hash"],
        "winner_constraint_absent": protocol["desired_winner_or_margin_constraint"] is None
        and audit["desired_winner_constraint_used"] is False,
        "ready_for_plot_plan": manifest["state"] == "ready_for_plot_plan",
        "plots_created_false": manifest["plots_created"] is False,
    }
    plot_files = [
        str(path)
        for pattern in ("*.png", "*.svg", "*.pdf")
        for path in destination.rglob(pattern)
    ]
    checks["no_plot_files"] = not plot_files
    hash_mismatches = [
        name
        for name, expected in manifest["output_hashes"].items()
        if v1.sha256_file(destination / name) != expected
    ]
    checks["output_hashes_valid"] = not hash_mismatches
    row_mismatches = {}
    for name, expected in EXPECTED_ROWS.items():
        actual = len(_csv(destination / name))
        if actual != expected:
            row_mismatches[name] = {"expected": expected, "actual": actual}
    checks["row_counts_valid"] = not row_mismatches
    invalid_scores = []
    for row in _csv(destination / "seven_dimension_scores.csv"):
        for index in range(1, 8):
            value = float(row[f"D{index}"])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                invalid_scores.append(f"{row['model']}/D{index}")
    checks["dimension_scores_valid"] = not invalid_scores
    invalid_weights = []
    for row in _csv(destination / "weight_sensitivity_grid.csv"):
        total = sum(float(row[f"weight_D{index}"]) for index in range(1, 8))
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            invalid_weights.append(f"{row['aggregation']}/{row['grid_id']}/{row['model']}")
    checks["weight_vectors_valid"] = not invalid_weights
    report = {
        "schema_version": "manuscript_training_aware_v2_independent_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "plot_files": plot_files,
        "hash_mismatches": hash_mismatches,
        "row_count_mismatches": row_mismatches,
        "invalid_scores": invalid_scores,
        "invalid_weight_vectors": invalid_weights,
        "protocol_hash": protocol["protocol_hash"],
        "manifest_hash": manifest["manifest_hash"],
    }
    report["audit_hash"] = v1.canonical_hash(report)
    v1._atomic_json(REPORT, report)
    if not report["passed"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent integrity audit for the pre-plot multi-objective package."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from uav_inspection.analysis import manuscript_multiobjective_v1 as multi


REPORT = (
    multi.OUTPUT
    / "analysis/manuscript_multiobjective_v1_independent_audit.json"
)
EXPECTED_CSV_ROWS = {
    "nominal_task_metrics.csv": 4392,
    "nominal_map_dimensions.csv": 400,
    "dimension_scores.csv": 19,
    "robustness_condition_dimensions.csv": 272,
    "robustness_model_dimensions.csv": 3,
    "mechanism_robustness_summary.csv": 11,
    "scenario_scores.csv": 152,
    "weight_sensitivity_grid.csv": 8550,
    "pareto_membership.csv": 19,
    "training_stability_seed.csv": 10,
    "training_stability_summary.csv": 2,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run() -> dict:
    destination = multi.DESTINATION
    manifest = json.loads(
        (destination / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (destination / "multiobjective_audit.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(multi.PROTOCOL.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "manifest_hash_valid": manifest["manifest_hash"]
        == multi.canonical_hash(
            {key: value for key, value in manifest.items() if key != "manifest_hash"}
        ),
        "analysis_audit_hash_valid": audit["audit_hash"]
        == multi.canonical_hash(
            {key: value for key, value in audit.items() if key != "audit_hash"}
        ),
        "protocol_hash_valid": protocol["protocol_hash"]
        == multi.canonical_hash(
            {key: value for key, value in protocol.items() if key != "protocol_hash"}
        ),
        "matrix_hash_valid": multi.sha256_file(multi.MATRIX)
        == multi.EXPECTED_MATRIX_SHA256,
        "result_hash_valid": multi.sha256_file(multi.FINAL_RESULTS)
        == multi.EXPECTED_RESULTS_SHA256,
        "implementation_hash_valid": multi.sha256_file(Path(multi.__file__))
        == protocol["implementation_sha256"],
        "ready_for_plot_plan": manifest["state"] == "ready_for_plot_plan",
        "plots_created_false": manifest["plots_created"] is False,
        "raw_rows_exact": audit["raw_result_row_count"] == multi.EXPECTED_ROWS,
        "raw_keys_unique": audit["raw_unique_identity_count"] == multi.EXPECTED_ROWS,
        "missing_robustness_not_imputed": audit["missing_robustness_imputed"] is False,
    }
    plot_files = [
        str(path)
        for pattern in ("*.png", "*.svg", "*.pdf")
        for path in destination.rglob(pattern)
    ]
    checks["no_plot_files"] = not plot_files
    hash_mismatches = []
    for name, expected in manifest["csv_and_audit_hashes"].items():
        actual = multi.sha256_file(destination / name)
        if actual != expected:
            hash_mismatches.append(name)
    checks["all_output_hashes_valid"] = not hash_mismatches

    row_count_mismatches = {}
    for name, expected in EXPECTED_CSV_ROWS.items():
        actual = len(_read_csv(destination / name))
        if actual != expected:
            row_count_mismatches[name] = {"expected": expected, "actual": actual}
    checks["csv_row_counts_valid"] = not row_count_mismatches

    dimensions = _read_csv(destination / "dimension_scores.csv")
    invalid_dimensions = []
    for row in dimensions:
        for name in ("D1", "D2", "D3", "D5"):
            value = float(row[name])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                invalid_dimensions.append(f"{row['scope']}/{row['model']}/{name}")
        d4 = float(row["D4"])
        should_exist = row["scope"] == "core_learning_complete"
        if should_exist != math.isfinite(d4):
            invalid_dimensions.append(f"{row['scope']}/{row['model']}/D4")
    checks["dimension_values_valid"] = not invalid_dimensions

    grids = _read_csv(destination / "weight_sensitivity_grid.csv")
    invalid_weights = []
    for row in grids:
        total = sum(float(row[f"weight_D{index}"]) for index in range(1, 6))
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            invalid_weights.append(
                f"{row['scope']}/{row['model']}/{row['grid_id']}"
            )
    checks["weight_vectors_sum_to_one"] = not invalid_weights
    passed = all(checks.values())
    report = {
        "schema_version": "manuscript_multiobjective_v1_independent_audit",
        "passed": passed,
        "checks": checks,
        "hash_mismatches": hash_mismatches,
        "row_count_mismatches": row_count_mismatches,
        "invalid_dimensions": invalid_dimensions,
        "invalid_weight_vectors": invalid_weights,
        "plot_files": plot_files,
        "protocol_hash": protocol["protocol_hash"],
        "manifest_hash": manifest["manifest_hash"],
    }
    report["audit_hash"] = multi.canonical_hash(report)
    multi._atomic_json(REPORT, report)
    if not passed:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

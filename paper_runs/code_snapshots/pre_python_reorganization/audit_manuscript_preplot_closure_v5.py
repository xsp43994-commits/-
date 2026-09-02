#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent final audit for the v5 pre-plot closure package."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import manuscript_multiobjective_v1 as v1
import manuscript_preplot_closure_v5 as v5


REPORT = v1.OUTPUT / "analysis/manuscript_preplot_closure_v5_independent_audit.json"
EXPECTED_ROWS = {
    "joint_normalization_weight_sensitivity.csv": 37410,
    "joint_sensitivity_summary.csv": 30,
    "paired_dimension_units.csv": 36,
    "paired_dimension_tests.csv": 3,
    "hierarchical_bootstrap_distribution.csv": 10000,
    "hierarchical_bootstrap_summary.csv": 6,
}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run() -> dict:
    destination = v5.DESTINATION
    manifest = json.loads(
        (destination / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (destination / "preplot_closure_audit.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(v5.PROTOCOL.read_text(encoding="utf-8"))
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
        == v1.sha256_file(Path(v5.__file__)),
        "source_manifest_chain_valid": protocol["source_manifest_hashes"]
        == v5._source_manifest_hashes(),
        "formal_results_hash_valid": v1.sha256_file(v1.FINAL_RESULTS)
        == v1.EXPECTED_RESULTS_SHA256,
        "formal_matrix_hash_valid": v1.sha256_file(v1.MATRIX)
        == v1.EXPECTED_MATRIX_SHA256,
        "ready_for_formal_plot_plan": manifest["state"]
        == "ready_for_formal_plot_plan",
        "plots_created_false": manifest["plots_created"] is False,
    }
    plot_files = [
        str(path)
        for source in [*v5.SOURCE_MANIFESTS.values(), destination / "analysis_manifest.json"]
        for pattern in ("*.png", "*.svg", "*.pdf")
        for path in source.parent.rglob(pattern)
    ]
    checks["no_plot_files_in_v1_to_v5"] = not plot_files
    hash_mismatches = [
        name
        for name, expected in manifest["output_hashes"].items()
        if v1.sha256_file(destination / name) != expected
    ]
    checks["all_output_hashes_valid"] = not hash_mismatches
    row_mismatches = {}
    loaded = {}
    for name, expected in EXPECTED_ROWS.items():
        rows = _csv(destination / name)
        loaded[name] = rows
        if len(rows) != expected:
            row_mismatches[name] = {"expected": expected, "actual": len(rows)}
    checks["all_row_counts_valid"] = not row_mismatches

    joint = loaded["joint_normalization_weight_sensitivity.csv"]
    joint_keys = {
        (
            row["operational_floor"], row["grid_id"], row["model"], row["aggregation"]
        )
        for row in joint
    }
    floor_counts = Counter(row["operational_floor"] for row in joint)
    checks["joint_keys_unique"] = len(joint_keys) == 37410
    checks["each_floor_has_7482_rows"] = set(floor_counts.values()) == {7482}
    invalid_weights = []
    invalid_scores = []
    for row in joint:
        total = sum(float(row[f"weight_D{index}"]) for index in range(1, 8))
        score = float(row["score_0_to_100"])
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            invalid_weights.append((row["operational_floor"], row["grid_id"], row["model"]))
        if not math.isfinite(score) or not 0.0 <= score <= 100.0:
            invalid_scores.append((row["operational_floor"], row["grid_id"], row["model"]))
    checks["joint_weight_vectors_valid"] = not invalid_weights
    checks["joint_scores_valid"] = not invalid_scores

    units = loaded["paired_dimension_units.csv"]
    unit_keys = {(row["dimension"], row["model"], row["unit_id"]) for row in units}
    checks["paired_units_unique"] = len(unit_keys) == 36
    tests = loaded["paired_dimension_tests.csv"]
    checks["holm_family_complete"] = {row["dimension"] for row in tests} == {"D4", "D6", "D7"}
    checks["holm_values_valid"] = all(
        float(row["p_value"]) <= float(row["p_holm"]) <= 1.0 for row in tests
    )

    bootstrap = loaded["hierarchical_bootstrap_distribution.csv"]
    replicate_ids = {int(row["bootstrap_replicate"]) for row in bootstrap}
    checks["bootstrap_replicates_unique_complete"] = replicate_ids == set(range(10000))
    checks["bootstrap_values_finite"] = all(
        math.isfinite(float(value))
        for row in bootstrap
        for key, value in row.items()
        if key != "bootstrap_replicate"
    )
    summary = loaded["hierarchical_bootstrap_summary.csv"]
    checks["bootstrap_summary_metrics_complete"] = {row["metric"] for row in summary} == {
        "full_score",
        "a2c_pointer_score",
        "full_minus_a2c_points",
        "D4_difference",
        "D6_difference",
        "D7_difference",
    }
    report = {
        "schema_version": "manuscript_preplot_closure_v5_independent_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "plot_files": sorted(set(plot_files)),
        "hash_mismatches": hash_mismatches,
        "row_count_mismatches": row_mismatches,
        "invalid_weight_vector_count": len(invalid_weights),
        "invalid_score_count": len(invalid_scores),
        "protocol_hash": protocol["protocol_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "formal_results_sha256": v1.EXPECTED_RESULTS_SHA256,
        "formal_matrix_sha256": v1.EXPECTED_MATRIX_SHA256,
    }
    report["audit_hash"] = v1.canonical_hash(report)
    v1._atomic_json(REPORT, report)
    if not report["passed"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

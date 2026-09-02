#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对training_curve_correction_v6执行独立结果审计。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from uav_inspection.analysis import manuscript_multiobjective_v1 as v1
from uav_inspection.analysis import training_curve_correction_v6 as v6


REPORT = v6.DESTINATION / "independent_audit.json"


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run() -> dict[str, object]:
    manifest = json.loads((v6.DESTINATION / "analysis_manifest.json").read_text(encoding="utf-8"))
    seed_rows = _csv(v6.DESTINATION / "training_seed_metrics.csv")
    model_rows = _csv(v6.DESTINATION / "training_dimension_scores.csv")
    m06 = _csv(v6.DESTINATION / "M06_source_data.csv")
    s06 = _csv(v6.DESTINATION / "S06_source_data.csv")
    joint = _csv(v6.DESTINATION / "joint_normalization_weight_sensitivity.csv")

    checks: dict[str, object] = {}
    checks["manifest_passed"] = manifest.get("passed") is True
    checks["seed_metric_rows"] = len(seed_rows) == 15
    checks["model_metric_rows"] = len(model_rows) == 3
    checks["joint_rows"] = len(joint) == 37410
    checks["m06_seed_rows"] = len([row for row in m06 if row["record_type"] == "seed"]) == 390
    checks["m06_summary_rows"] = len([row for row in m06 if row["record_type"] == "summary"]) == 78
    checks["m06_models"] = {row["model"] for row in m06} == set(v6.CORE_MODELS)
    checks["m06_no_exact_one"] = all(
        float(row["safe_weighted_coverage"]) < 1.0
        for row in m06
        if row["record_type"] == "seed"
    )
    checks["s06_seed_rows"] = len([row for row in s06 if row["record_type"] == "seed"]) == 6720
    checks["s06_models"] = {row["model"] for row in s06} == set(v6.LEARNING_MODELS)
    checks["validation_identity"] = all(
        row["validation_mode"] == v6.EXPECTED_VALIDATION_MODE
        and int(row["validation_instance_count"]) == v6.EXPECTED_VALIDATION_COUNT
        and row["validation_instances_hash"] == v6.EXPECTED_VALIDATION_HASH
        for row in seed_rows
    )
    checks["no_obsolete_threshold_fields"] = not any(
        "threshold" in key or "convergence" in key for key in seed_rows[0]
    )
    checks["common_window"] = all(
        math.isclose(float(row["auc_interaction_start"]), v6.COMMON_INTERACTION_START)
        and math.isclose(float(row["auc_interaction_end"]), v6.COMMON_INTERACTION_END)
        for row in seed_rows
    )

    # 不调用v6的D6聚合函数，直接从种子表独立复算三模型结果。
    reproduced = {}
    for model in v6.CORE_MODELS:
        selected = [row for row in seed_rows if row["model"] == model]
        tail_means = np.asarray([float(row["tail_mean_safe_weighted_coverage"]) for row in selected])
        temporal_sd = np.asarray([float(row["tail_temporal_sd"]) for row in selected])
        auc = np.asarray([float(row["validation_auc"]) for row in selected])
        seed_consistency = max(0.0, min(1.0, 1.0 - float(np.std(tail_means, ddof=1))))
        temporal_consistency = max(0.0, min(1.0, 1.0 - float(np.mean(temporal_sd))))
        reproduced[model] = {
            "D6": 0.60 * seed_consistency + 0.40 * temporal_consistency,
            "D7": float(np.mean(auc)),
        }
    stored = {row["model"]: row for row in model_rows}
    checks["d6_reproduced"] = all(
        math.isclose(reproduced[model]["D6"], float(stored[model]["D6_training_stability"]), abs_tol=1e-12)
        for model in v6.CORE_MODELS
    )
    checks["d7_reproduced"] = all(
        math.isclose(reproduced[model]["D7"], float(stored[model]["D7_sample_efficiency"]), abs_tol=1e-12)
        for model in v6.CORE_MODELS
    )
    checks["formal_results_unchanged"] = v6.sha256_file(v6.FINAL_RESULTS) == json.loads(
        v6.PROTOCOL.read_text(encoding="utf-8")
    )["formal_results_sha256"]
    legacy = {name: v6.tree_snapshot(path)["aggregate_sha256"] for name, path in v6.LEGACY_DIRECTORIES.items()}
    checks["legacy_directories_unchanged"] = legacy == json.loads(
        v6.PROTOCOL.read_text(encoding="utf-8")
    )["legacy_tree_hashes"]
    checks["output_hashes"] = all(
        v6.sha256_file(v6.DESTINATION / name) == digest
        for name, digest in manifest["output_hashes"].items()
    )

    passed = all(value is True for value in checks.values())
    report: dict[str, object] = {
        "schema_version": "training_curve_correction_v6_independent_audit",
        "passed": passed,
        "checks": checks,
        "reproduced_training_dimensions": reproduced,
        "manifest_hash": manifest["manifest_hash"],
    }
    report["audit_hash"] = v1.canonical_hash(report)
    v1._atomic_json(REPORT, report)
    if not passed:
        raise RuntimeError("training_curve_correction_v6 independent audit failed")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

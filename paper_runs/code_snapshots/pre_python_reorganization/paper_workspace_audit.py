#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复核论文专用工作区的模型、结果、任务、图像和哈希完整性。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import paper_v3_2_experiments as v32
from paper_workspace_cleanup import AUDIT_DIR, ROOT, critical_hashes


EXPECTED_MATRIX_SHA256 = "48a31ee9b58d41a617fff61acb6eba6a2d9a930767d7af15856f70a964686224"
EXPECTED_RESULTS_SHA256 = "4b620c21566c2e33c875f6bea2017b741b02a7d30d70aa50add60a6d06214a2c"
EXPECTED_RESULT_ROWS = 21648
EXPECTED_SYNTHETIC_TASKS = 216
EXPECTED_REAL_TASKS = 144
EXPECTED_MODELS = 35
EXPECTED_FIGURE_PANELS = 72


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _figure_audit(root: Path) -> dict[str, Any]:
    manifest = _json(root / "figure_manifest.json")
    qa = _json(root / "qa_report.json")
    hash_errors: list[str] = []
    for stem, record in manifest["panels"].items():
        for kind, info in record["files"].items():
            path = root / info["path"]
            if not path.is_file() or _sha256(path) != info["sha256"]:
                hash_errors.append(f"{stem}:{kind}")
        for kind in ("caption", "source_data"):
            info = record[kind]
            path = root / info["path"]
            if not path.is_file() or _sha256(path) != info["sha256"]:
                hash_errors.append(f"{stem}:{kind}")
    return {
        "panel_count": int(manifest["panel_count"]),
        "manifest_qa_passed": bool(manifest["qa_passed"]),
        "qa_passed": bool(qa["passed"]),
        "hash_errors": hash_errors,
        "revised_v1_present": "figV01_3d_taihang_route" in manifest["panels"],
        "old_v1_absent": "figV01_3d_route_a" not in manifest["panels"],
    }


def audit() -> dict[str, Any]:
    matrix = ROOT / "paper_runs/multimap_v3_2_14/formal_evaluation/evaluation_matrix.jsonl"
    results = ROOT / "paper_runs/multimap_v3_2_14/formal_evaluation/results/final_results.jsonl"
    final_status = _json(
        ROOT / "paper_runs/multimap_v3_2_14/formal_evaluation/results/final_audit_status.json"
    )
    synthetic = ROOT / "paper_runs/multimap_v3_2_14/manifests/synthetic_test/records.jsonl"
    real = ROOT / "paper_runs/multimap_v3_2_14/formal_evaluation/real_tasks_parallel/records.jsonl"
    figure_root = ROOT / "paper_runs/multimap_v3_2_14/figures/paper_final"

    catalog = v32.checkpoint_catalog()
    missing_training_evidence: list[str] = []
    for row in catalog["rows"]:
        run_dir = Path(row["checkpoint_path"]).parent
        for name in (
            "best_safe.pt",
            "status.json",
            "run_config.json",
            "training_metrics.jsonl",
            "training_summary.json",
        ):
            if not (run_dir / name).is_file():
                missing_training_evidence.append(str((run_dir / name).relative_to(ROOT)))

    before = _json(AUDIT_DIR / "critical_hashes_before.json")
    before_hashes = {row["path"]: row["sha256"] for row in before["files"]}
    after_rows = critical_hashes()
    preserved_hash_errors = [
        row["path"]
        for row in after_rows
        if row["path"] in before_hashes and before_hashes[row["path"]] != row["sha256"]
    ]

    trace_manifest = _json(
        ROOT
        / "paper_runs/multimap_v3_2_14/analysis/training_trace_inputs_v2/source_manifest.json"
    )
    trace_hash_errors = [
        row["preserved_path"]
        for row in trace_manifest["files"]
        if not (ROOT / row["preserved_path"]).is_file()
        or _sha256(ROOT / row["preserved_path"]) != row["sha256"]
    ]

    ppo_mlp_paths = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*ppo_mlp*")
        if "_cleanup_quarantine_20260802" not in path.parts
    ]
    figure = _figure_audit(figure_root)
    checks = {
        "model_count": int(catalog["model_count"]),
        "active_variant_count": int(catalog["active_variant_count"]),
        "missing_training_evidence": missing_training_evidence,
        "ppo_mlp_paths": ppo_mlp_paths,
        "matrix_rows": _line_count(matrix),
        "result_rows": _line_count(results),
        "matrix_sha256": _sha256(matrix),
        "results_sha256": _sha256(results),
        "final_audit_passed": bool(final_status["passed"]),
        "final_audit_rows": int(final_status["row_count"]),
        "final_audit_routes": int(final_status["route_count"]),
        "synthetic_tasks": _line_count(synthetic),
        "real_tasks": _line_count(real),
        "real_corridor_assets_present": (
            ROOT / "paper_runs/multimap_v3_2_4/real_corridor_assets"
        ).is_dir(),
        "protocol_chain_present": (ROOT / "paper_runs/protocols").is_dir(),
        "map_assets_present": (ROOT / "map_data/multimap_v3_1").is_dir(),
        "preserved_hash_errors": preserved_hash_errors,
        "frozen_training_trace_count": int(trace_manifest["file_count"]),
        "frozen_training_trace_hash_errors": trace_hash_errors,
        "figure": figure,
    }
    passed = all(
        (
            checks["model_count"] == EXPECTED_MODELS,
            checks["active_variant_count"] == 7,
            not checks["missing_training_evidence"],
            not checks["ppo_mlp_paths"],
            checks["matrix_rows"] == EXPECTED_RESULT_ROWS,
            checks["result_rows"] == EXPECTED_RESULT_ROWS,
            checks["matrix_sha256"] == EXPECTED_MATRIX_SHA256,
            checks["results_sha256"] == EXPECTED_RESULTS_SHA256,
            checks["final_audit_passed"],
            checks["final_audit_rows"] == EXPECTED_RESULT_ROWS,
            checks["final_audit_routes"] == EXPECTED_RESULT_ROWS,
            checks["synthetic_tasks"] == EXPECTED_SYNTHETIC_TASKS,
            checks["real_tasks"] == EXPECTED_REAL_TASKS,
            checks["real_corridor_assets_present"],
            checks["protocol_chain_present"],
            checks["map_assets_present"],
            not checks["preserved_hash_errors"],
            checks["frozen_training_trace_count"] == 10,
            not checks["frozen_training_trace_hash_errors"],
            checks["figure"]["panel_count"] == EXPECTED_FIGURE_PANELS,
            checks["figure"]["manifest_qa_passed"],
            checks["figure"]["qa_passed"],
            not checks["figure"]["hash_errors"],
            checks["figure"]["revised_v1_present"],
            checks["figure"]["old_v1_absent"],
        )
    )
    report = {
        "schema": "paper-workspace-post-cleanup-audit-v1",
        "passed": passed,
        "checks": checks,
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "post_cleanup_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)

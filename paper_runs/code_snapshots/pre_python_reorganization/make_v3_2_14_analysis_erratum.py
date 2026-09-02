#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze a schema-only analysis implementation erratum."""

from __future__ import annotations

import json
from pathlib import Path

import paper_v3_2_experiments as v32
import v3_2_14_evaluation_smoke as smoke
import v3_2_14_statistics as statistics


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14"
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "analysis_protocol.json"
)
DESTINATION = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "analysis_implementation_erratum.json"
)
FINAL_AUDIT = (
    OUTPUT / "formal_evaluation/results/final_audit.json"
)
FAILED_STDERR = OUTPUT / "diagnostics/analysis_chain.stderr.log"
PREFLIGHT_ERRATUM = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "analysis_implementation_erratum_superseded_preflight.json"
)


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    audit = json.loads(FINAL_AUDIT.read_text(encoding="utf-8"))
    if not audit.get("passed") or int(audit["row_count"]) != 21648:
        raise RuntimeError("final audit is not complete and valid")
    if statistics.DESTINATION.exists():
        raise RuntimeError("analysis outputs already exist")
    if not FAILED_STDERR.is_file():
        raise RuntimeError("failed analysis traceback is missing")
    payload = {
        "schema_version": 1,
        "parent_analysis_protocol_hash": protocol[
            "analysis_protocol_hash"
        ],
        "original_implementation_sha256": protocol[
            "implementation_sha256"
        ],
        "corrected_implementation_sha256": v32._sha256_file(
            Path(statistics.__file__)
        ),
        "failed_traceback_sha256": v32._sha256_file(FAILED_STDERR),
        "final_results_sha256": audit["results_sha256"],
        "final_audit_manifest_hash": audit["manifest_hash"],
        "correction_scope": "route_schema_adapter_only",
        "correction_description": (
            "accept the frozen traditional-planner integer "
            "constraint_violations count in addition to the learning-route "
            "list schema; use the explicit constraint_violation_count and "
            "do not infer unavailable constraint names"
        ),
        "statistical_rules_changed": False,
        "algorithm_scores_used_for_fix": False,
        "algorithm_scores_inspected_during_fix": False,
        "task_selection_changed": False,
        "result_rows_changed": False,
        "route_rows_changed": False,
        "plotting_performed": False,
    }
    if PREFLIGHT_ERRATUM.is_file():
        superseded = json.loads(
            PREFLIGHT_ERRATUM.read_text(encoding="utf-8")
        )
        payload["supersedes_preflight_erratum_hash"] = superseded[
            "erratum_hash"
        ]
    payload["erratum_hash"] = smoke._canonical_hash(payload)
    text = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if DESTINATION.exists() and DESTINATION.read_text(
        encoding="utf-8"
    ) != text:
        raise RuntimeError("analysis erratum already exists and differs")
    smoke._atomic_text(DESTINATION, text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

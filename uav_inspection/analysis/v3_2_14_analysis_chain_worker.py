#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wait for final audit, create statistics and plot inputs, then stop."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Dict, Sequence

from uav_inspection.experiments import paper_v3_2_experiments as v32
from uav_inspection.evaluation import v3_2_14_evaluation_smoke as smoke
from uav_inspection.analysis import v3_2_14_statistics as statistics


ROOT = WORKSPACE_ROOT
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14"
FINAL_AUDIT = (
    OUTPUT / "formal_evaluation/results/final_audit.json"
)
ANALYSIS_PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "analysis_protocol.json"
)
ANALYSIS_ERRATUM = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "analysis_implementation_erratum.json"
)
STATUS = OUTPUT / "analysis/analysis_chain_status.json"


def _protocol() -> Dict[str, Any]:
    payload = json.loads(
        ANALYSIS_PROTOCOL.read_text(encoding="utf-8")
    )
    expected = payload["analysis_protocol_hash"]
    actual = smoke._canonical_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "analysis_protocol_hash"
        }
    )
    current_implementation = v32._sha256_file(
        Path(statistics.__file__)
    )
    if expected != actual or not payload.get("plotting_forbidden"):
        raise RuntimeError("analysis protocol identity mismatch")
    if payload["implementation_sha256"] != current_implementation:
        if not ANALYSIS_ERRATUM.is_file():
            raise RuntimeError(
                "analysis implementation changed without an erratum"
            )
        erratum = json.loads(
            ANALYSIS_ERRATUM.read_text(encoding="utf-8")
        )
        erratum_hash = erratum["erratum_hash"]
        erratum_actual = smoke._canonical_hash(
            {
                key: value
                for key, value in erratum.items()
                if key != "erratum_hash"
            }
        )
        if (
            erratum_hash != erratum_actual
            or erratum["parent_analysis_protocol_hash"] != expected
            or erratum["original_implementation_sha256"]
            != payload["implementation_sha256"]
            or erratum["corrected_implementation_sha256"]
            != current_implementation
            or erratum.get("statistical_rules_changed") is not False
            or erratum.get("algorithm_scores_used_for_fix") is not False
        ):
            raise RuntimeError("analysis implementation erratum mismatch")
        payload["_analysis_erratum_hash"] = erratum_hash
    return payload


def _final_audit_passed() -> bool:
    if not FINAL_AUDIT.is_file():
        return False
    try:
        payload = json.loads(FINAL_AUDIT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("passed")
        and int(payload.get("row_count", -1)) == 21648
        and int(payload.get("unique_key_count", -1)) == 21648
        and int(payload.get("route_count", -1)) == 21648
    )


def run(*, poll_seconds: float) -> Dict[str, Any]:
    protocol = _protocol()
    while not _final_audit_passed():
        smoke._atomic_json(
            STATUS,
            {
                "schema_version": 1,
                "state": "waiting_for_final_audit",
                "analysis_protocol_hash": protocol[
                    "analysis_protocol_hash"
                ],
                "plots_created": False,
            },
        )
        time.sleep(poll_seconds)
    smoke._atomic_json(
        STATUS,
        {
            "schema_version": 1,
            "state": "running_statistics",
            "analysis_protocol_hash": protocol[
                "analysis_protocol_hash"
            ],
            "plots_created": False,
        },
    )
    manifest = statistics.run()
    if (
        manifest.get("state") != "ready_for_plotting"
        or manifest.get("plots_created")
        or int(manifest.get("row_count", -1)) != 21648
    ):
        raise RuntimeError("analysis did not reach pre-plot stop state")
    status = {
        "schema_version": 1,
        "state": "ready_for_plotting",
        "analysis_protocol_hash": protocol["analysis_protocol_hash"],
        "analysis_implementation_erratum_hash": protocol.get(
            "_analysis_erratum_hash"
        ),
        "analysis_manifest_hash": manifest["manifest_hash"],
        "row_count": manifest["row_count"],
        "plots_created": False,
        "action_required": (
            "stop and wait for the user's plotting plan"
        ),
    }
    smoke._atomic_json(STATUS, status)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    result = run(poll_seconds=float(args.poll_seconds))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

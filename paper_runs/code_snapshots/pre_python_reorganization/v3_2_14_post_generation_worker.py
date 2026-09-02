#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wait for 216+144 tasks, then merge, audit, and freeze the 21,648-row matrix."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import paper_v3_2_experiments as v32
import v3_2_1_real_task_shards as real_shards


ROOT = Path(__file__).resolve().parent
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/protocol.json"
)
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14"
MAP_ROOT = ROOT / "map_data/multimap_v3_1"
STATUS = OUTPUT / "formal_evaluation/post_generation_status.json"


def _write_status(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(STATUS)


def _snapshot(protocol_hash: str) -> dict:
    synthetic_records = (
        OUTPUT / "manifests/synthetic_test/records.jsonl"
    )
    synthetic_audit_path = (
        OUTPUT / "manifests/synthetic_test/audit.json"
    )
    synthetic_rows = (
        len(v32._read_jsonl(synthetic_records))
        if synthetic_records.is_file()
        else 0
    )
    synthetic_audit_passed = False
    if synthetic_audit_path.is_file():
        synthetic_audit_passed = bool(
            json.loads(
                synthetic_audit_path.read_text(encoding="utf-8")
            ).get("passed")
        )
    shard_root = (
        OUTPUT / "formal_evaluation/real_task_shards/formal"
    )
    real_rows = 0
    complete_shards = 0
    for map_index in range(8):
        directory = shard_root / (
            f"maps_{map_index:02d}_{map_index + 1:02d}"
        )
        records = directory / "records.jsonl"
        manifest = directory / "manifest.json"
        count = (
            len(v32._read_jsonl(records)) if records.is_file() else 0
        )
        real_rows += count
        if count == 18 and manifest.is_file():
            complete_shards += 1
    return {
        "schema_version": 1,
        "state": "waiting_for_task_generation",
        "protocol_hash": protocol_hash,
        "synthetic_rows": synthetic_rows,
        "synthetic_audit_passed": synthetic_audit_passed,
        "real_rows": real_rows,
        "complete_real_shards": complete_shards,
        "formal_algorithm_evaluation_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    args = parser.parse_args()
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    deadline = time.monotonic() + float(args.timeout_hours) * 3600.0
    while time.monotonic() < deadline:
        status = _snapshot(str(protocol["protocol_hash"]))
        _write_status(status)
        if (
            status["synthetic_rows"] == 216
            and status["synthetic_audit_passed"]
            and status["real_rows"] == 144
            and status["complete_real_shards"] == 8
        ):
            break
        time.sleep(float(args.poll_seconds))
    else:
        status = _snapshot(str(protocol["protocol_hash"]))
        status["state"] = "timeout_waiting_for_task_generation"
        _write_status(status)
        return 2

    merged = real_shards.merge_shards(PROTOCOL, OUTPUT, MAP_ROOT)
    if not merged["audit"]["passed"]:
        raise RuntimeError("merged real-task audit did not pass")
    synthetic_records = (
        OUTPUT / "manifests/synthetic_test/records.jsonl"
    )
    real_records = (
        OUTPUT / "formal_evaluation/real_tasks_parallel/records.jsonl"
    )
    matrix = v32.freeze_evaluation_matrix(
        PROTOCOL, OUTPUT, synthetic_records, real_records
    )
    if int(matrix["row_count"]) != 21648:
        raise RuntimeError("evaluation matrix is not exactly 21,648 rows")
    _write_status(
        {
            "schema_version": 1,
            "state": "completed",
            "protocol_hash": protocol["protocol_hash"],
            "synthetic_rows": 216,
            "synthetic_audit_passed": True,
            "real_rows": 144,
            "complete_real_shards": 8,
            "real_manifest_hash": merged["manifest"]["manifest_hash"],
            "matrix_row_count": matrix["row_count"],
            "matrix_sha256": matrix["matrix_sha256"],
            "formal_algorithm_evaluation_started": False,
        }
    )
    print(json.dumps(matrix, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

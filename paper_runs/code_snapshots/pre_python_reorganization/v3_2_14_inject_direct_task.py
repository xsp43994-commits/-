#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inject one audited direct-threshold task into a partial real shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import paper_multimap_experiments as multimap
import paper_v3_2_experiments as v32
import v3_2_13_certificate_composition as composition


ROOT = Path(__file__).resolve().parent
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/protocol.json"
)
OUTPUT_ROOT = ROOT / "paper_runs/multimap_v3_2_14"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-index", type=int, required=True)
    parser.add_argument("--road-index", type=int, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace the same frozen task ID after an archived certificate-only correction.",
    )
    args = parser.parse_args()
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    parent = json.loads(
        multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
    )
    record = composition.compose_direct_single_certificate(
        protocol,
        parent,
        output_root=OUTPUT_ROOT,
        map_index=int(args.map_index),
        road_index=int(args.road_index),
        task_index=int(args.task_index),
        attempt=int(args.attempt),
    )
    directory = (
        OUTPUT_ROOT
        / "formal_evaluation"
        / "real_task_shards"
        / "formal"
        / f"maps_{args.map_index:02d}_{args.map_index + 1:02d}"
    )
    records_path = directory / "records.jsonl"
    rows = v32._read_jsonl(records_path)
    records = {str(row["id"]): dict(row) for row in rows}
    task_id = str(record["id"])
    if (
        task_id in records
        and records[task_id] != record
        and not args.replace_existing
    ):
        raise RuntimeError("direct task conflicts with existing shard row")
    records[task_id] = record
    rows = [records[key] for key in sorted(records)]
    if any(
        multimap._canonical_hash(row, excluded=("task_hash",))
        != row.get("task_hash")
        for row in rows
    ):
        raise RuntimeError("post-injection task hash audit failed")
    records_path.write_text(v32._jsonl(rows), encoding="utf-8")
    checkpoint = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "map_index": int(args.map_index),
        "row_count": len(rows),
        "expected": 18,
        "state": "completed" if len(rows) == 18 else "partial",
        "injected_task_id": task_id,
        "injected_task_hash": record["task_hash"],
        "records_sha256": v32._sha256_file(records_path),
        "algorithm_results_used": False,
    }
    path = directory / "direct_injection_checkpoint.json"
    path.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

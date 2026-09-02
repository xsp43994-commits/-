#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Migrate real-task rows and inject the exact Colorado mixed certificate."""

from __future__ import annotations

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
SOURCE = (
    ROOT
    / "paper_runs/multimap_v3_2_13/formal_evaluation/"
    "real_task_shards/formal"
)
DESTINATION = (
    ROOT
    / "paper_runs/multimap_v3_2_14/formal_evaluation/"
    "real_task_shards/formal"
)


def main() -> int:
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    parent = json.loads(
        multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
    )
    catalog = v32.checkpoint_catalog(
        PROTOCOL, output_root=ROOT / "paper_runs/multimap_v3_2_14"
    )
    context_manifest = (
        Path(str(protocol["real_corridor_asset_root"])) / "manifest.json"
    )
    colorado = composition.compose_constructive_mixed_certificate(
        protocol,
        parent,
        output_root=ROOT / "paper_runs/multimap_v3_2_14",
    )
    colorado_id = str(colorado["id"])
    migrated = []
    for map_index in range(8):
        name = f"maps_{map_index:02d}_{map_index + 1:02d}"
        source_dir = SOURCE / name
        source_records = source_dir / "records.jsonl"
        rows = (
            v32._read_jsonl(source_records)
            if source_records.is_file()
            else []
        )
        records = {str(row["id"]): dict(row) for row in rows}
        if map_index == 5:
            existing = records.get(colorado_id)
            if existing is not None and existing != colorado:
                raise RuntimeError("Colorado task06 identity conflict")
            records[colorado_id] = colorado
        rows = [records[key] for key in sorted(records)]
        if any(
            multimap._canonical_hash(row, excluded=("task_hash",))
            != row.get("task_hash")
            for row in rows
        ) or len({str(row["id"]) for row in rows}) != len(rows):
            raise RuntimeError(f"{name} task-object migration failed")
        destination_dir = DESTINATION / name
        destination_records = destination_dir / "records.jsonl"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_records.write_text(
            v32._jsonl(rows), encoding="utf-8"
        )
        complete = len(rows) == 18
        if complete:
            manifest = {
                "schema_version": 1,
                "protocol_hash": protocol["protocol_hash"],
                "checkpoint_catalog_hash": catalog["catalog_hash"],
                "map_index_start": map_index,
                "map_index_stop": map_index + 1,
                "run_label": "formal",
                "task_limit": 9,
                "scenario_count": 18,
                "records_sha256": v32._sha256_file(
                    destination_records
                ),
                "context_manifest_sha256": v32._sha256_file(
                    context_manifest
                ),
                "algorithm_results_used": False,
                "migration": {
                    "source_records": str(source_records.resolve()),
                    "source_records_sha256": (
                        v32._sha256_file(source_records)
                        if source_records.is_file()
                        else None
                    ),
                    "task_objects_unchanged_except_registered_injection": True,
                },
            }
            manifest["manifest_hash"] = v32._canonical_hash(
                manifest, excluded=("manifest_hash",)
            )
            v32._write_json(destination_dir / "manifest.json", manifest)
        checkpoint = {
            "schema_version": 1,
            "protocol_hash": protocol["protocol_hash"],
            "map_index": map_index,
            "row_count": len(rows),
            "expected": 18,
            "state": "completed" if complete else "partial",
            "source_records_sha256": (
                v32._sha256_file(source_records)
                if source_records.is_file()
                else None
            ),
            "destination_records_sha256": v32._sha256_file(
                destination_records
            ),
            "injected_task_id": (
                colorado_id if map_index == 5 else None
            ),
            "algorithm_results_used": False,
        }
        v32._write_json(
            destination_dir / "migration_checkpoint.json", checkpoint
        )
        migrated.append(checkpoint)
    summary = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "row_count": sum(item["row_count"] for item in migrated),
        "complete_shards": sum(
            item["state"] == "completed" for item in migrated
        ),
        "partial_shards": sum(
            item["state"] == "partial" for item in migrated
        ),
        "injected_task_id": colorado_id,
        "injected_task_hash": colorado["task_hash"],
        "algorithm_results_used": False,
        "shards": migrated,
    }
    v32._write_json(
        DESTINATION.parent / "migration_manifest.json", summary
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

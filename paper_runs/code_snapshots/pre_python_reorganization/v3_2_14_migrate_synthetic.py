#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Migrate the fully audited 216 synthetic task objects to v3.2.14."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import paper_multimap_experiments as multimap
import paper_v3_2_experiments as v32
import v3_2_12_synthetic_tasks as base


ROOT = Path(__file__).resolve().parent
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/protocol.json"
)
SOURCE_DIR = (
    ROOT / "paper_runs/multimap_v3_2_13/manifests/synthetic_test"
)
DESTINATION_DIR = (
    ROOT / "paper_runs/multimap_v3_2_14/manifests/synthetic_test"
)
MAP_ROOT = ROOT / "map_data/multimap_v3_1"


def _overwrite_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    source_audit = json.loads(
        (SOURCE_DIR / "audit.json").read_text(encoding="utf-8")
    )
    if not source_audit.get("passed"):
        raise RuntimeError("v3.2.13 synthetic source audit did not pass")
    rows = v32._read_jsonl(SOURCE_DIR / "records.jsonl")
    if len(rows) != 216 or any(
        multimap._canonical_hash(row, excluded=("task_hash",))
        != row.get("task_hash")
        for row in rows
    ):
        raise RuntimeError("v3.2.13 synthetic task objects are invalid")
    source_protocol = multimap.load_protocol(base.SOURCE_PROTOCOL)
    original_rows = v32._read_jsonl(base.SOURCE_RECORDS)
    registry_path = (
        MAP_ROOT
        / "procedural"
        / "synthetic_test"
        / "map_registry.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    reuse_audit = base._reuse_audit(
        protocol,
        source_protocol,
        original_rows,
        registry,
        MAP_ROOT,
    )
    if not reuse_audit["passed"]:
        raise RuntimeError(
            "v3.2.14 synthetic input-semantics audit failed"
        )
    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    records_path = DESTINATION_DIR / "records.jsonl"
    text = multimap._jsonl_text(rows)
    records_path.write_text(text, encoding="utf-8")
    _overwrite_json(DESTINATION_DIR / "reuse_audit.json", reuse_audit)
    provider = multimap.FrozenMapProvider.from_registries(
        MAP_ROOT, [registry_path]
    )
    parent = json.loads(
        multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "parent_difficulty_protocol_hash": parent["protocol_hash"],
        "split": "synthetic_test",
        "map_count": 24,
        "tasks_per_map": 9,
        "scenario_count": 216,
        "map_registry_path": str(registry_path.resolve()),
        "map_registry_hash": registry["registry_hash"],
        "map_provider_hash": provider.provider_hash,
        "records_sha256": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
        "reused_record_count": 216,
        "new_record_count": 0,
        "source_records_path": str(
            (SOURCE_DIR / "records.jsonl").resolve()
        ),
        "source_records_file_sha256": v32._sha256_file(
            SOURCE_DIR / "records.jsonl"
        ),
        "source_audit_sha256": v32._sha256_file(
            SOURCE_DIR / "audit.json"
        ),
        "selection_used_algorithm_results": False,
        "sharded": False,
        "smoke": False,
    }
    manifest["manifest_hash"] = multimap._canonical_hash(
        manifest, excluded=("manifest_hash",)
    )
    _overwrite_json(DESTINATION_DIR / "manifest.json", manifest)
    audit = multimap.audit_task_manifest(
        PROTOCOL,
        MAP_ROOT,
        DESTINATION_DIR / "manifest.json",
        expected_map_count=24,
        expected_tasks_per_map=9,
    )
    _overwrite_json(DESTINATION_DIR / "audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(
            "v3.2.14 synthetic audit failed: "
            + "; ".join(audit["reasons"][:8])
        )
    checkpoint = {
        "schema_version": 1,
        "state": "completed",
        "protocol_hash": protocol["protocol_hash"],
        "reused": 216,
        "generated": 0,
        "completed": 216,
        "expected": 216,
        "records_sha256": manifest["records_sha256"],
        "manifest_hash": manifest["manifest_hash"],
        "algorithm_results_used": False,
    }
    _overwrite_json(
        DESTINATION_DIR / "generation_checkpoint.json", checkpoint
    )
    print(
        json.dumps(
            {"manifest": manifest, "audit": audit},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

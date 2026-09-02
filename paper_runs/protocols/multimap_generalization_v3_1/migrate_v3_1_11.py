#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把试训前v3.1.10注册表、validation封存和training断点迁移到v3.1.11。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paper_multimap_experiments as multimap


OLD_PROTOCOL_HASH = (
    "59ba71e12cd9edb55390f4961c844e30b8277815523cb002b6a77c07b183c79c"
)
NEW_PROTOCOL_HASH = (
    "68fde73db57d07f4c9451ed54baa978e09985c5017c43e68adac54d6f2a5131a"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = multimap.load_protocol()
    if protocol["protocol_hash"] != NEW_PROTOCOL_HASH:
        raise RuntimeError("v3.1.11协议哈希不匹配。")
    map_root = ROOT / "map_data" / "multimap_v3_1"
    output_root = ROOT / "paper_runs" / "multimap_v3_1"
    registry_paths = {
        "real": map_root / "real" / "map_registry.json",
        "training": (
            map_root / "procedural" / "training" / "map_registry.json"
        ),
        "validation": (
            map_root / "procedural" / "validation" / "map_registry.json"
        ),
    }
    migrated = {}
    for name, path in registry_paths.items():
        registry = json.loads(path.read_text(encoding="utf-8"))
        if registry.get("protocol_hash") != OLD_PROTOCOL_HASH:
            raise RuntimeError(f"{name}注册表不是v3.1.10身份。")
        old_hash = str(registry["registry_hash"])
        registry["protocol_hash"] = NEW_PROTOCOL_HASH
        registry["supersedes_registry_hash"] = old_hash
        registry["registry_hash"] = multimap._canonical_hash(
            registry, excluded=("registry_hash",)
        )
        multimap._atomic_json(path, registry)
        migrated[name] = {
            "old_registry_hash": old_hash,
            "new_registry_hash": registry["registry_hash"],
        }

    validation_dir = output_root / "manifests" / "validation"
    validation_records = validation_dir / "records.jsonl"
    training_dir = output_root / "manifests" / "training"
    training_records = training_dir / "records.jsonl"
    validation_before = sha256_file(validation_records)
    training_before = sha256_file(training_records)

    validation_manifest_path = validation_dir / "manifest.json"
    validation_manifest = json.loads(
        validation_manifest_path.read_text(encoding="utf-8")
    )
    if validation_manifest.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError("validation manifest不是v3.1.10身份。")
    provider = multimap.FrozenMapProvider.from_registries(
        map_root, [registry_paths["validation"]]
    )
    validation_registry = json.loads(
        registry_paths["validation"].read_text(encoding="utf-8")
    )
    validation_manifest["protocol_hash"] = NEW_PROTOCOL_HASH
    validation_manifest["map_registry_hash"] = validation_registry[
        "registry_hash"
    ]
    validation_manifest["map_provider_hash"] = provider.provider_hash
    validation_manifest["manifest_hash"] = multimap._canonical_hash(
        validation_manifest, excluded=("manifest_hash",)
    )
    multimap._atomic_json(validation_manifest_path, validation_manifest)
    validation_audit = multimap.audit_task_manifest(
        multimap.DEFAULT_PROTOCOL,
        map_root,
        validation_manifest_path,
        expected_map_count=12,
        expected_tasks_per_map=9,
    )
    if not validation_audit["passed"]:
        raise RuntimeError("迁移后validation任务审计失败。")
    multimap._atomic_json(
        validation_dir / "environment_audit.json", validation_audit
    )
    validation_checkpoint_path = validation_dir / "generation_checkpoint.json"
    validation_checkpoint = json.loads(
        validation_checkpoint_path.read_text(encoding="utf-8")
    )
    if (
        validation_checkpoint.get("protocol_hash") != OLD_PROTOCOL_HASH
        or int(validation_checkpoint.get("completed", -1)) != 108
        or not bool(validation_checkpoint.get("audit_passed", False))
    ):
        raise RuntimeError("validation断点身份或完成状态无效。")
    validation_checkpoint["protocol_hash"] = NEW_PROTOCOL_HASH
    validation_checkpoint["supersedes_protocol_hash"] = OLD_PROTOCOL_HASH
    validation_checkpoint["protocol_migration"] = (
        "v3.1.11 changes only pre-pilot certification scheduling; "
        "108 validation records are byte-identical"
    )
    multimap._atomic_json(
        validation_checkpoint_path, validation_checkpoint
    )

    training_checkpoint_path = training_dir / "generation_checkpoint.json"
    training_checkpoint = json.loads(
        training_checkpoint_path.read_text(encoding="utf-8")
    )
    if (
        training_checkpoint.get("protocol_hash") != OLD_PROTOCOL_HASH
        or int(training_checkpoint.get("completed", -1)) != 42
        or training_checkpoint.get("current_task_id")
        != "training__training__map_004__task_06"
        or int(training_checkpoint.get("current_attempt", -1)) != 40
    ):
        raise RuntimeError("training断点与已封存42条状态不一致。")
    training_checkpoint["protocol_hash"] = NEW_PROTOCOL_HASH
    training_checkpoint["supersedes_protocol_hash"] = OLD_PROTOCOL_HASH
    training_checkpoint["protocol_migration"] = (
        "v3.1.11 preserves 42 serial records and switches remaining maps "
        "to four audited shards"
    )
    multimap._atomic_json(training_checkpoint_path, training_checkpoint)

    validation_after = sha256_file(validation_records)
    training_after = sha256_file(training_records)
    if (
        validation_before != validation_after
        or training_before != training_after
    ):
        raise RuntimeError("迁移过程中任务records发生字节漂移。")
    report = {
        "old_protocol_hash": OLD_PROTOCOL_HASH,
        "new_protocol_hash": NEW_PROTOCOL_HASH,
        "validation_records_unchanged": 108,
        "training_records_unchanged": 42,
        "validation_records_sha256_before": validation_before,
        "validation_records_sha256_after": validation_after,
        "training_records_sha256_before": training_before,
        "training_records_sha256_after": training_after,
        "validation_manifest_hash": validation_manifest["manifest_hash"],
        "registries": migrated,
    }
    report["migration_hash"] = multimap._canonical_hash(
        report, excluded=("migration_hash",)
    )
    multimap._atomic_json(
        output_root / "audits" / "migration_v3_1_11.json", report
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在零回合试训失败后，把 v3.1.15 身份迁移到数值边界修复后的 v3.1.16。"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paper_multimap_experiments as multimap


OLD_PROTOCOL_HASH = "92b25776749a9430e71e47e0882970c5ea149ed778906104ad38604b870860ba"
NEW_PROTOCOL_HASH = "20c246fdc986f4fb6654449ef9f306188c410e7c3637e929e8c88b79ddb79c9b"
MIGRATION_LABEL = "v3.1.16"
MIGRATION_DESCRIPTION = (
    "v3.1.16 repairs only the zero-episode oracle-bound floating-point "
    "normalization guard; all maps, tasks, routes and certificates are "
    "byte-identical"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def migrate_registry(path: Path) -> Dict[str, str]:
    registry = read_json(path)
    if registry.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError(f"注册表协议身份异常：{path}")
    old_hash = str(registry["registry_hash"])
    registry["protocol_hash"] = NEW_PROTOCOL_HASH
    registry["supersedes_registry_hash"] = old_hash
    registry["registry_hash"] = multimap._canonical_hash(
        registry, excluded=("registry_hash",)
    )
    multimap._atomic_json(path, registry)
    return {"old_registry_hash": old_hash, "new_registry_hash": registry["registry_hash"]}


def migrate_manifest(
    path: Path,
    *,
    registry: Mapping[str, Any],
    provider_hash: str,
    shard_manifest_hashes: Mapping[str, str] | None = None,
) -> str:
    manifest = read_json(path)
    if manifest.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError(f"任务清单协议身份异常：{path}")
    old_hash = str(manifest["manifest_hash"])
    manifest["protocol_hash"] = NEW_PROTOCOL_HASH
    manifest["supersedes_manifest_hash"] = old_hash
    manifest["map_registry_hash"] = registry["registry_hash"]
    manifest["map_provider_hash"] = provider_hash
    if shard_manifest_hashes is not None:
        manifest["shard_manifest_hashes"] = dict(shard_manifest_hashes)
    manifest["manifest_hash"] = multimap._canonical_hash(
        manifest, excluded=("manifest_hash",)
    )
    multimap._atomic_json(path, manifest)
    return str(manifest["manifest_hash"])


def migrate_checkpoint(path: Path) -> int:
    checkpoint = read_json(path)
    if checkpoint.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError(f"检查点协议身份异常：{path}")
    checkpoint["protocol_hash"] = NEW_PROTOCOL_HASH
    checkpoint["supersedes_protocol_hash"] = OLD_PROTOCOL_HASH
    checkpoint["protocol_migration"] = MIGRATION_DESCRIPTION
    multimap._atomic_json(path, checkpoint)
    return int(checkpoint.get("completed", -1))


def archive_pre_pilot_artifacts(output_root: Path) -> Dict[str, str]:
    audits = output_root / "audits"
    freeze_path = output_root / "environment_freeze.json"
    pilot_path = output_root / "pilot" / "pilot_full_seed42_600ep"
    if not freeze_path.is_file() or not pilot_path.is_dir():
        raise RuntimeError("缺少待归档的零回合试训环境封存或试训目录")
    freeze = read_json(freeze_path)
    status = read_json(pilot_path / "status.json")
    if freeze.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError("待归档环境封存不是 v3.1.15")
    if status.get("state") != "failed" or int(status.get("completed", -1)) != 0:
        raise RuntimeError("失败试训不是零回合预检失败，拒绝迁移")

    archived_freeze = audits / "superseded_environment_freeze_v3_1_15.json"
    archived_pilot = audits / "pilot_full_seed42_600ep_preepisode_failure_v3_1_15"
    if archived_freeze.exists() or archived_pilot.exists():
        raise RuntimeError("v3.1.15 预试训归档已存在，拒绝覆盖")
    # 移动而非删除，保留失败预检与原封存的可追溯证据。
    freeze_path.replace(archived_freeze)
    pilot_path.replace(archived_pilot)
    failure = {
        "schema_version": 1,
        "protocol_hash": OLD_PROTOCOL_HASH,
        "state": "superseded_zero_episode_preflight_failure",
        "completed_episodes": 0,
        "error": str(status.get("error", "")),
        "archived_environment_freeze": str(archived_freeze),
        "archived_pilot_directory": str(archived_pilot),
        "new_protocol_hash": NEW_PROTOCOL_HASH,
        "no_algorithm_result_was_generated": True,
    }
    failure["archive_hash"] = multimap._canonical_hash(
        failure, excluded=("archive_hash",)
    )
    multimap._atomic_json(
        audits / "pilot_preflight_failure_v3_1_15.json", failure
    )
    return {
        "old_freeze_hash": str(freeze["freeze_hash"]),
        "archived_freeze": str(archived_freeze),
        "archived_pilot": str(archived_pilot),
        "failure_archive_hash": str(failure["archive_hash"]),
    }


def main() -> None:
    protocol = multimap.load_protocol()
    if protocol["protocol_hash"] != NEW_PROTOCOL_HASH:
        raise RuntimeError("v3.1.16 协议哈希不匹配")
    output_root = ROOT / "paper_runs" / "multimap_v3_1"
    map_root = ROOT / "map_data" / "multimap_v3_1"
    registry_paths = {
        "real": map_root / "real" / "map_registry.json",
        "training": map_root / "procedural" / "training" / "map_registry.json",
        "validation": map_root / "procedural" / "validation" / "map_registry.json",
    }
    validation_dir = output_root / "manifests" / "validation"
    training_dir = output_root / "manifests" / "training"
    shard_root = output_root / "manifests" / "training_shards"
    record_paths = {
        "training": training_dir / "records.jsonl",
        "validation": validation_dir / "records.jsonl",
        "serial_base_for_merge": training_dir / "serial_base_for_merge.jsonl",
    }
    for shard_name in ("shard_00", "shard_01", "shard_02", "shard_03"):
        record_paths[shard_name] = shard_root / shard_name / "records.jsonl"
    before_hashes = {name: sha256_file(path) for name, path in record_paths.items()}

    parent = read_json(multimap.PARENT_DIFFICULTY_PROTOCOL)
    expected_counts = {
        "training": 648,
        "validation": 108,
        "serial_base_for_merge": 36,
        "shard_00": 162,
        "shard_01": 153,
        "shard_02": 153,
        "shard_03": 144,
    }
    source_counts: Dict[str, int] = {}
    for name, path in record_paths.items():
        rows = multimap._read_jsonl(path)
        source_counts[name] = len(rows)
        if len(rows) != expected_counts[name]:
            raise RuntimeError(f"记录数异常：{name}={len(rows)}")
        for row in rows:
            if row.get("task_hash") != multimap._canonical_hash(
                row, excluded=("task_hash",)
            ):
                raise RuntimeError(f"任务哈希无效：{name}:{row.get('id')}")
            if not all_finite(row):
                raise RuntimeError(f"记录含非有限值：{name}:{row.get('id')}")
            reasons = multimap._audit_budget_transform_record(
                row, protocol, parent
            )
            if reasons:
                raise RuntimeError(f"预算审计失败：{name}:{row.get('id')}:{reasons}")

    registries = {
        name: migrate_registry(path) for name, path in registry_paths.items()
    }
    training_registry = read_json(registry_paths["training"])
    validation_registry = read_json(registry_paths["validation"])
    training_provider = multimap.FrozenMapProvider.from_registries(
        map_root, [registry_paths["training"]]
    )
    validation_provider = multimap.FrozenMapProvider.from_registries(
        map_root, [registry_paths["validation"]]
    )
    validation_manifest_hash = migrate_manifest(
        validation_dir / "manifest.json",
        registry=validation_registry,
        provider_hash=validation_provider.provider_hash,
    )
    shard_manifest_hashes = {}
    for shard_name in ("shard_00", "shard_01", "shard_02", "shard_03"):
        shard_manifest_hashes[shard_name] = migrate_manifest(
            shard_root / shard_name / "manifest.json",
            registry=training_registry,
            provider_hash=training_provider.provider_hash,
        )
    training_manifest_hash = migrate_manifest(
        training_dir / "manifest.json",
        registry=training_registry,
        provider_hash=training_provider.provider_hash,
        shard_manifest_hashes=shard_manifest_hashes,
    )
    checkpoint_counts = {
        "validation": migrate_checkpoint(validation_dir / "generation_checkpoint.json"),
        "training": migrate_checkpoint(training_dir / "generation_checkpoint.json"),
    }
    for shard_name in ("shard_00", "shard_01", "shard_02", "shard_03"):
        checkpoint_counts[shard_name] = migrate_checkpoint(
            shard_root / shard_name / "generation_checkpoint.json"
        )
    supervisor_path = output_root / "monitoring" / "parallel_training_supervisor.json"
    supervisor = read_json(supervisor_path)
    if supervisor.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError("监督状态协议身份异常")
    supervisor["protocol_hash"] = NEW_PROTOCOL_HASH
    supervisor["supersedes_protocol_hash"] = OLD_PROTOCOL_HASH
    supervisor["protocol_migration"] = MIGRATION_DESCRIPTION
    multimap._atomic_json(supervisor_path, supervisor)
    pre_pilot_archive = archive_pre_pilot_artifacts(output_root)

    after_hashes = {name: sha256_file(path) for name, path in record_paths.items()}
    if before_hashes != after_hashes:
        raise RuntimeError("迁移过程中原始任务记录发生字节漂移")
    report = {
        "schema_version": 1,
        "migration_label": MIGRATION_LABEL,
        "old_protocol_hash": OLD_PROTOCOL_HASH,
        "new_protocol_hash": NEW_PROTOCOL_HASH,
        "description": MIGRATION_DESCRIPTION,
        "records_byte_identical": True,
        "record_sha256_before": before_hashes,
        "record_sha256_after": after_hashes,
        "source_record_counts": source_counts,
        "checkpoint_counts": checkpoint_counts,
        "registries": registries,
        "training_manifest_hash": training_manifest_hash,
        "validation_manifest_hash": validation_manifest_hash,
        "shard_manifest_hashes": shard_manifest_hashes,
        "pre_pilot_archive": pre_pilot_archive,
    }
    report["migration_hash"] = multimap._canonical_hash(
        report, excluded=("migration_hash",)
    )
    multimap._atomic_json(
        output_root / "audits" / "migration_v3_1_16.json", report
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把试训前v3.1.11身份迁移到初始SOC边界修复后的v3.1.12。"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paper_multimap_experiments as multimap


OLD_PROTOCOL_HASH = (
    "68fde73db57d07f4c9451ed54baa978e09985c5017c43e68adac54d6f2a5131a"
)
NEW_PROTOCOL_HASH = (
    "662766dcca7b964e67d9a49603c8d8634f468b4b02c19409927dfe358c14580e"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    protocol = multimap.load_protocol()
    if protocol["protocol_hash"] != NEW_PROTOCOL_HASH:
        raise RuntimeError("v3.1.12协议哈希不匹配。")
    parent = json.loads(
        multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
    )
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
    records_paths = {
        "validation": output_root
        / "manifests"
        / "validation"
        / "records.jsonl",
        "training_serial": output_root
        / "manifests"
        / "training"
        / "records.jsonl",
    }
    shard_root = output_root / "manifests" / "training_shards"
    for shard_name in ("shard_00", "shard_01", "shard_02", "shard_03"):
        records_paths[shard_name] = shard_root / shard_name / "records.jsonl"
    before_hashes = {
        name: sha256_file(path) for name, path in records_paths.items()
    }

    # 在迁移身份前逐条验证：新SOC边界没有改变任何已接受任务。
    training_rows_by_id = {}
    diagnostic_overlap_ids = set()
    source_counts = {}
    minimum_soc = math.inf
    for source, path in records_paths.items():
        rows = multimap._read_jsonl(path)
        source_counts[source] = len(rows)
        for row in rows:
            task_id = str(row.get("id", ""))
            if row.get("task_hash") != multimap._canonical_hash(
                row, excluded=("task_hash",)
            ):
                raise RuntimeError(f"任务哈希无效：{source}:{task_id}")
            if not all_finite(row):
                raise RuntimeError(f"任务包含非有限值：{source}:{task_id}")
            reasons = multimap._audit_budget_transform_record(
                row, protocol, parent
            )
            if reasons:
                raise RuntimeError(
                    f"预算变换审计失败：{source}:{task_id}:{reasons}"
                )
            minimum_soc = min(minimum_soc, float(row["initial_soc"]))
            if source != "validation":
                existing = training_rows_by_id.get(task_id)
                if existing is not None and existing != row:
                    # 旧串行map004与分片重算受限时MILP证书影响，不进入同一合并输入。
                    diagnostic_overlap_ids.add(task_id)
                training_rows_by_id[task_id] = row
    expected_overlap_ids = {
        f"training__training__map_004__task_{index:02d}"
        for index in range(6)
    }
    if diagnostic_overlap_ids != expected_overlap_ids:
        raise RuntimeError(
            "诊断性重复任务集合异常："
            f"{sorted(diagnostic_overlap_ids)}"
        )
    if len(training_rows_by_id) != 134 or minimum_soc <= 0.251:
        raise RuntimeError(
            "迁移前记录门禁失败："
            f"unique={len(training_rows_by_id)} min_soc={minimum_soc}"
        )

    serial_rows = multimap._read_jsonl(records_paths["training_serial"])
    serial_merge_rows = [
        row
        for row in serial_rows
        if int(str(row["map_id"]).rsplit("_", 1)[1]) < 4
    ]
    if len(serial_rows) != 42 or len(serial_merge_rows) != 36:
        raise RuntimeError("串行正式合并输入不是预期的42保留、36进入合并。")
    serial_merge_path = (
        output_root
        / "manifests"
        / "training"
        / "serial_base_for_merge.jsonl"
    )
    multimap._atomic_text(
        serial_merge_path, multimap._jsonl_text(serial_merge_rows)
    )

    migrated_registries = {}
    for name, path in registry_paths.items():
        registry = json.loads(path.read_text(encoding="utf-8"))
        if registry.get("protocol_hash") != OLD_PROTOCOL_HASH:
            raise RuntimeError(f"{name}注册表不是v3.1.11身份。")
        old_hash = str(registry["registry_hash"])
        registry["protocol_hash"] = NEW_PROTOCOL_HASH
        registry["supersedes_registry_hash"] = old_hash
        registry["registry_hash"] = multimap._canonical_hash(
            registry, excluded=("registry_hash",)
        )
        multimap._atomic_json(path, registry)
        migrated_registries[name] = {
            "old_registry_hash": old_hash,
            "new_registry_hash": registry["registry_hash"],
        }

    validation_dir = output_root / "manifests" / "validation"
    validation_manifest_path = validation_dir / "manifest.json"
    validation_manifest = json.loads(
        validation_manifest_path.read_text(encoding="utf-8")
    )
    if validation_manifest.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError("validation manifest不是v3.1.11身份。")
    validation_registry = json.loads(
        registry_paths["validation"].read_text(encoding="utf-8")
    )
    validation_provider = multimap.FrozenMapProvider.from_registries(
        map_root, [registry_paths["validation"]]
    )
    validation_manifest["protocol_hash"] = NEW_PROTOCOL_HASH
    validation_manifest["map_registry_hash"] = validation_registry[
        "registry_hash"
    ]
    validation_manifest["map_provider_hash"] = (
        validation_provider.provider_hash
    )
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

    checkpoint_paths = {
        "validation": validation_dir / "generation_checkpoint.json",
        "training_serial": output_root
        / "manifests"
        / "training"
        / "generation_checkpoint.json",
    }
    for shard_name in ("shard_00", "shard_01", "shard_02", "shard_03"):
        checkpoint_paths[shard_name] = (
            shard_root / shard_name / "generation_checkpoint.json"
        )
    checkpoint_counts = {}
    for name, path in checkpoint_paths.items():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("protocol_hash") != OLD_PROTOCOL_HASH:
            raise RuntimeError(f"{name}断点不是v3.1.11身份。")
        checkpoint["protocol_hash"] = NEW_PROTOCOL_HASH
        checkpoint["supersedes_protocol_hash"] = OLD_PROTOCOL_HASH
        checkpoint["protocol_migration"] = (
            "v3.1.12 enforces initial_soc >= 0.251; retained records "
            "are byte-identical and all have initial_soc > 0.394"
        )
        multimap._atomic_json(path, checkpoint)
        checkpoint_counts[name] = int(checkpoint.get("completed", -1))

    after_hashes = {
        name: sha256_file(path) for name, path in records_paths.items()
    }
    if before_hashes != after_hashes:
        raise RuntimeError("迁移过程中records发生字节漂移。")
    report = {
        "old_protocol_hash": OLD_PROTOCOL_HASH,
        "new_protocol_hash": NEW_PROTOCOL_HASH,
        "unique_training_records_unchanged": 134,
        "minimum_retained_initial_soc": minimum_soc,
        "source_record_counts": source_counts,
        "serial_records_preserved": len(serial_rows),
        "serial_records_used_for_merge": len(serial_merge_rows),
        "serial_merge_records_sha256": sha256_file(serial_merge_path),
        "diagnostic_overlap_ids_excluded_from_merge": sorted(
            diagnostic_overlap_ids
        ),
        "checkpoint_counts": checkpoint_counts,
        "record_sha256_before": before_hashes,
        "record_sha256_after": after_hashes,
        "validation_manifest_hash": validation_manifest["manifest_hash"],
        "registries": migrated_registries,
    }
    report["migration_hash"] = multimap._canonical_hash(
        report, excluded=("migration_hash",)
    )
    multimap._atomic_json(
        output_root / "audits" / "migration_v3_1_12.json", report
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

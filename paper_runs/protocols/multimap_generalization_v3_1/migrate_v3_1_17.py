#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将瞬时 Windows 文件锁修复登记为 v3.1.17，并保留所有原始场景记录。"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paper_multimap_experiments as multimap


OLD_PROTOCOL_HASH = "20c246fdc986f4fb6654449ef9f306188c410e7c3637e929e8c88b79ddb79c9b"
NEW_PROTOCOL_HASH = "8014a94241779ca55745ebcf533784a51682a6ff8cfa1ad41af0ce84760e61ce"
LABEL = "v3.1.17"
DESCRIPTION = (
    "Bounded retry for transient Windows atomic-replace locks only; no map, "
    "scenario, reward, network, mask, optimizer, schedule, evaluator, or "
    "selection rule changed."
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _update_registry(path: Path) -> Dict[str, str]:
    registry = _read_json(path)
    if registry.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError(f"registry protocol mismatch: {path}")
    old_hash = str(registry["registry_hash"])
    registry["protocol_hash"] = NEW_PROTOCOL_HASH
    registry["supersedes_registry_hash"] = old_hash
    registry["registry_hash"] = multimap._canonical_hash(
        registry, excluded=("registry_hash",)
    )
    multimap._atomic_json(path, registry)
    return {"old": old_hash, "new": str(registry["registry_hash"])}


def _update_manifest(
    path: Path,
    registry_hash: str,
    provider_hash: str,
    shard_manifest_hashes: Mapping[str, str] | None = None,
) -> str:
    manifest = _read_json(path)
    if manifest.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError(f"manifest protocol mismatch: {path}")
    old_hash = str(manifest["manifest_hash"])
    manifest["protocol_hash"] = NEW_PROTOCOL_HASH
    manifest["supersedes_manifest_hash"] = old_hash
    manifest["map_registry_hash"] = registry_hash
    manifest["map_provider_hash"] = provider_hash
    if shard_manifest_hashes is not None:
        manifest["shard_manifest_hashes"] = dict(shard_manifest_hashes)
    manifest["manifest_hash"] = multimap._canonical_hash(
        manifest, excluded=("manifest_hash",)
    )
    multimap._atomic_json(path, manifest)
    return str(manifest["manifest_hash"])


def _update_checkpoint(path: Path) -> None:
    checkpoint = _read_json(path)
    if checkpoint.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError(f"checkpoint protocol mismatch: {path}")
    checkpoint["protocol_hash"] = NEW_PROTOCOL_HASH
    checkpoint["supersedes_protocol_hash"] = OLD_PROTOCOL_HASH
    checkpoint["protocol_migration"] = DESCRIPTION
    multimap._atomic_json(path, checkpoint)


def _archive_runtime_failure(output_root: Path) -> Dict[str, str]:
    audits = output_root / "audits"
    freeze_path = output_root / "environment_freeze.json"
    failed_run = output_root / "formal_training" / "formal_full_seed42_3000ep"
    archive_freeze = audits / "superseded_environment_freeze_v3_1_16.json"
    archive_run = audits / "formal_full_seed42_3000ep_runtime_io_failure_v3_1_16"
    if not freeze_path.is_file() or not failed_run.is_dir():
        raise RuntimeError("missing v3.1.16 freeze or failed formal run")
    if archive_freeze.exists() or archive_run.exists():
        raise RuntimeError("v3.1.16 runtime-failure archive already exists")
    freeze = _read_json(freeze_path)
    status = _read_json(failed_run / "status.json")
    if freeze.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError("freeze does not belong to v3.1.16")
    if status.get("state") != "failed" or int(status.get("completed", -1)) != 724:
        raise RuntimeError("failed formal run is not the registered episode-724 I/O failure")
    # 移动而非删除：失败断点和日志必须保留，但不能混入新的正式训练批次。
    freeze_path.replace(archive_freeze)
    failed_run.replace(archive_run)
    report = {
        "schema_version": 1,
        "old_protocol_hash": OLD_PROTOCOL_HASH,
        "new_protocol_hash": NEW_PROTOCOL_HASH,
        "state": "superseded_runtime_io_failure",
        "completed_episodes": 724,
        "error": str(status.get("error", "")),
        "archived_environment_freeze": str(archive_freeze),
        "archived_formal_run": str(archive_run),
        "paper_eligible": False,
        "no_formal_model_completed": True,
    }
    report["archive_hash"] = multimap._canonical_hash(
        report, excluded=("archive_hash",)
    )
    multimap._atomic_json(audits / "formal_runtime_io_failure_v3_1_16.json", report)
    return {
        "old_freeze_hash": str(freeze["freeze_hash"]),
        "archive_freeze": str(archive_freeze),
        "archive_run": str(archive_run),
        "failure_archive_hash": str(report["archive_hash"]),
    }


def _carry_forward_pilot_gate(output_root: Path) -> Dict[str, str]:
    pilot = output_root / "pilot"
    decision_path = pilot / "pilot_decision.json"
    archive_path = pilot / "pilot_decision_v3_1_16.json"
    if archive_path.exists():
        raise RuntimeError("v3.1.16 pilot decision already archived")
    decision = _read_json(decision_path)
    if decision.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError("pilot decision does not belong to v3.1.16")
    if decision.get("decision") not in {"pilot_passed", "pilot_passed_pointer_lag"}:
        raise RuntimeError("pilot gate did not permit formal training")
    original_hash = str(decision["decision_hash"])
    shutil.copy2(decision_path, archive_path)
    decision["protocol_hash"] = NEW_PROTOCOL_HASH
    decision["source_protocol_hash"] = OLD_PROTOCOL_HASH
    decision["source_decision_hash"] = original_hash
    decision["protocol_migration"] = DESCRIPTION
    decision["decision_hash"] = multimap._canonical_hash(
        decision, excluded=("decision_hash",)
    )
    multimap._atomic_json(decision_path, decision)
    return {
        "source_decision_hash": original_hash,
        "carried_decision_hash": str(decision["decision_hash"]),
        "archive": str(archive_path),
    }


def main() -> None:
    protocol = multimap.load_protocol()
    if protocol.get("protocol_hash") != NEW_PROTOCOL_HASH:
        raise RuntimeError("v3.1.17 protocol hash mismatch")
    output_root = ROOT / "paper_runs" / "multimap_v3_1"
    map_root = ROOT / "map_data" / "multimap_v3_1"
    registry_paths = {
        "real": map_root / "real" / "map_registry.json",
        "training": map_root / "procedural" / "training" / "map_registry.json",
        "validation": map_root / "procedural" / "validation" / "map_registry.json",
    }
    training = output_root / "manifests" / "training"
    validation = output_root / "manifests" / "validation"
    shard_root = output_root / "manifests" / "training_shards"
    records = {
        "training": training / "records.jsonl",
        "validation": validation / "records.jsonl",
        "serial_base_for_merge": training / "serial_base_for_merge.jsonl",
        **{
            name: shard_root / name / "records.jsonl"
            for name in ("shard_00", "shard_01", "shard_02", "shard_03")
        },
    }
    expected_counts = {
        "training": 648,
        "validation": 108,
        "serial_base_for_merge": 36,
        "shard_00": 162,
        "shard_01": 153,
        "shard_02": 153,
        "shard_03": 144,
    }
    before = {name: _sha256(path) for name, path in records.items()}
    counts = {name: len(multimap._read_jsonl(path)) for name, path in records.items()}
    if counts != expected_counts:
        raise RuntimeError(f"unexpected immutable record counts: {counts}")

    registries = {name: _update_registry(path) for name, path in registry_paths.items()}
    training_provider = multimap.FrozenMapProvider.from_registries(
        map_root, [registry_paths["training"]]
    )
    validation_provider = multimap.FrozenMapProvider.from_registries(
        map_root, [registry_paths["validation"]]
    )
    validation_manifest_hash = _update_manifest(
        validation / "manifest.json",
        registries["validation"]["new"],
        validation_provider.provider_hash,
    )
    shard_hashes = {
        name: _update_manifest(
            shard_root / name / "manifest.json",
            registries["training"]["new"],
            training_provider.provider_hash,
        )
        for name in ("shard_00", "shard_01", "shard_02", "shard_03")
    }
    training_manifest_hash = _update_manifest(
        training / "manifest.json",
        registries["training"]["new"],
        training_provider.provider_hash,
        shard_hashes,
    )
    for path in (
        validation / "generation_checkpoint.json",
        training / "generation_checkpoint.json",
        *[
            shard_root / name / "generation_checkpoint.json"
            for name in ("shard_00", "shard_01", "shard_02", "shard_03")
        ],
    ):
        _update_checkpoint(path)

    supervisor_path = output_root / "monitoring" / "parallel_training_supervisor.json"
    supervisor = _read_json(supervisor_path)
    if supervisor.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError("parallel certification supervisor protocol mismatch")
    supervisor["protocol_hash"] = NEW_PROTOCOL_HASH
    supervisor["supersedes_protocol_hash"] = OLD_PROTOCOL_HASH
    supervisor["protocol_migration"] = DESCRIPTION
    multimap._atomic_json(supervisor_path, supervisor)

    pilot_gate = _carry_forward_pilot_gate(output_root)
    failure_archive = _archive_runtime_failure(output_root)
    after = {name: _sha256(path) for name, path in records.items()}
    if before != after:
        raise RuntimeError("immutable task record bytes changed during migration")
    report = {
        "schema_version": 1,
        "migration_label": LABEL,
        "old_protocol_hash": OLD_PROTOCOL_HASH,
        "new_protocol_hash": NEW_PROTOCOL_HASH,
        "description": DESCRIPTION,
        "records_byte_identical": True,
        "record_sha256_before": before,
        "record_sha256_after": after,
        "source_record_counts": counts,
        "registries": registries,
        "training_manifest_hash": training_manifest_hash,
        "validation_manifest_hash": validation_manifest_hash,
        "shard_manifest_hashes": shard_hashes,
        "pilot_gate": pilot_gate,
        "runtime_failure_archive": failure_archive,
    }
    report["migration_hash"] = multimap._canonical_hash(
        report, excluded=("migration_hash",)
    )
    multimap._atomic_json(output_root / "audits" / "migration_v3_1_17.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

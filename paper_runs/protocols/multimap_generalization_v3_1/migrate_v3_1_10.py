#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把未进入试训的v3.1.9地图注册表和断点机械迁移到v3.1.10。"""

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
    "86e19710e1422712f2412e05bfa19c07f7dfb19aab91e17fb68427c5fca98680"
)
NEW_PROTOCOL_HASH = (
    "59ba71e12cd9edb55390f4961c844e30b8277815523cb002b6a77c07b183c79c"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = multimap.load_protocol()
    if protocol["protocol_hash"] != NEW_PROTOCOL_HASH:
        raise RuntimeError("v3.1.10协议哈希不匹配。")
    registry_paths = (
        ROOT / "map_data" / "multimap_v3_1" / "real" / "map_registry.json",
        ROOT
        / "map_data"
        / "multimap_v3_1"
        / "procedural"
        / "training"
        / "map_registry.json",
        ROOT
        / "map_data"
        / "multimap_v3_1"
        / "procedural"
        / "validation"
        / "map_registry.json",
    )
    records_path = (
        ROOT
        / "paper_runs"
        / "multimap_v3_1"
        / "manifests"
        / "validation"
        / "records.jsonl"
    )
    records_hash_before = file_sha256(records_path)
    records_count = sum(
        1
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if records_count != 78:
        raise RuntimeError("只允许迁移已确认的78条validation记录。")

    migrated = {}
    for path in registry_paths:
        registry = json.loads(path.read_text(encoding="utf-8"))
        if registry.get("protocol_hash") != OLD_PROTOCOL_HASH:
            raise RuntimeError(f"注册表不是待迁移的v3.1.9身份：{path}")
        old_registry_hash = str(registry["registry_hash"])
        registry["protocol_hash"] = NEW_PROTOCOL_HASH
        registry["supersedes_registry_hash"] = old_registry_hash
        registry["registry_hash"] = multimap._canonical_hash(
            registry, excluded=("registry_hash",)
        )
        multimap._atomic_json(path, registry)
        migrated[str(path.relative_to(ROOT))] = {
            "old_registry_hash": old_registry_hash,
            "new_registry_hash": registry["registry_hash"],
        }

    checkpoint_path = (
        ROOT
        / "paper_runs"
        / "multimap_v3_1"
        / "manifests"
        / "validation"
        / "generation_checkpoint.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("protocol_hash") != OLD_PROTOCOL_HASH:
        raise RuntimeError("validation断点不是待迁移的v3.1.9身份。")
    if (
        int(checkpoint.get("completed", -1)) != 78
        or checkpoint.get("current_task_id")
        != "validation__validation__map_008__task_06"
        or int(checkpoint.get("current_attempt", -1)) != 680
    ):
        raise RuntimeError("validation断点位置与已审计状态不一致。")
    checkpoint["protocol_hash"] = NEW_PROTOCOL_HASH
    checkpoint["supersedes_protocol_hash"] = OLD_PROTOCOL_HASH
    checkpoint["protocol_migration"] = (
        "v3.1.10 adds a model-independent non-intended-resource relaxation "
        "fallback; the 78 accepted task records are byte-identical"
    )
    multimap._atomic_json(checkpoint_path, checkpoint)

    records_hash_after = file_sha256(records_path)
    if records_hash_after != records_hash_before:
        raise RuntimeError("迁移过程中validation记录发生字节漂移。")
    report = {
        "old_protocol_hash": OLD_PROTOCOL_HASH,
        "new_protocol_hash": NEW_PROTOCOL_HASH,
        "accepted_validation_records_unchanged": records_count,
        "records_sha256_before": records_hash_before,
        "records_sha256_after": records_hash_after,
        "checkpoint_resume_task": checkpoint["current_task_id"],
        "checkpoint_resume_attempt": checkpoint["current_attempt"],
        "registries": migrated,
    }
    report["migration_hash"] = multimap._canonical_hash(
        report, excluded=("migration_hash",)
    )
    multimap._atomic_json(
        ROOT
        / "paper_runs"
        / "multimap_v3_1"
        / "audits"
        / "migration_v3_1_10.json",
        report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

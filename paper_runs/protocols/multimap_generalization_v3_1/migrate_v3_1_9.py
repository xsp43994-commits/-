#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把未进入试训的v3.1.8地图注册表和断点机械迁移到v3.1.9。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paper_multimap_experiments as multimap


OLD_PROTOCOL_HASH = (
    "a5010fff297142de5bf3b5e889df38b6598de61cb25c0686e3083396c5ad9e40"
)
NEW_PROTOCOL_HASH = (
    "86e19710e1422712f2412e05bfa19c07f7dfb19aab91e17fb68427c5fca98680"
)


def main() -> None:
    protocol = multimap.load_protocol()
    if protocol["protocol_hash"] != NEW_PROTOCOL_HASH:
        raise RuntimeError("v3.1.9协议哈希不匹配。")
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
    migrated = {}
    for path in registry_paths:
        registry = json.loads(path.read_text(encoding="utf-8"))
        if registry.get("protocol_hash") != OLD_PROTOCOL_HASH:
            raise RuntimeError(f"注册表不是待迁移的v3.1.8身份：{path}")
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
        raise RuntimeError("validation断点不是待迁移的v3.1.8身份。")
    if int(checkpoint.get("completed", -1)) != 69:
        raise RuntimeError("只允许迁移已确认的69条validation断点。")
    checkpoint["protocol_hash"] = NEW_PROTOCOL_HASH
    checkpoint["supersedes_protocol_hash"] = OLD_PROTOCOL_HASH
    checkpoint["protocol_migration"] = (
        "v3.1.9 adds a model-independent resource-threshold fallback; "
        "the 69 accepted task records are byte-identical"
    )
    multimap._atomic_json(checkpoint_path, checkpoint)
    report = {
        "old_protocol_hash": OLD_PROTOCOL_HASH,
        "new_protocol_hash": NEW_PROTOCOL_HASH,
        "accepted_validation_records_unchanged": 69,
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
        / "migration_v3_1_9.json",
        report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

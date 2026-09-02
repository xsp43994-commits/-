#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the v3.2.1 pre-test repair protocol without altering v3.2 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT


ROOT = WORKSPACE_ROOT
SOURCE = ROOT / "paper_runs/protocols/multimap_generalization_v3_2/protocol.json"
DESTINATION = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_1/protocol.json"


def canonical_hash(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "protocol_hash"}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_1"
    protocol["protocol_name"] = "multimap_generalization_v3_2_1_pretest_repair"
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_repair"] = {
        "reason": "v3.2 pre-test certification failed before any formal algorithm evaluation: one synthetic cell exhausted its model-free MILP candidate budget, and the sealed real-road tracks could not support the prespecified 24-node per-road task grid.",
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "archived_v3_2_output_root": str((ROOT / "paper_runs/multimap_v3_2").resolve()),
        "failed_synthetic_records_retained": 213,
        "failed_real_records_retained": 12,
        "old_records_excluded_from_v3_2_1_statistics": True,
    }
    protocol["real_corridor_contexts"] = {
        "definition": "two deterministic launch contexts per DSM, each using the complete clipped OSM road corridor for point placement and a distinct geometry-certified launch position; contexts are nested within map and are not independent geographic samples",
        "source": "existing sealed DSM and archived raw OSM JSON only",
        "base_candidate_stride": 4,
        "base_selection": "maximum feasible candidate capacity, then maximum spatial separation among remaining feasible bases",
        "all_node_counts_preflight": [16, 20, 24],
        "minimum_spacing_m": 120.0,
        "maximum_task_radius_m": 3200.0,
    }
    real = protocol["formal_evaluation"]["real_external"]
    real["road_contexts_per_map"] = real.pop("road_tracks_per_map")
    real["road_contexts_are_nested_within_map"] = True
    protocol["formal_evaluation"]["robustness"]["selection"]["road_balance_total"] = {
        "context_0": 12,
        "context_1": 12,
    }
    protocol["protocol_hash"] = canonical_hash(protocol)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(DESTINATION), "protocol_hash": protocol["protocol_hash"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

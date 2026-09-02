#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create v3.2.9's bounded two-resource grid for every task type."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT


ROOT = WORKSPACE_ROOT
SOURCE = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_8/protocol.json"
DESTINATION = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_9/protocol.json"


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_9"
    protocol["protocol_name"] = "multimap_generalization_v3_2_9_all_type_two_resource_grid"
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_all_type_two_resource_grid"] = {
        "reason": "a nominal single-resource task can still have a terrain-induced nuisance resource that prevents the target coverage band. Applying the same fixed two-resource grid supplies a candidate to the unchanged final certificate; it does not alter the registered task label or acceptance condition.",
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "archived_v3_2_8_output_root": str((ROOT / "paper_runs/multimap_v3_2_8").resolve()),
        "old_v3_2_8_records_excluded_from_v3_2_9_statistics": True,
        "all_constraint_types_use_two_resource_grid": True,
        "resource_order": "descending MILP utilization with deterministic tie order energy,distance,time",
        "maximum_grid_probes_per_geometry": 12,
        "final_certificate_limits_unchanged": [10.0, 60.0],
    }
    payload = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    protocol["protocol_hash"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(DESTINATION), "protocol_hash": protocol["protocol_hash"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

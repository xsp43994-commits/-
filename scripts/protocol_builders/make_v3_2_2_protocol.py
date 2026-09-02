#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create v3.2.2's coordinate-preflight protocol without overwriting prior evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT


ROOT = WORKSPACE_ROOT
SOURCE = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_1/protocol.json"
DESTINATION = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_2/protocol.json"


def canonical_hash(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "protocol_hash"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_2"
    protocol["protocol_name"] = "multimap_generalization_v3_2_2_coordinate_preflight"
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_coordinate_boundary_repair"] = {
        "reason": "v3.2.1 real corridor coordinates could equal a raster edge after geographic-to-pixel conversion; TerrainModel correctly rejects coordinates outside its half-open raster domain.",
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "archived_v3_2_1_output_root": str((ROOT / "paper_runs/multimap_v3_2_1").resolve()),
        "old_v3_2_1_records_excluded_from_v3_2_2_statistics": True,
        "coordinate_rule": "clip only source-road geometry to [epsilon, width-1-epsilon] x [epsilon, height-1-epsilon] before any task sampling",
        "boundary_epsilon_px": 0.001,
        "required_preflight": [
            "all clipped road coordinates and selected bases are in the valid raster domain",
            "each map context is feasible for node counts 16, 20, and 24",
            "one end-to-end certified task per map context completes before full task generation",
        ],
    }
    protocol["real_corridor_contexts"]["boundary_epsilon_px"] = 0.001
    protocol["real_corridor_contexts"]["preflight_scope"] = "coordinate-domain validation for all road vertices and bases, all-node-count geometry preflight, then one certified task per context"
    protocol["protocol_hash"] = canonical_hash(protocol)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(DESTINATION), "protocol_hash": protocol["protocol_hash"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

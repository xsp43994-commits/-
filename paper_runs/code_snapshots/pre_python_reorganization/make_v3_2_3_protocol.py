#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create v3.2.3's deterministic geometry-preselection protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_2/protocol.json"
DESTINATION = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_3/protocol.json"


def canonical_hash(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "protocol_hash"}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_3"
    protocol["protocol_name"] = "multimap_generalization_v3_2_3_geometry_preselection"
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_geometry_preselection"] = {
        "reason": "v3.2.2 exhaustively reran all-node-count geometry preflight for every possible launch sample although only two contexts per map are required; this is redundant and delays input construction without adding independent evidence.",
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "archived_v3_2_2_output_root": str((ROOT / "paper_runs/multimap_v3_2_2").resolve()),
        "old_v3_2_2_records_excluded_from_v3_2_3_statistics": True,
        "candidate_deck": "fixed top-48 source-road samples ranked by geometry-only reachable-road capacity, ties broken by source sample index",
        "acceptance": "each selected context still requires explicit 16/20/24 geometry preflight and one end-to-end MILP-certified smoke task before formal task generation",
    }
    contexts = protocol["real_corridor_contexts"]
    contexts["base_preflight_candidate_count"] = 48
    contexts["base_selection"] = "first geometry-feasible member of the fixed capacity-ranked candidate deck, then the remaining feasible member with maximum spatial separation (capacity then source-index tie-break)"
    protocol["protocol_hash"] = canonical_hash(protocol)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(DESTINATION), "protocol_hash": protocol["protocol_hash"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

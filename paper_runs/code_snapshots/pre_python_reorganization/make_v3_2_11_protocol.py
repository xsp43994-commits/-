#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create v3.2.11's fixed-budget final MILP handoff protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_10/protocol.json"
DESTINATION = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_11/protocol.json"


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_11"
    protocol["protocol_name"] = "multimap_generalization_v3_2_11_fixed_budget_final_milp"
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_fixed_budget_final_milp"] = {
        "reason": "the legacy acceptance adapter may recalibrate a bounded-search candidate before its final solve. For a fixed candidate whose ten-second interval intersects the band, the final sixty-second MILP must evaluate that exact candidate rather than a modified budget.",
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "archived_v3_2_10_output_root": str((ROOT / "paper_runs/multimap_v3_2_10").resolve()),
        "old_v3_2_10_records_excluded_from_v3_2_11_statistics": True,
        "fixed_candidate_budget_recalibration_forbidden": True,
        "final_acceptance": "unchanged strict 60-second MILP certificate with safe return, registered coverage band, and constraint checks",
    }
    payload = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    protocol["protocol_hash"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(DESTINATION), "protocol_hash": protocol["protocol_hash"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

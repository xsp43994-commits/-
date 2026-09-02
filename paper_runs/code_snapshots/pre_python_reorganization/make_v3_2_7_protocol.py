#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create v3.2.7's bounded rescue for fast MILP probes without a bound."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_6/protocol.json"
DESTINATION = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_7/protocol.json"


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_7"
    protocol["protocol_name"] = "multimap_generalization_v3_2_7_missing_bound_rescue"
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_missing_bound_rescue"] = {
        "reason": "a two-second screen has false negatives on a subset of terrain geometries: it returns no MILP bound although the unchanged ten-second screen quickly certifies a valid task. Rescue only those no-bound probes, never loosening final acceptance.",
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "archived_v3_2_6_output_root": str((ROOT / "paper_runs/multimap_v3_2_6").resolve()),
        "old_v3_2_6_records_excluded_from_v3_2_7_statistics": True,
        "fast_probe_limit_s": 2.0,
        "rescue_trigger": "missing_solver_bound_only",
        "rescue_limit_s": 10.0,
        "final_certificate_limits_unchanged": [10.0, 60.0],
    }
    protocol["pretest_bounded_certificate_search"]["missing_bound_rescue"] = True
    payload = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    protocol["protocol_hash"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(DESTINATION), "protocol_hash": protocol["protocol_hash"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

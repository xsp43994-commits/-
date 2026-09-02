#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create v3.2.8's complete rescue rule for fast-probe false negatives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT


ROOT = WORKSPACE_ROOT
SOURCE = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_7/protocol.json"
DESTINATION = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_8/protocol.json"


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_8"
    protocol["protocol_name"] = "multimap_generalization_v3_2_8_complete_fast_probe_rescue"
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_complete_fast_probe_rescue"] = {
        "reason": "a two-second solve can return a provisional non-acceptance reason other than missing bounds even when the same candidate is strictly accepted at ten seconds. All non-accepted fast probes therefore receive one ten-second rescue; only the fixed candidate grid is searched.",
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "archived_v3_2_7_output_root": str((ROOT / "paper_runs/multimap_v3_2_7").resolve()),
        "old_v3_2_7_records_excluded_from_v3_2_8_statistics": True,
        "fast_probe_limit_s": 2.0,
        "rescue_trigger": "any_fast_nonacceptance_once",
        "rescue_limit_s": 10.0,
        "fixed_mixed_grid_maximum": 12,
        "final_certificate_limits_unchanged": [10.0, 60.0],
    }
    protocol["pretest_bounded_certificate_search"]["fast_probe_rescue_all_nonaccepted"] = True
    payload = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    protocol["protocol_hash"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(DESTINATION), "protocol_hash": protocol["protocol_hash"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

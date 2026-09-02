#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create v3.2.10's complete bounded-certificate handoff protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_9/protocol.json"
DESTINATION = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_10/protocol.json"


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_10"
    protocol["protocol_name"] = "multimap_generalization_v3_2_10_complete_bounded_certificate_handoff"
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_complete_certificate_handoff"] = {
        "reason": "the bounded grid omitted the unmodified secondary resource and did not pass a ten-second interval that intersected the registered band to the existing sixty-second final MILP certificate. Both are required for complete bounded candidate evaluation.",
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "archived_v3_2_9_output_root": str((ROOT / "paper_runs/multimap_v3_2_9").resolve()),
        "old_v3_2_9_records_excluded_from_v3_2_10_statistics": True,
        "secondary_factor_order_above_band": [1.0, 0.85, 0.70, 0.55],
        "final_handoff": "run unchanged 10/60-second acceptance whenever a screen is accepted or its certified interval intersects the pre-registered difficulty band",
        "final_certificate_limits_unchanged": [10.0, 60.0],
    }
    protocol["pretest_bounded_certificate_search"]["secondary_factor_order_above_band"] = [1.0, 0.85, 0.70, 0.55]
    payload = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    protocol["protocol_hash"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(DESTINATION), "protocol_hash": protocol["protocol_hash"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

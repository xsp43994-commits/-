#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create v3.2.6's exact two-resource fast-probe protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_5/protocol.json"
DESTINATION = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_6/protocol.json"


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_6"
    protocol["protocol_name"] = "multimap_generalization_v3_2_6_fast_probe_grid"
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_fast_probe_grid"] = {
        "reason": "the v3.2.5 sequential first-resource search could miss a valid mixed task when a 2-second single-resource probe has no bound even though the jointly tightened two-resource candidate is decisive. The fixed grid evaluates those joint candidates directly.",
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "archived_v3_2_5_output_root": str((ROOT / "paper_runs/multimap_v3_2_5").resolve()),
        "old_v3_2_5_records_excluded_from_v3_2_6_statistics": True,
        "mixed_above_band_primary_factors": [0.70, 0.55, 0.46, 0.40],
        "mixed_above_band_secondary_factors": [0.85, 0.70, 0.55],
        "probe_time_limit_s": 2.0,
        "final_certificate_limits_unchanged": [10.0, 60.0],
    }
    protocol["pretest_bounded_certificate_search"]["mixed_grid"] = "4 primary factors x 3 secondary factors after resource-utilization ranking"
    payload = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    protocol["protocol_hash"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(DESTINATION), "protocol_hash": protocol["protocol_hash"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

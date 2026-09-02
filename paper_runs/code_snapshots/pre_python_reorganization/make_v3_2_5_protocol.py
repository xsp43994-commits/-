#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create v3.2.5's two-stage bounded certificate-search protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_4/protocol.json"
DESTINATION = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_5/protocol.json"


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_5"
    protocol["protocol_name"] = "multimap_generalization_v3_2_5_two_stage_certificate_search"
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_two_stage_search"] = {
        "reason": "v3.2.4 used the final 10-second MILP screen for every monotone probe. A 2-second screen is sufficient to discard obvious full-coverage and out-of-band candidates; the unchanged 10/60-second certificate is reserved for candidates that pass this fast screen.",
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "archived_v3_2_4_output_root": str((ROOT / "paper_runs/multimap_v3_2_4").resolve()),
        "old_v3_2_4_records_excluded_from_v3_2_5_statistics": True,
        "fast_probe_limit_s": 2.0,
        "final_certificate_limits_unchanged": [10.0, 60.0],
    }
    protocol["pretest_bounded_certificate_search"]["probe_time_limit_s"] = 2.0
    asset_root = ROOT / "paper_runs/multimap_v3_2_4/real_corridor_assets"
    asset_manifest = json.loads((asset_root / "manifest.json").read_text(encoding="utf-8"))
    protocol["real_corridor_asset_root"] = str(asset_root.resolve())
    protocol["real_corridor_asset_protocol_hash"] = asset_manifest["protocol_hash"]
    protocol["real_corridor_asset_manifest_hash"] = asset_manifest["manifest_hash"]
    payload = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    protocol["protocol_hash"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(DESTINATION), "protocol_hash": protocol["protocol_hash"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

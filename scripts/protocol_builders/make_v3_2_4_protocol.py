#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create v3.2.4's bounded certificate-search protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT


ROOT = WORKSPACE_ROOT
SOURCE = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_3/protocol.json"
DESTINATION = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_4/protocol.json"


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_4"
    protocol["protocol_name"] = "multimap_generalization_v3_2_4_bounded_certificate_search"
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_bounded_certificate_search"] = {
        "reason": "random joint sampling of geometry and multiple resource budgets can spend 2000 MILP probes on candidates that are trivially full-coverage or far below the prespecified band. A finite monotone certificate search reaches the same pre-registered band using fewer model-free probes.",
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "archived_v3_2_3_output_root": str((ROOT / "paper_runs/multimap_v3_2_3").resolve()),
        "old_v3_2_3_records_excluded_from_v3_2_4_statistics": True,
        "fixed_geometry_attempt_count": 16,
        "resource_factor_order": [1.0, 0.85, 0.70, 0.55, 0.46, 0.40, 0.32, 0.25],
        "mixed_rule": "move the most-utilized resource into the registered coverage band, then monotonically tighten only the second-most-utilized resource until the existing strict mixed-bottleneck certificate accepts",
        "final_acceptance": "unchanged existing MILP safety, difficulty-band, return, and bottleneck certificate",
    }
    protocol["protocol_hash"] = hashlib.sha256(json.dumps({k:v for k,v in protocol.items() if k != 'protocol_hash'}, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(DESTINATION), "protocol_hash": protocol["protocol_hash"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

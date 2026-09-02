#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the clean v3.2.12 parametric-certification protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_11/protocol.json"
)
DESTINATION = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_12/protocol.json"
)


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_12"
    protocol["protocol_name"] = (
        "multimap_generalization_v3_2_12_parametric_milp_certification"
    )
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_parametric_certificate_search"] = {
        "reason": (
            "formal difficulty certification and engineering integration smoke "
            "are separated; fixed geometry now uses a deterministic monotone "
            "resource-budget search instead of a coarse two-resource grid"
        ),
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "archived_v3_2_11_output_root": str(
            (ROOT / "paper_runs/multimap_v3_2_11").resolve()
        ),
        "old_v3_2_to_v3_2_11_records_excluded_from_v3_2_12_statistics": True,
        "engineering_sentinels_are_paper_ineligible": True,
        "fixed_geometry_attempt_count": 16,
        "fast_probe_time_limit_s": 2.0,
        "global_tighten_factors": [1.0, 0.85, 0.70, 0.55, 0.40, 0.25],
        "global_loosen_factors": [1.0, 1.15, 1.35, 1.60, 1.90, 2.20],
        "bracket_refinement_steps": 5,
        "activation_tighten_factors": [
            0.95,
            0.90,
            0.85,
            0.80,
            0.75,
            0.70,
            0.60,
            0.50,
        ],
        "single_constraint_absolute_fractions": [
            1.0,
            0.90,
            0.80,
            0.70,
            0.60,
            0.50,
            0.40,
            0.30,
            0.20,
            0.10,
            0.0,
        ],
        "probe_cache_key": (
            "geometry_and_frozen_inputs_plus_exact_budget_triple_and_time_limit"
        ),
        "final_acceptance": (
            "unchanged strict 60-second MILP safe-return, registered coverage "
            "band, solver-bound, and intended-bottleneck checks"
        ),
        "post_handoff_budget_recalibration_forbidden": True,
    }
    payload = {
        key: value for key, value in protocol.items() if key != "protocol_hash"
    }
    protocol["protocol_hash"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "protocol": str(DESTINATION),
                "protocol_hash": protocol["protocol_hash"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the final compact branch-and-cut certificate protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_13/protocol.json"
)
DESTINATION = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/protocol.json"
)


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_14"
    protocol["protocol_name"] = (
        "multimap_generalization_v3_2_14_compact_branch_cut_certificate"
    )
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_compact_branch_cut_certificate"] = {
        "reason": (
            "before formal algorithm evaluation, the Colorado mixed 24-node "
            "cell had a replay-safe lower-threshold route but the generic "
            "MILP repeatedly timed out without a bound; an exact optional-node "
            "assignment with iterative directed subtour cuts proved the first "
            "priority value above the registered band infeasible"
        ),
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "evaluation_semantics_changed": False,
        "formal_algorithm_evaluation_started": False,
        "archived_v3_2_13_output_root": str(
            (ROOT / "paper_runs/multimap_v3_2_13").resolve()
        ),
        "safe_lower_witness": {
            "minimum_resource_threshold_milp_time_limit_s": 60.0,
            "minimum_priority_is_discrete_band_lower_bound": True,
            "must_be_replayed_by_shared_mission_evaluator": True,
        },
        "two_resource_calibration": {
            "resources": ["distance", "time"],
            "target_utilization": 0.98,
            "route_fixed_before_calibration": True,
            "registered_parameter_bounds_unchanged": True,
            "minimum_active_resources": 2,
        },
        "high_threshold_proof": {
            "minimum_priority_is_first_discrete_value_above_band": True,
            "formulation": (
                "exact directed optional-node assignment with all additive "
                "energy, distance and time budgets plus iterative directed "
                "subtour elimination cuts"
            ),
            "time_limit_s": 180.0,
            "cut_sets_persisted_and_hashed": True,
            "proof_run_is_standalone": True,
            "required_terminal_status": "infeasible",
            "timeout_is_never_acceptance": True,
        },
        "proof_composition": (
            "safe low-threshold replay supplies the lower bound; exact "
            "infeasibility of the first priority value above the band supplies "
            "the upper bound"
        ),
        "acceptance_criteria_unchanged": True,
        "formal_evaluation_count_unchanged": 21648,
        "discarded_pre_freeze_draft_hash": (
            "bbab8af73d96684769ee4a7b92ae127fb9b2ead592966aa50832d69a674bf812"
        ),
        "post_handoff_tuning_from_algorithm_results_forbidden": True,
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
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "protocol": str(DESTINATION),
                "protocol_hash": protocol["protocol_hash"],
                "formal_evaluation_total": protocol["formal_evaluation"][
                    "counts"
                ]["total"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

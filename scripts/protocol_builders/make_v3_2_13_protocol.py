#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the v3.2.13 proof-composition amendment before formal evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT


ROOT = WORKSPACE_ROOT
SOURCE = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_12/protocol.json"
)
DESTINATION = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_13/protocol.json"
)


def main() -> int:
    protocol = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol["protocol_version"] = "multimap_generalization_v3_2_13"
    protocol["protocol_name"] = (
        "multimap_generalization_v3_2_13_constructive_witness_certification"
    )
    protocol["supersedes_protocol_hash"] = protocol["protocol_hash"]
    protocol["pretest_constructive_witness_certification"] = {
        "reason": (
            "before any formal algorithm evaluation, the last synthetic "
            "single-resource cell and one real mixed-resource cell exposed "
            "runtime-dependent MILP incumbent/bound variability; v3.2.13 "
            "separates the constructive safe-route lower bound from the "
            "independent MILP upper-bound proof"
        ),
        "algorithm_results_used": False,
        "training_semantics_changed": False,
        "evaluation_semantics_changed": False,
        "formal_algorithm_evaluation_started": False,
        "archived_v3_2_12_output_root": str(
            (ROOT / "paper_runs/multimap_v3_2_12").resolve()
        ),
        "old_v3_2_to_v3_2_12_outputs_are_archive_evidence_only": True,
        "safe_witness": {
            "method": (
                "deterministic certification-only multi-start resource-density "
                "construction followed by bounded beam search"
            ),
            "not_a_reported_or_compared_planner": True,
            "beam_width": 32768,
            "maximum_label_expansions": 5_000_000,
            "time_limit_s": 45.0,
            "acceptance_use": (
                "one route replayed by the shared frozen MissionEvaluator "
                "supplies only a feasible lower bound"
            ),
        },
        "single_resource_proof_transport": {
            "enabled": True,
            "rule": (
                "a threshold-MILP dual lower bound may be transported only "
                "to a tighter actual budget when geometry, frozen scenario, "
                "objective resource, discrete priority threshold, every "
                "non-target budget, and the relaxed threshold-MILP resource "
                "budget are identical"
            ),
            "source_proof_file_sha256_required": True,
            "source_dual_bound_must_exceed_target_budget": True,
            "safe_low_route_must_be_replayed_under_target_budget": True,
        },
        "mixed_resource_proof_composition": {
            "enabled": True,
            "witness_target_utilization": 0.92,
            "minimum_active_resources": 2,
            "high_threshold_rule": (
                "a replay-safe in-band witness supplies the lower bound; a "
                "minimum-resource MILP dual bound proving the first discrete "
                "priority value above the band infeasible supplies the upper "
                "bound"
            ),
            "calibration": (
                "start from registered upper budgets and monotonically tighten "
                "distance and time to the fixed witness route; no map, point, "
                "priority, wind, dynamics, reward, model, or algorithm result "
                "is changed or read"
            ),
        },
        "acceptance_criteria_unchanged": [
            "registered difficulty band",
            "safe return and all hard constraints",
            "complete coverage excluded",
            "intended single-resource bottleneck or two mixed bottlenecks",
            "frozen map task and scenario hashes",
        ],
        "formal_evaluation_count_unchanged": 21648,
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

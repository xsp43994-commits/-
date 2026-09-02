#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从只读 v3.1.17 生成 v3.2 子协议，不改写任何父协议或既有训练证据。"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paper_multimap_experiments as multimap


PARENT_PROTOCOL = (
    ROOT
    / "paper_runs"
    / "protocols"
    / "multimap_generalization_v3_1"
    / "protocol.json"
)
OUTPUT_PROTOCOL = Path(__file__).resolve().parent / "protocol.json"
AMENDMENT = Path(__file__).resolve().parent / "AMENDMENT_v3_2.md"
EXPECTED_PARENT_HASH = (
    "8014a94241779ca55745ebcf533784a51682a6ff8cfa1ad41af0ce84760e61ce"
)

ACTIVE_VARIANTS = [
    "full",
    "traditional_ppo",
    "a2c_pointer",
    "no_priority_bias",
    "no_domain_randomization",
    "no_resource_shaping",
    "no_return_reserve",
]
TRAINING_SEEDS = [42, 43, 44, 45, 46]
MAIN_SYNTHETIC_BASELINES = {
    "nearest_feasible": [42],
    "priority_resource_greedy": [42],
    "aco": TRAINING_SEEDS,
    "ga": TRAINING_SEEDS,
    "sa": TRAINING_SEEDS,
    "milp": [42],
}
SUPPLEMENTARY_BASELINES = {
    "a_star": [42],
    "pso": TRAINING_SEEDS,
    "exact_pareto_dp": [42],
}
REAL_BASELINES = {
    "nearest_feasible": [42],
    "priority_resource_greedy": [42],
    "aco": TRAINING_SEEDS,
    "milp": [42],
}


def _evaluation_design() -> Dict[str, Any]:
    synthetic_learning = 216 * len(ACTIVE_VARIANTS) * len(TRAINING_SEEDS)
    synthetic_main = 216 * sum(
        len(seeds) for seeds in MAIN_SYNTHETIC_BASELINES.values()
    )
    synthetic_supplementary = 72 * sum(
        len(seeds) for seeds in SUPPLEMENTARY_BASELINES.values()
    )
    real_learning = 144 * len(ACTIVE_VARIANTS) * len(TRAINING_SEEDS)
    real_baselines = 144 * sum(len(seeds) for seeds in REAL_BASELINES.values())
    known_members = [
        "full",
        "traditional_ppo",
        "a2c_pointer",
        "no_domain_randomization",
    ]
    mismatch_members = [
        "full",
        "traditional_ppo",
        "a2c_pointer",
        "no_domain_randomization",
        "no_return_reserve",
    ]
    known_rows = 24 * 2 * (len(known_members) * 5 + 1)
    mismatch_rows = 24 * 4 * (len(mismatch_members) * 5 + 1)
    counts = {
        "synthetic_learning": synthetic_learning,
        "synthetic_main_baselines": synthetic_main,
        "synthetic_supplementary": synthetic_supplementary,
        "synthetic_total": (
            synthetic_learning + synthetic_main + synthetic_supplementary
        ),
        "real_learning": real_learning,
        "real_baselines": real_baselines,
        "real_total": real_learning + real_baselines,
        "known_domain_shift": known_rows,
        "hidden_model_perception_mismatch": mismatch_rows,
        "robustness_total": known_rows + mismatch_rows,
    }
    counts["total"] = (
        counts["synthetic_total"]
        + counts["real_total"]
        + counts["robustness_total"]
    )
    if counts["total"] != 21_648:
        raise AssertionError(f"v3.2 formal row count drifted: {counts}")
    return {
        "row_definition": (
            "one checkpoint or planner seed on one task under one evaluation condition"
        ),
        "active_learning_variants": ACTIVE_VARIANTS,
        "training_seeds": TRAINING_SEEDS,
        "deterministic_learning_decode": True,
        "synthetic": {
            "map_count": 24,
            "task_count": 216,
            "all_learning_variants_on_all_tasks": True,
            "main_baselines": MAIN_SYNTHETIC_BASELINES,
            "supplementary_task_count": 72,
            "supplementary_selection": (
                "deterministic_stratification_before_any_algorithm_result"
            ),
            "supplementary_baselines": SUPPLEMENTARY_BASELINES,
        },
        "real_external": {
            "map_count": 8,
            "road_tracks_per_map": 2,
            "task_count": 144,
            "all_learning_variants_on_all_tasks": True,
            "all_four_ablations_on_all_tasks": True,
            "baseline_planners": REAL_BASELINES,
        },
        "robustness": {
            "task_count": 24,
            "selection": {
                "tasks_per_map": 3,
                "node_counts_per_map": [16, 20, 24],
                "road_balance_total": {"road_0": 12, "road_1": 12},
                "balance_fields": [
                    "difficulty",
                    "constraint_type",
                    "priority_layout",
                ],
                "algorithm_results_forbidden": True,
            },
            "known_domain_shift": {
                "factors": ["wind", "power_model"],
                "learning_variants": known_members,
                "baseline": "priority_resource_greedy",
                "planner_and_execution_share_shifted_truth": True,
            },
            "hidden_model_perception_mismatch": {
                "factors": ["wind", "power_model", "dem_error", "localization"],
                "learning_variants": mismatch_members,
                "baseline": "priority_resource_greedy",
                "route_locked_before_truth_evaluation": True,
            },
            "common_random_realizations_across_algorithms": True,
            "factorial_perturbations_forbidden": True,
        },
        "counts": counts,
    }


def _statistics_design() -> Dict[str, Any]:
    return {
        "primary_independent_unit": "map",
        "aggregation_order": ["repeat_within_task", "task_within_map"],
        "primary_metric": "safe_weighted_coverage",
        "omnibus": "Friedman_on_map_aggregates",
        "pairwise": "two_sided_paired_Wilcoxon_on_map_aggregates",
        "multiplicity": "Holm_within_each_family",
        "effect_sizes": ["rank_biserial", "Hodges_Lehmann"],
        "confidence_interval": {
            "method": "hierarchical_bootstrap",
            "replicates": 10000,
            "outer_unit": "map",
            "inner_units": ["task", "repeat"],
        },
        "families": {
            "synthetic_main": {
                "reference": "full",
                "members": [
                    "full",
                    "traditional_ppo",
                    "a2c_pointer",
                    *MAIN_SYNTHETIC_BASELINES.keys(),
                ],
            },
            "synthetic_ablation": {
                "reference": "full",
                "members": [
                    "full",
                    "no_priority_bias",
                    "no_domain_randomization",
                    "no_resource_shaping",
                    "no_return_reserve",
                ],
            },
            "real_main": {
                "reference": "full",
                "members": [
                    "full",
                    "traditional_ppo",
                    "a2c_pointer",
                    *REAL_BASELINES.keys(),
                ],
            },
            "real_ablation": {
                "reference": "full",
                "members": [
                    "full",
                    "no_priority_bias",
                    "no_domain_randomization",
                    "no_resource_shaping",
                    "no_return_reserve",
                ],
            },
            "known_domain_shift": {
                "reference": "full",
                "members": [
                    "full",
                    "traditional_ppo",
                    "a2c_pointer",
                    "no_domain_randomization",
                    "priority_resource_greedy",
                ],
            },
            "hidden_model_perception_mismatch": {
                "reference": "full",
                "members": [
                    "full",
                    "traditional_ppo",
                    "a2c_pointer",
                    "no_domain_randomization",
                    "no_return_reserve",
                    "priority_resource_greedy",
                ],
            },
        },
        "supplementary_and_interactions_are_descriptive": True,
        "scenario_rows_are_not_independent_samples": True,
    }


def build_protocol() -> Dict[str, Any]:
    parent = multimap.load_protocol(PARENT_PROTOCOL)
    if parent["protocol_hash"] != EXPECTED_PARENT_HASH:
        raise RuntimeError("v3.1.17 parent protocol identity mismatch")
    protocol = copy.deepcopy(parent)
    protocol.pop("protocol_hash", None)
    protocol["protocol_version"] = "multimap_generalization_v3_2"
    protocol["parent_protocol_hash"] = EXPECTED_PARENT_HASH
    protocol["asset_parent_protocol_hash"] = EXPECTED_PARENT_HASH
    protocol["amendment_reason"] = (
        "replace the archived shared-node PPO-MLP baseline with a true fixed-slot "
        "flat MLP PPO baseline; close real-terrain ablation, robustness semantics, "
        "and map-level statistics gaps before any formal test is generated"
    )
    protocol["pilot_training"] = {
        "variants": ["traditional_ppo"],
        "seed": 42,
        "episodes": 600,
        "monitor_episodes": [100, 200, 400, 600],
        "engineering_only": True,
        "paper_eligible": False,
        "score_based_tuning_forbidden": True,
        "required_node_counts": [16, 20, 24],
        "required_checks": [
            "finite_metrics",
            "valid_padding_and_masks",
            "checkpoint_roundtrip",
            "at_least_one_safe_return",
        ],
    }
    protocol["formal_training"] = {
        "variants": ACTIVE_VARIANTS,
        "seeds": TRAINING_SEEDS,
        "episodes": 3000,
        "monitor_episodes": [250, 500, 1000, 1500, 2000, 2500, 3000],
        "restart_all_after_material_change": True,
        "reused_parent_variants": [
            variant for variant in ACTIVE_VARIANTS if variant != "traditional_ppo"
        ],
        "new_training_variants": ["traditional_ppo"],
        "archived_excluded_variants": ["ppo_mlp"],
        "paper_eligible_model_count": 35,
        "paper_eligible_episode_count": 105000,
        "cumulative_executed_formal_episode_count_after_completion": 120000,
    }
    protocol["formal_evaluation"] = _evaluation_design()
    protocol["statistics"] = _statistics_design()
    protocol["claim_boundaries"] = {
        "node_counts": [16, 20, 24],
        "out_of_range_scale_generalization_claim_forbidden": True,
        "synthetic_claim": "unseen_procedural_map_generalization",
        "real_claim": "zero_shot_geographic_DSM_simulation_transfer",
        "real_flight_or_safety_certification_claim_forbidden": True,
        "mask_claim": "return_aware_multi_resource_feasibility_mask",
        "individual_mask_component_claim_forbidden": True,
        "no_priority_bias_removes_only_explicit_attention_bias": True,
        "no_domain_randomization_retains_multimap_training": True,
        "raw_reward_cross_variant_ranking_forbidden": True,
    }
    protocol["forbidden"] = sorted(
        set(protocol.get("forbidden", ()))
        | {
            "evaluating_archived_ppo_mlp",
            "out_of_range_scale_generalization_claim",
            "scenario_row_pseudoreplication",
            "hidden_and_observed_perturbation_semantics_mixing",
            "real_flight_claim",
            "individual_submask_ablation_claim",
        }
    )
    protocol["protocol_hash"] = multimap._canonical_hash(
        protocol, excluded=("protocol_hash",)
    )
    return protocol


def main() -> None:
    if OUTPUT_PROTOCOL.exists() or AMENDMENT.exists():
        raise FileExistsError("v3.2 protocol artifacts already exist")
    protocol = build_protocol()
    multimap._atomic_json(OUTPUT_PROTOCOL, protocol)
    AMENDMENT.write_text(
        "\n".join(
            [
                "# AMENDMENT v3.2",
                "",
                f"- parent_protocol_hash: `{EXPECTED_PARENT_HASH}`",
                f"- protocol_hash: `{protocol['protocol_hash']}`",
                "- v3.1.17 remains immutable and all 35 old checkpoints remain archived.",
                "- `ppo_mlp` is excluded from v3.2 evaluation and replaced by `traditional_ppo`.",
                "- Only the new traditional PPO is trained; six parent variants are reused.",
                "- Formal evaluation is frozen at 21,648 rows with all four ablations on all 144 real tasks.",
                "- Robustness separates known domain shifts from hidden model/perception mismatch.",
                "- Maps are the primary independent statistical units.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

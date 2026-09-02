#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the versioned manuscript multi-objective analysis protocol."""

from __future__ import annotations

import json

import manuscript_multiobjective_v1 as analysis


def build_protocol() -> dict:
    value = {
        "schema_version": "manuscript_multiobjective_v1",
        "frozen_at_utc": "2026-07-31T00:00:00Z",
        "analysis_role": "post_result_manuscript_draft_revision",
        "scientific_boundary": (
            "This analysis was specified after formal results existed. It is a "
            "versioned manuscript-stage multi-objective analysis, not the original "
            "preregistered confirmatory test, and all component metrics remain visible."
        ),
        "task_definition": (
            "Visit fixed inspection points on two mountain national-highway corridors; "
            "continuous road coverage is not required; the airport lies at the road "
            "intersection; planning obeys energy, distance, time, wind, terrain, "
            "dynamics and forced-return constraints."
        ),
        "parent_analysis_protocol_hash": (
            "0b3cdcce77429875978c68fd9500c43de5de123d5fb7691847d4a6a1e6d93d22"
        ),
        "matrix_sha256": analysis.EXPECTED_MATRIX_SHA256,
        "final_results_sha256": analysis.EXPECTED_RESULTS_SHA256,
        "implementation_sha256": analysis.sha256_file(analysis.Path(analysis.__file__)),
        "dimensions": analysis.INTERNAL_WEIGHTS,
        "default_overall_weights": analysis.DEFAULT_WEIGHTS,
        "weight_ranges": analysis.WEIGHT_RANGES,
        "weight_step": 0.05,
        "scenario_weights": analysis.SCENARIOS,
        "default_online_deadline_s": analysis.DEFAULT_DEADLINE_S,
        "online_deadline_sensitivity_s": analysis.DEADLINE_SENSITIVITY,
        "aggregation_methods": {
            "primary_manuscript_composite": "weighted_geometric",
            "sensitivity": "weighted_arithmetic",
        },
        "hierarchy": "repeat_mean_then_task_mean_then_map_mean",
        "scopes": {
            "synthetic_all": "D1,D2,D3,D5; no D4 imputation",
            "real_all": "D1,D2,D3,D5; no D4 imputation",
            "core_learning_complete": "full,traditional_ppo,a2c_pointer; D1-D5",
            "mechanism_robustness": "descriptive specialized comparisons only",
        },
        "raw_oracle_rule": "D1 uses weighted_coverage/oracle_upper, clipped to [0,1]",
        "unsafe_cost_rule": "resource cost metrics use safe routes only; no-safe task utility is zero",
        "robustness_rule": (
            "Six frozen perturbation conditions, paired to nominal task/model/seed; "
            "no missing-value imputation and no post-hoc condition selection."
        ),
        "plots_forbidden": True,
        "stopping_state": "ready_for_plot_plan",
    }
    value["protocol_hash"] = analysis.canonical_hash(value)
    return value


def main() -> None:
    protocol = build_protocol()
    analysis._atomic_json(analysis.PROTOCOL, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

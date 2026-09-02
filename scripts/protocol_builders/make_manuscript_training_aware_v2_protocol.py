#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the training-aware v2 manuscript protocol."""

from __future__ import annotations

import json
from pathlib import Path

from uav_inspection.analysis import manuscript_multiobjective_v1 as v1
from uav_inspection.analysis import manuscript_training_aware_v2 as v2


def build_protocol() -> dict:
    parent = json.loads(v2.V1_MANIFEST.read_text(encoding="utf-8"))
    protocol = {
        "schema_version": "manuscript_training_aware_v2",
        "analysis_role": "post_result_training_aware_manuscript_extension",
        "scientific_boundary": (
            "D6 and D7 were added after formal outcomes existed. Their definitions and "
            "weights are frozen without a required winning model or score margin."
        ),
        "parent_v1_manifest_hash": parent["manifest_hash"],
        "parent_v1_protocol_hash": parent["protocol_hash"],
        "implementation_sha256": v1.sha256_file(Path(v2.__file__)),
        "training_source_hashes": v2.training_source_hashes(),
        "D6_training_stability": {
            "tail_start_episode_exclusive": v2.TAIL_START_EPISODE,
            "seed_consistency": "1 - sample SD of five seed tail means",
            "temporal_consistency": "1 - mean within-seed tail sample SD",
            "weights": v2.TRAINING_DIMENSION_WEIGHTS["D6"],
        },
        "D7_sample_efficiency": {
            "auc": "trapezoidal AUC of training mean_weighted_coverage over 0-3000 episodes",
            "threshold": v2.CONVERGENCE_THRESHOLD,
            "rolling_window_updates": v2.CONVERGENCE_WINDOW_UPDATES,
            "interaction_budget": v2.INTERACTION_BUDGET,
            "unreached_threshold_utility": 0.0,
            "weights": v2.TRAINING_DIMENSION_WEIGHTS["D7"],
        },
        "default_weights": v2.DEFAULT_WEIGHTS,
        "weight_ranges": v2.WEIGHT_RANGES,
        "weight_step": 0.05,
        "scenarios": v2.SCENARIOS,
        "eligible_scope": list(v2.CORE_MODELS),
        "traditional_baseline_rule": "D6/D7 unavailable; no imputation and no seven-dimension score",
        "desired_winner_or_margin_constraint": None,
        "plots_forbidden": True,
        "stopping_state": "ready_for_plot_plan",
    }
    protocol["protocol_hash"] = v1.canonical_hash(protocol)
    return protocol


if __name__ == "__main__":
    value = build_protocol()
    v1._atomic_json(v2.PROTOCOL, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))

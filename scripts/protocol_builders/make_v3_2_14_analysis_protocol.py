#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze v3.2.14 analysis rules before the complete result set exists."""

from __future__ import annotations

import json
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT

from uav_inspection.experiments import paper_v3_2_experiments as v32
from uav_inspection.evaluation import v3_2_14_evaluation_smoke as smoke
from uav_inspection.analysis import v3_2_14_statistics as analysis


ROOT = WORKSPACE_ROOT
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14"
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/protocol.json"
)
MATRIX = OUTPUT / "formal_evaluation/evaluation_matrix.jsonl"
FINAL_RESULTS = (
    OUTPUT / "formal_evaluation/results/final_results.jsonl"
)
FINAL_AUDIT = OUTPUT / "formal_evaluation/results/final_audit.json"
DESTINATION = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "analysis_protocol.json"
)


def main() -> int:
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    if FINAL_RESULTS.exists() or FINAL_AUDIT.exists():
        raise RuntimeError(
            "analysis protocol must be frozen before final merged results"
        )
    if analysis.DESTINATION.exists():
        raise RuntimeError(
            "analysis protocol must be frozen before analysis output"
        )
    families = {
        name: {
            **config,
            "sources": list(config["sources"]),
            "algorithms": list(config["algorithms"]),
            "conditions": list(config["conditions"]),
        }
        for name, config in analysis.FAMILIES.items()
    }
    payload = {
        "schema_version": 1,
        "parent_protocol_hash": protocol["protocol_hash"],
        "matrix_sha256": v32._sha256_file(MATRIX),
        "matrix_row_count": 21648,
        "implementation_path": str(
            Path(analysis.__file__).resolve()
        ),
        "implementation_sha256": v32._sha256_file(
            Path(analysis.__file__)
        ),
        "generator_sha256": v32._sha256_file(Path(__file__)),
        "frozen_before_final_results_complete": True,
        "final_results_existing_at_freeze": False,
        "final_audit_existing_at_freeze": False,
        "algorithm_scores_used_for_design": False,
        "source_of_rules": (
            "v3.2 preregistered plan supplied before formal evaluation"
        ),
        "confirmatory_metric": "safe_weighted_coverage",
        "reference_algorithm": "full",
        "alpha": 0.05,
        "omnibus_test": "Friedman on paired map-level aggregates",
        "pairwise_test": (
            "two-sided paired Wilcoxon signed-rank on map-level aggregates"
        ),
        "zero_method": "wilcox",
        "multiple_testing": (
            "Holm correction separately within each of six families"
        ),
        "effect_sizes": (
            "rank-biserial correlation and Hodges-Lehmann paired shift"
        ),
        "bootstrap": {
            "samples": 10000,
            "seed": 20260731,
            "confidence_level": 0.95,
            "outer_unit": "map",
            "middle_unit": "task nested within map",
            "inner_unit": (
                "training or planning repeats resampled independently "
                "within algorithm and task-condition cell"
            ),
            "fixed_robustness_conditions_resampled": False,
        },
        "aggregation_order": (
            "repeat mean within task-condition, fixed-condition mean "
            "within task, task mean within map"
        ),
        "statistical_families": families,
        "auxiliary_metrics_role": "descriptive_only",
        "iqm_role": "exploratory_only",
        "interaction_role": "exploratory_only",
        "resource_utilization_rule": "summarize only safe routes",
        "robustness_drop_rule": (
            "same model, task, and repeat seed: nominal safe weighted "
            "coverage minus perturbed safe weighted coverage"
        ),
        "priority_coverage_rule": (
            "derive low/medium/high visited fractions from frozen route "
            "visit order and frozen task priorities"
        ),
        "dangerous_proposal_rule": (
            "derive from observation-plan constraint_violation_count; "
            "do not infer from executed truth route"
        ),
        "plotting_forbidden": True,
        "stop_state": "ready_for_plotting",
    }
    payload["analysis_protocol_hash"] = smoke._canonical_hash(payload)
    text = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if DESTINATION.exists() and DESTINATION.read_text(
        encoding="utf-8"
    ) != text:
        raise RuntimeError("analysis protocol already exists and differs")
    smoke._atomic_text(DESTINATION, text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

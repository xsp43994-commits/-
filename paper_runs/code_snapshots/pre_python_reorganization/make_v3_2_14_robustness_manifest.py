#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the numerical robustness implementation before any robust result."""

from __future__ import annotations

import json
from pathlib import Path

import paper_v3_2_experiments as v32
import v3_2_14_evaluation_smoke as smoke


ROOT = Path(__file__).resolve().parent
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/protocol.json"
)
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14"
MATRIX_MANIFEST = (
    OUTPUT / "formal_evaluation/evaluation_matrix_manifest.json"
)
REAL = OUTPUT / "formal_evaluation/real_tasks_parallel/records.jsonl"
IMPLEMENTATION = ROOT / "v3_2_14_robustness_worker.py"
DESTINATION = (
    OUTPUT / "formal_evaluation/robustness_implementation_manifest.json"
)


def main() -> int:
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    matrix = json.loads(MATRIX_MANIFEST.read_text(encoding="utf-8"))
    real = v32._read_jsonl(REAL)
    selected = v32.select_robustness_tasks(real)
    robustness_roots = [
        OUTPUT
        / "formal_evaluation"
        / "results"
        / family
        for family in (
            "known_domain_shift",
            "hidden_model_perception_mismatch",
        )
    ]
    existing_results = sum(
        len(list(path.rglob("results.jsonl"))) if path.exists() else 0
        for path in robustness_roots
    )
    if existing_results:
        raise RuntimeError(
            "robustness manifest must be frozen before robust results"
        )
    payload = {
        "schema_version": 3,
        "parent_protocol_hash": protocol["protocol_hash"],
        "matrix_sha256": matrix["matrix_sha256"],
        "real_records_sha256": v32._sha256_file(REAL),
        "implementation_path": str(IMPLEMENTATION.resolve()),
        "implementation_sha256": v32._sha256_file(IMPLEMENTATION),
        "generator_sha256": v32._sha256_file(Path(__file__)),
        "algorithm_results_used_for_design": False,
        "robustness_results_existing_at_freeze": existing_results,
        "created_after_nominal_evaluation_started": True,
        "supersedes_archived_manifest_hash": (
            "e999d2bc5aeb57952c66e65788774091de4c9dfcd41949556c1ac98485b46a4f"
        ),
        "revision_reason": (
            "the pre-result hidden-DEM production probe showed that positive "
            "DEM error can also place the observed launch point below "
            "observed terrain; the v2 manifest and its 45 known-shift rows "
            "were archived, hidden rows were zero, and v3 applies the same "
            "1 cm geometric-consistency floor to the observed launch "
            "altitude for every hidden condition while leaving the frozen "
            "execution-truth launch point unchanged"
        ),
        "reason_for_separate_manifest": (
            "the frozen parent protocol registered robustness families, "
            "conditions, row counts, and visibility semantics but did not "
            "serialize every numerical perturbation implementation detail"
        ),
        "same_magnitude_known_vs_hidden_rule": (
            "wind and power use identical magnitudes; only observability "
            "and route-locking semantics differ"
        ),
        "common_random_seed_derivation": (
            "little_endian_uint64(first_8_bytes(sha256("
            "protocol_hash|task_hash|condition|v3_2_14)))"
        ),
        "common_random_realizations_across_algorithms": True,
        "factorial_combinations_forbidden": True,
        "robustness_task_ids": [str(row["id"]) for row in selected],
        "robustness_task_hashes": [
            str(row["task_hash"]) for row in selected
        ],
        "perturbations": {
            "wind": {
                "speed_scale": 1.2,
                "rotation_deg": 15.0,
                "vertical_bias_mps": 0.5,
            },
            "power_model": {"coefficient_scale": 1.1},
            "dem_error": {
                "sigma_m": 3.0,
                "correlation_length_m": 100.0,
                "generator": (
                    "seeded_standard_normal_then_scipy_gaussian_filter_"
                    "and_exact_sample_std_normalization"
                ),
                "gaussian_filter_sigma_rule": (
                    "correlation_length_m/(sqrt(2)*coordinate_scale_m_per_unit)"
                ),
                "boundary_mode": "reflect",
                "observed_start_minimum_altitude_rule": (
                    "max(observed_start_z, observed_ground_z+0.01m)"
                ),
                "truth_uses_frozen_dsm": True,
            },
            "localization": {
                "horizontal_sigma_m": 10.0,
                "vertical_sigma_m": 3.0,
                "distribution": "independent_zero_mean_gaussian",
                "horizontal_coordinates_clipped_to_map": True,
                "observed_points_use_flight_altitude_mode": True,
                "observed_start_minimum_altitude_rule": (
                    "max(raw_noisy_start_z, observed_ground_z+0.01m)"
                ),
                "truth_uses_frozen_coordinates": True,
            },
        },
        "known_domain_shift": {
            "planner_and_execution_share_shifted_truth": True,
            "observed_input_hash_equals_execution_truth_hash": True,
        },
        "hidden_model_perception_mismatch": {
            "route_locked_before_truth_evaluation": True,
            "observed_input_hash_differs_from_execution_truth_hash": True,
            "single_factor_only": True,
        },
    }
    payload["manifest_hash"] = smoke._canonical_hash(payload)
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
        raise RuntimeError("robustness manifest already exists and differs")
    smoke._atomic_text(DESTINATION, text)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

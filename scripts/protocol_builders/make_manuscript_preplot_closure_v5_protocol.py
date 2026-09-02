#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the final pre-plot statistical closure protocol."""

from __future__ import annotations

import json
from pathlib import Path

from uav_inspection.analysis import manuscript_multiobjective_v1 as v1
from uav_inspection.analysis import manuscript_preplot_closure_v5 as v5
from uav_inspection.analysis import manuscript_training_aware_v2 as v2
from uav_inspection.analysis import manuscript_operational_band_v4 as v4


def build_protocol() -> dict:
    value = {
        "schema_version": "manuscript_preplot_closure_v5",
        "analysis_role": "final_preplot_statistical_closure",
        "source_manifest_hashes": v5._source_manifest_hashes(),
        "implementation_sha256": v1.sha256_file(Path(v5.__file__)),
        "joint_sensitivity": {
            "floors": v4.FLOOR_SENSITIVITY,
            "weight_vectors": len(v2.enumerate_weight_grid()),
            "models": 3,
            "aggregations": ["arithmetic", "geometric"],
            "expected_rows": 37410,
        },
        "paired_dimension_family": ["D4", "D6", "D7"],
        "paired_tests": "two-sided Wilcoxon with Holm correction across D4,D6,D7",
        "effect_sizes": ["rank_biserial", "Hodges_Lehmann", "direction_consistency"],
        "bootstrap": {
            "replicates": v5.BOOTSTRAP_REPLICATES,
            "seed": v5.BOOTSTRAP_SEED,
            "ci_level": v5.CI_LEVEL,
            "paired_units": {
                "synthetic_maps": 24,
                "real_maps": 8,
                "robustness_maps": 8,
                "training_seeds": 5,
            },
            "domain_weighting": "synthetic and real nominal domains each receive 0.5",
        },
        "selected_operational_floor": v4.SELECTED_OPERATIONAL_FLOOR,
        "primary_aggregation": "arithmetic",
        "formal_result_row_count": 21648,
        "formal_result_sha256": v1.EXPECTED_RESULTS_SHA256,
        "plots_forbidden": True,
        "stopping_state": "ready_for_formal_plot_plan",
    }
    value["protocol_hash"] = v1.canonical_hash(value)
    return value


if __name__ == "__main__":
    protocol = build_protocol()
    v1._atomic_json(v5.PROTOCOL, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))

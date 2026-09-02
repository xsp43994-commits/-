#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the internal operational-band v4 protocol."""

from __future__ import annotations

import json
from pathlib import Path

from uav_inspection.analysis import manuscript_multiobjective_v1 as v1
from uav_inspection.analysis import manuscript_operational_band_v4 as v4
from uav_inspection.analysis import manuscript_training_priority_v3 as v3


def build_protocol() -> dict:
    parent = json.loads(v4.PARENT_MANIFEST.read_text(encoding="utf-8"))
    value = {
        "schema_version": "manuscript_operational_band_v4",
        "analysis_role": "internal_post_result_operational_band_sensitivity",
        "scientific_boundary": (
            "The 0.60 floor is a disclosed internal high-performance operational-band "
            "scenario. It does not replace the untransformed v2 score. All five floors "
            "must be retained together when this scenario is shown."
        ),
        "parent_v3_manifest_hash": parent["manifest_hash"],
        "implementation_sha256": v1.sha256_file(Path(v4.__file__)),
        "score_scale": v4.SCORE_SCALE,
        "selected_operational_floor": v4.SELECTED_OPERATIONAL_FLOOR,
        "floor_sensitivity": v4.FLOOR_SENSITIVITY,
        "rescaled_dimensions": v4.RESCALED_DIMENSIONS,
        "rescale_formula": "clip((value-floor)/(1-floor),0,1)",
        "unchanged_dimensions": ["D1", "D2", "D3", "D5"],
        "priority_weights": v3.PRIORITY_WEIGHTS,
        "primary_aggregation": "arithmetic",
        "geometric_aggregation_role": (
            "diagnostic_only; clipping below-floor dimensions to zero makes the "
            "traditional PPO geometric composite degenerate at the 0.60 floor"
        ),
        "parent_results_modified": False,
        "plots_forbidden": True,
        "stopping_state": "ready_for_plot_plan",
    }
    value["protocol_hash"] = v1.canonical_hash(value)
    return value


if __name__ == "__main__":
    protocol = build_protocol()
    v1._atomic_json(v4.PROTOCOL, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))

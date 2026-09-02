#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the v3 training-and-robustness-priority draft protocol."""

from __future__ import annotations

import json
from pathlib import Path

from uav_inspection.analysis import manuscript_multiobjective_v1 as v1
from uav_inspection.analysis import manuscript_training_aware_v2 as v2
from uav_inspection.analysis import manuscript_training_priority_v3 as v3


def build_protocol() -> dict:
    parent = json.loads(v3.PARENT_MANIFEST.read_text(encoding="utf-8"))
    value = {
        "schema_version": "manuscript_training_priority_v3",
        "analysis_role": "post_result_training_robustness_priority_draft_scenario",
        "scientific_boundary": (
            "This is a disclosed secondary priority scenario, not a replacement for "
            "the balanced v2 score or the original confirmatory analysis."
        ),
        "parent_v2_manifest_hash": parent["manifest_hash"],
        "implementation_sha256": v1.sha256_file(Path(v3.__file__)),
        "score_scale": v3.SCORE_SCALE,
        "priority_weights": v3.PRIORITY_WEIGHTS,
        "parent_weight_ranges": v2.WEIGHT_RANGES,
        "sensitivity_reuse_rule": (
            "The selected vector already exists in the frozen 1247-vector v2 grid; "
            "scores are rescaled by 100 and ranks are unchanged."
        ),
        "parent_results_modified": False,
        "plots_forbidden": True,
        "stopping_state": "ready_for_plot_plan",
    }
    value["protocol_hash"] = v1.canonical_hash(value)
    return value


if __name__ == "__main__":
    protocol = build_protocol()
    v1._atomic_json(v3.PROTOCOL, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))

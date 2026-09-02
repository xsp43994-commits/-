#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct and screen a mixed task from an explicit safe witness."""

from __future__ import annotations

import json
import math
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT

from uav_inspection.experiments import paper_difficulty_experiments as difficulty
from uav_inspection.experiments import paper_multimap_experiments as multimap
from uav_inspection.experiments import paper_v3_2_experiments as v32
from uav_inspection.generation import v3_2_12_parametric_certificate_search as search
from uav_inspection.generation import v3_2_13_certificate_witness as witness
from python_classical_algs.common import MissionEvaluator


ROOT = WORKSPACE_ROOT
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_12/protocol.json"
)
MAP_ROOT = ROOT / "map_data/multimap_v3_1"
OUTPUT = (
    ROOT
    / "paper_runs/multimap_v3_2_12/diagnostics/witness_search/"
    "colorado_task06_attempt17_calibrated.json"
)
TARGET_UTILIZATION = 0.92


def _candidate(
    protocol: dict, parent: dict, registry: dict, contexts: dict
) -> dict:
    map_index, road_index, task_index, attempt = 5, 0, 6, 17
    map_record = dict(registry["maps"][map_index])
    full_bundle = multimap._load_map_bundle(MAP_ROOT, map_record)
    context_record = contexts[str(map_record["map_id"])]
    corridor = v32._corridor_bundle(full_bundle, context_record)
    depot = list(context_record["contexts"][road_index]["start_xy"])
    design = multimap._task_design(map_index, task_index)
    minimum, interval = multimap._effective_task_radius_range(
        map_record,
        corridor,
        protocol,
        node_count=int(design["node_count"]),
        difficulty=str(design["difficulty"]),
        depot_override_xy=depot,
    )
    result = multimap._task_candidate(
        map_record,
        corridor,
        protocol,
        parent,
        split="real_test",
        map_index=map_index,
        task_index=task_index,
        attempt=attempt,
        master_seed=int(protocol["map_splits"]["synthetic_test"]["seed"]),
        geometry_radius_range_m=interval,
        geometry_minimum_feasible_radius_m=minimum,
        seed_namespace=f"road_{road_index:02d}",
        road_index=road_index,
        depot_override_xy=depot,
    )
    result["road_context_hash"] = str(context_record["context_hash"])
    result["road_context_definition"] = str(
        protocol["real_corridor_contexts"]["definition"]
    )
    return result


def main() -> int:
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    parent = json.loads(
        multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
    )
    registry_path = MAP_ROOT / "real" / "map_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    provider = multimap.FrozenMapProvider.from_registries(
        MAP_ROOT, [registry_path]
    )
    contexts = v32._load_real_corridor_contexts(
        ROOT / "paper_runs/multimap_v3_2_12", protocol
    )
    raw = _candidate(protocol, parent, registry, contexts)
    upper = search._with_values(
        raw,
        {
            parameter: search._bounds(protocol, parameter)[1]
            for parameter in search.RESOURCE_PARAMETERS
        },
        protocol,
        {"stage": "constructive_witness_registered_upper_bounds"},
    )
    upper_problem = witness.build_frozen_problem(upper, provider)
    upper_evaluator = MissionEvaluator(upper_problem)
    total_priority = float(sum(upper["priorities"]))
    band_low = float(
        parent["difficulty_bands"][str(upper["difficulty"])][0]
    )
    low_required = int(math.ceil(band_low * total_priority - 1e-9))
    found = witness.construct_threshold_witness(
        upper_problem, minimum_priority_weight=float(low_required)
    )
    if found.evaluation is None:
        raise RuntimeError("registered upper bounds have no witness route")
    route = found.evaluation
    upper_metrics = {
        "energy": route.energy_wh / upper_evaluator.energy_budget_wh,
        "distance": route.distance_m / upper_evaluator.distance_budget_m,
        "time": route.time_s / upper_evaluator.time_budget_s,
    }
    # 距离与时间预算对各自 scale 线性；把同一安全路线放在 92% 利用率，
    # 形成两个独立活跃瓶颈，能量保持注册上界。
    calibrated = search._with_values(
        upper,
        {
            "distance_budget_scale": float(
                upper["distance_budget_scale"]
                * upper_metrics["distance"]
                / TARGET_UTILIZATION
            ),
            "time_budget_scale": float(
                upper["time_budget_scale"]
                * upper_metrics["time"]
                / TARGET_UTILIZATION
            ),
        },
        protocol,
        {
            "stage": "constructive_witness_two_resource_calibration",
            "resources": ["distance", "time"],
            "target_utilization": TARGET_UTILIZATION,
        },
    )
    calibrated_problem = witness.build_frozen_problem(calibrated, provider)
    calibrated_evaluator = MissionEvaluator(calibrated_problem)
    replay = calibrated_evaluator.evaluate_order(route.order)
    metrics = {
        "energy_utilization": replay.energy_wh
        / calibrated_evaluator.energy_budget_wh,
        "distance_utilization": replay.distance_m
        / calibrated_evaluator.distance_budget_m,
        "time_utilization": replay.time_s
        / calibrated_evaluator.time_budget_s,
    }
    bottlenecks = difficulty._resource_bottlenecks(
        metrics,
        minimum=float(
            parent["certification"]["bottleneck_utilization_min"]
        ),
        max_gap=float(
            parent["certification"]["single_bottleneck_max_gap"]
        ),
    )
    _accepted, certificate, reason = multimap._certify_multimap_task(
        calibrated,
        provider,
        parent,
        time_limit_s=float(
            protocol["certification"]["candidate_screening_time_limit_s"]
        ),
    )
    payload = {
        "schema_version": 1,
        "task_id": calibrated["id"],
        "generation_attempt": calibrated["generation_attempt"],
        "algorithm_results_used": False,
        "target_utilization": TARGET_UTILIZATION,
        "calibrated_parameters": {
            name: calibrated[name]
            for name in search.RESOURCE_PARAMETERS
        },
        "witness": found.as_dict(),
        "replay": {
            "returned": replay.returned,
            "visit_order": list(replay.order),
            "weighted_coverage": replay.weighted_coverage,
            **metrics,
            "bottleneck_resources": list(bottlenecks),
            "scenario_hash": calibrated_problem.scenario_hash,
        },
        "screen": {
            "reason": reason,
            "weighted_coverage_lower_bound": certificate.get(
                "weighted_coverage_lower_bound"
            ),
            "weighted_coverage_upper_bound": certificate.get(
                "weighted_coverage_upper_bound"
            ),
            "solver_status": certificate.get("solver_status"),
            "solver_message": certificate.get("solver_message"),
            "scenario_hash": certificate.get("scenario_hash"),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    upper_bound = certificate.get("weighted_coverage_upper_bound")
    band_high = float(
        parent["difficulty_bands"][str(calibrated["difficulty"])][1]
    )
    return (
        0
        if replay.returned
        and len(bottlenecks) >= 2
        and upper_bound is not None
        and float(upper_bound) <= band_high + 1e-9
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())

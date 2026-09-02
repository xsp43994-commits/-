#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct minimum-resource proofs for one blocked real single-resource task."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import paper_multimap_experiments as multimap
import paper_v3_2_experiments as v32
import v3_2_12_parametric_certificate_search as search
import v3_2_13_certificate_witness as witness
from python_classical_algs.milp import solve_resource_threshold_milp


ROOT = Path(__file__).resolve().parent
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/protocol.json"
)
OUTPUT_ROOT = ROOT / "paper_runs/multimap_v3_2_14"
MAP_ROOT = ROOT / "map_data/multimap_v3_1"


def build_candidate(
    protocol: dict,
    parent: dict,
    registry: dict,
    contexts: dict,
    *,
    map_index: int,
    road_index: int,
    task_index: int,
    attempt: int,
) -> dict:
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
    candidate = multimap._task_candidate(
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
    candidate["road_context_hash"] = str(context_record["context_hash"])
    candidate["road_context_definition"] = str(
        protocol["real_corridor_contexts"]["definition"]
    )
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-index", type=int, required=True)
    parser.add_argument("--road-index", type=int, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--threshold", choices=("low", "high"), required=True)
    parser.add_argument("--time-limit-s", type=float, required=True)
    args = parser.parse_args()
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    parent = json.loads(
        multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
    )
    registry_path = MAP_ROOT / "real" / "map_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    provider = multimap.FrozenMapProvider.from_registries(
        MAP_ROOT, [registry_path]
    )
    contexts = v32._load_real_corridor_contexts(OUTPUT_ROOT, protocol)
    raw = build_candidate(
        protocol,
        parent,
        registry,
        contexts,
        map_index=int(args.map_index),
        road_index=int(args.road_index),
        task_index=int(args.task_index),
        attempt=int(args.attempt),
    )
    resource = str(raw["constraint_type"])
    parameter = search.NAME_TO_PARAMETER.get(resource)
    if parameter is None:
        raise RuntimeError("direct threshold probe requires a single resource")
    candidate = search._with_values(
        raw,
        {
            name: search._bounds(protocol, name)[1]
            for name in search.RESOURCE_PARAMETERS
        },
        protocol,
        {
            "stage": "direct_threshold_registered_upper_bounds",
            "intended_resource": resource,
        },
    )
    problem = witness.build_frozen_problem(candidate, provider)
    total_priority = float(sum(candidate["priorities"]))
    band_low, band_high = (
        float(value)
        for value in parent["difficulty_bands"][
            str(candidate["difficulty"])
        ]
    )
    required = (
        int(math.ceil(band_low * total_priority - 1e-9))
        if args.threshold == "low"
        # 高阈值不可达应推出严格低于难度带上界；整数边界不能再额外加 1。
        else int(math.ceil(band_high * total_priority - 1e-9))
    )
    proof = solve_resource_threshold_milp(
        problem,
        resource_name=resource,
        minimum_priority_weight=float(required),
        time_limit_s=float(args.time_limit_s),
    )
    payload = {
        "schema_version": 1,
        "task_id": candidate["id"],
        "algorithm_results_used": False,
        "map_index": int(args.map_index),
        "road_index": int(args.road_index),
        "task_index": int(args.task_index),
        "generation_attempt": int(args.attempt),
        "threshold": str(args.threshold),
        "requested_time_limit_s": float(args.time_limit_s),
        "intended_resource": resource,
        "intended_parameter": parameter,
        "required_priority": required,
        "total_priority": total_priority,
        "scenario_hash": problem.scenario_hash,
        "upper_budget_candidate": {
            name: candidate[name]
            for name in search.RESOURCE_PARAMETERS
        },
        "proof": proof,
    }
    directory = OUTPUT_ROOT / "diagnostics" / "direct_threshold"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / (
        f"map{args.map_index:02d}_road{args.road_index:02d}_"
        f"task{args.task_index:02d}_attempt{args.attempt:02d}_"
        f"{args.threshold}_limit{int(round(args.time_limit_s))}s.json"
    )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    evaluation = proof.get("actual_budget_evaluation") or {}
    return (
        0
        if (
            args.threshold == "low" and evaluation.get("returned")
        )
        or (
            args.threshold == "high"
            and proof.get("resource_dual_bound") is not None
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())

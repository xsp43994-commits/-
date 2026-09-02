#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict proof composition for the two v3.2.12 certification blockers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

import paper_difficulty_experiments as difficulty
import paper_multimap_experiments as multimap
import paper_v3_2_experiments as v32
import v3_2_12_parametric_certificate_search as search
import v3_2_13_certificate_witness as witness
import v3_2_13_calibrated_mixed_diagnostic as mixed_source
import v3_2_14_direct_threshold_probe as direct_probe
from python_classical_algs.common import MissionEvaluator


ROOT = Path(__file__).resolve().parent
MAP_ROOT = ROOT / "map_data/multimap_v3_1"
SOURCE_SINGLE_PROOF = (
    ROOT
    / "paper_runs/multimap_v3_2_12/diagnostics/threshold_probe/"
    "synthetic_task06_attempt01_fraction0.500000.json"
)
TARGET_SINGLE_PROOF = (
    ROOT
    / "paper_runs/multimap_v3_2_12/diagnostics/threshold_probe/"
    "synthetic_task06_attempt01_fraction0.492034.json"
)
MIXED_LOW_PROOF = (
    ROOT
    / "paper_runs/multimap_v3_2_12/diagnostics/witness_search/"
    "colorado_task06_attempt17_low_distance_upper.json"
)
MIXED_CALIBRATION = (
    ROOT
    / "paper_runs/multimap_v3_2_12/diagnostics/witness_search/"
    "colorado_task06_attempt17_calibrated_from_distance.json"
)
MIXED_INITIAL_CUT_PROOF = (
    ROOT
    / "paper_runs/multimap_v3_2_12/diagnostics/witness_search/"
    "colorado_task06_attempt17_subtour_from_distance.json"
)
MIXED_FINAL_CUT_PROOF = (
    ROOT
    / "paper_runs/multimap_v3_2_12/diagnostics/witness_search/"
    "colorado_task06_attempt17_subtour_from_distance_resume.json"
)


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _synthetic_candidate(
    protocol: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    attempt: int,
    fraction: float,
) -> Tuple[Dict[str, Any], Any]:
    registry_path = (
        MAP_ROOT / "procedural" / "synthetic_test" / "map_registry.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    provider = multimap.FrozenMapProvider.from_registries(
        MAP_ROOT, [registry_path]
    )
    map_index, task_index = 23, 6
    map_record = dict(registry["maps"][map_index])
    bundle = multimap._load_map_bundle(MAP_ROOT, map_record)
    design = multimap._task_design(map_index, task_index)
    minimum, interval = multimap._effective_task_radius_range(
        map_record,
        bundle,
        protocol,
        node_count=int(design["node_count"]),
        difficulty=str(design["difficulty"]),
    )
    candidate = multimap._task_candidate(
        map_record,
        bundle,
        protocol,
        parent,
        split="synthetic_test",
        map_index=map_index,
        task_index=task_index,
        attempt=int(attempt),
        master_seed=int(protocol["map_splits"]["synthetic_test"]["seed"]),
        geometry_radius_range_m=interval,
        geometry_minimum_feasible_radius_m=minimum,
    )
    parameter = "distance_budget_scale"
    lower, upper = search._bounds(protocol, parameter)
    candidate = search._with_values(
        candidate,
        {
            "initial_soc": search._bounds(protocol, "initial_soc")[1],
            "time_budget_scale": search._bounds(
                protocol, "time_budget_scale"
            )[1],
            parameter: lower + float(fraction) * (upper - lower),
        },
        protocol,
        {
            "stage": "v3_2_13_transported_resource_threshold",
            "registered_range_fraction": float(fraction),
        },
    )
    return candidate, provider


def _problem_invariant_payload(
    record: Mapping[str, Any],
    evaluator: MissionEvaluator,
) -> Dict[str, Any]:
    """Hash every threshold-MILP input except the target actual budget."""

    segments = []
    for key, segment in sorted(evaluator._segments.items()):
        segments.append(
            {
                "arc": list(key),
                "feasible": bool(segment.feasible),
                "energy_wh": float(segment.energy_wh),
                "distance_m": float(segment.distance_m),
                "time_s": float(segment.time_s),
            }
        )
    return {
        "task_id": str(record["id"]),
        "map_id": str(record["map_id"]),
        "map_hash": str(record["map_hash"]),
        "inspection_points_xyz": record["inspection_points_xyz"],
        "priorities": record["priorities"],
        "service_times_s": record["service_times_s"],
        "constraint_type": record["constraint_type"],
        "initial_soc": record["initial_soc"],
        "time_budget_scale": record["time_budget_scale"],
        "energy_budget_wh": evaluator.energy_budget_wh,
        "time_budget_s": evaluator.time_budget_s,
        "segments": segments,
    }


def compose_transported_single_certificate(
    protocol: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compose task 06 from a replayed low route and a transported dual bound."""

    source_result = json.loads(
        SOURCE_SINGLE_PROOF.read_text(encoding="utf-8")
    )
    target_result = json.loads(
        TARGET_SINGLE_PROOF.read_text(encoding="utf-8")
    )
    source_candidate, provider = _synthetic_candidate(
        protocol, parent, attempt=1, fraction=0.5
    )
    target_fraction = (1.16375 - 0.16) / (2.2 - 0.16)
    target_candidate, _ = _synthetic_candidate(
        protocol, parent, attempt=1, fraction=target_fraction
    )
    source_problem = witness.build_frozen_problem(
        source_candidate, provider
    )
    target_problem = witness.build_frozen_problem(
        target_candidate, provider
    )
    source_evaluator = MissionEvaluator(source_problem)
    target_evaluator = MissionEvaluator(target_problem)
    source_invariant = _problem_invariant_payload(
        source_candidate, source_evaluator
    )
    target_invariant = _problem_invariant_payload(
        target_candidate, target_evaluator
    )
    source_invariant_hash = _canonical(source_invariant)
    target_invariant_hash = _canonical(target_invariant)
    if source_invariant_hash != target_invariant_hash:
        raise RuntimeError("threshold MILP invariant inputs changed")

    source_high = dict(source_result["certificate"]["high_threshold"])
    target_low = dict(target_result["certificate"]["low_threshold"])
    low_order = list(target_low.get("visit_order") or ())
    low_evaluation = target_evaluator.evaluate_order(low_order)
    if not low_evaluation.returned:
        raise RuntimeError("transported certificate low route is not safe")
    target_budget = float(target_evaluator.distance_budget_m)
    source_dual = float(source_high["resource_dual_bound"])
    if not source_dual > target_budget + 1e-7:
        raise RuntimeError("source dual bound does not exclude target budget")
    if not math.isclose(
        float(source_high["relaxed_resource_budget"]),
        float(target_result["certificate"]["high_threshold"][
            "relaxed_resource_budget"
        ]),
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("relaxed threshold-MILP budget changed")

    priorities = np.asarray(target_candidate["priorities"], dtype=np.float64)
    total_priority = float(np.sum(priorities))
    band_low, band_high = (
        float(value)
        for value in parent["difficulty_bands"][
            str(target_candidate["difficulty"])
        ]
    )
    low_required = int(math.ceil(band_low * total_priority - 1e-9))
    # 证明该离散权重不可达后，可得最大权重严格小于难度带上界。
    high_required = int(math.ceil(band_high * total_priority - 1e-9))
    lower = float(low_evaluation.weighted_coverage)
    upper = float(high_required - 1) / total_priority
    metrics = {
        "energy_utilization": low_evaluation.energy_wh
        / target_evaluator.energy_budget_wh,
        "distance_utilization": low_evaluation.distance_m
        / target_evaluator.distance_budget_m,
        "time_utilization": low_evaluation.time_s
        / target_evaluator.time_budget_s,
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
    if "distance" not in bottlenecks:
        raise RuntimeError("transported task has no distance bottleneck")
    transported_high = copy.deepcopy(source_high)
    transported_high["source_actual_resource_budget"] = float(
        source_high["actual_resource_budget"]
    )
    transported_high["actual_resource_budget"] = target_budget
    transported_high["threshold_impossible_under_actual_budget"] = True
    certificate = {
        "algorithm": "milp_weighted_coverage",
        "solver_status": source_high["solver_status"],
        "solver_success": True,
        "solver_message": (
            "replayed low-threshold route plus invariant MILP dual-bound "
            "transport to a tighter actual distance budget"
        ),
        "status": str(low_evaluation.termination_reason),
        "mip_gap": max(0.0, upper - lower) / max(lower, 1e-12),
        "weighted_coverage_lower_bound": lower,
        "weighted_coverage_upper_bound": upper,
        "optimality_certified": math.isclose(
            lower, upper, rel_tol=0.0, abs_tol=1e-12
        ),
        "visit_order": list(low_evaluation.order),
        "visited_count": len(low_evaluation.order),
        "returned": True,
        **metrics,
        "runtime_s": float(target_low["runtime_s"])
        + float(source_high["runtime_s"]),
        "scenario_hash": str(target_problem.scenario_hash),
        "map_id": str(target_candidate["map_id"]),
        "map_hash": str(target_candidate["map_hash"]),
        "difficulty_certificate": (
            "replayed_lower_plus_invariant_transported_threshold_dual"
        ),
        "bottleneck_resources": list(bottlenecks),
        "screening": {
            "time_limit_s": float(
                protocol["certification"][
                    "candidate_screening_time_limit_s"
                ]
            ),
            "reason": "transported_threshold_proof_precheck",
            "weighted_coverage_lower_bound": lower,
            "weighted_coverage_upper_bound": upper,
            "mip_gap": max(0.0, upper - lower)
            / max(lower, 1e-12),
        },
        "certification_source": (
            "v3_2_13_transported_resource_threshold"
        ),
        "certification_time_limit_s_used": float(
            protocol["task_generation"]["resource_threshold_fallback"][
                "lower_time_limit_s"
            ]
        )
        + float(
            protocol["task_generation"]["resource_threshold_fallback"][
                "upper_time_limit_s"
            ]
        ),
        "transported_resource_threshold_proof": {
            "resource_name": "distance",
            "low_required_priority": low_required,
            "high_required_priority": high_required,
            "total_priority": total_priority,
            "low_threshold": target_low,
            "source_high_threshold": source_high,
            "transported_high_threshold": transported_high,
            "source_proof_path": str(SOURCE_SINGLE_PROOF.resolve()),
            "source_proof_sha256": v32._sha256_file(SOURCE_SINGLE_PROOF),
            "source_invariant_hash": source_invariant_hash,
            "target_invariant_hash": target_invariant_hash,
            "transport_rule": (
                "identical relaxed threshold MILP; only actual distance "
                "budget tightened"
            ),
        },
    }
    result = copy.deepcopy(target_candidate)
    result["certificate"] = certificate
    result["task_hash"] = multimap._canonical_hash(
        result, excluded=("task_hash",)
    )
    return result


def compose_constructive_mixed_certificate(
    protocol: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    output_root: Path,
) -> Dict[str, Any]:
    """Compose the Colorado task from a safe lower route and exact SEC proof."""

    registry_path = MAP_ROOT / "real" / "map_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    provider = multimap.FrozenMapProvider.from_registries(
        MAP_ROOT, [registry_path]
    )
    contexts = v32._load_real_corridor_contexts(output_root, protocol)
    raw = mixed_source._candidate(
        dict(protocol), dict(parent), registry, contexts
    )
    calibration = json.loads(
        MIXED_CALIBRATION.read_text(encoding="utf-8")
    )
    if not math.isclose(
        float(calibration["target_utilization"]),
        float(
            protocol["pretest_compact_branch_cut_certificate"][
                "two_resource_calibration"
            ]["target_utilization"]
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("mixed calibration target is not registered")
    candidate = search._with_values(
        raw,
        calibration["calibrated_parameters"],
        protocol,
        {
            "stage": "minimum_resource_witness_two_resource_calibration",
            "source_resource": "distance",
            "resources": ["distance", "time"],
            "target_utilization": float(
                calibration["target_utilization"]
            ),
        },
    )
    problem = witness.build_frozen_problem(candidate, provider)
    evaluator = MissionEvaluator(problem)
    low_source = json.loads(MIXED_LOW_PROOF.read_text(encoding="utf-8"))
    low_proof = dict(low_source["proof"])
    low_order = list(low_proof.get("visit_order") or ())
    low_evaluation = evaluator.evaluate_order(low_order)
    if not low_evaluation.returned:
        raise RuntimeError("mixed lower-threshold route is not replay safe")
    final_cut = json.loads(
        MIXED_FINAL_CUT_PROOF.read_text(encoding="utf-8")
    )
    high_proof = dict(final_cut["proof"])
    if (
        not bool(high_proof.get("threshold_infeasible"))
        or int(high_proof.get("solver_status", -1)) != 2
        or high_proof.get("connected_route") is not None
    ):
        raise RuntimeError("mixed high-threshold proof is not infeasible")
    final_sets = {
        tuple(int(node) for node in item)
        for item in high_proof.get("subtour_cut_node_sets", ())
    }
    if not final_sets or len(final_sets) != int(
        high_proof.get("subtour_cut_count", -1)
    ):
        raise RuntimeError("subtour-cut proof serialization is invalid")

    priorities = np.asarray(candidate["priorities"], dtype=np.float64)
    total_priority = float(np.sum(priorities))
    band_low, band_high = (
        float(value)
        for value in parent["difficulty_bands"][
            str(candidate["difficulty"])
        ]
    )
    low_required = int(math.ceil(band_low * total_priority - 1e-9))
    # 证明该离散权重不可达后，可得最大权重严格小于难度带上界。
    high_required = int(math.ceil(band_high * total_priority - 1e-9))
    if not math.isclose(
        float(low_proof["minimum_priority_weight"]),
        float(low_required),
        rel_tol=0.0,
        abs_tol=1e-9,
    ) or not math.isclose(
        float(high_proof["minimum_priority_weight"]),
        float(high_required),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("mixed discrete thresholds changed")
    lower = float(low_evaluation.weighted_coverage)
    upper = float(high_required - 1) / total_priority
    metrics = {
        "energy_utilization": low_evaluation.energy_wh
        / evaluator.energy_budget_wh,
        "distance_utilization": low_evaluation.distance_m
        / evaluator.distance_budget_m,
        "time_utilization": low_evaluation.time_s
        / evaluator.time_budget_s,
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
    if len(bottlenecks) < 2:
        raise RuntimeError("mixed route does not activate two bottlenecks")
    registered_total = (
        float(
            protocol["pretest_compact_branch_cut_certificate"][
                "safe_lower_witness"
            ]["minimum_resource_threshold_milp_time_limit_s"]
        )
        + float(
            protocol["pretest_compact_branch_cut_certificate"][
                "high_threshold_proof"
            ]["time_limit_s"]
        )
    )
    certificate = {
        "algorithm": "milp_weighted_coverage",
        "solver_status": 2,
        "solver_success": True,
        "solver_message": (
            "safe minimum-resource lower-threshold route plus exact "
            "subtour-cut infeasibility proof"
        ),
        "status": str(low_evaluation.termination_reason),
        "mip_gap": max(0.0, upper - lower) / max(lower, 1e-12),
        "weighted_coverage_lower_bound": lower,
        "weighted_coverage_upper_bound": upper,
        "optimality_certified": math.isclose(
            lower, upper, rel_tol=0.0, abs_tol=1e-12
        ),
        "visit_order": list(low_evaluation.order),
        "visited_count": len(low_evaluation.order),
        "returned": True,
        **metrics,
        "runtime_s": float(low_proof["runtime_s"])
        + float(high_proof["runtime_s"]),
        "scenario_hash": str(problem.scenario_hash),
        "map_id": str(candidate["map_id"]),
        "map_hash": str(candidate["map_hash"]),
        "difficulty_certificate": (
            "safe_lower_plus_exact_high_threshold_infeasibility"
        ),
        "bottleneck_resources": list(bottlenecks),
        "screening": {
            "time_limit_s": float(
                protocol["certification"][
                    "candidate_screening_time_limit_s"
                ]
            ),
            "reason": "constructive_mixed_threshold_precheck",
            "weighted_coverage_lower_bound": lower,
            "weighted_coverage_upper_bound": upper,
            "mip_gap": max(0.0, upper - lower)
            / max(lower, 1e-12),
        },
        "certification_source": (
            "v3_2_14_constructive_mixed_threshold"
        ),
        "certification_time_limit_s_used": registered_total,
        "constructive_mixed_threshold_proof": {
            "low_required_priority": low_required,
            "high_required_priority": high_required,
            "total_priority": total_priority,
            "registered_solver_time_limit_s": registered_total,
            "low_threshold_proof": low_proof,
            "low_threshold_proof_path": str(
                MIXED_LOW_PROOF.resolve()
            ),
            "low_threshold_proof_sha256": v32._sha256_file(
                MIXED_LOW_PROOF
            ),
            "calibration_path": str(MIXED_CALIBRATION.resolve()),
            "calibration_sha256": v32._sha256_file(
                MIXED_CALIBRATION
            ),
            "final_cut_proof_path": str(
                MIXED_FINAL_CUT_PROOF.resolve()
            ),
            "final_cut_proof_sha256": v32._sha256_file(
                MIXED_FINAL_CUT_PROOF
            ),
            "final_cut_count": len(final_sets),
            "high_threshold_proof": high_proof,
        },
    }
    result = copy.deepcopy(candidate)
    result["certificate"] = certificate
    result["task_hash"] = multimap._canonical_hash(
        result, excluded=("task_hash",)
    )
    return result


def compose_direct_single_certificate(
    protocol: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    output_root: Path,
    map_index: int,
    road_index: int,
    task_index: int,
    attempt: int,
) -> Dict[str, Any]:
    """Place a single-resource budget strictly between low primal and high dual."""

    directory = Path(output_root) / "diagnostics" / "direct_threshold"
    stem = (
        f"map{map_index:02d}_road{road_index:02d}_"
        f"task{task_index:02d}_attempt{attempt:02d}"
    )
    low_path = directory / f"{stem}_low_limit60s.json"
    high_path = directory / f"{stem}_high_limit300s.json"
    low_source = json.loads(low_path.read_text(encoding="utf-8"))
    high_source = json.loads(high_path.read_text(encoding="utf-8"))
    low_proof = dict(low_source["proof"])
    high_proof = dict(high_source["proof"])
    registry_path = MAP_ROOT / "real" / "map_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    provider = multimap.FrozenMapProvider.from_registries(
        MAP_ROOT, [registry_path]
    )
    contexts = v32._load_real_corridor_contexts(output_root, protocol)
    raw = direct_probe.build_candidate(
        dict(protocol),
        dict(parent),
        registry,
        contexts,
        map_index=map_index,
        road_index=road_index,
        task_index=task_index,
        attempt=attempt,
    )
    resource = str(raw["constraint_type"])
    parameter = search.NAME_TO_PARAMETER.get(resource)
    if parameter not in {
        "initial_soc",
        "distance_budget_scale",
        "time_budget_scale",
    }:
        raise RuntimeError(
            "direct composer requires a registered single-resource budget"
        )
    upper_candidate = search._with_values(
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
    upper_problem = witness.build_frozen_problem(
        upper_candidate, provider
    )
    upper_evaluator = MissionEvaluator(upper_problem)
    low_order = list(low_proof.get("visit_order") or ())
    upper_low_evaluation = upper_evaluator.evaluate_order(low_order)
    if not upper_low_evaluation.returned:
        raise RuntimeError("direct low-threshold route is not safe")
    if resource == "energy":
        low_resource = float(upper_low_evaluation.energy_wh)
    elif resource == "distance":
        low_resource = float(upper_low_evaluation.distance_m)
    else:
        low_resource = float(upper_low_evaluation.time_s)
    high_dual = float(high_proof["resource_dual_bound"])
    if not high_dual > low_resource + 1e-6:
        raise RuntimeError(
            "direct threshold interval is empty; try another geometry"
        )
    target_budget = (low_resource + high_dual) / 2.0
    if resource == "energy":
        upper_budget = float(upper_evaluator.energy_budget_wh)
        capacity = float(upper_evaluator.template["battery_capacity"])
        reserve_ratio = float(
            upper_evaluator.template["cfg"]["battery_reserve_ratio"]
        )
        # 能量预算 = 电池容量 × (初始 SOC - 返航预留率)，据此精确反解 SOC。
        target_parameter = reserve_ratio + target_budget / capacity
    elif resource == "distance":
        upper_budget = float(upper_evaluator.distance_budget_m)
        target_parameter = float(
            upper_candidate[parameter] * target_budget / upper_budget
        )
    else:
        upper_budget = float(upper_evaluator.time_budget_s)
        target_parameter = float(
            upper_candidate[parameter] * target_budget / upper_budget
        )
    candidate = search._with_values(
        upper_candidate,
        {parameter: target_parameter},
        protocol,
        {
            "stage": "direct_threshold_dual_primal_midpoint",
            "parameter": parameter,
            "low_primal_resource": low_resource,
            "high_dual_resource": high_dual,
            "target_resource_budget": target_budget,
        },
    )
    problem = witness.build_frozen_problem(candidate, provider)
    evaluator = MissionEvaluator(problem)
    low_evaluation = evaluator.evaluate_order(low_order)
    if not low_evaluation.returned:
        raise RuntimeError("direct midpoint route is not safe")
    if resource == "energy":
        actual_budget = float(evaluator.energy_budget_wh)
    elif resource == "distance":
        actual_budget = float(evaluator.distance_budget_m)
    else:
        actual_budget = float(evaluator.time_budget_s)
    if not math.isclose(
        actual_budget, target_budget, rel_tol=0.0, abs_tol=1e-6
    ) or not high_dual > actual_budget + 1e-7:
        raise RuntimeError("direct midpoint budget replay failed")
    priorities = np.asarray(candidate["priorities"], dtype=np.float64)
    total_priority = float(np.sum(priorities))
    band_low, band_high = (
        float(value)
        for value in parent["difficulty_bands"][
            str(candidate["difficulty"])
        ]
    )
    low_required = int(math.ceil(band_low * total_priority - 1e-9))
    # 证明该离散权重不可达后，可得最大权重严格小于难度带上界。
    high_required = int(math.ceil(band_high * total_priority - 1e-9))
    if int(high_source.get("required_priority", -1)) != high_required:
        raise RuntimeError("direct high-threshold priority does not match protocol")
    lower = float(low_evaluation.weighted_coverage)
    upper = float(high_required - 1) / total_priority
    metrics = {
        "energy_utilization": low_evaluation.energy_wh
        / evaluator.energy_budget_wh,
        "distance_utilization": low_evaluation.distance_m
        / evaluator.distance_budget_m,
        "time_utilization": low_evaluation.time_s
        / evaluator.time_budget_s,
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
    if resource not in bottlenecks:
        raise RuntimeError("direct certificate intended bottleneck inactive")
    fallback = protocol["task_generation"]["resource_threshold_fallback"]
    registered_total = float(fallback["lower_time_limit_s"]) + float(
        fallback["upper_time_limit_s"]
    )
    if not math.isclose(
        float(low_source.get("requested_time_limit_s", math.nan)),
        float(fallback["lower_time_limit_s"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(high_source.get("requested_time_limit_s", math.nan)),
        float(fallback["upper_time_limit_s"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("direct threshold solver limits are not registered")
    transported_high = copy.deepcopy(high_proof)
    transported_high["source_actual_resource_budget"] = float(
        high_proof["actual_resource_budget"]
    )
    transported_high["actual_resource_budget"] = actual_budget
    transported_high["threshold_impossible_under_actual_budget"] = True
    certificate = {
        "algorithm": "milp_weighted_coverage",
        "solver_status": high_proof["solver_status"],
        "solver_success": True,
        "solver_message": (
            "safe low-threshold route plus high-threshold minimum-resource "
            "dual bound"
        ),
        "status": str(low_evaluation.termination_reason),
        "mip_gap": max(0.0, upper - lower) / max(lower, 1e-12),
        "weighted_coverage_lower_bound": lower,
        "weighted_coverage_upper_bound": upper,
        "optimality_certified": math.isclose(
            lower, upper, rel_tol=0.0, abs_tol=1e-12
        ),
        "visit_order": list(low_evaluation.order),
        "visited_count": len(low_evaluation.order),
        "returned": True,
        **metrics,
        "runtime_s": registered_total,
        "scenario_hash": str(problem.scenario_hash),
        "map_id": str(candidate["map_id"]),
        "map_hash": str(candidate["map_hash"]),
        "difficulty_certificate": (
            "direct_low_primal_high_dual_same_band"
        ),
        "bottleneck_resources": list(bottlenecks),
        "screening": {
            "time_limit_s": float(
                protocol["certification"][
                    "candidate_screening_time_limit_s"
                ]
            ),
            "reason": "direct_threshold_precheck",
            "weighted_coverage_lower_bound": lower,
            "weighted_coverage_upper_bound": upper,
            "mip_gap": max(0.0, upper - lower)
            / max(lower, 1e-12),
        },
        "certification_source": "resource_threshold_fallback",
        "certification_time_limit_s_used": registered_total,
        "resource_threshold_proof": {
            "resource_name": resource,
            "low_required_priority": low_required,
            "high_required_priority": high_required,
            "total_priority": total_priority,
            "low_threshold": low_proof,
            "high_threshold": transported_high,
            "direct_midpoint_transport": {
                "parameter": parameter,
                "source_high_threshold": high_proof,
            },
            "low_proof_path": str(low_path.resolve()),
            "low_proof_sha256": v32._sha256_file(low_path),
            "high_proof_path": str(high_path.resolve()),
            "high_proof_sha256": v32._sha256_file(high_path),
            "low_primal_resource": low_resource,
            "high_dual_resource": high_dual,
            "target_resource_budget": target_budget,
        },
    }
    result = copy.deepcopy(candidate)
    result["certificate"] = certificate
    result["task_hash"] = multimap._canonical_hash(
        result, excluded=("task_hash",)
    )
    return result


__all__ = [
    "compose_constructive_mixed_certificate",
    "compose_direct_single_certificate",
    "compose_transported_single_certificate",
]

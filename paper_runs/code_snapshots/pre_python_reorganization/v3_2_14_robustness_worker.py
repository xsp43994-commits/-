#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execute one model/seed job from a frozen v3.2.14 robustness family."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

import final_python_ppo_pointer as ppo
import paper_experiments as paper
import paper_multimap_experiments as multimap
import paper_v3_2_experiments as v32
import v3_2_14_evaluation_smoke as smoke
from python_classical_algs import run_planner
from python_classical_algs.common import MissionEvaluator, make_problem


ROOT = Path(__file__).resolve().parent
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/protocol.json"
)
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14"
MAP_ROOT = ROOT / "map_data/multimap_v3_1"
MATRIX = OUTPUT / "formal_evaluation/evaluation_matrix.jsonl"
MATRIX_MANIFEST = (
    OUTPUT / "formal_evaluation/evaluation_matrix_manifest.json"
)
ROBUSTNESS_MANIFEST = (
    OUTPUT / "formal_evaluation/robustness_implementation_manifest.json"
)
REAL = OUTPUT / "formal_evaluation/real_tasks_parallel/records.jsonl"
REAL_REGISTRY = MAP_ROOT / "real/map_registry.json"
FAMILIES = {
    "known_domain_shift": 1008,
    "hidden_model_perception_mismatch": 2496,
}


def _sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _seed(protocol_hash: str, task_hash: str, condition: str) -> int:
    payload = f"{protocol_hash}|{task_hash}|{condition}|v3_2_14"
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8],
        "little",
        signed=False,
    )


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["family"]),
        str(row["task_id"]),
        str(row["model"]),
        row.get("training_seed"),
        row.get("planner_seed"),
        str(row["condition"]),
    )


def _domain_shift(
    task: Mapping[str, Any],
    *,
    condition: str,
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    shifted = copy.deepcopy(dict(task))
    if condition == "wind":
        wind = spec["wind"]
        shifted["wind_scale"] = float(
            shifted.get("wind_scale", 1.0)
        ) * float(wind["speed_scale"])
        shifted["wind_rotation_deg"] = float(
            shifted.get("wind_rotation_deg", 0.0)
        ) + float(wind["rotation_deg"])
        shifted["wind_vertical_bias_mps"] = float(
            shifted.get("wind_vertical_bias_mps", 0.0)
        ) + float(wind["vertical_bias_mps"])
    elif condition == "power_model":
        shifted["power_scale"] = float(
            shifted.get("power_scale", 1.0)
        ) * float(spec["power_model"]["coefficient_scale"])
    else:
        raise ValueError(f"not a domain shift: {condition}")
    return shifted


def _input_hash(
    *,
    task: Mapping[str, Any],
    start: np.ndarray,
    points: np.ndarray,
    terrain: np.ndarray,
    condition: str,
    layer: str,
    realization_seed: int,
) -> str:
    return smoke._canonical_hash(
        {
            "task_hash": task["task_hash"],
            "condition": condition,
            "realization_seed": realization_seed,
            "start_sha256": _sha_array(start),
            "points_sha256": _sha_array(points),
            "terrain_sha256": _sha_array(terrain),
            "wind_scale": float(task.get("wind_scale", 1.0)),
            "wind_rotation_deg": float(
                task.get("wind_rotation_deg", 0.0)
            ),
            "wind_vertical_bias_mps": float(
                task.get("wind_vertical_bias_mps", 0.0)
            ),
            "power_scale": float(task.get("power_scale", 1.0)),
        }
    )


def _condition_inputs(
    *,
    task: Mapping[str, Any],
    context: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    family: str,
    condition: str,
    protocol_hash: str,
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    seed = _seed(protocol_hash, str(task["task_hash"]), condition)
    rng = np.random.default_rng(seed)
    truth_task = copy.deepcopy(dict(task))
    observed_task = copy.deepcopy(dict(task))
    truth_terrain = np.asarray(context["terrain"], dtype=np.float32)
    observed_terrain = truth_terrain
    truth_start = np.asarray(context["start_pos"], dtype=np.float32)
    observed_start = truth_start.copy()
    truth_points = np.asarray(
        task["inspection_points_xyz"], dtype=np.float32
    )
    observed_points = truth_points.copy()
    observed_cfg = dict(base_cfg)
    realization: Dict[str, Any] = {
        "seed": seed,
        "condition": condition,
        "family": family,
    }

    if family == "known_domain_shift":
        observed_task = _domain_shift(
            task, condition=condition, spec=spec
        )
        truth_task = copy.deepcopy(observed_task)
    elif condition in {"wind", "power_model"}:
        truth_task = _domain_shift(
            task, condition=condition, spec=spec
        )
    elif condition == "dem_error":
        dem = spec["dem_error"]
        coordinate_scale = float(
            context["cfg_overrides"]["coordinate_scale_m_per_unit"]
        )
        sigma_px = float(dem["correlation_length_m"]) / (
            math.sqrt(2.0) * coordinate_scale
        )
        field = gaussian_filter(
            rng.normal(size=truth_terrain.shape),
            sigma=sigma_px,
            mode=str(dem["boundary_mode"]),
        )
        field -= float(np.mean(field))
        standard = float(np.std(field))
        if standard <= 1e-12:
            raise RuntimeError("degenerate DEM perturbation field")
        field *= float(dem["sigma_m"]) / standard
        observed_terrain = (
            truth_terrain.astype(np.float64) + field
        ).astype(np.float32)
        realization.update(
            {
                "field_sha256": _sha_array(observed_terrain),
                "field_mean_m": float(np.mean(field)),
                "field_std_m": float(np.std(field)),
                "gaussian_filter_sigma_px": sigma_px,
            }
        )
    elif condition == "localization":
        localization = spec["localization"]
        coordinate_scale = float(
            context["cfg_overrides"]["coordinate_scale_m_per_unit"]
        )
        horizontal_sigma_units = float(
            localization["horizontal_sigma_m"]
        ) / coordinate_scale
        node_xy_noise = rng.normal(
            0.0, horizontal_sigma_units, size=(len(observed_points), 2)
        )
        node_z_noise = rng.normal(
            0.0,
            float(localization["vertical_sigma_m"]),
            size=len(observed_points),
        )
        start_xy_noise = rng.normal(
            0.0, horizontal_sigma_units, size=2
        )
        start_z_noise = float(
            rng.normal(0.0, float(localization["vertical_sigma_m"]))
        )
        observed_points[:, :2] += node_xy_noise.astype(np.float32)
        observed_points[:, 0] = np.clip(
            observed_points[:, 0], 0.0, truth_terrain.shape[1] - 1.0
        )
        observed_points[:, 1] = np.clip(
            observed_points[:, 1], 0.0, truth_terrain.shape[0] - 1.0
        )
        clearance = float(base_cfg["terrain_clearance_m"])
        for index, point in enumerate(observed_points):
            ground = ppo.height_at(
                observed_terrain, float(point[0]), float(point[1])
            )
            point[2] = ground + clearance + float(node_z_noise[index])
        observed_start[:2] += start_xy_noise.astype(np.float32)
        observed_start[0] = np.clip(
            observed_start[0], 0.0, truth_terrain.shape[1] - 1.0
        )
        observed_start[1] = np.clip(
            observed_start[1], 0.0, truth_terrain.shape[0] - 1.0
        )
        raw_observed_start_z = float(
            observed_start[2] + start_z_noise
        )
        minimum_observed_start_z = (
            ppo.height_at(
                observed_terrain,
                float(observed_start[0]),
                float(observed_start[1]),
            )
            + 1e-2
        )
        # 数值安全前置条件：受噪观测起点不能位于其观测地面以下。
        # 1 cm 裕度用于覆盖 float32 在高海拔 DSM 上的量化误差。
        observed_start[2] = max(
            raw_observed_start_z, minimum_observed_start_z
        )
        observed_cfg["point_z_mode"] = "flight_altitude"
        realization.update(
            {
                "node_xy_noise_sha256": _sha_array(node_xy_noise),
                "node_z_noise_sha256": _sha_array(node_z_noise),
                "start_xy_noise_sha256": _sha_array(start_xy_noise),
                "start_z_noise_m": start_z_noise,
                "raw_observed_start_z_m": raw_observed_start_z,
                "minimum_observed_start_z_m": minimum_observed_start_z,
                "observed_start_z_clipped": bool(
                    raw_observed_start_z < minimum_observed_start_z
                ),
            }
        )
    else:
        raise ValueError(f"unsupported robustness condition {condition}")

    if family == "hidden_model_perception_mismatch":
        geometry_raw_start_z = float(observed_start[2])
        geometry_ground_z = float(
            ppo.height_at(
                observed_terrain,
                float(observed_start[0]),
                float(observed_start[1]),
            )
        )
        geometry_minimum_start_z = geometry_ground_z + 1e-2
        # DEM 和定位误差都可能令观测起点落到观测地面以下。
        # 只修正策略可见的观测高度；冻结真值起点保持不变。
        observed_start[2] = max(
            geometry_raw_start_z, geometry_minimum_start_z
        )
        realization.update(
            {
                "geometry_raw_observed_start_z_m": geometry_raw_start_z,
                "geometry_observed_ground_z_m": geometry_ground_z,
                "geometry_minimum_observed_start_z_m": (
                    geometry_minimum_start_z
                ),
                "geometry_observed_start_z_clipped": bool(
                    geometry_raw_start_z < geometry_minimum_start_z
                ),
            }
        )

    observed_hash = _input_hash(
        task=observed_task,
        start=observed_start,
        points=observed_points,
        terrain=observed_terrain,
        condition=condition,
        layer="observed",
        realization_seed=seed,
    )
    truth_hash = _input_hash(
        task=truth_task,
        start=truth_start,
        points=truth_points,
        terrain=truth_terrain,
        condition=condition,
        layer="execution_truth",
        realization_seed=seed,
    )
    if family == "known_domain_shift" and observed_hash != truth_hash:
        raise RuntimeError("known shift observation/truth must match")
    if (
        family == "hidden_model_perception_mismatch"
        and observed_hash == truth_hash
    ):
        raise RuntimeError("hidden mismatch observation/truth must differ")
    realization["realization_hash"] = smoke._canonical_hash(realization)
    return {
        "observed_task": observed_task,
        "truth_task": truth_task,
        "observed_terrain": observed_terrain,
        "truth_terrain": truth_terrain,
        "observed_start": observed_start,
        "truth_start": truth_start,
        "observed_points": observed_points,
        "truth_points": truth_points,
        "observed_cfg": observed_cfg,
        "observed_hash": observed_hash,
        "truth_hash": truth_hash,
        "realization": realization,
    }


def _truth_metrics(
    *,
    order: Sequence[int],
    task: Mapping[str, Any],
    context: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    cfg = ppo.resolve_config(
        {**dict(base_cfg), **dict(context["cfg_overrides"])}
    )
    truth_cfg, truth_wind = ppo.apply_frozen_domain_instance(
        cfg, context["wind_data"], inputs["truth_task"]
    )
    problem = make_problem(
        inputs["truth_start"],
        inputs["truth_points"],
        np.asarray(task["priorities"], dtype=np.float32),
        inputs["truth_terrain"],
        truth_cfg,
        truth_wind,
        name=f"{task['id']}__truth",
    )
    evaluator = MissionEvaluator(problem)
    evaluation = evaluator.evaluate_order(order)
    return {
        "returned": bool(evaluation.returned),
        "weighted_coverage": float(evaluation.weighted_coverage),
        "coverage": float(evaluation.coverage),
        "visited_count": len(evaluation.order),
        "termination_reason": str(evaluation.termination_reason),
        "energy_wh": float(evaluation.energy_wh),
        "energy_budget_wh": float(evaluator.energy_budget_wh),
        "energy_utilization": float(
            evaluation.energy_wh / evaluator.energy_budget_wh
        ),
        "distance_m": float(evaluation.distance_m),
        "distance_budget_m": float(evaluator.distance_budget_m),
        "distance_utilization": float(
            evaluation.distance_m / evaluator.distance_budget_m
        ),
        "time_s": float(evaluation.time_s),
        "time_budget_s": float(evaluator.time_budget_s),
        "time_utilization": float(
            evaluation.time_s / evaluator.time_budget_s
        ),
        "min_remaining_soc": float(
            inputs["truth_task"]["initial_soc"]
            - evaluation.energy_wh
            / float(evaluator.template["battery_capacity"])
        ),
        "energy_violation": bool(
            evaluation.energy_wh > evaluator.energy_budget_wh + 1e-6
        ),
        "distance_violation": bool(
            evaluation.distance_m > evaluator.distance_budget_m + 1e-6
        ),
        "time_violation": bool(
            evaluation.time_s > evaluator.time_budget_s + 1e-6
        ),
        "dynamics_violation": bool(
            not evaluation.returned
            and evaluation.termination_reason
            in {"infeasible_candidate", "constraint_failure"}
        ),
    }


def _result_row(
    *,
    matrix_row: Mapping[str, Any],
    task: Mapping[str, Any],
    checkpoint_hash: str,
    metrics: Mapping[str, Any],
    observation_metrics: Mapping[str, Any],
    inputs: Mapping[str, Any],
    route_hash: str,
    planning_time_s: float,
    evaluations: int,
    planner_status: str,
    protocol_hash: str,
) -> Dict[str, Any]:
    safe = bool(metrics.get("returned")) and not any(
        bool(metrics.get(field, False))
        for field in (
            "energy_violation",
            "distance_violation",
            "time_violation",
            "dynamics_violation",
        )
    )
    certificate = dict(task["certificate"])
    row = {
        "schema_version": 1,
        "paper_eligible": True,
        "protocol_hash": protocol_hash,
        "matrix_row_hash": smoke._canonical_hash(matrix_row),
        "family": str(matrix_row["family"]),
        "condition": str(matrix_row["condition"]),
        "perturbation_layer": (
            "known_domain_shift"
            if matrix_row["family"] == "known_domain_shift"
            else "hidden_model_perception_mismatch"
        ),
        "perturbation_type": str(matrix_row["condition"]),
        "task_id": str(task["id"]),
        "task_hash": str(task["task_hash"]),
        "map_id": str(task["map_id"]),
        "road_index": int(task["road_index"]),
        "task_index": int(task["task_index"]),
        "node_count": int(task["node_count"]),
        "difficulty": str(task["difficulty"]),
        "constraint_type": str(task["constraint_type"]),
        "priority_layout": str(task["priority_layout"]),
        "model": str(matrix_row["model"]),
        "training_seed": matrix_row.get("training_seed"),
        "planner_seed": matrix_row.get("planner_seed"),
        "checkpoint_hash": checkpoint_hash,
        "nominal_input_hash": str(task["task_hash"]),
        "observed_input_hash": str(inputs["observed_hash"]),
        "execution_truth_hash": str(inputs["truth_hash"]),
        "perturbation_realization_hash": str(
            inputs["realization"]["realization_hash"]
        ),
        "route_hash": route_hash,
        "observation_returned": bool(
            observation_metrics.get("returned", False)
        ),
        "observation_weighted_coverage": float(
            observation_metrics.get("weighted_coverage", 0.0)
        ),
        "safe": safe,
        "safe_weighted_coverage": (
            float(metrics.get("weighted_coverage", 0.0)) if safe else 0.0
        ),
        "weighted_coverage": float(
            metrics.get("weighted_coverage", 0.0)
        ),
        "coverage": float(metrics.get("coverage", 0.0)),
        "returned": bool(metrics.get("returned", False)),
        "visited_count": int(metrics.get("visited_count", 0)),
        "termination_reason": str(
            metrics.get("termination_reason", "unknown")
        ),
        "energy_wh": float(metrics.get("energy_wh", 0.0)),
        "energy_budget_wh": float(
            metrics.get("energy_budget_wh", 0.0)
        ),
        "energy_utilization": float(
            metrics.get("energy_utilization", 0.0)
        ),
        "distance_m": float(metrics.get("distance_m", 0.0)),
        "distance_budget_m": float(
            metrics.get("distance_budget_m", 0.0)
        ),
        "distance_utilization": float(
            metrics.get("distance_utilization", 0.0)
        ),
        "time_s": float(metrics.get("time_s", 0.0)),
        "time_budget_s": float(metrics.get("time_budget_s", 0.0)),
        "time_utilization": float(
            metrics.get("time_utilization", 0.0)
        ),
        "min_remaining_soc": float(
            metrics.get("min_remaining_soc", 0.0)
        ),
        "energy_violation": bool(
            metrics.get("energy_violation", False)
        ),
        "distance_violation": bool(
            metrics.get("distance_violation", False)
        ),
        "time_violation": bool(
            metrics.get("time_violation", False)
        ),
        "dynamics_violation": bool(
            metrics.get("dynamics_violation", False)
        ),
        "planning_time_s": float(planning_time_s),
        "evaluations": int(evaluations),
        "planner_status": str(planner_status),
        "oracle_lower": float(
            certificate["weighted_coverage_lower_bound"]
        ),
        "oracle_upper": float(
            certificate["weighted_coverage_upper_bound"]
        ),
    }
    for value in row.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("robustness result contains non-finite data")
    row["result_hash"] = smoke._canonical_hash(row)
    return row


def run(
    *,
    family: str,
    model_name: str,
    seed: int,
    resume: bool,
    max_new_rows: int | None,
    device: str,
) -> Dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(f"unsupported robustness family {family!r}")
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    matrix_manifest = json.loads(
        MATRIX_MANIFEST.read_text(encoding="utf-8")
    )
    robustness = json.loads(
        ROBUSTNESS_MANIFEST.read_text(encoding="utf-8")
    )
    if (
        robustness["parent_protocol_hash"] != protocol["protocol_hash"]
        or robustness["matrix_sha256"]
        != matrix_manifest["matrix_sha256"]
        or robustness["implementation_sha256"]
        != v32._sha256_file(Path(__file__))
    ):
        raise RuntimeError("robustness implementation identity mismatch")
    task_rows = v32._read_jsonl(REAL)
    tasks = {str(row["id"]): row for row in task_rows}
    matrix_rows = [
        row
        for row in v32._read_jsonl(MATRIX)
        if str(row["family"]) == family
        and str(row["model"]) == model_name
        and (
            int(row["training_seed"]) == seed
            if row.get("training_seed") is not None
            else int(row["planner_seed"]) == seed
        )
    ]
    if not matrix_rows:
        raise RuntimeError("robustness job absent from matrix")
    expected_task_hashes = set(robustness["robustness_task_hashes"])
    if {
        str(row["task_hash"]) for row in matrix_rows
    } != expected_task_hashes:
        raise RuntimeError("robustness task subset drift")

    provider = multimap.FrozenMapProvider.from_registries(
        MAP_ROOT, [REAL_REGISTRY]
    )
    catalog = v32.checkpoint_catalog(PROTOCOL, output_root=OUTPUT)
    checkpoint = next(
        (
            row
            for row in catalog["rows"]
            if row["variant"] == model_name
            and int(row["training_seed"]) == seed
        ),
        None,
    )
    learning = checkpoint is not None
    if not learning and not (
        model_name == "priority_resource_greedy" and seed == 42
    ):
        raise RuntimeError("unsupported robustness algorithm identity")
    if learning:
        model, payload = ppo.load_checkpoint(
            checkpoint["checkpoint_path"], map_location=device
        )
        checkpoint_hash = str(checkpoint["checkpoint_sha256"])
    else:
        full = next(
            row
            for row in catalog["rows"]
            if row["variant"] == "full"
            and int(row["training_seed"]) == 42
        )
        model = None
        _, payload = ppo.load_checkpoint(
            full["checkpoint_path"], map_location="cpu"
        )
        checkpoint_hash = ""
    base_cfg = dict(payload["cfg"])

    seed_label = f"train{seed}" if learning else f"plan{seed}"
    run_dir = (
        OUTPUT
        / "formal_evaluation"
        / "results"
        / family
        / "jobs"
        / f"{model_name}__{seed_label}"
    )
    results_path = run_dir / "results.jsonl"
    if results_path.exists() and not resume:
        raise FileExistsError("robustness job exists; use --resume")
    writer = paper.DurableResultJsonlWriter(
        results_path, resume=resume, repair_trailing=resume
    )
    existing = writer.records()
    completed: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for row in existing:
        key = _key(row)
        if key in completed:
            raise RuntimeError("duplicate robustness job key")
        completed[key] = row
    smoke._atomic_json(
        run_dir / "run_config.json",
        {
            "schema_version": 1,
            "paper_eligible": True,
            "protocol_hash": protocol["protocol_hash"],
            "robustness_manifest_hash": robustness[
                "manifest_hash"
            ],
            "family": family,
            "model": model_name,
            "seed": seed,
            "expected_rows": len(matrix_rows),
            "common_random_realizations": True,
            "route_locked_before_truth_evaluation": (
                family == "hidden_model_perception_mismatch"
            ),
        },
    )
    started_run = time.time()
    new_rows = 0
    for matrix_row in sorted(
        matrix_rows,
        key=lambda row: (str(row["condition"]), str(row["task_id"])),
    ):
        key = _key(matrix_row)
        if key in completed:
            continue
        if max_new_rows is not None and new_rows >= max_new_rows:
            break
        task = tasks[str(matrix_row["task_id"])]
        context = provider(task)
        inputs = _condition_inputs(
            task=task,
            context=context,
            base_cfg=base_cfg,
            family=family,
            condition=str(matrix_row["condition"]),
            protocol_hash=str(protocol["protocol_hash"]),
            spec=robustness["perturbations"],
        )
        observed_cfg = ppo.resolve_config(
            {
                **base_cfg,
                **dict(inputs["observed_cfg"]),
                **dict(context["cfg_overrides"]),
            }
        )
        plan_cfg, plan_wind = ppo.apply_frozen_domain_instance(
            observed_cfg,
            context["wind_data"],
            inputs["observed_task"],
        )
        if learning:
            started = time.perf_counter()
            detail = ppo.plan_with_policy_improved(
                model,
                inputs["observed_start"],
                inputs["observed_points"],
                np.asarray(task["priorities"], dtype=np.float32),
                inputs["observed_terrain"],
                plan_cfg,
                plan_wind,
                return_details=True,
                decode_mode="deterministic",
            )
            planning_time_s = time.perf_counter() - started
            order = list(detail.get("visit_order", ()))
            observation_metrics = dict(detail["metrics"])
            observation_payload: Mapping[str, Any] = detail
            evaluations = 1
            planner_status = "ok"
        else:
            problem = make_problem(
                inputs["observed_start"],
                inputs["observed_points"],
                np.asarray(task["priorities"], dtype=np.float32),
                inputs["observed_terrain"],
                plan_cfg,
                plan_wind,
                name=f"{task['id']}__observed",
            )
            result = run_planner(
                "priority_resource_greedy", problem, seed=seed
            )
            planning_time_s = float(result.runtime_s)
            order = list(result.visit_order)
            observation_metrics = dict(result.metrics)
            observation_payload = result.as_dict()
            evaluations = int(result.evaluations)
            planner_status = str(result.status)
        if family == "known_domain_shift":
            metrics = observation_metrics
        else:
            metrics = _truth_metrics(
                order=order,
                task=task,
                context=context,
                base_cfg=base_cfg,
                inputs=inputs,
            )
        route_payload = smoke._jsonable(
            {
                "schema_version": 1,
                "protocol_hash": protocol["protocol_hash"],
                "robustness_manifest_hash": robustness[
                    "manifest_hash"
                ],
                "matrix_row": matrix_row,
                "task_hash": task["task_hash"],
                "checkpoint_hash": checkpoint_hash,
                "observed_input_hash": inputs["observed_hash"],
                "execution_truth_hash": inputs["truth_hash"],
                "realization": inputs["realization"],
                "visit_order_locked": order,
                "observation_plan": observation_payload,
                "truth_metrics": metrics,
            }
        )
        route_hash = smoke._canonical_hash(route_payload)
        route_name = (
            f"{matrix_row['condition']}__"
            f"{str(task['id']).replace(':', '_')}.json"
        )
        smoke._atomic_json(
            run_dir / "routes" / route_name, route_payload
        )
        row = _result_row(
            matrix_row=matrix_row,
            task=task,
            checkpoint_hash=checkpoint_hash,
            metrics=metrics,
            observation_metrics=observation_metrics,
            inputs=inputs,
            route_hash=route_hash,
            planning_time_s=planning_time_s,
            evaluations=evaluations,
            planner_status=planner_status,
            protocol_hash=str(protocol["protocol_hash"]),
        )
        writer.append(row)
        completed[key] = row
        new_rows += 1
        smoke._atomic_json(
            run_dir / "status.json",
            {
                "state": "running",
                "family": family,
                "model": model_name,
                "seed": seed,
                "completed": len(completed),
                "total": len(matrix_rows),
                "last_key": list(key),
                "elapsed_s": time.time() - started_run,
            },
        )
    state = (
        "completed" if len(completed) == len(matrix_rows) else "partial"
    )
    smoke._atomic_json(
        run_dir / "status.json",
        {
            "state": state,
            "family": family,
            "model": model_name,
            "seed": seed,
            "completed": len(completed),
            "total": len(matrix_rows),
            "new_rows": new_rows,
            "elapsed_s": time.time() - started_run,
        },
    )
    return {
        "state": state,
        "family": family,
        "model": model_name,
        "seed": seed,
        "completed": len(completed),
        "total": len(matrix_rows),
        "new_rows": new_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=tuple(FAMILIES), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-rows", type=int)
    args = parser.parse_args(argv)
    report = run(
        family=str(args.family),
        model_name=str(args.model),
        seed=int(args.seed),
        resume=bool(args.resume),
        max_new_rows=args.max_new_rows,
        device=str(args.device),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

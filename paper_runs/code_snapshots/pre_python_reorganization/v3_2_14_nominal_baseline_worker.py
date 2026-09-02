#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execute one planner/seed job from a frozen nominal baseline family."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

import final_python_ppo_pointer as ppo
import paper_experiments as paper
import paper_multimap_experiments as multimap
import paper_v3_2_experiments as v32
import v3_2_14_evaluation_smoke as smoke
from python_classical_algs import PLANNER_SPECS, run_planner
from python_classical_algs.common import make_problem


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
SYNTHETIC = OUTPUT / "manifests/synthetic_test/records.jsonl"
REAL = OUTPUT / "formal_evaluation/real_tasks_parallel/records.jsonl"
REGISTRIES = (
    MAP_ROOT / "procedural/synthetic_test/map_registry.json",
    MAP_ROOT / "real/map_registry.json",
)
FAMILIES = {
    "synthetic_main_baselines": 3888,
    "synthetic_supplementary": 504,
    "real_baselines": 1152,
}
PLANNER_ALIASES = {"milp": "milp_orienteering"}


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["family"]),
        str(row["task_id"]),
        str(row["model"]),
        int(row["planner_seed"]),
        str(row["condition"]),
    )


def _safe(metrics: Mapping[str, Any]) -> tuple[bool, float]:
    safe = bool(metrics.get("returned")) and not any(
        bool(metrics.get(field, False))
        for field in (
            "energy_violation",
            "distance_violation",
            "time_violation",
            "dynamics_violation",
        )
    )
    return (
        safe,
        float(metrics.get("weighted_coverage", 0.0)) if safe else 0.0,
    )


def _result_row(
    *,
    matrix_row: Mapping[str, Any],
    task: Mapping[str, Any],
    result: Any,
    route_hash: str,
    environment_config_hash: str,
    protocol_hash: str,
) -> Dict[str, Any]:
    metrics = dict(result.metrics)
    metadata = dict(result.metadata or {})
    safe, safe_coverage = _safe(metrics)
    nominal_hash = str(task["task_hash"])
    certificate = dict(task["certificate"])
    row = {
        "schema_version": 1,
        "paper_eligible": True,
        "protocol_hash": protocol_hash,
        "matrix_row_hash": smoke._canonical_hash(matrix_row),
        "family": str(matrix_row["family"]),
        "condition": "nominal",
        "perturbation_layer": "none",
        "perturbation_type": "none",
        "task_id": str(task["id"]),
        "task_hash": nominal_hash,
        "map_id": str(task["map_id"]),
        "road_index": task.get("road_index"),
        "task_index": int(task["task_index"]),
        "node_count": int(task["node_count"]),
        "difficulty": str(task["difficulty"]),
        "constraint_type": str(task["constraint_type"]),
        "priority_layout": str(task["priority_layout"]),
        "model": str(matrix_row["model"]),
        "training_seed": None,
        "planner_seed": int(matrix_row["planner_seed"]),
        "checkpoint_hash": "",
        "environment_config_hash": environment_config_hash,
        "nominal_input_hash": nominal_hash,
        "observed_input_hash": nominal_hash,
        "execution_truth_hash": nominal_hash,
        "route_hash": route_hash,
        "safe": safe,
        "safe_weighted_coverage": safe_coverage,
        "weighted_coverage": float(metrics.get("weighted_coverage", 0.0)),
        "coverage": float(metrics.get("coverage", 0.0)),
        "returned": bool(metrics.get("returned", False)),
        "visited_count": int(metrics.get("visited_count", 0)),
        "termination_reason": str(metrics.get("termination_reason", "unknown")),
        "energy_wh": float(metrics.get("energy_wh", 0.0)),
        "energy_budget_wh": float(metrics.get("energy_budget_wh", 0.0)),
        "energy_utilization": float(metrics.get("energy_utilization", 0.0)),
        "distance_m": float(metrics.get("distance_m", 0.0)),
        "distance_budget_m": float(metrics.get("distance_budget_m", 0.0)),
        "distance_utilization": float(
            metrics.get("distance_utilization", 0.0)
        ),
        "time_s": float(metrics.get("time_s", 0.0)),
        "time_budget_s": float(metrics.get("time_budget_s", 0.0)),
        "time_utilization": float(metrics.get("time_utilization", 0.0)),
        "min_remaining_soc": float(metrics.get("min_remaining_soc", 0.0)),
        "energy_violation": bool(metrics.get("energy_violation", False)),
        "distance_violation": bool(
            metrics.get("distance_violation", False)
        ),
        "time_violation": bool(metrics.get("time_violation", False)),
        "dynamics_violation": bool(
            metrics.get("dynamics_violation", False)
        ),
        "planning_time_s": float(result.runtime_s),
        "evaluations": int(result.evaluations),
        "planner_status": str(result.status),
        "solver_status": metadata.get("solver_status"),
        "solver_success": metadata.get("solver_success"),
        "solver_gap": metadata.get(
            "mip_gap", metadata.get("optimality_gap")
        ),
        "solver_dual_bound": metadata.get(
            "objective_dual_bound",
            metadata.get("weighted_coverage_upper_bound"),
        ),
        "optimality_certified": metadata.get("optimality_certified"),
        "oracle_lower": float(
            certificate["weighted_coverage_lower_bound"]
        ),
        "oracle_upper": float(
            certificate["weighted_coverage_upper_bound"]
        ),
    }
    for value in row.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("formal baseline result contains non-finite data")
    row["result_hash"] = smoke._canonical_hash(row)
    return row


def run(
    *,
    family: str,
    model: str,
    planner_seed: int,
    resume: bool,
    max_new_rows: int | None,
) -> Dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(f"unsupported baseline family {family!r}")
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    manifest = json.loads(MATRIX_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest["protocol_hash"] != protocol["protocol_hash"]
        or int(manifest["row_count"]) != 21648
        or manifest["matrix_sha256"] != v32._sha256_file(MATRIX)
    ):
        raise RuntimeError("frozen matrix identity mismatch")
    task_rows = v32._read_jsonl(SYNTHETIC) + v32._read_jsonl(REAL)
    tasks = {str(row["id"]): row for row in task_rows}
    rows = [
        row
        for row in v32._read_jsonl(MATRIX)
        if str(row["family"]) == family
        and str(row["model"]) == model
        and int(row["planner_seed"]) == int(planner_seed)
    ]
    if not rows:
        raise RuntimeError("baseline job is absent from frozen matrix")
    for row in rows:
        task = tasks.get(str(row["task_id"]))
        if (
            task is None
            or str(task["task_hash"]) != str(row["task_hash"])
            or row.get("training_seed") is not None
            or str(row["condition"]) != "nominal"
        ):
            raise RuntimeError("baseline matrix/task semantics mismatch")

    planner_name = PLANNER_ALIASES.get(model, model)
    if planner_name not in PLANNER_SPECS:
        raise RuntimeError(f"unregistered planner {model!r}")
    provider = multimap.FrozenMapProvider.from_registries(
        MAP_ROOT, REGISTRIES
    )
    catalog = v32.checkpoint_catalog(PROTOCOL, output_root=OUTPUT)
    full = next(
        row
        for row in catalog["rows"]
        if row["variant"] == "full" and int(row["training_seed"]) == 42
    )
    _, payload = ppo.load_checkpoint(
        full["checkpoint_path"], map_location="cpu"
    )
    base_cfg = dict(payload["cfg"])
    environment_config_hash = smoke._canonical_hash(base_cfg)

    job_name = f"{model}__seed{int(planner_seed)}"
    run_dir = (
        OUTPUT
        / "formal_evaluation"
        / "results"
        / family
        / "jobs"
        / job_name
    )
    results_path = run_dir / "results.jsonl"
    if results_path.exists() and not resume:
        raise FileExistsError("baseline job output exists; use --resume")
    writer = paper.DurableResultJsonlWriter(
        results_path, resume=resume, repair_trailing=resume
    )
    existing = writer.records()
    completed: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for row in existing:
        key = _key(row)
        if key in completed:
            raise RuntimeError("duplicate baseline job key")
        completed[key] = row

    smoke._atomic_json(
        run_dir / "run_config.json",
        {
            "schema_version": 1,
            "paper_eligible": True,
            "protocol_hash": protocol["protocol_hash"],
            "matrix_sha256": manifest["matrix_sha256"],
            "family": family,
            "model": model,
            "planner_name": planner_name,
            "planner_seed": int(planner_seed),
            "planner_budget": {
                "max_evaluations": PLANNER_SPECS[
                    planner_name
                ].max_evaluations,
                "time_limit_s": PLANNER_SPECS[
                    planner_name
                ].time_limit_s,
            },
            "environment_config_hash": environment_config_hash,
            "expected_rows": len(rows),
        },
    )
    new_rows = 0
    started_run = time.time()
    for matrix_row in sorted(rows, key=lambda row: str(row["task_id"])):
        key = _key(matrix_row)
        if key in completed:
            continue
        if max_new_rows is not None and new_rows >= max_new_rows:
            break
        task = tasks[str(matrix_row["task_id"])]
        context = provider(task)
        cfg = ppo.resolve_config(
            {**base_cfg, **dict(context["cfg_overrides"])}
        )
        scenario_cfg, wind = ppo.apply_frozen_domain_instance(
            cfg, context["wind_data"], task
        )
        problem = make_problem(
            context["start_pos"],
            np.asarray(task["inspection_points_xyz"], dtype=np.float32),
            np.asarray(task["priorities"], dtype=np.float32),
            context["terrain"],
            scenario_cfg,
            wind,
            name=str(task["id"]),
        )
        result = run_planner(
            planner_name,
            problem,
            seed=int(planner_seed),
        )
        route_payload = smoke._jsonable(
            {
                "schema_version": 1,
                "protocol_hash": protocol["protocol_hash"],
                "matrix_row": matrix_row,
                "task_hash": task["task_hash"],
                "environment_config_hash": environment_config_hash,
                "result": result.as_dict(),
            }
        )
        route_hash = smoke._canonical_hash(route_payload)
        smoke._atomic_json(
            run_dir
            / "routes"
            / f"{str(task['id']).replace(':', '_')}.json",
            route_payload,
        )
        row = _result_row(
            matrix_row=matrix_row,
            task=task,
            result=result,
            route_hash=route_hash,
            environment_config_hash=environment_config_hash,
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
                "model": model,
                "planner_seed": int(planner_seed),
                "completed": len(completed),
                "total": len(rows),
                "last_key": list(key),
                "elapsed_s": time.time() - started_run,
            },
        )

    state = "completed" if len(completed) == len(rows) else "partial"
    smoke._atomic_json(
        run_dir / "status.json",
        {
            "state": state,
            "family": family,
            "model": model,
            "planner_seed": int(planner_seed),
            "completed": len(completed),
            "total": len(rows),
            "new_rows": new_rows,
            "elapsed_s": time.time() - started_run,
        },
    )
    return {
        "state": state,
        "family": family,
        "model": model,
        "planner_seed": int(planner_seed),
        "completed": len(completed),
        "total": len(rows),
        "new_rows": new_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=tuple(FAMILIES), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--planner-seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-rows", type=int)
    args = parser.parse_args(argv)
    report = run(
        family=args.family,
        model=str(args.model),
        planner_seed=int(args.planner_seed),
        resume=bool(args.resume),
        max_new_rows=args.max_new_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

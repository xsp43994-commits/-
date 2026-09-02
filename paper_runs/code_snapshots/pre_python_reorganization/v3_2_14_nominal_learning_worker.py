#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execute one frozen v3.2.14 nominal learning-family matrix."""

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
POST_STATUS = OUTPUT / "formal_evaluation/post_generation_status.json"
SYNTHETIC = OUTPUT / "manifests/synthetic_test/records.jsonl"
REAL = OUTPUT / "formal_evaluation/real_tasks_parallel/records.jsonl"
REGISTRIES = (
    MAP_ROOT / "procedural/synthetic_test/map_registry.json",
    MAP_ROOT / "real/map_registry.json",
)
EXPECTED = {"synthetic_learning": 7560, "real_learning": 5040}


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["family"]),
        str(row["task_id"]),
        str(row["model"]),
        int(row["training_seed"]),
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


def _result(
    *,
    matrix_row: Mapping[str, Any],
    task: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    metrics: Mapping[str, Any],
    route_hash: str,
    planning_time_s: float,
    protocol_hash: str,
) -> Dict[str, Any]:
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
        "training_seed": int(matrix_row["training_seed"]),
        "planner_seed": None,
        "checkpoint_hash": str(checkpoint["checkpoint_sha256"]),
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
        "planning_time_s": float(planning_time_s),
        "oracle_lower": float(
            certificate["weighted_coverage_lower_bound"]
        ),
        "oracle_upper": float(
            certificate["weighted_coverage_upper_bound"]
        ),
    }
    for value in row.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("formal learning result contains non-finite data")
    row["result_hash"] = smoke._canonical_hash(row)
    return row


def run(
    *,
    family: str,
    device: str,
    resume: bool,
    max_new_rows: int | None,
    shard_index: int,
    shard_count: int,
) -> Dict[str, Any]:
    if family not in EXPECTED:
        raise ValueError(f"unsupported family {family!r}")
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    manifest = json.loads(MATRIX_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest["protocol_hash"] != protocol["protocol_hash"]
        or manifest["matrix_sha256"] != v32._sha256_file(MATRIX)
        or int(manifest["row_count"]) != 21648
    ):
        raise RuntimeError("frozen evaluation matrix identity mismatch")
    task_rows = v32._read_jsonl(SYNTHETIC) + v32._read_jsonl(REAL)
    tasks = {str(row["id"]): row for row in task_rows}
    matrix_rows = [
        row
        for row in v32._read_jsonl(MATRIX)
        if str(row["family"]) == family
    ]
    if len(matrix_rows) != EXPECTED[family]:
        raise RuntimeError("formal family row count mismatch")
    for row in matrix_rows:
        task = tasks.get(str(row["task_id"]))
        if task is None or str(task["task_hash"]) != str(row["task_hash"]):
            raise RuntimeError("formal matrix task identity mismatch")
        if (
            row.get("planner_seed") is not None
            or row.get("training_seed") is None
            or str(row.get("condition")) != "nominal"
        ):
            raise RuntimeError("nominal learning matrix semantics mismatch")

    catalog = v32.checkpoint_catalog(PROTOCOL, output_root=OUTPUT)
    checkpoints = {
        (str(row["variant"]), int(row["training_seed"])): row
        for row in catalog["rows"]
    }
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid shard index/count")
    family_identities = sorted(
        {
            (str(row["model"]), int(row["training_seed"]))
            for row in matrix_rows
        }
    )
    shard_identities = {
        identity
        for position, identity in enumerate(family_identities)
        if position % shard_count == shard_index
    }
    matrix_rows = [
        row
        for row in matrix_rows
        if (str(row["model"]), int(row["training_seed"]))
        in shard_identities
    ]
    shard_expected = len(matrix_rows)
    if shard_expected < 1:
        raise RuntimeError("formal learning shard is empty")
    provider = multimap.FrozenMapProvider.from_registries(
        MAP_ROOT, REGISTRIES
    )
    base_dir = OUTPUT / "formal_evaluation" / "results" / family
    run_dir = (
        base_dir
        if shard_count == 1
        else base_dir
        / "shards"
        / f"shard_{shard_index:02d}_of_{shard_count:02d}"
    )
    results_path = run_dir / "results.jsonl"
    if results_path.exists() and not resume:
        raise FileExistsError("formal family output exists; use --resume")
    writer = paper.DurableResultJsonlWriter(
        results_path, resume=resume, repair_trailing=resume
    )
    existing = writer.records()
    completed: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for row in existing:
        key = _key(row)
        if key in completed:
            raise RuntimeError("duplicate formal result key")
        completed[key] = row

    ordered = sorted(
        matrix_rows,
        key=lambda row: (
            str(row["model"]),
            int(row["training_seed"]),
            str(row["task_id"]),
        ),
    )
    smoke._atomic_json(
        run_dir / "run_config.json",
        {
            "schema_version": 1,
            "paper_eligible": True,
            "family": family,
            "expected_rows": shard_expected,
            "global_expected_rows": EXPECTED[family],
            "shard_index": shard_index,
            "shard_count": shard_count,
            "checkpoint_identities": [
                list(identity) for identity in sorted(shard_identities)
            ],
            "protocol_hash": protocol["protocol_hash"],
            "matrix_sha256": manifest["matrix_sha256"],
            "checkpoint_catalog_hash": catalog["catalog_hash"],
            "provider_hash": provider.provider_hash,
            "deterministic_decode": True,
            "device": device,
        },
    )
    new_rows = 0
    active_identity: tuple[str, int] | None = None
    model: Any = None
    payload: Mapping[str, Any] | None = None
    checkpoint: Mapping[str, Any] | None = None
    started_run = time.time()
    formal_start_marked = bool(existing)
    try:
        for matrix_row in ordered:
            key = _key(matrix_row)
            if key in completed:
                continue
            if max_new_rows is not None and new_rows >= max_new_rows:
                break
            if not formal_start_marked:
                post_status = json.loads(
                    POST_STATUS.read_text(encoding="utf-8")
                )
                if (
                    post_status.get("state") != "completed"
                    or int(post_status.get("matrix_row_count", -1)) != 21648
                    or post_status.get("matrix_sha256")
                    != manifest["matrix_sha256"]
                ):
                    raise RuntimeError(
                        "formal evaluation start gate is not complete"
                    )
                if not bool(
                    post_status.get(
                        "formal_algorithm_evaluation_started", False
                    )
                ):
                    post_status[
                        "formal_algorithm_evaluation_started"
                    ] = True
                    post_status[
                        "formal_algorithm_evaluation_started_at_unix"
                    ] = time.time()
                    post_status["first_formal_family"] = family
                    smoke._atomic_json(POST_STATUS, post_status)
                formal_start_marked = True
            identity = (
                str(matrix_row["model"]),
                int(matrix_row["training_seed"]),
            )
            if identity != active_identity:
                model = None
                payload = None
                checkpoint = checkpoints.get(identity)
                if checkpoint is None:
                    raise RuntimeError(f"checkpoint missing for {identity}")
                path = Path(str(checkpoint["checkpoint_path"]))
                if v32._sha256_file(path) != checkpoint["checkpoint_sha256"]:
                    raise RuntimeError(f"checkpoint hash drift for {identity}")
                model, payload = ppo.load_checkpoint(
                    path, map_location=device
                )
                if str(payload.get("checkpoint_kind")) != "best_safe":
                    raise RuntimeError("formal evaluation requires best_safe.pt")
                active_identity = identity
            task = tasks[str(matrix_row["task_id"])]
            context = provider(task)
            cfg = ppo.resolve_config(
                {
                    **dict(payload["cfg"]),
                    **dict(context["cfg_overrides"]),
                }
            )
            scenario_cfg, wind = ppo.apply_frozen_domain_instance(
                cfg, context["wind_data"], task
            )
            started = time.perf_counter()
            detail = ppo.plan_with_policy_improved(
                model,
                context["start_pos"],
                np.asarray(
                    task["inspection_points_xyz"], dtype=np.float32
                ),
                np.asarray(task["priorities"], dtype=np.float32),
                context["terrain"],
                scenario_cfg,
                wind,
                return_details=True,
                decode_mode="deterministic",
            )
            elapsed = time.perf_counter() - started
            route_payload = {
                "schema_version": 1,
                "protocol_hash": protocol["protocol_hash"],
                "matrix_row": matrix_row,
                "task_hash": task["task_hash"],
                "checkpoint_hash": checkpoint["checkpoint_sha256"],
                "detail": detail,
            }
            route_payload = smoke._jsonable(route_payload)
            route_hash = smoke._canonical_hash(route_payload)
            route_name = (
                f"{identity[0]}__seed{identity[1]}__"
                f"{str(task['id']).replace(':', '_')}.json"
            )
            smoke._atomic_json(
                run_dir / "routes" / route_name, route_payload
            )
            result = _result(
                matrix_row=matrix_row,
                task=task,
                checkpoint=checkpoint,
                metrics=dict(detail["metrics"]),
                route_hash=route_hash,
                planning_time_s=elapsed,
                protocol_hash=str(protocol["protocol_hash"]),
            )
            writer.append(result)
            completed[key] = result
            new_rows += 1
            if new_rows == 1 or new_rows % 10 == 0:
                smoke._atomic_json(
                    run_dir / "status.json",
                    {
                        "state": "running",
                        "family": family,
                        "completed": len(completed),
                        "total": shard_expected,
                        "global_total": EXPECTED[family],
                        "shard_index": shard_index,
                        "shard_count": shard_count,
                        "last_key": list(key),
                        "elapsed_s": time.time() - started_run,
                    },
                )
    finally:
        model = None
        payload = None
        checkpoint = None

    state = (
        "completed"
        if len(completed) == shard_expected
        else "partial"
    )
    smoke._atomic_json(
        run_dir / "status.json",
        {
            "state": state,
            "family": family,
            "completed": len(completed),
            "total": shard_expected,
            "global_total": EXPECTED[family],
            "shard_index": shard_index,
            "shard_count": shard_count,
            "new_rows": new_rows,
            "elapsed_s": time.time() - started_run,
        },
    )
    return {
        "state": state,
        "family": family,
        "completed": len(completed),
        "total": shard_expected,
        "global_total": EXPECTED[family],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "new_rows": new_rows,
        "results_path": str(results_path.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=tuple(EXPECTED),
        required=True,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-rows", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    report = run(
        family=args.family,
        device=args.device,
        resume=bool(args.resume),
        max_new_rows=args.max_new_rows,
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation-only end-to-end smoke test for the frozen v3.2.14 evaluator.

This script never reads the formal synthetic/real task manifests and all
outputs are marked ``paper_eligible=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from uav_inspection.core import final_python_ppo_pointer as ppo
from uav_inspection.experiments import paper_multimap_experiments as multimap
from uav_inspection.experiments import paper_v3_2_experiments as v32
from python_classical_algs import run_planner
from python_classical_algs.common import make_problem


ROOT = WORKSPACE_ROOT
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/protocol.json"
)
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14"
MAP_ROOT = ROOT / "map_data/multimap_v3_1"
VALIDATION_RECORDS = (
    ROOT / "paper_runs/multimap_v3_1/manifests/validation/records.jsonl"
)
VALIDATION_REGISTRY = (
    MAP_ROOT / "procedural/validation/map_registry.json"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        newline="\n",
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    try:
        # Windows 上杀毒扫描或监控进程可能短暂占用状态文件。
        # 仅重试原子替换，不改变结果内容或评价语义。
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(
            _jsonable(row),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )


def _safe_metrics(metrics: Mapping[str, Any]) -> tuple[bool, float]:
    safe = bool(metrics.get("returned")) and not any(
        bool(metrics.get(field, False))
        for field in (
            "energy_violation",
            "distance_violation",
            "time_violation",
            "dynamics_violation",
        )
    )
    coverage = float(metrics.get("weighted_coverage", 0.0)) if safe else 0.0
    return safe, coverage


def _result_row(
    *,
    record: Mapping[str, Any],
    model: str,
    training_seed: int | None,
    planner_seed: int | None,
    checkpoint_hash: str,
    metrics: Mapping[str, Any],
    planning_time_s: float,
    route_hash: str,
    protocol_hash: str,
) -> Dict[str, Any]:
    safe, coverage = _safe_metrics(metrics)
    nominal_hash = str(record["task_hash"])
    row = {
        "schema_version": 1,
        "paper_eligible": False,
        "stage": "validation_evaluation_smoke",
        "protocol_hash": protocol_hash,
        "task_id": str(record["id"]),
        "task_hash": nominal_hash,
        "map_id": str(record["map_id"]),
        "road_index": record.get("road_index"),
        "task_index": int(record["task_index"]),
        "model": model,
        "training_seed": training_seed,
        "planner_seed": planner_seed,
        "checkpoint_hash": checkpoint_hash,
        "condition": "nominal",
        "perturbation_layer": "none",
        "perturbation_type": "none",
        "nominal_input_hash": nominal_hash,
        "observed_input_hash": nominal_hash,
        "execution_truth_hash": nominal_hash,
        "route_hash": route_hash,
        "safe": safe,
        "safe_weighted_coverage": coverage,
        "weighted_coverage": float(metrics.get("weighted_coverage", 0.0)),
        "returned": bool(metrics.get("returned", False)),
        "visited_count": int(metrics.get("visited_count", 0)),
        "termination_reason": str(metrics.get("termination_reason", "unknown")),
        "energy_wh": float(metrics.get("energy_wh", 0.0)),
        "distance_m": float(metrics.get("distance_m", 0.0)),
        "time_s": float(metrics.get("time_s", 0.0)),
        "min_remaining_soc": float(metrics.get("min_remaining_soc", 0.0)),
        "planning_time_s": float(planning_time_s),
    }
    numeric = [
        value
        for value in row.values()
        if isinstance(value, float) and not isinstance(value, bool)
    ]
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError("validation smoke produced a non-finite result")
    row["result_hash"] = _canonical_hash(row)
    return row


def _selected_validation_records() -> list[Dict[str, Any]]:
    rows = _read_jsonl(VALIDATION_RECORDS)
    selected = []
    for node_count in (16, 20, 24):
        candidates = sorted(
            (
                row
                for row in rows
                if int(row.get("node_count", -1)) == node_count
            ),
            key=lambda row: (str(row["map_id"]), str(row["task_hash"])),
        )
        if not candidates:
            raise RuntimeError(f"validation lacks N={node_count}")
        selected.append(dict(candidates[0]))
    return selected


def run(
    *,
    output_root: Path,
    device: str,
    resume: bool,
    max_new_rows: int | None,
) -> Dict[str, Any]:
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    catalog = v32.checkpoint_catalog(PROTOCOL, output_root=OUTPUT)
    records = _selected_validation_records()
    provider = multimap.FrozenMapProvider.from_registries(
        MAP_ROOT, [VALIDATION_REGISTRY]
    )
    run_dir = Path(output_root) / "smoke" / "validation_evaluation_v1"
    results_path = run_dir / "results.jsonl"
    existing = _read_jsonl(results_path) if resume else []
    if results_path.exists() and not resume:
        raise FileExistsError("smoke output exists; use --resume")
    completed: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for row in existing:
        key = (
            row["task_id"],
            row["model"],
            row.get("training_seed"),
            row.get("planner_seed"),
        )
        if key in completed:
            raise RuntimeError("duplicate smoke result key")
        completed[key] = row

    seed = int(protocol["formal_evaluation"]["training_seeds"][0])
    checkpoint_rows = [
        row
        for row in catalog["rows"]
        if int(row["training_seed"]) == seed
    ]
    tasks = [
        ("learning", row, record)
        for row in checkpoint_rows
        for record in records
    ]
    tasks.extend(
        ("baseline", {"variant": "priority_resource_greedy"}, record)
        for record in records
    )
    if len(tasks) != 24:
        raise RuntimeError("validation smoke must contain exactly 24 rows")

    _atomic_json(
        run_dir / "run_config.json",
        {
            "schema_version": 1,
            "paper_eligible": False,
            "protocol_hash": protocol["protocol_hash"],
            "checkpoint_catalog_hash": catalog["catalog_hash"],
            "validation_records_sha256": v32._sha256_file(
                VALIDATION_RECORDS
            ),
            "validation_provider_hash": provider.provider_hash,
            "expected_rows": 24,
            "device": device,
        },
    )
    new_rows = 0
    current_model_path: str | None = None
    current_model: Any = None
    current_payload: Mapping[str, Any] | None = None
    try:
        for kind, identity, record in tasks:
            model_name = str(identity["variant"])
            training_seed = seed if kind == "learning" else None
            planner_seed = None if kind == "learning" else 42
            key = (
                str(record["id"]),
                model_name,
                training_seed,
                planner_seed,
            )
            if key in completed:
                continue
            if max_new_rows is not None and new_rows >= max_new_rows:
                break
            context = provider(record)
            if kind == "learning":
                checkpoint = str(identity["checkpoint_path"])
                if checkpoint != current_model_path:
                    current_model = None
                    current_payload = None
                    current_model, current_payload = ppo.load_checkpoint(
                        checkpoint, map_location=device
                    )
                    if (
                        str(current_payload.get("checkpoint_kind"))
                        != "best_safe"
                    ):
                        raise RuntimeError("formal smoke requires best_safe.pt")
                    current_model_path = checkpoint
                cfg = ppo.resolve_config(
                    {
                        **dict(current_payload["cfg"]),
                        **dict(context["cfg_overrides"]),
                    }
                )
                scenario_cfg, wind = ppo.apply_frozen_domain_instance(
                    cfg, context["wind_data"], record
                )
                started = time.perf_counter()
                detail = ppo.plan_with_policy_improved(
                    current_model,
                    context["start_pos"],
                    np.asarray(
                        record["inspection_points_xyz"], dtype=np.float32
                    ),
                    np.asarray(record["priorities"], dtype=np.float32),
                    context["terrain"],
                    scenario_cfg,
                    wind,
                    return_details=True,
                    decode_mode="deterministic",
                )
                elapsed = time.perf_counter() - started
                metrics = dict(detail["metrics"])
                route_payload = {
                    "task": record,
                    "model": model_name,
                    "training_seed": seed,
                    "detail": detail,
                }
                checkpoint_hash = str(identity["checkpoint_sha256"])
            else:
                # 基线采用完整模型检查点中的冻结环境配置，不读取其策略输出。
                full = next(
                    item
                    for item in checkpoint_rows
                    if item["variant"] == "full"
                )
                _, payload = ppo.load_checkpoint(
                    full["checkpoint_path"], map_location="cpu"
                )
                cfg = ppo.resolve_config(
                    {
                        **dict(payload["cfg"]),
                        **dict(context["cfg_overrides"]),
                    }
                )
                scenario_cfg, wind = ppo.apply_frozen_domain_instance(
                    cfg, context["wind_data"], record
                )
                problem = make_problem(
                    context["start_pos"],
                    np.asarray(
                        record["inspection_points_xyz"], dtype=np.float32
                    ),
                    np.asarray(record["priorities"], dtype=np.float32),
                    context["terrain"],
                    scenario_cfg,
                    wind,
                    name=str(record["id"]),
                )
                result = run_planner(model_name, problem, seed=42)
                elapsed = float(result.runtime_s)
                metrics = dict(result.metrics)
                route_payload = {
                    "task": record,
                    "model": model_name,
                    "planner_seed": 42,
                    "result": result.as_dict(),
                }
                checkpoint_hash = ""
            route_payload = _jsonable(route_payload)
            route_hash = _canonical_hash(route_payload)
            route_name = "__".join(
                (
                    model_name,
                    (
                        f"train{training_seed}"
                        if training_seed is not None
                        else f"plan{planner_seed}"
                    ),
                    str(record["id"]),
                )
            )
            _atomic_json(
                run_dir / "routes" / f"{route_name}.json", route_payload
            )
            row = _result_row(
                record=record,
                model=model_name,
                training_seed=training_seed,
                planner_seed=planner_seed,
                checkpoint_hash=checkpoint_hash,
                metrics=metrics,
                planning_time_s=elapsed,
                route_hash=route_hash,
                protocol_hash=str(protocol["protocol_hash"]),
            )
            existing.append(row)
            completed[key] = row
            _atomic_text(results_path, _jsonl(existing))
            new_rows += 1
            _atomic_json(
                run_dir / "status.json",
                {
                    "state": "running",
                    "completed": len(existing),
                    "total": len(tasks),
                    "last_key": list(key),
                    "paper_eligible": False,
                },
            )
    finally:
        current_model = None
        current_payload = None

    state = "completed" if len(existing) == len(tasks) else "partial"
    _atomic_json(
        run_dir / "status.json",
        {
            "state": state,
            "completed": len(existing),
            "total": len(tasks),
            "paper_eligible": False,
        },
    )
    return {
        "state": state,
        "completed": len(existing),
        "total": len(tasks),
        "new_rows": new_rows,
        "run_dir": str(run_dir.resolve()),
        "paper_eligible": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-rows", type=int)
    args = parser.parse_args(argv)
    report = run(
        output_root=args.output_root,
        device=args.device,
        resume=bool(args.resume),
        max_new_rows=args.max_new_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

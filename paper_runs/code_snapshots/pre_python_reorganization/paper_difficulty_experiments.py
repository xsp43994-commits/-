#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""困难约束纠偏实验：场景认证、现有检查点资格验证与分阶段门禁。

本入口不会修改旧正式实验。正式 test 的生成在训练协议冻结前被显式禁止。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

import final_python_ppo_pointer as ppo
import paper_experiments as legacy
from python_classical_algs.common import PlannerBudget, build_context
from python_classical_algs.greedy import (
    plan_nearest_feasible,
    plan_priority_resource_greedy,
)
from python_classical_algs.milp import plan_milp_orienteering
from python_classical_algs import planner_names, run_planner


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    ROOT / "paper_runs" / "protocols" / "difficulty_test_v2_1" / "protocol.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "paper_runs" / "difficulty_v2_1"
OLD_FROZEN_PROTOCOL = (
    ROOT / "paper_runs" / "protocols" / "frozen_test_v1" / "protocol.json"
)
VIOLATION_FIELDS = (
    "energy_violation",
    "distance_violation",
    "time_violation",
    "dynamics_violation",
)


def _canonical_hash(payload: Mapping[str, Any], excluded: Sequence[str] = ()) -> str:
    normalized = {key: value for key, value in payload.items() if key not in excluded}
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}必须是JSON对象。")
    return payload


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}必须是JSON对象。")
            rows.append(value)
    return rows


def _all_numeric_values_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_values_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> Dict[str, Any]:
    protocol = _read_json(path)
    actual = _canonical_hash(protocol, excluded=("protocol_hash",))
    expected = str(protocol.get("protocol_hash", ""))
    if actual != expected:
        raise RuntimeError(
            f"困难实验协议哈希不一致：expected={expected}, actual={actual}"
        )
    if float(protocol.get("power_scale", -1.0)) != 1.0:
        raise RuntimeError("困难实验必须保持名义功率power_scale=1.0。")
    if "power_sensitivity" not in set(protocol.get("forbidden", ())):
        raise RuntimeError("困难实验协议必须继续禁止功率敏感性。")
    return protocol


def _seed_for(master_seed: int, *parts: Any) -> int:
    text = "|".join([str(master_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def _road_arms(scenario: Any) -> List[np.ndarray]:
    start = np.asarray(scenario.start_pos, dtype=np.float64)
    arms: List[np.ndarray] = []
    for road in (scenario.road_1, scenario.road_2):
        road_arr = np.asarray(road, dtype=np.float64)
        index = int(np.argmin(np.linalg.norm(road_arr[:, :2] - start[:2], axis=1)))
        for raw_path in (road_arr[: index + 1][::-1], road_arr[index:]):
            path = raw_path.copy()
            if np.linalg.norm(path[0, :2] - start[:2]) > 1e-7:
                path = np.vstack([start, path])
            else:
                path[0] = start
            arms.append(path)
    if len(arms) != 4:
        raise RuntimeError(f"预期4条道路分支，实际{len(arms)}。")
    return arms


def _path_lengths(path: np.ndarray, coordinate_scale: float) -> np.ndarray:
    return np.concatenate(
        (
            [0.0],
            np.cumsum(
                np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
                * float(coordinate_scale)
            ),
        )
    )


def _interpolate_path(
    path: np.ndarray,
    distances_m: np.ndarray,
    coordinate_scale: float,
) -> np.ndarray:
    cumulative = _path_lengths(path, coordinate_scale)
    clipped = np.clip(distances_m, 0.0, cumulative[-1])
    return np.column_stack(
        [
            np.interp(clipped, cumulative, path[:, dimension])
            for dimension in range(3)
        ]
    )


def _priority_counts(node_count: int) -> Tuple[int, int, int]:
    raw = np.asarray([5.0, 6.0, 5.0], dtype=np.float64) * node_count / 16.0
    counts = np.floor(raw).astype(int)
    remainder = node_count - int(np.sum(counts))
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _priority_layout(
    node_count: int,
    arm_ids: np.ndarray,
    distances_m: np.ndarray,
    layout: str,
    rng: np.random.Generator,
) -> np.ndarray:
    high_count, medium_count, _ = _priority_counts(node_count)
    priorities = np.ones(node_count, dtype=np.int32)
    jitter = rng.uniform(0.0, 1e-6, node_count)
    normalized_distance = distances_m / max(float(np.max(distances_m)), 1.0)

    if layout == "clustered":
        cluster_arm = int(rng.integers(0, 4))
        high_score = (
            (arm_ids == cluster_arm).astype(np.float64) * 2.0
            + normalized_distance
            + jitter
        )
        high_order = np.argsort(-high_score, kind="stable")
    elif layout == "dispersed":
        # 先保证四条分支都出现高优先级，再按远端程度补齐。
        high_order_list: List[int] = []
        for arm in range(4):
            indices = np.flatnonzero(arm_ids == arm)
            indices = indices[np.argsort(-distances_m[indices], kind="stable")]
            high_order_list.extend(int(value) for value in indices[:1])
        remaining = [
            int(value)
            for value in np.argsort(-(normalized_distance + jitter), kind="stable")
            if int(value) not in high_order_list
        ]
        high_order = np.asarray(
            (high_order_list + remaining)[:node_count], dtype=np.int64
        )
    elif layout == "far_high_conflict":
        high_order = np.argsort(-(normalized_distance + jitter), kind="stable")
    else:
        raise ValueError(f"未知优先级布局：{layout}")

    high_indices = high_order[:high_count]
    priorities[high_indices] = 3
    remaining_indices = np.asarray(
        [index for index in range(node_count) if index not in set(high_indices)],
        dtype=np.int64,
    )
    if layout == "far_high_conflict":
        medium_order = remaining_indices[
            np.argsort(distances_m[remaining_indices], kind="stable")
        ]
    else:
        medium_order = remaining_indices[
            np.argsort(
                -(0.35 * normalized_distance[remaining_indices] + jitter[remaining_indices]),
                kind="stable",
            )
        ]
    priorities[medium_order[:medium_count]] = 2
    return priorities


def _candidate_record(
    scenario: Any,
    protocol: Mapping[str, Any],
    *,
    split: str,
    node_count: int,
    difficulty: str,
    constraint_type: str,
    priority_layout: str,
    replicate: int,
    attempt: int,
    master_seed: int,
) -> Dict[str, Any]:
    seed = _seed_for(
        master_seed,
        split,
        node_count,
        difficulty,
        constraint_type,
        priority_layout,
        replicate,
        attempt,
    )
    rng = np.random.default_rng(seed)
    arms = _road_arms(scenario)
    per_arm = int(node_count) // 4
    if per_arm * 4 != int(node_count):
        raise ValueError("困难场景节点数必须能被四条道路分支均分。")
    coordinate_scale = float(scenario.coordinate_scale_m_per_unit)
    extent_low, extent_high = protocol["candidate_ranges"][
        "radial_extent_fraction"
    ]
    extent_fraction = float(rng.uniform(extent_low, extent_high))
    points: List[np.ndarray] = []
    arm_ids: List[int] = []
    distances: List[float] = []
    for arm_id, arm in enumerate(arms):
        arm_length = float(_path_lengths(arm, coordinate_scale)[-1])
        usable_extent = extent_fraction * arm_length
        edges = np.linspace(0.12 * usable_extent, usable_extent, per_arm + 1)
        sampled = np.asarray(
            [
                rng.uniform(edges[index], edges[index + 1])
                for index in range(per_arm)
            ],
            dtype=np.float64,
        )
        sampled.sort()
        points.extend(_interpolate_path(arm, sampled, coordinate_scale))
        arm_ids.extend([arm_id] * per_arm)
        distances.extend(float(value) for value in sampled)

    point_array = np.asarray(points, dtype=np.float32)
    arm_array = np.asarray(arm_ids, dtype=np.int32)
    distance_array = np.asarray(distances, dtype=np.float64)
    priorities = _priority_layout(
        int(node_count), arm_array, distance_array, priority_layout, rng
    )
    ranges = protocol["difficulty_candidate_ranges"][difficulty][constraint_type]
    offsets = protocol["node_count_budget_offsets"][str(node_count)]
    initial_soc = float(rng.uniform(*ranges["initial_soc"]))
    distance_scale = float(rng.uniform(*ranges["distance_budget_scale"]))
    time_scale = float(rng.uniform(*ranges["time_budget_scale"]))
    initial_soc = min(1.0, initial_soc + float(offsets["initial_soc"]))
    distance_scale = min(
        1.2, distance_scale + float(offsets["distance_budget_scale"])
    )
    time_scale = min(1.2, time_scale + float(offsets["time_budget_scale"]))
    wind_scale = float(rng.uniform(*protocol["candidate_ranges"]["wind_scale"]))
    vertical = float(
        rng.uniform(*protocol["candidate_ranges"]["wind_vertical_bias_mps"])
    )
    identifier = (
        f"{split}__n{node_count}__{difficulty}__{constraint_type}__"
        f"{priority_layout}__r{replicate:02d}"
    )
    return {
        "id": identifier,
        "split": split,
        "replicate_id": int(replicate),
        "instance_seed": int(seed % (2**32)),
        "node_count": int(node_count),
        "difficulty": difficulty,
        "constraint_type": constraint_type,
        "priority_layout": priority_layout,
        "generation_attempt": int(attempt),
        "radial_extent_fraction": extent_fraction,
        "inspection_points_xyz": point_array.tolist(),
        "point_arm_ids": arm_array.tolist(),
        "point_along_arm_distances_m": distance_array.tolist(),
        "priorities": priorities.tolist(),
        "service_times_s": np.full(int(node_count), 20.0).tolist(),
        "initial_soc": initial_soc,
        "distance_budget_scale": distance_scale,
        "time_budget_scale": time_scale,
        "wind_scale": wind_scale,
        "wind_rotation_deg": float(rng.uniform(-15.0, 15.0)),
        "wind_vertical_bias_mps": vertical,
        "power_scale": 1.0,
    }


def _problem_from_record(
    scenario_file: Path,
    scenario: Any,
    base_cfg: Mapping[str, Any],
    record: Mapping[str, Any],
) -> Any:
    cfg = legacy._instance_cfg(base_cfg, scenario, record, 1.0)
    cfg.update(
        {
            "inspection_points_xyz": record["inspection_points_xyz"],
            "priorities": record["priorities"],
            "service_times_s": record["service_times_s"],
            "wind_data": legacy._transform_wind(scenario.wind_data, record),
            "id": record["id"],
            "split": record["split"],
            "instance_seed": record["instance_seed"],
            "node_count": record["node_count"],
        }
    )
    return build_context(scenario_file, cfg)


def _resource_bottlenecks(
    metrics: Mapping[str, Any],
    *,
    minimum: float,
    max_gap: float,
) -> Tuple[str, ...]:
    values = {
        "energy": float(metrics["energy_utilization"]),
        "distance": float(metrics["distance_utilization"]),
        "time": float(metrics["time_utilization"]),
    }
    maximum = max(values.values())
    return tuple(
        name
        for name, value in values.items()
        if value >= minimum and maximum - value <= max_gap
    )


def certify_record(
    scenario_file: Path,
    scenario: Any,
    base_cfg: Mapping[str, Any],
    protocol: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    time_limit_s: Optional[float] = None,
) -> Tuple[bool, Dict[str, Any], str]:
    certification = protocol["certification"]
    problem = _problem_from_record(scenario_file, scenario, base_cfg, record)
    result = plan_milp_orienteering(
        problem,
        seed=42,
        budget=PlannerBudget(
            max_evaluations=None,
            time_limit_s=float(
                time_limit_s
                if time_limit_s is not None
                else certification["time_limit_s"]
            ),
        ),
        params={
            "objective_mode": "weighted_coverage",
            "mip_rel_gap": float(certification["mip_rel_gap"]),
            "presolve": True,
        },
    )
    metadata = result.metadata
    lower = metadata.get("weighted_coverage_incumbent")
    upper = metadata.get("weighted_coverage_upper_bound")
    gap = metadata.get("mip_gap")
    metrics = dict(result.metrics)
    certificate: Dict[str, Any] = {
        "algorithm": "milp_weighted_coverage",
        "solver_status": metadata.get("solver_status"),
        "solver_success": metadata.get("solver_success"),
        "solver_message": metadata.get("solver_message"),
        "status": result.status,
        "mip_gap": gap,
        "weighted_coverage_lower_bound": lower,
        "weighted_coverage_upper_bound": upper,
        "optimality_certified": bool(metadata.get("optimality_certified")),
        "visit_order": list(result.visit_order),
        "visited_count": int(metrics.get("visited_count", len(result.visit_order))),
        "returned": bool(metrics.get("returned", False)),
        "energy_utilization": float(metrics.get("energy_utilization", math.inf)),
        "distance_utilization": float(metrics.get("distance_utilization", math.inf)),
        "time_utilization": float(metrics.get("time_utilization", math.inf)),
        "runtime_s": float(result.runtime_s),
        "scenario_hash": str(result.scenario_hash),
    }
    if lower is None or upper is None or gap is None:
        return False, certificate, "missing_solver_bound"
    lower_value, upper_value, gap_value = float(lower), float(upper), float(gap)
    if not all(math.isfinite(value) for value in (lower_value, upper_value, gap_value)):
        return False, certificate, "nonfinite_solver_bound"
    if lower_value > upper_value + 1e-7:
        return False, certificate, "inverted_solver_bounds"
    if not certificate["returned"] or certificate["visited_count"] < int(
        certification["minimum_visited_count"]
    ):
        return False, certificate, "no_safe_partial_route"
    band_low, band_high = (
        float(value)
        for value in protocol["difficulty_bands"][str(record["difficulty"])]
    )
    tolerance = float(certification["band_tolerance"])
    # 上下界必须共同落在预注册区间；这样无需查看任何候选算法结果。
    if lower_value < band_low - tolerance or lower_value > band_high + tolerance:
        return False, certificate, "incumbent_outside_band"
    if upper_value >= float(certification["full_coverage_upper_bound_max"]) - tolerance:
        return False, certificate, "full_coverage_not_excluded"
    same_band = upper_value <= band_high + tolerance
    small_gap = gap_value <= float(certification["mip_rel_gap"]) + 1e-10
    certificate["difficulty_certificate"] = (
        "bounds_same_band" if same_band else "mip_gap"
    )
    if not (same_band or small_gap):
        return False, certificate, "bounds_and_gap_insufficient"

    bottlenecks = _resource_bottlenecks(
        metrics,
        minimum=float(certification["bottleneck_utilization_min"]),
        max_gap=float(certification["single_bottleneck_max_gap"]),
    )
    certificate["bottleneck_resources"] = list(bottlenecks)
    intended = str(record["constraint_type"])
    if intended == "mixed":
        if len(bottlenecks) < int(certification["mixed_min_active_resources"]):
            return False, certificate, "mixed_bottleneck_not_active"
    elif intended not in bottlenecks:
        return False, certificate, "intended_bottleneck_not_active"
    return True, certificate, "accepted"


def _cell_product(protocol: Mapping[str, Any]) -> Iterable[Tuple[int, str, str, str]]:
    for node_count in protocol["node_counts"]:
        for difficulty in protocol["difficulty_bands"]:
            for constraint_type in protocol["constraint_types"]:
                for layout in protocol["priority_layouts"]:
                    yield int(node_count), str(difficulty), str(constraint_type), str(layout)


def _manifest_hash(metadata: Mapping[str, Any], records_hash: str) -> str:
    payload = {key: value for key, value in metadata.items() if key != "manifest_hash"}
    payload["records_sha256"] = records_hash
    return _canonical_hash(payload)


def generate_split(
    protocol_path: Path,
    output_root: Path,
    split: str,
    *,
    resume_existing: bool = False,
    dry_run: bool = False,
    quick_limit: Optional[int] = None,
    certification_time_limit_s: Optional[float] = None,
    max_attempts_per_cell: Optional[int] = None,
    training_freeze_path: Optional[Path] = None,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if split == "formal_test":
        if training_freeze_path is None:
            raise RuntimeError("正式test必须等训练协议冻结后由独立命令生成。")
        _verify_training_freeze(training_freeze_path, protocol_path)
    elif split not in {"training_pool", "validation"}:
        raise ValueError("split只能是training_pool、validation或formal_test。")
    design = protocol["split_design"][split]
    expected_count = int(design["count"])
    replicates = int(design["replicates_per_cell"])
    master_seed = int(design["seed"])
    scenario_file = ROOT / str(protocol["base_scenario_file"])
    scenario = legacy._load_scenario(scenario_file)
    base_cfg = ppo.resolve_config(dict(scenario.as_training_inputs()["cfg"]))
    run_dir = (
        output_root / "smoke" / split
        if quick_limit is not None
        else output_root / "manifests" / split
    )
    checkpoint_path = run_dir / "generation_checkpoint.json"

    if dry_run:
        result = {
            "action": "prepare_difficulty_split",
            "dry_run": True,
            "split": split,
            "expected_count": expected_count,
            "cells": len(tuple(_cell_product(protocol))),
            "replicates_per_cell": replicates,
            "protocol_hash": protocol["protocol_hash"],
            "output": str(run_dir),
        }
        return result

    if run_dir.exists() and not resume_existing:
        raise FileExistsError(
            f"困难场景目录已存在：{run_dir}；恢复请使用--resume-existing。"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    accepted: List[Dict[str, Any]] = []
    attempts_by_cell: Dict[str, int] = {}
    rejections: Counter[str] = Counter()
    if resume_existing and checkpoint_path.exists():
        checkpoint = _read_json(checkpoint_path)
        if checkpoint.get("protocol_hash") != protocol["protocol_hash"]:
            raise RuntimeError("恢复清单的协议哈希不一致。")
        accepted = [dict(row) for row in checkpoint.get("accepted", ())]
        attempts_by_cell = {
            str(key): int(value)
            for key, value in dict(checkpoint.get("attempts_by_cell", {})).items()
        }
        rejections.update(
            {
                str(key): int(value)
                for key, value in dict(checkpoint.get("rejections", {})).items()
            }
        )
    accepted_ids = {str(row["id"]) for row in accepted}
    target_ids: List[Tuple[str, int, str, str, str, int]] = []
    for node_count, difficulty, constraint_type, layout in _cell_product(protocol):
        for replicate in range(replicates):
            identifier = (
                f"{split}__n{node_count}__{difficulty}__{constraint_type}__"
                f"{layout}__r{replicate:02d}"
            )
            target_ids.append(
                (
                    identifier,
                    node_count,
                    difficulty,
                    constraint_type,
                    layout,
                    replicate,
                )
            )
    if quick_limit is not None:
        limit = min(len(target_ids), max(1, int(quick_limit)))
        probe_indices = np.linspace(0, len(target_ids) - 1, limit, dtype=int)
        target_ids = [target_ids[int(index)] for index in probe_indices]

    if max_attempts_per_cell is not None and quick_limit is None:
        raise ValueError("仅smoke模式允许缩短候选尝试次数。")
    maximum_attempts = int(
        max_attempts_per_cell
        if max_attempts_per_cell is not None
        else protocol["certification"]["max_candidate_attempts_per_cell"]
    )
    started = time.perf_counter()
    for (
        identifier,
        node_count,
        difficulty,
        constraint_type,
        layout,
        replicate,
    ) in target_ids:
        if identifier in accepted_ids:
            continue
        cell_key = f"n{node_count}|{difficulty}|{constraint_type}|{layout}|r{replicate}"
        start_attempt = int(attempts_by_cell.get(cell_key, 0))
        selected: Optional[Dict[str, Any]] = None
        for attempt in range(start_attempt, maximum_attempts):
            record = _candidate_record(
                scenario,
                protocol,
                split=split,
                node_count=node_count,
                difficulty=difficulty,
                constraint_type=constraint_type,
                priority_layout=layout,
                replicate=replicate,
                attempt=attempt,
                master_seed=master_seed,
            )
            accepted_flag, certificate, reason = certify_record(
                scenario_file,
                scenario,
                base_cfg,
                protocol,
                record,
                time_limit_s=certification_time_limit_s,
            )
            attempts_by_cell[cell_key] = attempt + 1
            if accepted_flag:
                record["certificate"] = certificate
                selected = record
                accepted.append(record)
                accepted_ids.add(identifier)
                break
            rejections[reason] += 1
            _atomic_json(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "protocol_hash": protocol["protocol_hash"],
                    "split": split,
                    "accepted": accepted,
                    "attempts_by_cell": attempts_by_cell,
                    "rejections": dict(rejections),
                },
            )
        if selected is None:
            raise RuntimeError(
                f"{cell_key}在{maximum_attempts}次候选内未找到合格场景。"
            )
        _atomic_json(
            checkpoint_path,
            {
                "schema_version": 1,
                "protocol_hash": protocol["protocol_hash"],
                "split": split,
                "accepted": accepted,
                "attempts_by_cell": attempts_by_cell,
                "rejections": dict(rejections),
            },
        )

    records_text = _jsonl_text(sorted(accepted, key=lambda row: str(row["id"])))
    records_path = run_dir / "instances.jsonl"
    _atomic_text(records_path, records_text)
    records_hash = hashlib.sha256(records_text.encode("utf-8")).hexdigest()
    metadata: Dict[str, Any] = {
        "schema_version": 2,
        "created_by": "paper_difficulty_experiments.prepare",
        "protocol_hash": protocol["protocol_hash"],
        "parent_frozen_protocol_hash": protocol["parent_frozen_protocol_hash"],
        "base_scenario_file": str(scenario_file.resolve()),
        "base_scenario_hash": str(scenario.scenario_hash),
        "split": split,
        "record_count": len(accepted),
        "expected_record_count": (
            min(expected_count, int(quick_limit))
            if quick_limit is not None
            else expected_count
        ),
        "records_file": "instances.jsonl",
        "records_sha256": records_hash,
        "generation_elapsed_s": time.perf_counter() - started,
        "attempt_count": int(sum(attempts_by_cell.values())),
        "rejections": dict(rejections),
        "candidate_algorithms_used_for_selection": [],
        "generator_code_sha256": _sha256_file(Path(__file__).resolve()),
        "formal_test_generated": split == "formal_test",
        "training_freeze_hash": (
            _read_json(training_freeze_path)["freeze_hash"]
            if training_freeze_path is not None
            else None
        ),
        "smoke_only": quick_limit is not None,
    }
    metadata["manifest_hash"] = _manifest_hash(metadata, records_hash)
    _atomic_json(run_dir / "manifest.json", metadata)
    audit = audit_environment(
        run_dir / "manifest.json",
        protocol_path,
        require_full_design=quick_limit is None,
    )
    if split == "formal_test" and training_freeze_path is not None:
        freeze = _read_json(training_freeze_path)
        _training_meta, training_rows = load_manifest(
            Path(freeze["training_manifest_path"])
        )[:2]
        _validation_meta, validation_rows = load_manifest(
            Path(freeze["validation_manifest_path"])
        )[:2]
        _assert_train_validation_disjoint(training_rows, accepted)
        _assert_train_validation_disjoint(validation_rows, accepted)
    return {"manifest": metadata, "audit": audit}


def load_manifest(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Path]:
    root = path if path.is_dir() else path.parent
    metadata_path = root / "manifest.json"
    metadata = _read_json(metadata_path)
    records_path = root / str(metadata.get("records_file", "instances.jsonl"))
    text = records_path.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != metadata["records_sha256"]:
        raise RuntimeError("困难场景instances.jsonl哈希不一致。")
    if _manifest_hash(metadata, metadata["records_sha256"]) != metadata["manifest_hash"]:
        raise RuntimeError("困难场景manifest哈希不一致。")
    return metadata, _read_jsonl(records_path), root


def audit_environment(
    manifest_path: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
    *,
    require_full_design: bool = True,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    metadata, records, root = load_manifest(manifest_path)
    checks: Dict[str, bool] = {
        "protocol_hash": metadata.get("protocol_hash") == protocol["protocol_hash"],
        "parent_protocol_hash": metadata.get("parent_frozen_protocol_hash")
        == protocol["parent_frozen_protocol_hash"],
        "no_candidate_algorithms_used_for_selection": not metadata.get(
            "candidate_algorithms_used_for_selection"
        ),
        "record_count": len(records) == int(metadata["expected_record_count"]),
        "unique_ids": len({row["id"] for row in records}) == len(records),
        "power_scale_nominal": all(
            math.isclose(float(row["power_scale"]), 1.0, abs_tol=1e-12)
            for row in records
        ),
    }
    reasons: List[str] = []
    bottleneck_counts: Counter[str] = Counter()
    band_counts: Counter[str] = Counter()
    scenario_hashes = set()
    content_hashes = set()
    cell_counts: Counter[str] = Counter()
    for row in records:
        if not _all_numeric_values_finite(row):
            reasons.append(f"{row.get('id', '<unknown>')}:nonfinite_value")
        certificate = dict(row.get("certificate") or {})
        lower = certificate.get("weighted_coverage_lower_bound")
        upper = certificate.get("weighted_coverage_upper_bound")
        if lower is None or upper is None:
            reasons.append(f"{row['id']}:missing_bounds")
            continue
        band_low, band_high = protocol["difficulty_bands"][row["difficulty"]]
        lower_value = float(lower)
        upper_value = float(upper)
        gap_value = float(certificate.get("mip_gap", math.inf))
        same_band = upper_value <= float(band_high) + 1e-9
        small_gap = gap_value <= float(
            protocol["certification"]["mip_rel_gap"]
        ) + 1e-10
        if not (
            lower_value >= float(band_low) - 1e-9
            and lower_value <= float(band_high) + 1e-9
            and upper_value
            < float(protocol["certification"]["full_coverage_upper_bound_max"])
            - 1e-9
            and (same_band or small_gap)
        ):
            reasons.append(f"{row['id']}:band_violation")
        if not certificate.get("returned") or int(
            certificate.get("visited_count", 0)
        ) < 1:
            reasons.append(f"{row['id']}:no_partial_return")
        scenario_hash = str(certificate.get("scenario_hash", ""))
        if not scenario_hash or scenario_hash in scenario_hashes:
            reasons.append(f"{row['id']}:duplicate_or_empty_scenario_hash")
        scenario_hashes.add(scenario_hash)
        content_hash = _canonical_hash(
            {
                "points": row.get("inspection_points_xyz"),
                "priorities": row.get("priorities"),
                "service_times_s": row.get("service_times_s"),
                "initial_soc": row.get("initial_soc"),
                "distance_budget_scale": row.get("distance_budget_scale"),
                "time_budget_scale": row.get("time_budget_scale"),
                "wind_scale": row.get("wind_scale"),
                "wind_rotation_deg": row.get("wind_rotation_deg"),
                "wind_vertical_bias_mps": row.get("wind_vertical_bias_mps"),
            }
        )
        if content_hash in content_hashes:
            reasons.append(f"{row['id']}:duplicate_content")
        content_hashes.add(content_hash)
        if (
            len(row.get("inspection_points_xyz", ())) != int(row["node_count"])
            or len(row.get("priorities", ())) != int(row["node_count"])
            or len(row.get("service_times_s", ())) != int(row["node_count"])
        ):
            reasons.append(f"{row['id']}:node_field_length")
        intended = str(row["constraint_type"])
        active = set(certificate.get("bottleneck_resources", ()))
        if intended == "mixed":
            if len(active) < int(
                protocol["certification"]["mixed_min_active_resources"]
            ):
                reasons.append(f"{row['id']}:mixed_bottleneck")
        elif intended not in active:
            reasons.append(f"{row['id']}:intended_bottleneck")
        cell_counts[
            "|".join(
                (
                    str(row["node_count"]),
                    str(row["difficulty"]),
                    intended,
                    str(row["priority_layout"]),
                )
            )
        ] += 1
        band_counts[str(row["difficulty"])] += 1
        bottleneck_counts.update(certificate.get("bottleneck_resources", ()))
    checks["all_certificates_valid"] = not reasons
    if require_full_design:
        expected_replicates = int(
            protocol["split_design"][metadata["split"]]["replicates_per_cell"]
        )
        checks["complete_factorial_cells"] = all(
            cell_counts[
                "|".join((str(node_count), difficulty, constraint, layout))
            ]
            == expected_replicates
            for node_count, difficulty, constraint, layout in _cell_product(
                protocol
            )
        )
        checks["all_difficulty_layers_present"] = all(
            band_counts[name] > 0 for name in protocol["difficulty_bands"]
        )
        checks["at_least_two_resource_bottlenecks"] = sum(
            count > 0 for count in bottleneck_counts.values()
        ) >= 2
    report = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "protocol_hash": protocol["protocol_hash"],
        "manifest_hash": metadata["manifest_hash"],
        "manifest_root": str(root.resolve()),
        "checks": checks,
        "difficulty_counts": dict(band_counts),
        "bottleneck_counts": dict(bottleneck_counts),
        "failures": reasons[:100],
    }
    report["audit_hash"] = _canonical_hash(report, excluded=("audit_hash",))
    _atomic_json(root / "environment_audit.json", report)
    if not report["passed"]:
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"困难环境审计失败：{failed}")
    return report


def _old_checkpoint_grid(old_protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    checkpoints = old_protocol.get("checkpoints")
    if isinstance(checkpoints, list):
        return [dict(item) for item in checkpoints]
    if isinstance(checkpoints, dict):
        values: List[Dict[str, Any]] = []
        for items in checkpoints.values():
            if isinstance(items, list):
                values.extend(dict(item) for item in items)
        return values
    raise RuntimeError("旧冻结协议缺少检查点清单。")


def _challenge_row(
    record: Mapping[str, Any],
    *,
    variant: str,
    seed: int,
    checkpoint_hash: str,
    detail: Mapping[str, Any],
    runtime_s: float,
    challenge_manifest_hash: str,
    challenge_protocol_hash: str,
) -> Dict[str, Any]:
    metrics = dict(detail.get("metrics", detail))
    safe = bool(metrics.get("returned")) and not any(
        bool(metrics.get(field, False)) for field in VIOLATION_FIELDS
    )
    weighted = float(metrics.get("weighted_coverage", 0.0))
    return {
        "scenario_id": str(record["id"]),
        "split": str(record["split"]),
        "node_count": int(record["node_count"]),
        "difficulty": str(record["difficulty"]),
        "constraint_type": str(record["constraint_type"]),
        "priority_layout": str(record["priority_layout"]),
        "algorithm": variant,
        "variant": variant,
        "training_seed": int(seed),
        "checkpoint_hash": checkpoint_hash,
        "challenge_manifest_hash": challenge_manifest_hash,
        "challenge_protocol_hash": challenge_protocol_hash,
        "returned": bool(metrics.get("returned", False)),
        "safe": safe,
        "weighted_coverage": weighted,
        "safe_weighted_coverage": weighted if safe else 0.0,
        "coverage": float(metrics.get("coverage", 0.0)),
        "visited_count": int(metrics.get("visited_count", len(detail.get("visit_order", ())))),
        "termination_reason": str(
            metrics.get(
                "termination_reason",
                detail.get("termination_reason", "unknown"),
            )
        ),
        "energy_utilization": float(metrics.get("energy_utilization", math.nan)),
        "distance_utilization": float(metrics.get("distance_utilization", math.nan)),
        "time_utilization": float(metrics.get("time_utilization", math.nan)),
        "planning_time_s": float(runtime_s),
    }


def qualify_existing_checkpoints(
    protocol_path: Path,
    manifest_path: Path,
    output_root: Path,
    *,
    device: str = "cuda",
    resume_existing: bool = False,
    quick_limit: Optional[int] = None,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    metadata, records, _ = load_manifest(manifest_path)
    if metadata["split"] != "validation":
        raise ValueError("资格验证只允许困难validation清单。")
    if bool(metadata.get("smoke_only")):
        raise ValueError("smoke清单不得用于35个检查点资格验证。")
    if quick_limit is not None:
        records = records[: int(quick_limit)]
    old_protocol = _read_json(OLD_FROZEN_PROTOCOL)
    if old_protocol.get("protocol_hash") != protocol["parent_frozen_protocol_hash"]:
        raise RuntimeError("旧冻结协议身份与困难协议父哈希不一致。")
    grid = _old_checkpoint_grid(old_protocol)
    expected_pairs = {
        (variant, seed)
        for variant in protocol["qualification"]["all_variants"]
        for seed in protocol["qualification"]["training_seeds"]
    }
    indexed = {
        (str(item["variant"]), int(item["training_seed"])): item
        for item in grid
        if (str(item.get("variant")), int(item.get("training_seed", -1)))
        in expected_pairs
    }
    if set(indexed) != expected_pairs:
        missing = sorted(expected_pairs - set(indexed))
        raise RuntimeError(f"旧冻结协议缺少资格检查点：{missing}")
    scenario_file = ROOT / str(protocol["base_scenario_file"])
    scenario = legacy._load_scenario(scenario_file)
    map_location = getattr(ppo, "device", "cpu") if device == "auto" else device
    summary_rows: List[Dict[str, Any]] = []
    for variant, seed in sorted(expected_pairs):
        item = indexed[(variant, seed)]
        checkpoint = ROOT / str(item["path"])
        if _sha256_file(checkpoint) != str(item["sha256"]):
            raise RuntimeError(f"检查点哈希漂移：{checkpoint}")
        model, payload = ppo.load_checkpoint(checkpoint, map_location=map_location)
        if str(payload.get("checkpoint_kind", "")) != "best_safe":
            raise RuntimeError(f"资格验证只接受best_safe：{checkpoint}")
        base_cfg = ppo.resolve_config(dict(payload["cfg"]))
        run_dir = output_root / "qualification" / f"{variant}__seed{seed}"
        result_path = run_dir / "results.jsonl"
        if run_dir.exists() and not resume_existing:
            raise FileExistsError(
                f"资格目录已存在：{run_dir}；恢复请使用--resume-existing。"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        existing = _read_jsonl(result_path) if resume_existing else []
        by_id = {str(row["scenario_id"]): row for row in existing}
        if len(by_id) != len(existing):
            raise RuntimeError(f"{result_path}包含重复场景。")
        rows = list(existing)
        for record in records:
            if str(record["id"]) in by_id:
                continue
            cfg = legacy._instance_cfg(base_cfg, scenario, record, 1.0)
            wind = legacy._transform_wind(scenario.wind_data, record)
            started = time.perf_counter()
            detail = ppo.plan_with_policy_improved(
                model,
                scenario.start_pos,
                np.asarray(record["inspection_points_xyz"], dtype=np.float32),
                np.asarray(record["priorities"], dtype=np.float32),
                scenario.terrain,
                cfg,
                wind,
                return_details=True,
                decode_mode="deterministic",
            )
            elapsed = time.perf_counter() - started
            row = _challenge_row(
                record,
                variant=variant,
                seed=seed,
                checkpoint_hash=str(item["sha256"]),
                detail=detail,
                runtime_s=elapsed,
                challenge_manifest_hash=str(metadata["manifest_hash"]),
                challenge_protocol_hash=str(protocol["protocol_hash"]),
            )
            rows.append(row)
            _atomic_text(result_path, _jsonl_text(rows))
            _atomic_json(
                run_dir / "status.json",
                {
                    "state": "running",
                    "completed": len(rows),
                    "total": len(records),
                    "variant": variant,
                    "training_seed": seed,
                },
            )
        _write_csv(run_dir / "results.csv", rows)
        _atomic_json(
            run_dir / "status.json",
            {
                "state": "completed",
                "completed": len(rows),
                "total": len(records),
                "variant": variant,
                "training_seed": seed,
            },
        )
        summary_rows.append(
            _qualification_summary(
                rows,
                variant=variant,
                seed=seed,
                protocol=protocol,
            )
        )
        del model
    _write_csv(output_root / "qualification" / "qualification_summary.csv", summary_rows)
    decision = qualification_decision(summary_rows, protocol)
    _atomic_json(output_root / "qualification" / "qualification_decision.json", decision)
    return decision


def _qualification_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    seed: int,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = int(protocol["split_design"]["validation"]["count"])
    finite_fields = (
        "safe_weighted_coverage",
        "coverage",
        "energy_utilization",
        "distance_utilization",
        "time_utilization",
        "planning_time_s",
    )
    finite = all(
        all(math.isfinite(float(row[field])) for field in finite_fields) for row in rows
    )
    safe_rate = float(np.mean([bool(row["safe"]) for row in rows])) if rows else 0.0
    zero_rate = (
        float(np.mean([int(row["visited_count"]) == 0 for row in rows]))
        if rows
        else 1.0
    )
    partial_rate = (
        float(
            np.mean(
                [
                    bool(row["safe"])
                    and str(row["termination_reason"]) == "returned_partial"
                    for row in rows
                ]
            )
        )
        if rows
        else 0.0
    )
    median_visited = (
        float(statistics.median(int(row["visited_count"]) for row in rows))
        if rows
        else 0.0
    )
    median_effectiveness = (
        float(
            statistics.median(float(row["safe_weighted_coverage"]) for row in rows)
        )
        if rows
        else 0.0
    )
    return {
        "variant": variant,
        "training_seed": int(seed),
        "row_count": len(rows),
        "expected_row_count": expected,
        "complete": len(rows) == expected,
        "finite": finite,
        "safe_rate": safe_rate,
        "zero_visit_rate": zero_rate,
        "partial_return_rate": partial_rate,
        "median_visited_count": median_visited,
        "median_safe_weighted_coverage": median_effectiveness,
    }


def qualification_decision(
    summaries: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    qualification = protocol["qualification"]
    core = set(str(value) for value in qualification["core_variants"])
    low, high = (
        float(value)
        for value in qualification["core_median_safe_weighted_coverage"]
    )
    assessed: List[Dict[str, Any]] = []
    for raw in summaries:
        row = dict(raw)
        reasons: List[str] = []
        if not bool(row["complete"]):
            reasons.append("incomplete")
        if not bool(row["finite"]):
            reasons.append("nonfinite")
        if str(row["variant"]) in core:
            if float(row["safe_rate"]) < float(qualification["core_safe_rate_min"]):
                reasons.append("safe_rate")
            if float(row["zero_visit_rate"]) > float(
                qualification["core_zero_visit_rate_max"]
            ):
                reasons.append("zero_visit_rate")
            if float(row["partial_return_rate"]) < float(
                qualification["core_partial_return_rate_min"]
            ):
                reasons.append("partial_return_rate")
            if float(row["median_visited_count"]) < float(
                qualification["core_median_visited_count_min"]
            ):
                reasons.append("median_visited_count")
            value = float(row["median_safe_weighted_coverage"])
            if value < low or value > high:
                reasons.append("median_safe_weighted_coverage")
        row["passed"] = not reasons
        row["failure_reasons"] = reasons
        assessed.append(row)
    core_passed = all(
        bool(row["passed"]) for row in assessed if str(row["variant"]) in core
    )
    all_complete = all(
        bool(row["complete"]) and bool(row["finite"]) for row in assessed
    )
    decision = "keep_all_35" if core_passed and all_complete else "retrain_all_35"
    result = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "decision": decision,
        "core_passed": core_passed,
        "all_variants_complete_and_finite": all_complete,
        "partial_retraining_allowed": False,
        "summaries": assessed,
    }
    result["decision_hash"] = _canonical_hash(result, excluded=("decision_hash",))
    return result


def _require_passed_environment(
    manifest_path: Path,
    protocol_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metadata, records, root = load_manifest(manifest_path)
    audit_path = root / "environment_audit.json"
    audit = _read_json(audit_path) if audit_path.exists() else audit_environment(
        manifest_path, protocol_path
    )
    if not bool(audit.get("passed")):
        raise RuntimeError(f"环境审计未通过：{audit_path}")
    if bool(metadata.get("smoke_only")):
        raise RuntimeError("smoke清单不能进入资格验证或训练。")
    return metadata, records


def _assert_train_validation_disjoint(
    training_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
) -> None:
    training_ids = {str(row["id"]) for row in training_records}
    validation_ids = {str(row["id"]) for row in validation_records}
    if training_ids & validation_ids:
        raise RuntimeError("困难training_pool与validation存在重复ID。")
    training_hashes = {
        _canonical_hash(
            {
                "points": row["inspection_points_xyz"],
                "priorities": row["priorities"],
                "domain": {
                    field: row[field]
                    for field in (
                        "initial_soc",
                        "distance_budget_scale",
                        "time_budget_scale",
                        "wind_scale",
                        "wind_rotation_deg",
                        "wind_vertical_bias_mps",
                    )
                },
            }
        )
        for row in training_records
    }
    validation_hashes = {
        _canonical_hash(
            {
                "points": row["inspection_points_xyz"],
                "priorities": row["priorities"],
                "domain": {
                    field: row[field]
                    for field in (
                        "initial_soc",
                        "distance_budget_scale",
                        "time_budget_scale",
                        "wind_scale",
                        "wind_rotation_deg",
                        "wind_vertical_bias_mps",
                    )
                },
            }
        )
        for row in validation_records
    }
    if training_hashes & validation_hashes:
        raise RuntimeError("困难training_pool与validation存在内容重复。")


def _difficulty_training_cfg(
    scenario: Any,
    protocol: Mapping[str, Any],
    *,
    variant: str,
    seed: int,
    episodes: int,
    monitor_episodes: Sequence[int],
    run_dir: Path,
    training_manifest_hash: str,
    validation_manifest_hash: str,
    stage: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(scenario.as_training_inputs()["cfg"]))
    cfg.update(
        {
            "experiment_variant": variant,
            "seed": int(seed),
            "max_episodes": int(episodes),
            "checkpoint_dir": str(run_dir.resolve()),
            "experiment_stage": stage,
            "difficulty_protocol_hash": protocol["protocol_hash"],
            "difficulty_training_manifest_hash": training_manifest_hash,
            "difficulty_validation_manifest_hash": validation_manifest_hash,
            "monitor_episodes": [int(value) for value in monitor_episodes],
            "persist_monitor_checkpoints": stage == "pilot",
        }
    )
    return ppo.resolve_config(cfg)


def _health_callback(
    run_dir: Path,
    *,
    protocol: Mapping[str, Any],
    stage: str,
    variant: str,
    seed: int,
) -> Any:
    metrics_path = run_dir / "training_metrics.jsonl"
    existing = _read_jsonl(metrics_path)
    by_episode = {
        int(float(row["episodes_seen"])): dict(row)
        for row in existing
    }
    monitor_set = {
        int(value)
        for value in (
            protocol["pilot_training"]["monitor_episodes"]
            if stage == "pilot"
            else protocol["formal_training"]["monitor_episodes"]
        )
    }

    def callback(raw: Mapping[str, Any]) -> None:
        row = copy.deepcopy(dict(raw))
        episode = int(float(row["episodes_seen"]))
        by_episode[episode] = row
        ordered = [by_episode[key] for key in sorted(by_episode)]
        _atomic_text(metrics_path, _jsonl_text(ordered))
        validation = dict(row.get("validation") or {})
        finite_values = [
            float(value)
            for value in (
                row.get("mean_return"),
                row.get("mean_coverage"),
                row.get("mean_weighted_coverage"),
                row.get("return_rate"),
            )
            if value is not None
        ]
        alerts: List[str] = []
        if not all(math.isfinite(value) for value in finite_values):
            alerts.append("nonfinite_training_metric")
        if validation:
            if float(validation.get("return_rate", 0.0)) < float(
                protocol["pilot_training"]["floor_safe_rate"]
            ):
                alerts.append("validation_safe_rate_below_80pct")
            if float(validation.get("zero_visit_rate", 0.0)) > float(
                protocol["pilot_training"]["zero_visit_warning_rate"]
            ):
                alerts.append("validation_zero_visit_above_10pct")
            attainment = validation.get("median_oracle_attainment_lower")
            if attainment is not None and float(attainment) < float(
                protocol["pilot_training"]["floor_median_oracle_attainment"]
            ):
                alerts.append("validation_oracle_attainment_below_20pct")
        if episode in monitor_set:
            report = {
                "schema_version": 1,
                "protocol_hash": protocol["protocol_hash"],
                "variant": variant,
                "training_seed": int(seed),
                "episode": episode,
                "training_node_count": row.get("training_node_count"),
                "training_metrics": {
                    key: row.get(key)
                    for key in (
                        "mean_return",
                        "mean_coverage",
                        "mean_weighted_coverage",
                        "return_rate",
                        "mean_energy_utilization",
                        "mean_distance_utilization",
                        "mean_time_utilization",
                        "termination_reason_counts",
                    )
                },
                "validation": validation,
                "alerts": alerts,
                "single_model_alerts_do_not_authorize_protocol_changes": True,
            }
            report["health_hash"] = _canonical_hash(
                report, excluded=("health_hash",)
            )
            _atomic_json(
                run_dir / "health" / f"episode_{episode:04d}.json",
                report,
            )
        _atomic_json(
            run_dir / "status.json",
            {
                "state": "running",
                "variant": variant,
                "training_seed": int(seed),
                "completed": episode,
                "total": int(
                    protocol["pilot_training"]["episodes"]
                    if stage == "pilot"
                    else protocol["formal_training"]["episodes"]
                ),
                "latest_checkpoint": row.get("latest_checkpoint"),
                "monitor_checkpoint": row.get("monitor_checkpoint"),
                "alerts": alerts,
            },
        )

    return callback


def run_training_grid(
    protocol_path: Path,
    training_manifest_path: Path,
    validation_manifest_path: Path,
    output_root: Path,
    *,
    stage: str,
    device: str = "cuda",
    resume_existing: bool = False,
    variants: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    training_metadata, training_records = _require_passed_environment(
        training_manifest_path, protocol_path
    )
    validation_metadata, validation_records = _require_passed_environment(
        validation_manifest_path, protocol_path
    )
    if training_metadata["split"] != "training_pool":
        raise ValueError("training-manifest必须是training_pool。")
    if validation_metadata["split"] != "validation":
        raise ValueError("validation-manifest必须是validation。")
    _assert_train_validation_disjoint(training_records, validation_records)

    qualification_path = output_root / "qualification" / "qualification_decision.json"
    qualification = _read_json(qualification_path)
    if qualification.get("protocol_hash") != protocol["protocol_hash"]:
        raise RuntimeError("资格决策与当前困难协议不一致。")
    if qualification.get("decision") != "retrain_all_35":
        raise RuntimeError("现有35模型通过资格验证时不得启动重训分支。")

    if stage == "pilot":
        expected_variants = list(protocol["pilot_training"]["variants"])
        expected_seeds = [int(protocol["pilot_training"]["seed"])]
        episodes = int(protocol["pilot_training"]["episodes"])
        monitors = list(protocol["pilot_training"]["monitor_episodes"])
        run_root = output_root / "pilot"
    elif stage == "formal":
        pilot_decision = _read_json(
            output_root / "pilot" / "pilot_decision.json"
        )
        if pilot_decision.get("decision") not in {
            "pilot_passed",
            "pilot_passed_pointer_lag",
        }:
            raise RuntimeError("试训尚未通过，不得启动35个正式重训任务。")
        expected_variants = list(protocol["qualification"]["all_variants"])
        expected_seeds = [
            int(value) for value in protocol["qualification"]["training_seeds"]
        ]
        episodes = int(protocol["formal_training"]["episodes"])
        monitors = list(protocol["formal_training"]["monitor_episodes"])
        run_root = output_root / "formal_training"
    else:
        raise ValueError("stage只能是pilot或formal。")

    selected_variants = list(variants or expected_variants)
    selected_seeds = [int(value) for value in (seeds or expected_seeds)]
    if not set(selected_variants).issubset(expected_variants):
        raise ValueError("请求了当前阶段未注册的学习变体。")
    if not set(selected_seeds).issubset(expected_seeds):
        raise ValueError("请求了当前阶段未注册的训练种子。")
    planned = [
        (variant, seed)
        for variant in expected_variants
        if variant in selected_variants
        for seed in expected_seeds
        if seed in selected_seeds
    ]
    if dry_run:
        return {
            "action": "difficulty_training_grid",
            "dry_run": True,
            "stage": stage,
            "episodes_per_model": episodes,
            "planned_models": len(planned),
            "planned_episodes": len(planned) * episodes,
            "protocol_hash": protocol["protocol_hash"],
        }

    scenario_file = ROOT / str(protocol["base_scenario_file"])
    scenario = legacy._load_scenario(scenario_file)
    completed: List[str] = []
    for variant, seed in planned:
        run_dir = run_root / f"{stage}_{variant}_seed{seed}_{episodes}ep"
        latest = run_dir / "latest.pt"
        if run_dir.exists() and not resume_existing:
            raise FileExistsError(
                f"训练目录已存在：{run_dir}；恢复请使用--resume-existing。"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = _difficulty_training_cfg(
            scenario,
            protocol,
            variant=variant,
            seed=seed,
            episodes=episodes,
            monitor_episodes=monitors,
            run_dir=run_dir,
            training_manifest_hash=str(training_metadata["manifest_hash"]),
            validation_manifest_hash=str(validation_metadata["manifest_hash"]),
            stage=stage,
        )
        run_config = {
            "schema_version": 1,
            "stage": stage,
            "variant": variant,
            "training_seed": seed,
            "episodes": episodes,
            "protocol_hash": protocol["protocol_hash"],
            "training_manifest_hash": training_metadata["manifest_hash"],
            "validation_manifest_hash": validation_metadata["manifest_hash"],
            "training_config": cfg,
            "paper_eligible": stage == "formal",
        }
        _atomic_json(run_dir / "run_config.json", run_config)
        _atomic_json(
            run_dir / "status.json",
            {
                "state": "starting",
                "variant": variant,
                "training_seed": seed,
                "completed": 0,
                "total": episodes,
            },
        )
        try:
            model, _returns = ppo.train_policy_improved(
                scenario.start_pos,
                scenario.inspection_points,
                scenario.priorities,
                scenario.terrain,
                cfg,
                scenario.wind_data,
                resume_from=(
                    latest
                    if resume_existing and latest.exists()
                    else None
                ),
                metrics_callback=_health_callback(
                    run_dir,
                    protocol=protocol,
                    stage=stage,
                    variant=variant,
                    seed=seed,
                ),
                target_device=device,
                validation_instances=validation_records,
                training_instances=training_records,
            )
            summary = dict(getattr(model, "training_summary", {}) or {})
            _atomic_json(run_dir / "training_summary.json", summary)
            _atomic_json(
                run_dir / "status.json",
                {
                    "state": "completed",
                    "variant": variant,
                    "training_seed": seed,
                    "completed": episodes,
                    "total": episodes,
                    "selection_kind": summary.get("selection_kind"),
                },
            )
            completed.append(f"{variant}__seed{seed}")
        except Exception as exc:
            _atomic_json(
                run_dir / "status.json",
                {
                    "state": "failed",
                    "variant": variant,
                    "training_seed": seed,
                    "completed": len(_read_jsonl(run_dir / "training_metrics.jsonl")),
                    "total": episodes,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
    return {
        "stage": stage,
        "completed_models": completed,
        "protocol_hash": protocol["protocol_hash"],
    }


def _evaluate_monitor_checkpoint(
    checkpoint: Path,
    records: Sequence[Mapping[str, Any]],
    scenario: Any,
    *,
    device: str,
    algorithm: str,
    episode: int,
) -> List[Dict[str, Any]]:
    map_location = getattr(ppo, "device", "cpu") if device == "auto" else device
    model, payload = ppo.load_checkpoint(checkpoint, map_location=map_location)
    if str(payload.get("checkpoint_kind")) != "monitor":
        raise RuntimeError(f"试训评估只接受monitor检查点：{checkpoint}")
    cfg = ppo.resolve_config(dict(payload["cfg"]))
    rows: List[Dict[str, Any]] = []
    for record in records:
        scenario_cfg, wind = ppo.apply_frozen_domain_instance(
            cfg, scenario.wind_data, record
        )
        detail = ppo.plan_with_policy_improved(
            model,
            scenario.start_pos,
            np.asarray(record["inspection_points_xyz"], dtype=np.float32),
            np.asarray(record["priorities"], dtype=np.float32),
            scenario.terrain,
            scenario_cfg,
            wind,
            return_details=True,
            decode_mode="deterministic",
        )
        metrics = dict(detail["metrics"])
        safe = bool(metrics.get("returned")) and not any(
            bool(metrics.get(field, False)) for field in VIOLATION_FIELDS
        )
        achieved = float(metrics.get("weighted_coverage", 0.0)) if safe else 0.0
        upper = float(
            record["certificate"]["weighted_coverage_upper_bound"]
        )
        rows.append(
            {
                "scenario_id": record["id"],
                "algorithm": algorithm,
                "episode": int(episode),
                "safe": safe,
                "safe_weighted_coverage": achieved,
                "visited_count": int(metrics.get("visited_count", 0)),
                "partial_return": safe
                and str(metrics.get("termination_reason")) == "returned_partial",
                "oracle_upper": upper,
                "oracle_attainment_lower": min(1.0, achieved / max(upper, 1e-12)),
                "within_one_percent_of_oracle": achieved >= upper - 0.01 - 1e-12,
            }
        )
    return rows


def assess_pilot(
    protocol_path: Path,
    validation_manifest_path: Path,
    output_root: Path,
    *,
    device: str = "cuda",
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    _metadata, records = _require_passed_environment(
        validation_manifest_path, protocol_path
    )
    scenario = legacy._load_scenario(
        ROOT / str(protocol["base_scenario_file"])
    )
    pilot = protocol["pilot_training"]
    monitor_episodes = [int(value) for value in pilot["monitor_episodes"]]
    rows: List[Dict[str, Any]] = []
    for variant in pilot["variants"]:
        run_dir = (
            output_root
            / "pilot"
            / f"pilot_{variant}_seed{pilot['seed']}_{pilot['episodes']}ep"
        )
        status = _read_json(run_dir / "status.json")
        if status.get("state") != "completed":
            raise RuntimeError(f"试训尚未完成：{run_dir}")
        for episode in monitor_episodes:
            rows.extend(
                _evaluate_monitor_checkpoint(
                    run_dir / f"monitor_ep{episode:04d}.pt",
                    records,
                    scenario,
                    device=device,
                    algorithm=str(variant),
                    episode=episode,
                )
            )

    base_cfg = ppo.resolve_config(dict(scenario.as_training_inputs()["cfg"]))
    for algorithm, planner in (
        ("nearest_feasible", plan_nearest_feasible),
        ("priority_resource_greedy", plan_priority_resource_greedy),
    ):
        for record in records:
            problem = _problem_from_record(
                ROOT / str(protocol["base_scenario_file"]),
                scenario,
                base_cfg,
                record,
            )
            result = planner(
                problem,
                seed=42,
                budget=PlannerBudget(max_evaluations=None, time_limit_s=None),
            )
            metrics = result.metrics
            safe = bool(metrics.get("returned")) and not any(
                bool(metrics.get(field, False)) for field in VIOLATION_FIELDS
            )
            achieved = (
                float(metrics.get("weighted_coverage", 0.0)) if safe else 0.0
            )
            upper = float(
                record["certificate"]["weighted_coverage_upper_bound"]
            )
            rows.append(
                {
                    "scenario_id": record["id"],
                    "algorithm": algorithm,
                    "episode": int(pilot["episodes"]),
                    "safe": safe,
                    "safe_weighted_coverage": achieved,
                    "visited_count": int(metrics.get("visited_count", 0)),
                    "partial_return": safe
                    and str(metrics.get("termination_reason")) == "returned_partial",
                    "oracle_upper": upper,
                    "oracle_attainment_lower": min(
                        1.0, achieved / max(upper, 1e-12)
                    ),
                    "within_one_percent_of_oracle": achieved
                    >= upper - 0.01 - 1e-12,
                }
            )
    _write_csv(output_root / "pilot" / "pilot_validation_rows.csv", rows)

    summaries: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["algorithm"]), int(row["episode"]))].append(row)
    for (algorithm, episode), group in sorted(grouped.items()):
        summaries.append(
            {
                "algorithm": algorithm,
                "episode": episode,
                "row_count": len(group),
                "safe_rate": float(np.mean([row["safe"] for row in group])),
                "zero_visit_rate": float(
                    np.mean([int(row["visited_count"]) == 0 for row in group])
                ),
                "partial_return_rate": float(
                    np.mean([row["partial_return"] for row in group])
                ),
                "median_safe_weighted_coverage": float(
                    statistics.median(
                        float(row["safe_weighted_coverage"]) for row in group
                    )
                ),
                "median_oracle_attainment_lower": float(
                    statistics.median(
                        float(row["oracle_attainment_lower"]) for row in group
                    )
                ),
                "oracle_close_scene_share": float(
                    np.mean(
                        [row["within_one_percent_of_oracle"] for row in group]
                    )
                ),
            }
        )
    _write_csv(output_root / "pilot" / "pilot_summary.csv", summaries)
    final_episode = int(pilot["episodes"])
    final = {
        row["algorithm"]: row
        for row in summaries
        if int(row["episode"]) == final_episode
    }
    core_names = [str(value) for value in pilot["variants"]]
    comparison_names = core_names + [
        "nearest_feasible",
        "priority_resource_greedy",
    ]
    too_easy = all(
        float(final[name]["oracle_close_scene_share"])
        >= float(pilot["ceiling_scene_share"])
        for name in comparison_names
    )
    too_hard_attainment = all(
        float(final[name]["median_oracle_attainment_lower"])
        < float(pilot["floor_median_oracle_attainment"])
        for name in core_names
    )
    too_hard_safety = all(
        float(final[name]["safe_rate"]) < float(pilot["floor_safe_rate"])
        for name in core_names
    )
    zero_warning = any(
        float(final[name]["zero_visit_rate"])
        > float(pilot["zero_visit_warning_rate"])
        for name in core_names
    )
    pointer_lag = (
        float(final["full"]["median_safe_weighted_coverage"])
        < min(
            float(final["ppo_mlp"]["median_safe_weighted_coverage"]),
            float(final["a2c_pointer"]["median_safe_weighted_coverage"]),
        )
        and not (too_easy or too_hard_attainment or too_hard_safety)
    )
    if too_easy:
        decision = "revise_environment_too_easy"
    elif too_hard_attainment or too_hard_safety:
        decision = "revise_environment_too_hard"
    elif zero_warning:
        decision = "diagnose_legality_before_formal_training"
    elif pointer_lag:
        decision = "pilot_passed_pointer_lag"
    else:
        decision = "pilot_passed"
    report = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "decision": decision,
        "task_too_easy": too_easy,
        "task_too_hard_attainment": too_hard_attainment,
        "task_too_hard_safety": too_hard_safety,
        "zero_visit_warning": zero_warning,
        "pointer_only_lag_does_not_authorize_changes": True,
        "pilot_models_paper_eligible": False,
        "summaries": summaries,
    }
    report["decision_hash"] = _canonical_hash(
        report, excluded=("decision_hash",)
    )
    _atomic_json(output_root / "pilot" / "pilot_decision.json", report)
    return report


def _difficulty_code_fingerprints() -> List[Dict[str, Any]]:
    paths = (
        ROOT / "final_python_ppo_pointer.py",
        ROOT / "paper_difficulty_experiments.py",
        ROOT / "paper_experiments.py",
        ROOT / "python_classical_algs" / "common.py",
        ROOT / "python_classical_algs" / "milp.py",
        ROOT / "python_classical_algs" / "greedy.py",
        ROOT / "ppo_training_scenario.py",
    )
    return [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]


def freeze_training_branch(
    protocol_path: Path,
    training_manifest_path: Path,
    validation_manifest_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    training_metadata, training_records = _require_passed_environment(
        training_manifest_path, protocol_path
    )
    validation_metadata, validation_records = _require_passed_environment(
        validation_manifest_path, protocol_path
    )
    _assert_train_validation_disjoint(training_records, validation_records)
    decision = _read_json(
        output_root / "qualification" / "qualification_decision.json"
    )
    if decision.get("protocol_hash") != protocol["protocol_hash"]:
        raise RuntimeError("资格决策与当前协议不一致。")

    checkpoints: List[Dict[str, Any]] = []
    branch = str(decision.get("decision"))
    if branch == "keep_all_35":
        old_protocol = _read_json(OLD_FROZEN_PROTOCOL)
        for item in _old_checkpoint_grid(old_protocol):
            path = ROOT / str(item["path"])
            actual = _sha256_file(path)
            if actual != str(item["sha256"]):
                raise RuntimeError(f"保留分支检查点哈希漂移：{path}")
            checkpoints.append(
                {
                    "variant": item["variant"],
                    "training_seed": int(item["training_seed"]),
                    "path": str(path.resolve()),
                    "sha256": actual,
                    "checkpoint_kind": "best_safe",
                    "source_branch": "legacy_qualified_on_difficulty_v2_1",
                }
            )
    elif branch == "retrain_all_35":
        pilot_decision = _read_json(
            output_root / "pilot" / "pilot_decision.json"
        )
        if pilot_decision.get("decision") not in {
            "pilot_passed",
            "pilot_passed_pointer_lag",
        }:
            raise RuntimeError("试训未通过，不能冻结正式重训分支。")
        episodes = int(protocol["formal_training"]["episodes"])
        for variant in protocol["qualification"]["all_variants"]:
            for seed in protocol["qualification"]["training_seeds"]:
                run_dir = (
                    output_root
                    / "formal_training"
                    / f"formal_{variant}_seed{seed}_{episodes}ep"
                )
                status = _read_json(run_dir / "status.json")
                if status.get("state") != "completed":
                    raise RuntimeError(f"正式重训尚未完成：{run_dir}")
                summary = _read_json(run_dir / "training_summary.json")
                selection_kind = str(summary.get("selection_kind", ""))
                checkpoint_name = (
                    "best_safe.pt"
                    if selection_kind == "best_safe"
                    else "best_candidate.pt"
                )
                checkpoint = run_dir / checkpoint_name
                _model, payload = ppo.load_checkpoint(
                    checkpoint, map_location="cpu"
                )
                if (
                    payload["cfg"].get("difficulty_protocol_hash")
                    != protocol["protocol_hash"]
                ):
                    raise RuntimeError(f"正式检查点协议身份不一致：{checkpoint}")
                checkpoints.append(
                    {
                        "variant": variant,
                        "training_seed": int(seed),
                        "path": str(checkpoint.resolve()),
                        "sha256": _sha256_file(checkpoint),
                        "checkpoint_kind": str(
                            payload.get("checkpoint_kind", "")
                        ),
                        "source_branch": "difficulty_v2_1_retrained",
                    }
                )
    else:
        raise RuntimeError(f"未知资格决策：{branch}")
    if len(checkpoints) != 35:
        raise RuntimeError("训练冻结必须恰好包含35个检查点。")

    freeze = {
        "schema_version": 1,
        "state": "frozen",
        "protocol_hash": protocol["protocol_hash"],
        "parent_frozen_protocol_hash": protocol["parent_frozen_protocol_hash"],
        "training_branch": branch,
        "training_manifest_path": str(training_manifest_path.resolve()),
        "training_manifest_hash": training_metadata["manifest_hash"],
        "validation_manifest_path": str(validation_manifest_path.resolve()),
        "validation_manifest_hash": validation_metadata["manifest_hash"],
        "formal_test_seed": int(protocol["split_design"]["formal_test"]["seed"]),
        "checkpoints": checkpoints,
        "code_fingerprints": _difficulty_code_fingerprints(),
    }
    freeze["freeze_hash"] = _canonical_hash(
        freeze, excluded=("freeze_hash",)
    )
    path = output_root / "freezes" / "training_freeze.json"
    if path.exists():
        existing = _read_json(path)
        if existing != freeze:
            raise RuntimeError("训练冻结文件已存在且内容不同；禁止覆盖。")
        return existing
    _atomic_json(path, freeze)
    return freeze


def _verify_training_freeze(
    path: Path,
    protocol_path: Path,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    freeze = _read_json(path)
    if freeze.get("state") != "frozen":
        raise RuntimeError("训练分支尚未冻结。")
    if freeze.get("protocol_hash") != protocol["protocol_hash"]:
        raise RuntimeError("训练冻结与当前协议哈希不一致。")
    expected_hash = _canonical_hash(freeze, excluded=("freeze_hash",))
    if expected_hash != freeze.get("freeze_hash"):
        raise RuntimeError("训练冻结文件哈希不一致。")
    for item in freeze.get("code_fingerprints", ()):
        local = ROOT / str(item["path"])
        if _sha256_file(local) != str(item["sha256"]):
            raise RuntimeError(f"正式test前代码漂移：{local}")
    for item in freeze.get("checkpoints", ()):
        checkpoint = Path(str(item["path"]))
        if _sha256_file(checkpoint) != str(item["sha256"]):
            raise RuntimeError(f"正式test前检查点漂移：{checkpoint}")
    if len(freeze.get("checkpoints", ())) != 35:
        raise RuntimeError("训练冻结检查点数量不是35。")
    return freeze


def _priority_stratum_coverage(
    visit_order: Sequence[int],
    priorities: Sequence[float],
) -> Dict[str, float]:
    visited = {int(value) for value in visit_order}
    values = np.asarray(priorities, dtype=np.float64).reshape(-1)
    result: Dict[str, float] = {}
    for label, priority in (("high", 3), ("medium", 2), ("low", 1)):
        indices = np.flatnonzero(np.isclose(values, float(priority)))
        result[f"{label}_priority_coverage"] = (
            float(sum(int(index) in visited for index in indices) / len(indices))
            if len(indices)
            else math.nan
        )
    return result


def _formal_row(
    *,
    record: Mapping[str, Any],
    algorithm: str,
    metrics: Mapping[str, Any],
    visit_order: Sequence[int],
    planning_time_s: float,
    evaluations: int,
    status: str,
    training_seed: Optional[int],
    planner_seed: Optional[int],
    checkpoint_hash: str,
    manifest_hash: str,
    protocol_hash: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    safe = bool(metrics.get("returned")) and not any(
        bool(metrics.get(field, False)) for field in VIOLATION_FIELDS
    )
    weighted = float(metrics.get("weighted_coverage", 0.0))
    row: Dict[str, Any] = {
        "scenario_id": str(record["id"]),
        "split": "formal_test",
        "node_count": int(record["node_count"]),
        "difficulty": str(record["difficulty"]),
        "constraint_type": str(record["constraint_type"]),
        "priority_layout": str(record["priority_layout"]),
        "algorithm": algorithm,
        "training_seed": training_seed,
        "planner_seed": planner_seed,
        "checkpoint_hash": checkpoint_hash,
        "manifest_hash": manifest_hash,
        "protocol_hash": protocol_hash,
        "safe": safe,
        "returned": bool(metrics.get("returned", False)),
        "coverage": float(metrics.get("coverage", 0.0)),
        "weighted_coverage": weighted,
        "safe_coverage": float(metrics.get("coverage", 0.0)) if safe else 0.0,
        "safe_weighted_coverage": weighted if safe else 0.0,
        "visited_count": int(metrics.get("visited_count", len(visit_order))),
        "energy_wh": float(metrics.get("energy_wh", math.nan)),
        "distance_m": float(metrics.get("distance_m", math.nan)),
        "time_s": float(metrics.get("time_s", math.nan)),
        "energy_utilization": float(
            metrics.get("energy_utilization", math.nan)
        ),
        "distance_utilization": float(
            metrics.get("distance_utilization", math.nan)
        ),
        "time_utilization": float(metrics.get("time_utilization", math.nan)),
        "min_remaining_soc": float(
            metrics.get("min_remaining_soc", math.nan)
        ),
        "termination_reason": str(
            metrics.get("termination_reason", "unknown")
        ),
        "energy_violation": bool(metrics.get("energy_violation", False)),
        "distance_violation": bool(metrics.get("distance_violation", False)),
        "time_violation": bool(metrics.get("time_violation", False)),
        "dynamics_violation": bool(metrics.get("dynamics_violation", False)),
        "planning_time_s": float(planning_time_s),
        "evaluations": int(evaluations),
        "planner_status": status,
        "oracle_lower": float(
            record["certificate"]["weighted_coverage_lower_bound"]
        ),
        "oracle_upper": float(
            record["certificate"]["weighted_coverage_upper_bound"]
        ),
        "optimality_gap": (
            None if metadata is None else metadata.get("optimality_gap")
        ),
        "solver_dual_bound": (
            None if metadata is None else metadata.get("objective_dual_bound")
        ),
        "optimality_certified": (
            None if metadata is None else metadata.get("optimality_certified")
        ),
    }
    row.update(_priority_stratum_coverage(visit_order, record["priorities"]))
    return row


def evaluate_formal(
    protocol_path: Path,
    formal_manifest_path: Path,
    training_freeze_path: Path,
    output_root: Path,
    *,
    family: str,
    device: str = "cuda",
    resume_existing: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    freeze = _verify_training_freeze(training_freeze_path, protocol_path)
    metadata, records = _require_passed_environment(
        formal_manifest_path, protocol_path
    )
    if metadata["split"] != "formal_test" or len(records) != 216:
        raise RuntimeError("正式评估只接受已审计的216场景formal_test。")
    if metadata.get("training_freeze_hash") != freeze["freeze_hash"]:
        raise RuntimeError("formal_test与训练冻结身份不一致。")
    if family not in {"learning", "main", "supplementary", "all"}:
        raise ValueError("family只能是learning、main、supplementary或all。")
    families = (
        ["learning", "main", "supplementary"]
        if family == "all"
        else [family]
    )
    scenario = legacy._load_scenario(
        ROOT / str(protocol["base_scenario_file"])
    )
    base_cfg = ppo.resolve_config(dict(scenario.as_training_inputs()["cfg"]))
    map_location = getattr(ppo, "device", "cpu") if device == "auto" else device
    reports: Dict[str, Any] = {}

    for current_family in families:
        run_dir = output_root / "formal_evaluation" / current_family
        if run_dir.exists() and not resume_existing and not dry_run:
            raise FileExistsError(
                f"正式评估目录已存在：{run_dir}；恢复请使用--resume-existing。"
            )
        if current_family == "learning":
            tasks = [
                ("learning", item, record, None)
                for item in freeze["checkpoints"]
                for record in records
            ]
        else:
            names = list(planner_names(current_family))
            tasks = []
            for name in names:
                planner_seeds = (
                    list(range(42, 52))
                    if name in {"aco", "ga", "sa", "pso"}
                    else [42]
                )
                for record in records:
                    for planner_seed in planner_seeds:
                        tasks.append(
                            ("baseline", name, record, planner_seed)
                        )
        expected = (
            7560
            if current_family == "learning"
            else 7128
            if current_family == "main"
            else 2592
        )
        if len(tasks) != expected:
            raise AssertionError(
                f"{current_family}任务数{len(tasks)}不等于{expected}。"
            )
        if dry_run:
            reports[current_family] = {
                "planned_rows": expected,
                "run_dir": str(run_dir),
            }
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "results.jsonl"
        rows = _read_jsonl(result_path) if resume_existing else []
        completed: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for row in rows:
            key = (
                row["algorithm"],
                row.get("training_seed"),
                row.get("planner_seed"),
                row["scenario_id"],
            )
            if key in completed:
                raise RuntimeError(f"{result_path}含重复正式任务键。")
            completed[key] = row
        _atomic_json(
            run_dir / "status.json",
            {
                "state": "running",
                "family": current_family,
                "completed": len(rows),
                "total": expected,
            },
        )

        model_cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        for kind, identity, record, planner_seed in tasks:
            if kind == "learning":
                checkpoint_item = dict(identity)
                algorithm = str(checkpoint_item["variant"])
                training_seed = int(checkpoint_item["training_seed"])
                key = (algorithm, training_seed, None, str(record["id"]))
            else:
                algorithm = str(identity)
                training_seed = None
                key = (algorithm, None, int(planner_seed), str(record["id"]))
            if key in completed:
                continue

            if kind == "learning":
                checkpoint_hash = str(checkpoint_item["sha256"])
                checkpoint = str(checkpoint_item["path"])
                if checkpoint not in model_cache:
                    model_cache[checkpoint] = ppo.load_checkpoint(
                        checkpoint, map_location=map_location
                    )
                model, payload = model_cache[checkpoint]
                cfg, wind = ppo.apply_frozen_domain_instance(
                    ppo.resolve_config(dict(payload["cfg"])),
                    scenario.wind_data,
                    record,
                )
                started = time.perf_counter()
                detail = ppo.plan_with_policy_improved(
                    model,
                    scenario.start_pos,
                    np.asarray(
                        record["inspection_points_xyz"], dtype=np.float32
                    ),
                    np.asarray(record["priorities"], dtype=np.float32),
                    scenario.terrain,
                    cfg,
                    wind,
                    return_details=True,
                    decode_mode="deterministic",
                )
                elapsed = time.perf_counter() - started
                metrics = dict(detail["metrics"])
                visit_order = list(detail.get("visit_order", ()))
                evaluations = 1
                status = "ok"
                result_metadata: Optional[Mapping[str, Any]] = None
                route_payload = {
                    "record": record,
                    "detail": detail,
                }
            else:
                problem = _problem_from_record(
                    ROOT / str(protocol["base_scenario_file"]),
                    scenario,
                    base_cfg,
                    record,
                )
                result = run_planner(
                    algorithm,
                    problem,
                    seed=int(planner_seed),
                )
                elapsed = float(result.runtime_s)
                metrics = dict(result.metrics)
                visit_order = list(result.visit_order)
                evaluations = int(result.evaluations)
                status = str(result.status)
                result_metadata = result.metadata
                checkpoint_hash = ""
                route_payload = {
                    "record": record,
                    "result": result.as_dict(),
                }
            row = _formal_row(
                record=record,
                algorithm=algorithm,
                metrics=metrics,
                visit_order=visit_order,
                planning_time_s=elapsed,
                evaluations=evaluations,
                status=status,
                training_seed=training_seed,
                planner_seed=(
                    int(planner_seed) if planner_seed is not None else None
                ),
                checkpoint_hash=checkpoint_hash,
                manifest_hash=str(metadata["manifest_hash"]),
                protocol_hash=str(protocol["protocol_hash"]),
                metadata=result_metadata,
            )
            if not _all_numeric_values_finite(row):
                raise RuntimeError(f"正式评估产生非有限值：{key}")
            route_name = "__".join(
                (
                    algorithm,
                    (
                        f"train{training_seed}"
                        if training_seed is not None
                        else f"plan{planner_seed}"
                    ),
                    str(record["id"]),
                )
            )
            _atomic_json(run_dir / "routes" / f"{route_name}.json", route_payload)
            rows.append(row)
            _atomic_text(result_path, _jsonl_text(rows))
            completed[key] = row
            _atomic_json(
                run_dir / "status.json",
                {
                    "state": "running",
                    "family": current_family,
                    "completed": len(rows),
                    "total": expected,
                    "last_key": list(key),
                },
            )
        _write_csv(run_dir / "results.csv", rows)
        _atomic_json(
            run_dir / "status.json",
            {
                "state": "completed",
                "family": current_family,
                "completed": len(rows),
                "total": expected,
            },
        )
        reports[current_family] = {
            "completed": len(rows),
            "expected": expected,
            "run_dir": str(run_dir),
        }
    return {
        "protocol_hash": protocol["protocol_hash"],
        "formal_manifest_hash": metadata["manifest_hash"],
        "families": reports,
    }


def _holm_adjust(p_values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (float(p_values[key]), key))
    adjusted: Dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (total - rank) * float(p_values[key]))
        adjusted[key] = min(1.0, running)
    return adjusted


def _rank_biserial(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.abs(values) > 1e-15]
    if values.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(values), method="average")
    denominator = float(np.sum(ranks))
    return float(
        (np.sum(ranks[values > 0.0]) - np.sum(ranks[values < 0.0]))
        / denominator
    )


def _hodges_lehmann_paired(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    walsh = [
        (float(values[left]) + float(values[right])) / 2.0
        for left in range(values.size)
        for right in range(left, values.size)
    ]
    return float(statistics.median(walsh))


def _hierarchical_difference_ci(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    replicates: int = 10_000,
) -> Tuple[float, float]:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape or left_arr.ndim != 2:
        raise ValueError("层级bootstrap要求相同的[场景,种子]矩阵。")
    rng = np.random.default_rng(seed)
    scenario_count, seed_count = left_arr.shape
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        scene_indices = rng.integers(0, scenario_count, scenario_count)
        seed_indices = rng.integers(
            0, seed_count, size=(scenario_count, seed_count)
        )
        selected_left = left_arr[scene_indices]
        selected_right = right_arr[scene_indices]
        rows = np.arange(scenario_count)[:, None]
        samples[index] = float(
            np.mean(
                selected_left[rows, seed_indices]
                - selected_right[rows, seed_indices]
            )
        )
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def analyze_formal(
    protocol_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    learning_path = (
        output_root / "formal_evaluation" / "learning" / "results.jsonl"
    )
    learning_rows = _read_jsonl(learning_path)
    if len(learning_rows) != 7560:
        raise RuntimeError("学习模型正式结果必须完整包含7560行。")
    if not all(_all_numeric_values_finite(row) for row in learning_rows):
        raise RuntimeError("学习模型正式结果包含非有限值。")
    core = ("full", "ppo_mlp", "a2c_pointer")
    scenarios = sorted({str(row["scenario_id"]) for row in learning_rows})
    seeds = [int(value) for value in protocol["qualification"]["training_seeds"]]
    if len(scenarios) != 216:
        raise RuntimeError("正式学习结果的场景数不是216。")

    def matrix(algorithm: str, field: str) -> np.ndarray:
        lookup = {
            (str(row["scenario_id"]), int(row["training_seed"])): float(
                row[field]
            )
            for row in learning_rows
            if str(row["algorithm"]) == algorithm
        }
        expected = {(scenario, seed) for scenario in scenarios for seed in seeds}
        if set(lookup) != expected:
            raise RuntimeError(f"{algorithm}的{field}任务键不完整。")
        return np.asarray(
            [[lookup[(scenario, seed)] for seed in seeds] for scenario in scenarios],
            dtype=np.float64,
        )

    effectiveness = {
        algorithm: matrix(algorithm, "safe_weighted_coverage")
        for algorithm in core
    }
    safety = {
        algorithm: matrix(algorithm, "safe")
        for algorithm in core
    }
    planning = {
        algorithm: matrix(algorithm, "planning_time_s")
        for algorithm in core
    }
    scene_means = {
        algorithm: values.mean(axis=1)
        for algorithm, values in effectiveness.items()
    }
    friedman_inputs = [scene_means[algorithm] for algorithm in core]
    if all(
        np.allclose(friedman_inputs[0], values, rtol=0.0, atol=1e-15)
        for values in friedman_inputs[1:]
    ):
        friedman_statistic, friedman_p_value = 0.0, 1.0
    else:
        friedman_result = stats.friedmanchisquare(*friedman_inputs)
        friedman_statistic = float(friedman_result.statistic)
        friedman_p_value = float(friedman_result.pvalue)
        if not math.isfinite(friedman_statistic) or not math.isfinite(
            friedman_p_value
        ):
            raise RuntimeError("Friedman检验返回非有限值，拒绝生成判优结论。")
    raw_p: Dict[str, float] = {}
    comparisons: Dict[str, Dict[str, Any]] = {}
    for index, comparator in enumerate(("ppo_mlp", "a2c_pointer")):
        differences = scene_means["full"] - scene_means[comparator]
        if np.all(np.abs(differences) <= 1e-15):
            wilcoxon_statistic, p_value = 0.0, 1.0
        else:
            test = stats.wilcoxon(
                scene_means["full"],
                scene_means[comparator],
                alternative="two-sided",
                zero_method="wilcox",
            )
            wilcoxon_statistic, p_value = float(test.statistic), float(test.pvalue)
        raw_p[comparator] = p_value
        ci_low, ci_high = _hierarchical_difference_ci(
            effectiveness["full"],
            effectiveness[comparator],
            seed=2026072700 + index,
        )
        seed_directions = [
            float(
                np.mean(
                    effectiveness["full"][:, seed_index]
                    - effectiveness[comparator][:, seed_index]
                )
            )
            for seed_index in range(len(seeds))
        ]
        comparisons[comparator] = {
            "wilcoxon_statistic": wilcoxon_statistic,
            "raw_p": p_value,
            "mean_difference": float(np.mean(differences)),
            "median_difference": float(np.median(differences)),
            "rank_biserial": _rank_biserial(differences),
            "hodges_lehmann": _hodges_lehmann_paired(differences),
            "hierarchical_bootstrap_95ci": [ci_low, ci_high],
            "seed_mean_differences": seed_directions,
            "all_five_seed_directions_positive": all(
                value > 0.0 for value in seed_directions
            ),
        }
    adjusted = _holm_adjust(raw_p)
    for comparator, value in adjusted.items():
        comparisons[comparator]["holm_p"] = value

    safety_ci = _hierarchical_difference_ci(
        safety["full"], safety["ppo_mlp"], seed=2026072799
    )
    aggregate = {
        algorithm: {
            "safe_weighted_coverage_mean": float(
                np.mean(effectiveness[algorithm])
            ),
            "safe_rate": float(np.mean(safety[algorithm])),
            "planning_time_median_s": float(np.median(planning[algorithm])),
        }
        for algorithm in core
    }
    full_vector = aggregate["full"]
    dominated_by: List[str] = []
    for comparator in ("ppo_mlp", "a2c_pointer"):
        other = aggregate[comparator]
        no_worse = (
            other["safe_weighted_coverage_mean"]
            >= full_vector["safe_weighted_coverage_mean"] - 1e-12
            and other["safe_rate"] >= full_vector["safe_rate"] - 1e-12
            and other["planning_time_median_s"]
            <= full_vector["planning_time_median_s"] + 1e-12
        )
        strictly = (
            other["safe_weighted_coverage_mean"]
            > full_vector["safe_weighted_coverage_mean"] + 1e-12
            or other["safe_rate"] > full_vector["safe_rate"] + 1e-12
            or other["planning_time_median_s"]
            < full_vector["planning_time_median_s"] - 1e-12
        )
        if no_worse and strictly:
            dominated_by.append(comparator)

    highest_effectiveness = full_vector[
        "safe_weighted_coverage_mean"
    ] > max(
        aggregate["ppo_mlp"]["safe_weighted_coverage_mean"],
        aggregate["a2c_pointer"]["safe_weighted_coverage_mean"],
    ) + 1e-12
    friedman_significant = friedman_p_value < 0.05
    pairwise_supported = all(
        float(result["holm_p"]) < 0.05
        and float(result["rank_biserial"]) > 0.0
        and float(result["hodges_lehmann"]) > 0.0
        and float(result["hierarchical_bootstrap_95ci"][0]) > 0.0
        and bool(result["all_five_seed_directions_positive"])
        for result in comparisons.values()
    )
    safety_noninferior = float(safety_ci[0]) >= -0.02
    pareto_not_dominated = not dominated_by
    best_learning = all(
        (
            highest_effectiveness,
            friedman_significant,
            pairwise_supported,
            safety_noninferior,
            pareto_not_dominated,
        )
    )

    traditional_summary: Dict[str, Any] = {}
    main_path = output_root / "formal_evaluation" / "main" / "results.jsonl"
    if main_path.exists():
        main_rows = _read_jsonl(main_path)
        if len(main_rows) != 7128:
            raise RuntimeError("主传统基线结果存在但不是完整7128行。")
        for algorithm in planner_names("main"):
            values = [
                float(row["safe_weighted_coverage"])
                for row in main_rows
                if str(row["algorithm"]) == algorithm
            ]
            times = [
                float(row["planning_time_s"])
                for row in main_rows
                if str(row["algorithm"]) == algorithm
            ]
            traditional_summary[algorithm] = {
                "safe_weighted_coverage_mean": float(np.mean(values)),
                "planning_time_median_s": float(np.median(times)),
            }

    report = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "primary_metric": "safe_weighted_coverage",
        "statistical_unit": "scenario",
        "friedman": {
            "statistic": friedman_statistic,
            "p_value": friedman_p_value,
            "significant": friedman_significant,
        },
        "comparisons": comparisons,
        "aggregate": aggregate,
        "safety_noninferiority": {
            "margin": -0.02,
            "full_minus_ppo_mlp_95ci": list(safety_ci),
            "passed": safety_noninferior,
        },
        "pareto": {
            "full_dominated_by": dominated_by,
            "not_dominated": pareto_not_dominated,
        },
        "champion_conditions": {
            "highest_safe_weighted_coverage": highest_effectiveness,
            "friedman_significant": friedman_significant,
            "holm_effect_ci_seed_direction_supported": pairwise_supported,
            "safety_noninferior_to_ppo_mlp": safety_noninferior,
            "not_pareto_dominated_by_learning_algorithm": pareto_not_dominated,
        },
        "ppo_pointer_best_learning_algorithm": best_learning,
        "allowed_conclusion": (
            "在复杂山区多约束无人机巡检仿真场景中，"
            "PPO＋Pointer是本研究比较范围内综合证据支持的最佳学习算法。"
            if best_learning
            else "预注册条件未全部满足，不能宣称PPO＋Pointer为最佳学习算法。"
        ),
        "traditional_methods": traditional_summary,
    }
    report["analysis_hash"] = _canonical_hash(
        report, excluded=("analysis_hash",)
    )
    analysis_dir = output_root / "formal_analysis"
    _atomic_json(analysis_dir / "primary_analysis.json", report)
    source_rows = [
        {
            "scenario_id": scenario,
            **{
                f"{algorithm}_scene_mean": float(
                    scene_means[algorithm][index]
                )
                for algorithm in core
            },
        }
        for index, scenario in enumerate(scenarios)
    ]
    _write_csv(analysis_dir / "primary_source_data.csv", source_rows)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPO+Pointer困难约束纠偏实验的模型无关认证与资格门禁。"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--split",
        choices=("training_pool", "validation", "formal_test"),
        required=True,
    )
    prepare.add_argument("--resume-existing", action="store_true")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.add_argument("--quick-limit", type=int)
    prepare.add_argument("--certification-time-limit-s", type=float)
    prepare.add_argument("--max-attempts-per-cell", type=int)

    audit = subparsers.add_parser("audit-environment")
    audit.add_argument("--manifest", type=Path, required=True)

    qualify = subparsers.add_parser("qualify-existing")
    qualify.add_argument("--manifest", type=Path, required=True)
    qualify.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    qualify.add_argument("--resume-existing", action="store_true")
    qualify.add_argument("--quick-limit", type=int)

    train_grid = subparsers.add_parser("train-grid")
    train_grid.add_argument("--stage", choices=("pilot", "formal"), required=True)
    train_grid.add_argument("--training-manifest", type=Path, required=True)
    train_grid.add_argument("--validation-manifest", type=Path, required=True)
    train_grid.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    train_grid.add_argument("--resume-existing", action="store_true")
    train_grid.add_argument("--variant", action="append", dest="variants")
    train_grid.add_argument("--seed", action="append", dest="seeds", type=int)
    train_grid.add_argument("--dry-run", action="store_true")

    assess = subparsers.add_parser("assess-pilot")
    assess.add_argument("--validation-manifest", type=Path, required=True)
    assess.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")

    freeze = subparsers.add_parser("freeze-training")
    freeze.add_argument("--training-manifest", type=Path, required=True)
    freeze.add_argument("--validation-manifest", type=Path, required=True)

    formal_test = subparsers.add_parser("prepare-formal-test")
    formal_test.add_argument("--training-freeze", type=Path, required=True)
    formal_test.add_argument("--resume-existing", action="store_true")
    formal_test.add_argument("--certification-time-limit-s", type=float)

    formal_eval = subparsers.add_parser("evaluate-formal")
    formal_eval.add_argument("--formal-manifest", type=Path, required=True)
    formal_eval.add_argument("--training-freeze", type=Path, required=True)
    formal_eval.add_argument(
        "--family",
        choices=("learning", "main", "supplementary", "all"),
        required=True,
    )
    formal_eval.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    formal_eval.add_argument("--resume-existing", action="store_true")
    formal_eval.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("analyze-formal")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    protocol_path = args.protocol.resolve()
    output_root = args.output_root.resolve()
    if args.command == "prepare":
        result = generate_split(
            protocol_path,
            output_root,
            args.split,
            resume_existing=bool(args.resume_existing),
            dry_run=bool(args.dry_run),
            quick_limit=args.quick_limit,
            certification_time_limit_s=args.certification_time_limit_s,
            max_attempts_per_cell=args.max_attempts_per_cell,
        )
    elif args.command == "audit-environment":
        result = audit_environment(args.manifest.resolve(), protocol_path)
    elif args.command == "qualify-existing":
        result = qualify_existing_checkpoints(
            protocol_path,
            args.manifest.resolve(),
            output_root,
            device=args.device,
            resume_existing=bool(args.resume_existing),
            quick_limit=args.quick_limit,
        )
    elif args.command == "train-grid":
        result = run_training_grid(
            protocol_path,
            args.training_manifest.resolve(),
            args.validation_manifest.resolve(),
            output_root,
            stage=args.stage,
            device=args.device,
            resume_existing=bool(args.resume_existing),
            variants=args.variants,
            seeds=args.seeds,
            dry_run=bool(args.dry_run),
        )
    elif args.command == "assess-pilot":
        result = assess_pilot(
            protocol_path,
            args.validation_manifest.resolve(),
            output_root,
            device=args.device,
        )
    elif args.command == "freeze-training":
        result = freeze_training_branch(
            protocol_path,
            args.training_manifest.resolve(),
            args.validation_manifest.resolve(),
            output_root,
        )
    elif args.command == "prepare-formal-test":
        result = generate_split(
            protocol_path,
            output_root,
            "formal_test",
            resume_existing=bool(args.resume_existing),
            certification_time_limit_s=args.certification_time_limit_s,
            training_freeze_path=args.training_freeze.resolve(),
        )
    elif args.command == "evaluate-formal":
        result = evaluate_formal(
            protocol_path,
            args.formal_manifest.resolve(),
            args.training_freeze.resolve(),
            output_root,
            family=args.family,
            device=args.device,
            resume_existing=bool(args.resume_existing),
            dry_run=bool(args.dry_run),
        )
    elif args.command == "analyze-formal":
        result = analyze_formal(protocol_path, output_root)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

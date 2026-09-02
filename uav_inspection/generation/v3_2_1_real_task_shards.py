#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel, resumable v3.2.1 real-task certification with disjoint map shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from uav_inspection.core import final_python_ppo_pointer as ppo
from uav_inspection.experiments import paper_multimap_experiments as multimap
from uav_inspection.experiments import paper_v3_2_experiments as v32
from uav_inspection.generation import v3_2_12_parametric_certificate_search as certificate_search
from python_classical_algs.common import (
    MissionEvaluator,
    PlannerBudget,
    SearchController,
    make_problem,
)


ROOT = WORKSPACE_ROOT
DEFAULT_PROTOCOL = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_12/protocol.json"
DEFAULT_OUTPUT = ROOT / "paper_runs/multimap_v3_2_12"
DEFAULT_MAP_ROOT = ROOT / "map_data/multimap_v3_1"


def _shard_dir(output_root: Path, run_label: str, start: int, stop: int) -> Path:
    return Path(output_root) / "formal_evaluation" / "real_task_shards" / str(run_label) / f"maps_{start:02d}_{stop:02d}"


def generate_shard(
    protocol_path: Path,
    output_root: Path,
    map_root: Path,
    start: int,
    stop: int,
    *,
    task_limit: int = 9,
    run_label: str = "formal",
) -> Dict[str, Any]:
    protocol = v32.load_v3_2_protocol(protocol_path)
    if protocol.get("protocol_version") not in {
        "multimap_generalization_v3_2_1",
        "multimap_generalization_v3_2_2",
        "multimap_generalization_v3_2_3",
        "multimap_generalization_v3_2_4",
        "multimap_generalization_v3_2_5",
        "multimap_generalization_v3_2_6",
        "multimap_generalization_v3_2_7",
        "multimap_generalization_v3_2_8",
        "multimap_generalization_v3_2_9",
        "multimap_generalization_v3_2_10",
        "multimap_generalization_v3_2_11",
        "multimap_generalization_v3_2_12",
        "multimap_generalization_v3_2_13",
        "multimap_generalization_v3_2_14",
    }:
        raise RuntimeError("并行真实任务分片只接受v3.2.1")
    registry_path = Path(map_root) / "real" / "map_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    maps = list(registry.get("maps") or ())
    if not 0 <= int(start) < int(stop) <= len(maps):
        raise ValueError("map shard range is invalid")
    catalog = v32.checkpoint_catalog(protocol_path, output_root=output_root)
    provider = multimap.FrozenMapProvider.from_registries(map_root, [registry_path])
    contexts = v32._load_real_corridor_contexts(output_root, protocol)
    if not 1 <= int(task_limit) <= 9:
        raise ValueError("task_limit must be between 1 and 9")
    if not str(run_label).replace("_", "").isalnum():
        raise ValueError("run_label must contain only letters, digits, and underscores")
    directory = _shard_dir(output_root, run_label, start, stop)
    destination = directory / "records.jsonl"
    existing = v32._read_jsonl(destination) if destination.is_file() else []
    accepted = {str(row["id"]): dict(row) for row in existing}
    parent = json.loads(multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8"))
    max_attempts = int(protocol.get("pretest_bounded_certificate_search", {}).get("fixed_geometry_attempt_count", protocol["task_generation"]["maximum_candidate_attempts_per_task"]))
    for map_index in range(int(start), int(stop)):
        record = maps[map_index]
        full_bundle = multimap._load_map_bundle(map_root, record)
        context_record = contexts[str(record["map_id"])]
        corridor = v32._corridor_bundle(full_bundle, context_record)
        for road_index in (0, 1):
            context = dict(context_record["contexts"][road_index])
            depot = list(context["start_xy"])
            for task_index in range(int(task_limit)):
                task_id = f"real_test__{record['map_id']}__road_{road_index:02d}__task_{task_index:02d}"
                if task_id in accepted:
                    continue
                design = multimap._task_design(map_index, task_index)
                minimum, interval = multimap._effective_task_radius_range(
                    record, corridor, protocol,
                    node_count=int(design["node_count"]),
                    difficulty=str(design["difficulty"]),
                    depot_override_xy=depot,
                )
                result: Dict[str, Any] | None = None
                for attempt in range(max_attempts):
                    try:
                        candidate = multimap._task_candidate(
                            record, corridor, protocol, parent,
                            split="real_test", map_index=map_index, task_index=task_index,
                            attempt=attempt, master_seed=int(protocol["map_splits"]["synthetic_test"]["seed"]),
                            geometry_radius_range_m=interval,
                            geometry_minimum_feasible_radius_m=minimum,
                            seed_namespace=f"road_{road_index:02d}", road_index=road_index,
                            depot_override_xy=depot,
                        )
                    except RuntimeError:
                        continue
                    candidate["road_context_hash"] = str(context_record["context_hash"])
                    candidate["road_context_definition"] = str(protocol["real_corridor_contexts"]["definition"])
                    result = certificate_search.certify_candidate_with_parametric_search(candidate, provider, parent, protocol)
                    if result is not None:
                        break
                if result is None:
                    raise RuntimeError(f"{task_id} exhausted {max_attempts} model-free MILP candidates")
                accepted[task_id] = result
                directory.mkdir(parents=True, exist_ok=True)
                destination.write_text(v32._jsonl(accepted[key] for key in sorted(accepted)), encoding="utf-8")
    rows = [accepted[key] for key in sorted(accepted)]
    if len(rows) != (int(stop) - int(start)) * 2 * int(task_limit):
        raise RuntimeError("real task shard has an incomplete task count")
    manifest = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "checkpoint_catalog_hash": catalog["catalog_hash"],
        "map_index_start": int(start),
        "map_index_stop": int(stop),
        "run_label": str(run_label),
        "task_limit": int(task_limit),
        "scenario_count": len(rows),
        "records_sha256": v32._sha256_file(destination),
        "context_manifest_sha256": v32._sha256_file(
            Path(str(protocol.get("real_corridor_asset_root", Path(output_root) / "real_corridor_assets"))) / "manifest.json"
        ),
        "algorithm_results_used": False,
    }
    manifest["manifest_hash"] = v32._canonical_hash(manifest, excluded=("manifest_hash",))
    v32._write_json(directory / "manifest.json", manifest)
    return manifest


def merge_shards(protocol_path: Path, output_root: Path, map_root: Path) -> Dict[str, Any]:
    protocol = v32.load_v3_2_protocol(protocol_path)
    registry_path = Path(map_root) / "real" / "map_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    directories = sorted((Path(output_root) / "formal_evaluation" / "real_task_shards" / "formal").glob("maps_*"))
    rows: List[Dict[str, Any]] = []
    manifests: List[Dict[str, Any]] = []
    for directory in directories:
        manifest_path = directory / "manifest.json"
        records_path = directory / "records.jsonl"
        if not manifest_path.is_file() or not records_path.is_file():
            raise RuntimeError(f"incomplete real-task shard: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_hash") != protocol["protocol_hash"]:
            raise RuntimeError("real-task shard protocol mismatch")
        if manifest.get("run_label") != "formal" or manifest.get("task_limit") != 9:
            raise RuntimeError("only complete formal real-task shards can be merged")
        if manifest.get("records_sha256") != v32._sha256_file(records_path):
            raise RuntimeError("real-task shard record hash mismatch")
        manifests.append(manifest)
        rows.extend(v32._read_jsonl(records_path))
    ids = [str(row.get("id", "")) for row in rows]
    if len(rows) != 144 or len(ids) != len(set(ids)):
        raise RuntimeError("real-task shards do not form exactly 144 unique rows")
    audit = v32.audit_real_test_records(rows, registry, protocol)
    if not audit["passed"]:
        raise RuntimeError("merged real-task audit failed: " + "; ".join(audit["reasons"][:5]))
    destination = Path(output_root) / "formal_evaluation" / "real_tasks_parallel" / "records.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(v32._jsonl(sorted(rows, key=lambda row: str(row["id"]))), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "scenario_count": 144,
        "records_sha256": v32._sha256_file(destination),
        "shard_manifest_hashes": sorted(str(item["manifest_hash"]) for item in manifests),
        "algorithm_results_used": False,
    }
    manifest["manifest_hash"] = v32._canonical_hash(manifest, excluded=("manifest_hash",))
    v32._write_json(destination.parent / "manifest.json", manifest)
    v32._write_json(destination.parent / "audit.json", audit)
    return {"manifest": manifest, "audit": audit}


def _sentinel_problem(
    candidate: Mapping[str, Any],
    provider: Any,
    protocol: Mapping[str, Any],
):
    """构造仅用于接口连通性检查的宽预算问题，不执行难度认证。"""

    working = dict(candidate)
    for parameter in certificate_search.RESOURCE_PARAMETERS:
        _lower, upper = certificate_search._bounds(protocol, parameter)
        working[parameter] = float(upper)
    context = provider(working)
    base_cfg = ppo.resolve_config(
        {
            "reward_schema": "multimap_v3_1",
            "coordinate_scale_m_per_unit": context["cfg_overrides"][
                "coordinate_scale_m_per_unit"
            ],
            "point_z_mode": "terrain",
            "terrain_clearance_m": 18.0,
            "service_times_s": working["service_times_s"],
        }
    )
    scenario_cfg, scenario_wind = ppo.apply_frozen_domain_instance(
        base_cfg, context["wind_data"], working
    )
    return make_problem(
        context["start_pos"],
        np.asarray(working["inspection_points_xyz"], dtype=np.float32),
        np.asarray(working["priorities"], dtype=np.float32),
        context["terrain"],
        scenario_cfg,
        scenario_wind,
        name=f"engineering_sentinel__{working['id']}",
    )


def generate_sentinel_shard(
    protocol_path: Path,
    output_root: Path,
    map_root: Path,
    start: int,
    stop: int,
) -> Dict[str, Any]:
    """每个真实 DSM 上下文生成一条论文外的端到端接口哨兵。"""

    protocol = v32.load_v3_2_protocol(protocol_path)
    if protocol.get("protocol_version") != "multimap_generalization_v3_2_12":
        raise RuntimeError("工程哨兵只属于 v3.2.12，且不得写入旧协议目录")
    registry_path = Path(map_root) / "real" / "map_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    maps = list(registry.get("maps") or ())
    if not 0 <= int(start) < int(stop) <= len(maps):
        raise ValueError("map shard range is invalid")
    provider = multimap.FrozenMapProvider.from_registries(
        map_root, [registry_path]
    )
    contexts = v32._load_real_corridor_contexts(output_root, protocol)
    parent = json.loads(
        multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
    )
    directory = (
        Path(output_root)
        / "engineering_sentinels"
        / f"maps_{int(start):02d}_{int(stop):02d}"
    )
    destination = directory / "sentinels.jsonl"
    rows: List[Dict[str, Any]] = []
    maximum_attempts = int(
        protocol["pretest_parametric_certificate_search"][
            "fixed_geometry_attempt_count"
        ]
    )
    for map_index in range(int(start), int(stop)):
        map_record = maps[map_index]
        full_bundle = multimap._load_map_bundle(map_root, map_record)
        context_record = contexts[str(map_record["map_id"])]
        corridor = v32._corridor_bundle(full_bundle, context_record)
        for road_index in (0, 1):
            context = dict(context_record["contexts"][road_index])
            depot = list(context["start_xy"])
            design = multimap._task_design(map_index, 0)
            minimum, interval = multimap._effective_task_radius_range(
                map_record,
                corridor,
                protocol,
                node_count=int(design["node_count"]),
                difficulty=str(design["difficulty"]),
                depot_override_xy=depot,
            )
            candidate = None
            for attempt in range(maximum_attempts):
                try:
                    candidate = multimap._task_candidate(
                        map_record,
                        corridor,
                        protocol,
                        parent,
                        split="engineering_sentinel",
                        map_index=map_index,
                        task_index=0,
                        attempt=attempt,
                        master_seed=int(
                            protocol["map_splits"]["synthetic_test"]["seed"]
                        ),
                        geometry_radius_range_m=interval,
                        geometry_minimum_feasible_radius_m=minimum,
                        seed_namespace=f"sentinel_road_{road_index:02d}",
                        road_index=road_index,
                        depot_override_xy=depot,
                    )
                    break
                except RuntimeError:
                    continue
            if candidate is None:
                raise RuntimeError(
                    f"{map_record['map_id']} context {road_index} has no "
                    "geometry-feasible engineering sentinel"
                )
            candidate["road_context_hash"] = str(context_record["context_hash"])
            problem = _sentinel_problem(candidate, provider, protocol)
            evaluator = MissionEvaluator(problem)
            start_xy = np.asarray(problem.start_pos[:2], dtype=np.float64)
            point_xy = np.asarray(problem.points[:, :2], dtype=np.float64)
            order = np.argsort(
                np.linalg.norm(point_xy - start_xy[None, :], axis=1),
                kind="mergesort",
            )
            chosen = None
            evaluation = None
            for node in order:
                trial = evaluator.evaluate_order([int(node)])
                if trial.returned:
                    chosen, evaluation = int(node), trial
                    break
            if chosen is None or evaluation is None:
                raise RuntimeError(
                    f"{map_record['map_id']} context {road_index} cannot "
                    "complete a one-node safe-return sentinel"
                )
            controller = SearchController(PlannerBudget(max_evaluations=1))
            if not controller.consume():
                raise AssertionError("sentinel evaluation counter failed")
            replay = evaluator.build_result(
                "engineering_sentinel",
                evaluation,
                controller,
                seed=0,
                metadata={"paper_eligible": False},
            )
            payload = replay.as_dict()
            if (
                not bool(payload["metrics"].get("returned"))
                or int(payload["metrics"].get("visited_count", 0)) != 1
            ):
                raise RuntimeError("engineering sentinel replay was not safe")
            row = {
                "schema_version": 1,
                "id": (
                    f"engineering_sentinel__{map_record['map_id']}"
                    f"__context_{road_index:02d}"
                ),
                "protocol_hash": protocol["protocol_hash"],
                "map_id": str(map_record["map_id"]),
                "map_hash": str(map_record["map_hash"]),
                "road_index": int(road_index),
                "road_context_hash": str(context_record["context_hash"]),
                "geometry_hash": multimap._canonical_hash(
                    candidate,
                    excluded=(
                        "certificate",
                        "certificate_search_trace",
                        "task_hash",
                    ),
                ),
                "selected_node": chosen,
                "returned": True,
                "visited_count": 1,
                "scenario_hash": str(payload["scenario_hash"]),
                "paper_eligible": False,
                "formal_task": False,
                "difficulty_certificate_attempted": False,
                "algorithm_results_used": False,
            }
            row["sentinel_hash"] = v32._canonical_hash(
                row, excluded=("sentinel_hash",)
            )
            rows.append(row)
    directory.mkdir(parents=True, exist_ok=True)
    destination.write_text(v32._jsonl(rows), encoding="utf-8")
    expected = (int(stop) - int(start)) * 2
    if len(rows) != expected:
        raise RuntimeError("engineering sentinel shard count mismatch")
    manifest = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "map_index_start": int(start),
        "map_index_stop": int(stop),
        "sentinel_count": len(rows),
        "records_sha256": v32._sha256_file(destination),
        "paper_eligible": False,
        "formal_task": False,
        "algorithm_results_used": False,
    }
    manifest["manifest_hash"] = v32._canonical_hash(
        manifest, excluded=("manifest_hash",)
    )
    v32._write_json(directory / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v3.2.1 parallel real task certification")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("generate")
    shard.add_argument("--map-start", type=int, required=True)
    shard.add_argument("--map-stop", type=int, required=True)
    shard.add_argument("--task-limit", type=int, default=9)
    shard.add_argument("--run-label", type=str, default="formal")
    sentinel = commands.add_parser("sentinel")
    sentinel.add_argument("--map-start", type=int, required=True)
    sentinel.add_argument("--map-stop", type=int, required=True)
    commands.add_parser("merge")
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = generate_shard(
            args.protocol,
            args.output_root,
            args.map_root,
            args.map_start,
            args.map_stop,
            task_limit=args.task_limit,
            run_label=args.run_label,
        )
    elif args.command == "sentinel":
        result = generate_sentinel_shard(
            args.protocol,
            args.output_root,
            args.map_root,
            args.map_start,
            args.map_stop,
        )
    else:
        result = merge_shards(args.protocol, args.output_root, args.map_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

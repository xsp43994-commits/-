#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3.2 正式评价的冻结编排与检查点审计。

本模块刻意只负责把预注册任务、模型检查点和扰动条件展开为唯一结果行。
执行器必须消费本模块写出的行清单，不能一边看结果一边增删任务或重复次数。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from uav_inspection.experiments import paper_multimap_experiments as multimap


ROOT = WORKSPACE_ROOT
DEFAULT_PROTOCOL = ROOT / "paper_runs/protocols/multimap_generalization_v3_2/protocol.json"
DEFAULT_OUTPUT = ROOT / "paper_runs/multimap_v3_2"
PARENT_OUTPUT = ROOT / "paper_runs/multimap_v3_1"


def _canonical_hash(value: Mapping[str, Any], excluded: Sequence[str] = ()) -> str:
    payload = {key: item for key, item in value.items() if key not in set(excluded)}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v3_2_protocol(path: Path = DEFAULT_PROTOCOL) -> Dict[str, Any]:
    protocol = multimap.load_protocol(path)
    if protocol.get("protocol_version") not in {
        "multimap_generalization_v3_2",
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
        raise RuntimeError("正式评价编排只接受 multimap_generalization_v3_2 协议。")
    formal = dict(protocol.get("formal_evaluation") or {})
    if formal.get("counts", {}).get("total") != 21648:
        raise RuntimeError("v3.2 正式评价总行数必须固定为 21,648。")
    if "ppo_mlp" in set(formal.get("active_learning_variants") or ()):
        raise RuntimeError("归档 ppo_mlp 不得进入 v3.2 正式评价。")
    return protocol


def _load_real_corridor_contexts(output_root: Path, protocol: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Load post-v3.2 preflight-certified real-road contexts."""
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
        raise RuntimeError("真实道路走廊任务必须使用预检资产")
    asset_root = Path(str(protocol.get("real_corridor_asset_root", Path(output_root) / "real_corridor_assets")))
    path = asset_root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少真实道路走廊资产：{path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_asset_protocol_hash = str(protocol.get("real_corridor_asset_protocol_hash", protocol.get("protocol_hash")))
    if manifest.get("protocol_hash") != expected_asset_protocol_hash:
        raise RuntimeError("真实道路走廊资产与当前协议哈希不一致")
    rows = {str(row["map_id"]): dict(row) for row in manifest.get("maps", ())}
    if len(rows) != 8 or any(len(row.get("contexts", ())) != 2 for row in rows.values()):
        raise RuntimeError("真实道路走廊资产必须为8张地图各提供两个上下文")
    return rows


def _corridor_bundle(full_bundle: Mapping[str, Any], context_record: Mapping[str, Any]) -> Dict[str, Any]:
    asset = Path(str(context_record["asset_path"]))
    if not asset.is_file() or _sha256_file(asset) != str(context_record["asset_sha256"]):
        raise RuntimeError(f"道路走廊资产哈希漂移：{asset}")
    with np.load(asset, allow_pickle=False) as data:
        points = np.asarray(data["road_points"], dtype=np.float32)
        offsets = np.asarray(data["road_offsets"], dtype=np.int32)
    roads = [points[left:right].copy() for left, right in zip(offsets[:-1], offsets[1:]) if int(right) - int(left) >= 2]
    if not roads:
        raise RuntimeError("道路走廊资产不包含有效道路段")
    result = dict(full_bundle)
    result["roads"] = roads
    return result


def checkpoint_catalog(protocol_path: Path = DEFAULT_PROTOCOL, *, output_root: Path = DEFAULT_OUTPUT) -> Dict[str, Any]:
    """审计35个论文有效 checkpoint，旧 ppo_mlp 仅可作为归档证据存在。"""
    protocol = load_v3_2_protocol(protocol_path)
    seeds = [int(seed) for seed in protocol["formal_evaluation"]["training_seeds"]]
    active = list(protocol["formal_evaluation"]["active_learning_variants"])
    new_variants = set(protocol["formal_training"]["new_training_variants"])
    new_variant_root = (
        Path(output_root)
        if protocol.get("protocol_version") == "multimap_generalization_v3_2"
        else ROOT / "paper_runs/multimap_v3_2"
    )
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for variant in active:
        root = new_variant_root if variant in new_variants else PARENT_OUTPUT
        for seed in seeds:
            checkpoint = root / "formal_training" / f"formal_{variant}_seed{seed}_3000ep" / "best_safe.pt"
            status = checkpoint.parent / "status.json"
            if not checkpoint.is_file() or not status.is_file():
                missing.append(str(checkpoint))
                continue
            status_payload = json.loads(status.read_text(encoding="utf-8"))
            if str(status_payload.get("state")) not in {"completed", "complete"}:
                missing.append(f"incomplete:{checkpoint}")
                continue
            rows.append({
                "variant": variant,
                "training_seed": seed,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "status_sha256": _sha256_file(status),
                "source_protocol": "v3_2" if variant in new_variants else "v3_1_17_parent",
            })
    if missing:
        raise RuntimeError("有效检查点尚未齐全：" + "; ".join(missing[:5]))
    if len(rows) != 35 or len({(row["variant"], row["training_seed"]) for row in rows}) != 35:
        raise RuntimeError("检查点目录必须恰为7变体×5训练种子。")
    catalog = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "active_variant_count": 7,
        "model_count": 35,
        "archived_excluded_variants": list(protocol["formal_training"]["archived_excluded_variants"]),
        "rows": sorted(rows, key=lambda row: (row["variant"], row["training_seed"])),
    }
    catalog["catalog_hash"] = _canonical_hash(catalog, excluded=("catalog_hash",))
    return catalog


def _task_key(task: Mapping[str, Any]) -> str:
    task_id = str(task.get("id", ""))
    task_hash = str(task.get("task_hash", ""))
    if not task_id or not task_hash:
        raise ValueError("测试任务缺少 id 或 task_hash。")
    return task_id


def select_supplementary_tasks(synthetic: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """每张未见地图选3个不同节点规模任务，固定为72条且不看算法结果。"""
    by_map: Dict[str, List[Dict[str, Any]]] = {}
    for raw in synthetic:
        row = dict(raw)
        by_map.setdefault(str(row["map_id"]), []).append(row)
    chosen: List[Dict[str, Any]] = []
    for map_id, rows in sorted(by_map.items()):
        for node_count in (16, 20, 24):
            candidates = sorted(
                (row for row in rows if int(row["node_count"]) == node_count),
                key=lambda row: hashlib.sha256((str(row["task_hash"]) + "|supplementary_v3_2").encode()).hexdigest(),
            )
            if not candidates:
                raise RuntimeError(f"补充基线分层缺少 {map_id}/N={node_count}。")
            chosen.append(candidates[0])
    if len(chosen) != 72:
        raise RuntimeError("补充基线任务必须恰为24地图×3节点规模=72。")
    return chosen


def select_robustness_tasks(real: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """每张真实地图选择16/20/24各一条，同时冻结12:12道路平衡。"""
    by_map: Dict[str, List[Dict[str, Any]]] = {}
    for raw in real:
        row = dict(raw)
        if "road_index" not in row:
            raise RuntimeError("真实任务必须记录 road_index，才能验证道路层级平衡。")
        by_map.setdefault(str(row["map_id"]), []).append(row)
    chosen: List[Dict[str, Any]] = []
    for map_position, (map_id, rows) in enumerate(sorted(by_map.items())):
        for node_position, node_count in enumerate((16, 20, 24)):
            desired_road = (map_position + node_position) % 2
            candidates = [row for row in rows if int(row["node_count"]) == node_count and int(row["road_index"]) == desired_road]
            if not candidates:
                raise RuntimeError(f"鲁棒性分层缺少 {map_id}/road={desired_road}/N={node_count}。")
            chosen.append(sorted(candidates, key=lambda row: str(row["task_hash"]))[0])
    road_counts = {road: sum(int(row["road_index"]) == road for row in chosen) for road in (0, 1)}
    if len(chosen) != 24 or road_counts != {0: 12, 1: 12}:
        raise RuntimeError(f"鲁棒性子集必须为24条且道路平衡12:12，实际={road_counts}")
    return chosen


def _add_learning_rows(rows: List[Dict[str, Any]], *, family: str, tasks: Sequence[Mapping[str, Any]], variants: Sequence[str], seeds: Sequence[int], condition: str) -> None:
    for task in tasks:
        for variant in variants:
            for seed in seeds:
                rows.append({"family": family, "task_id": _task_key(task), "task_hash": task["task_hash"], "model": variant, "training_seed": int(seed), "planner_seed": None, "condition": condition})


def _add_planner_rows(rows: List[Dict[str, Any]], *, family: str, tasks: Sequence[Mapping[str, Any]], planners: Mapping[str, Sequence[int]], condition: str) -> None:
    for task in tasks:
        for planner, seeds in planners.items():
            for seed in seeds:
                rows.append({"family": family, "task_id": _task_key(task), "task_hash": task["task_hash"], "model": planner, "training_seed": None, "planner_seed": int(seed), "condition": condition})


def compile_evaluation_rows(protocol_path: Path, synthetic: Sequence[Mapping[str, Any]], real: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """展开唯一行键；这里的结果条数是执行前硬门，不是执行后的估计。"""
    protocol = load_v3_2_protocol(protocol_path)
    formal = protocol["formal_evaluation"]
    seeds = [int(seed) for seed in formal["training_seeds"]]
    variants = list(formal["active_learning_variants"])
    if len(synthetic) != 216 or len(real) != 144:
        raise RuntimeError("正式清单必须是216个合成任务和144个真实任务。")
    supplementary = select_supplementary_tasks(synthetic)
    robust = select_robustness_tasks(real)
    rows: List[Dict[str, Any]] = []
    _add_learning_rows(rows, family="synthetic_learning", tasks=synthetic, variants=variants, seeds=seeds, condition="nominal")
    _add_planner_rows(rows, family="synthetic_main_baselines", tasks=synthetic, planners=formal["synthetic"]["main_baselines"], condition="nominal")
    _add_planner_rows(rows, family="synthetic_supplementary", tasks=supplementary, planners=formal["synthetic"]["supplementary_baselines"], condition="nominal")
    _add_learning_rows(rows, family="real_learning", tasks=real, variants=variants, seeds=seeds, condition="nominal")
    _add_planner_rows(rows, family="real_baselines", tasks=real, planners=formal["real_external"]["baseline_planners"], condition="nominal")
    known = formal["robustness"]["known_domain_shift"]
    for condition in known["factors"]:
        _add_learning_rows(rows, family="known_domain_shift", tasks=robust, variants=known["learning_variants"], seeds=seeds, condition=condition)
        _add_planner_rows(rows, family="known_domain_shift", tasks=robust, planners={known["baseline"]: [42]}, condition=condition)
    hidden = formal["robustness"]["hidden_model_perception_mismatch"]
    for condition in hidden["factors"]:
        _add_learning_rows(rows, family="hidden_model_perception_mismatch", tasks=robust, variants=hidden["learning_variants"], seeds=seeds, condition=condition)
        _add_planner_rows(rows, family="hidden_model_perception_mismatch", tasks=robust, planners={hidden["baseline"]: [42]}, condition=condition)
    ids = [(row["family"], row["task_id"], row["model"], row["training_seed"], row["planner_seed"], row["condition"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("正式评价行键重复。")
    actual = {family: sum(row["family"] == family for row in rows) for family in formal["counts"] if family != "total" and family != "synthetic_total" and family != "real_total" and family != "robustness_total"}
    expected = {key: value for key, value in formal["counts"].items() if key in actual}
    if actual != expected or len(rows) != int(formal["counts"]["total"]):
        raise RuntimeError(f"评价矩阵行数不符合预注册协议：actual={actual}, expected={expected}, total={len(rows)}")
    return rows


def freeze_evaluation_matrix(protocol_path: Path, output_root: Path, synthetic_records: Path, real_records: Path) -> Dict[str, Any]:
    protocol = load_v3_2_protocol(protocol_path)
    rows = compile_evaluation_rows(protocol_path, _read_jsonl(synthetic_records), _read_jsonl(real_records))
    destination = Path(output_root) / "formal_evaluation" / "evaluation_matrix.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = _jsonl(rows)
    if destination.exists() and destination.read_text(encoding="utf-8") != text:
        raise RuntimeError("正式评价矩阵已经存在且与当前冻结输入不一致。")
    destination.write_text(text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "synthetic_records_sha256": _sha256_file(synthetic_records),
        "real_records_sha256": _sha256_file(real_records),
        "row_count": len(rows),
        "matrix_sha256": _sha256_file(destination),
        "algorithm_results_used": False,
    }
    manifest["matrix_manifest_hash"] = _canonical_hash(manifest, excluded=("matrix_manifest_hash",))
    (destination.parent / "evaluation_matrix_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """评价工件一次写入；中断后只可按完全相同的输入恢复。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"冻结工件已存在且内容不一致：{path}")
    path.write_text(text, encoding="utf-8")


def _accept_real_candidate(
    candidate: Dict[str, Any],
    provider: Any,
    parent_protocol: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> Dict[str, Any] | None:
    """只用 MILP 证书认证真实任务；绝不读取待比较算法输出。"""
    screening_limit = float(protocol["certification"]["candidate_screening_time_limit_s"])
    final_limit = float(protocol["certification"]["time_limit_s"])
    # 与训练/验证任务相同：只依据MILP证书做有限次单调校准，
    # 不接触任何学习策略或传统规划器的输出。
    working = copy.deepcopy(candidate)
    maximum_calibrations = int(
        protocol["task_generation"][
            "mixed_budget_calibration"
            if str(working.get("constraint_type")) == "mixed"
            else "single_constraint_budget_calibration"
        ]["maximum_iterations"]
    )
    for iteration in range(maximum_calibrations + 1):
        screen_ok, screen_certificate, screen_reason = multimap._certify_multimap_task(
            working, provider, parent_protocol, time_limit_s=screening_limit
        )
        if screen_ok or iteration >= maximum_calibrations:
            break
        if str(working.get("constraint_type")) == "mixed":
            calibrated = multimap._calibrate_mixed_candidate(
                working, screen_certificate, protocol, parent_protocol, iteration=iteration + 1
            )
        else:
            calibrated = multimap._calibrate_single_constraint_candidate(
                working, screen_certificate, protocol, parent_protocol, iteration=iteration + 1
            )
        if calibrated is None:
            break
        working = calibrated
    if screen_ok:
        final_ok, certificate, reason = screen_ok, screen_certificate, screen_reason
        source, used = "screening_sufficient", screening_limit
    elif multimap._screening_bounds_intersect_band(working, screen_certificate, parent_protocol):
        final_ok, certificate, reason = multimap._certify_multimap_task(
            candidate, provider, parent_protocol, time_limit_s=final_limit
        )
        source, used = "extended_final", final_limit
    else:
        return None
    if not final_ok:
        return None
    accepted = copy.deepcopy(working)
    certificate = dict(certificate)
    certificate["screening"] = {
        "time_limit_s": screening_limit,
        "reason": screen_reason,
        "weighted_coverage_lower_bound": screen_certificate.get("weighted_coverage_lower_bound"),
        "weighted_coverage_upper_bound": screen_certificate.get("weighted_coverage_upper_bound"),
        "mip_gap": screen_certificate.get("mip_gap"),
    }
    certificate["certification_source"] = source
    certificate["certification_time_limit_s_used"] = used
    accepted["certificate"] = certificate
    accepted["task_hash"] = multimap._canonical_hash(accepted, excluded=("task_hash",))
    return accepted


def prepare_real_test_manifest(
    protocol_path: Path = DEFAULT_PROTOCOL,
    *,
    map_root: Path = ROOT / "map_data/multimap_v3_1",
    output_root: Path = DEFAULT_OUTPUT,
    resume_existing: bool = False,
) -> Dict[str, Any]:
    """冻结8张DSM的2道路×9任务=144条独立认证任务。"""
    protocol = load_v3_2_protocol(protocol_path)
    # 只有35个模型审计通过后才可接触外部测试地图并生成任务。
    catalog = checkpoint_catalog(protocol_path, output_root=output_root)
    parent_protocol = json.loads(multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8"))
    registry_path = Path(map_root) / "real" / "map_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("protocol_hash") != protocol["asset_parent_protocol_hash"]:
        raise RuntimeError("真实DSM注册表不是协议声明的父资产。")
    maps = list(registry.get("maps") or ())
    if len(maps) != 8:
        raise RuntimeError("真实DSM注册表必须恰含8张地图。")
    provider = multimap.FrozenMapProvider.from_registries(map_root, [registry_path])
    contexts_by_map = _load_real_corridor_contexts(output_root, protocol)
    destination = Path(output_root) / "formal_evaluation" / "real_tasks" / "records.jsonl"
    existing = _read_jsonl(destination) if destination.is_file() else []
    if existing and not resume_existing:
        raise FileExistsError("真实测试任务已存在；恢复必须显式指定 resume_existing。")
    accepted = {str(row["id"]): dict(row) for row in existing}
    max_attempts = int(protocol["task_generation"]["maximum_candidate_attempts_per_task"])
    for map_index, map_record in enumerate(maps):
        full_bundle = multimap._load_map_bundle(map_root, map_record)
        context_record = contexts_by_map.get(str(map_record["map_id"]))
        if context_record is None:
            raise RuntimeError(f"缺少地图道路走廊上下文：{map_record['map_id']}")
        if len(full_bundle["roads"]) != 2:
            raise RuntimeError(f"{map_record['map_id']} 未保留两条道路轨迹。")
        for road_index in (0, 1):
            # 只将对应道路传给布点器；地形真值仍由完整、只读DSM提供。
            road_bundle = dict(full_bundle)
            road_bundle["roads"] = [full_bundle["roads"][road_index]]
            road_context = dict(context_record["contexts"][road_index])
            road_bundle = _corridor_bundle(full_bundle, context_record)
            depot_override_xy = list(road_context["start_xy"])
            for task_index in range(9):
                task_id = f"real_test__{map_record['map_id']}__road_{road_index:02d}__task_{task_index:02d}"
                if task_id in accepted:
                    continue
                design = multimap._task_design(map_index, task_index)
                minimum_radius, radius_range = multimap._effective_task_radius_range(
                    map_record, road_bundle, protocol,
                    node_count=int(design["node_count"]), difficulty=str(design["difficulty"]),
                    depot_override_xy=depot_override_xy,
                )
                result: Dict[str, Any] | None = None
                for attempt in range(max_attempts):
                    try:
                        candidate = multimap._task_candidate(
                            map_record, road_bundle, protocol, parent_protocol,
                            split="real_test", map_index=map_index, task_index=task_index,
                            attempt=attempt, master_seed=int(protocol["map_splits"]["synthetic_test"]["seed"]),
                            geometry_radius_range_m=radius_range,
                            geometry_minimum_feasible_radius_m=minimum_radius,
                            seed_namespace=f"road_{road_index:02d}", road_index=road_index,
                            depot_override_xy=depot_override_xy,
                        )
                    except RuntimeError:
                        continue
                    candidate["road_context_hash"] = str(context_record["context_hash"])
                    candidate["road_context_definition"] = str(protocol["real_corridor_contexts"]["definition"])
                    result = _accept_real_candidate(candidate, provider, parent_protocol, protocol)
                    if result is not None:
                        break
                if result is None:
                    raise RuntimeError(f"{task_id} 在 {max_attempts} 次MILP候选中未获得认证。")
                if result["id"] != task_id:
                    raise RuntimeError("真实任务ID生成漂移。")
                accepted[task_id] = result
                ordered = [accepted[key] for key in sorted(accepted)]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(_jsonl(ordered), encoding="utf-8")
    records = [accepted[key] for key in sorted(accepted)]
    audit = audit_real_test_records(records, registry, protocol)
    if not audit["passed"]:
        raise RuntimeError("真实测试任务审计失败：" + "; ".join(audit["reasons"][:5]))
    manifest = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "checkpoint_catalog_hash": catalog["catalog_hash"],
        "map_registry_hash": registry["registry_hash"],
        "real_corridor_context_manifest_sha256": _sha256_file(
            Path(str(protocol.get("real_corridor_asset_root", Path(output_root) / "real_corridor_assets"))) / "manifest.json"
        ),
        "map_count": 8,
        "road_tracks_per_map": 2,
        "tasks_per_road": 9,
        "scenario_count": len(records),
        "records_sha256": _sha256_file(destination),
        "selection_used_algorithm_results": False,
    }
    manifest["manifest_hash"] = _canonical_hash(manifest, excluded=("manifest_hash",))
    _write_json(destination.parent / "manifest.json", manifest)
    _write_json(destination.parent / "audit.json", audit)
    return {"manifest": manifest, "audit": audit}


def audit_real_test_records(
    records: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    reasons: List[str] = []
    maps = {str(row["map_id"]): row for row in registry.get("maps", ())}
    map_order = {
        str(row["map_id"]): index
        for index, row in enumerate(registry.get("maps", ()))
    }
    parent = json.loads(
        multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
    )
    certification = parent["certification"]
    if len(records) != 144:
        reasons.append(f"scenario_count={len(records)}")
    identities = set()
    task_hashes = set()
    scenario_hashes = set()
    groups: Dict[tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in records:
        identity = str(row.get("id", ""))
        if not identity or identity in identities:
            reasons.append("duplicate_or_empty_task_id")
        identities.add(identity)
        map_id = str(row.get("map_id", ""))
        road = row.get("road_index")
        if map_id not in maps or road not in {0, 1}:
            reasons.append(f"invalid_map_or_road={identity}")
            continue
        if row.get("task_hash") != multimap._canonical_hash(row, excluded=("task_hash",)):
            reasons.append(f"task_hash_mismatch={identity}")
        if row.get("task_hash") in task_hashes:
            reasons.append(f"duplicate_task_hash={identity}")
        task_hashes.add(row.get("task_hash"))
        if any(str(row.get(field)) != str(maps[map_id].get(field)) for field in ("map_hash", "map_file_sha256")):
            reasons.append(f"map_identity_mismatch={identity}")
        expected_design = multimap._task_design(
            map_order[map_id], int(row.get("task_index", -1))
        )
        for field in (
            "node_count",
            "difficulty",
            "constraint_type",
            "priority_layout",
        ):
            if row.get(field) != expected_design[field]:
                reasons.append(f"task_design_mismatch={identity}:{field}")
        if not str(row.get("road_context_hash", "")):
            reasons.append(f"missing_road_context_hash={identity}")
        groups.setdefault((map_id, int(road)), []).append(row)
    if len(groups) != 16 or any(len(rows) != 9 for rows in groups.values()):
        reasons.append("not_exactly_8_maps_x_2_roads_x_9_tasks")
    for group, rows in groups.items():
        if sorted(int(row.get("task_index", -1)) for row in rows) != list(range(9)):
            reasons.append(f"task_design_grid_mismatch={group}")
        if {int(row.get("node_count", -1)) for row in rows} != {16, 20, 24}:
            reasons.append(f"node_count_grid_mismatch={group}")
        for row in rows:
            certificate = dict(row.get("certificate") or {})
            if not bool(certificate.get("returned")) or int(certificate.get("visited_count", 0)) < 1:
                reasons.append(f"invalid_milp_certificate={row.get('id')}")
                continue
            try:
                lower = float(certificate["weighted_coverage_lower_bound"])
                upper = float(certificate["weighted_coverage_upper_bound"])
                gap = float(certificate["mip_gap"])
            except (KeyError, TypeError, ValueError):
                reasons.append(f"missing_milp_bounds={row.get('id')}")
                continue
            difficulty = str(row.get("difficulty", ""))
            band = parent["difficulty_bands"].get(difficulty)
            tolerance = float(certification["band_tolerance"])
            if (
                band is None
                or not all(math.isfinite(value) for value in (lower, upper, gap))
                or lower > upper + 1e-7
                or lower < float(band[0]) - tolerance
                or lower > float(band[1]) + tolerance
                or upper
                >= float(certification["full_coverage_upper_bound_max"])
                - tolerance
                or not (
                    upper <= float(band[1]) + tolerance
                    or gap <= float(certification["mip_rel_gap"]) + 1e-10
                )
            ):
                reasons.append(f"strict_certificate_bounds_failed={row.get('id')}")
            bottlenecks = set(certificate.get("bottleneck_resources") or ())
            intended = str(row.get("constraint_type", ""))
            if intended == "mixed":
                if len(bottlenecks) < int(
                    certification["mixed_min_active_resources"]
                ):
                    reasons.append(f"mixed_bottleneck_failed={row.get('id')}")
            elif intended not in bottlenecks:
                reasons.append(f"intended_bottleneck_failed={row.get('id')}")
            source = str(certificate.get("certification_source", ""))
            if source == "v3_2_12_parametric_fixed_budget_final":
                expected_limit = 60.0
            elif source == "resource_threshold_fallback":
                expected_limit = float(
                    protocol["task_generation"][
                        "resource_threshold_fallback"
                    ]["lower_time_limit_s"]
                ) + float(
                    protocol["task_generation"][
                        "resource_threshold_fallback"
                    ]["upper_time_limit_s"]
                )
                proof = dict(
                    certificate.get("resource_threshold_proof") or {}
                )
                if not bool(
                    dict(proof.get("high_threshold") or {}).get(
                        "threshold_impossible_under_actual_budget", False
                    )
                ):
                    reasons.append(
                        f"resource_threshold_proof_invalid={row.get('id')}"
                    )
            elif source == "v3_2_14_constructive_mixed_threshold":
                proof = dict(
                    certificate.get(
                        "constructive_mixed_threshold_proof"
                    )
                    or {}
                )
                expected_limit = float(
                    proof.get("registered_solver_time_limit_s", math.nan)
                )
                high_proof = dict(
                    proof.get("high_threshold_proof") or {}
                )
                if (
                    int(high_proof.get("solver_status", -1)) != 2
                    or not bool(
                        high_proof.get("threshold_infeasible", False)
                    )
                    or int(proof.get("final_cut_count", -1))
                    != int(high_proof.get("subtour_cut_count", -2))
                ):
                    reasons.append(
                        "constructive_mixed_proof_invalid="
                        f"{row.get('id')}"
                    )
            else:
                expected_limit = math.nan
            if source not in {
                "v3_2_12_parametric_fixed_budget_final",
                "resource_threshold_fallback",
                "v3_2_14_constructive_mixed_threshold",
            } or not math.isclose(
                float(
                    certificate.get(
                        "certification_time_limit_s_used", math.nan
                    )
                ),
                expected_limit,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                reasons.append(
                    f"final_certificate_identity_failed={row.get('id')}"
                )
            scenario_hash = str(certificate.get("scenario_hash", ""))
            if not scenario_hash or scenario_hash in scenario_hashes:
                reasons.append(
                    f"duplicate_or_empty_scenario_hash={row.get('id')}"
                )
            scenario_hashes.add(scenario_hash)
    return {"schema_version": 1, "passed": not reasons, "scenario_count": len(records), "reasons": sorted(set(reasons))}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v3.2正式评价冻结编排")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit-checkpoints")
    matrix = commands.add_parser("freeze-matrix")
    matrix.add_argument("--synthetic-records", type=Path, required=True)
    matrix.add_argument("--real-records", type=Path, required=True)
    real = commands.add_parser("prepare-real-tasks")
    real.add_argument("--map-root", type=Path, default=ROOT / "map_data/multimap_v3_1")
    real.add_argument("--resume-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "audit-checkpoints":
        result = checkpoint_catalog(args.protocol, output_root=args.output_root)
    elif args.command == "freeze-matrix":
        result = freeze_evaluation_matrix(args.protocol, args.output_root, args.synthetic_records, args.real_records)
    else:
        result = prepare_real_test_manifest(args.protocol, map_root=args.map_root, output_root=args.output_root, resume_existing=bool(args.resume_existing))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

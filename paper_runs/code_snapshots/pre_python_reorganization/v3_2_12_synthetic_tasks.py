#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit-reuse 213 sealed synthetic tasks and repair the final three.

The 213 task rows are copied byte-for-byte at the JSON-object level.  Their
task hashes remain unchanged.  Reuse is allowed only after all input-defining
protocol sections and all 24 map files/registry identities are proved equal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import paper_multimap_experiments as multimap
import paper_v3_2_experiments as v32
import v3_2_12_parametric_certificate_search as search


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_12/protocol.json"
)
SOURCE_PROTOCOL = (
    ROOT / "paper_runs/protocols/multimap_generalization_v3_2/protocol.json"
)
SOURCE_RECORDS = (
    ROOT
    / "paper_runs/multimap_v3_2/manifests/synthetic_test/records.jsonl"
)
DEFAULT_OUTPUT = ROOT / "paper_runs/multimap_v3_2_12"
DEFAULT_MAP_ROOT = ROOT / "map_data/multimap_v3_1"
IDENTITY_SECTIONS = (
    "map_splits",
    "procedural_terrain",
    "node_counts",
    "tasks_per_node_count_per_map",
    "constraint_types",
    "priority_layouts",
    "difficulty_bands",
    "task_generation",
    "certification",
)
PRECERTIFIED_REJECTED_ATTEMPTS = {
    "synthetic_test__synthetic_test__map_023__task_06": (0, 1, 2, 3, 4),
}


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )


def _single_axis_binary_search(
    candidate: Mapping[str, Any],
    provider: Any,
    parent: Mapping[str, Any],
    protocol: Mapping[str, Any],
    cache: search.ProbeCache,
    *,
    progress_path: Path,
    task_id: str,
    attempt: int,
) -> Dict[str, Any] | None:
    """释放非目标资源后，对目标预算做有限次确定性二分夹逼。"""

    intended = str(candidate["constraint_type"])
    parameter = search.NAME_TO_PARAMETER.get(intended)
    if parameter is None:
        return None
    values = {
        other: search._bounds(protocol, other)[1]
        for other in search.RESOURCE_PARAMETERS
        if other != parameter
    }
    relaxed = search._with_values(
        candidate,
        values,
        protocol,
        {
            "stage": "synthetic_repair_release_nuisance_resources",
            "intended_parameter": parameter,
        },
    )
    parameter_low, parameter_high = search._bounds(protocol, parameter)
    observations: List[Dict[str, Any]] = []

    def evaluate(fraction: float) -> tuple[Dict[str, Any] | None, str]:
        trial = search._with_values(
            relaxed,
            {
                parameter: parameter_low
                + float(fraction) * (parameter_high - parameter_low)
            },
            protocol,
            {
                "stage": "synthetic_repair_single_axis_bisection",
                "parameter": parameter,
                "registered_range_fraction": float(fraction),
            },
        )
        probe_ok, probe_certificate, probe_reason = search._probe(
            trial, provider, parent, protocol, cache
        )
        relation = search._relation(
            probe_certificate, str(trial["difficulty"]), parent
        )
        observation = {
            "fraction": float(fraction),
            "parameter": parameter,
            "value": float(trial[parameter]),
            "probe_ok": bool(probe_ok),
            "probe_reason": str(probe_reason),
            "relation": relation,
            "lower": probe_certificate.get("weighted_coverage_lower_bound"),
            "upper": probe_certificate.get("weighted_coverage_upper_bound"),
        }
        observations.append(observation)
        _write_progress(
            progress_path,
            {
                "schema_version": 1,
                "state": "single_axis_bisection",
                "task_id": task_id,
                "attempt": int(attempt),
                "observations": observations,
                "algorithm_results_used": False,
            },
        )
        if not probe_ok and relation != "intersects":
            return None, relation
        accepted, strict_certificate, strict_reason = search._strict_outcome(
            trial,
            probe_certificate,
            probe_reason,
            provider,
            parent,
            protocol,
            cache,
        )
        observation["strict_reason"] = strict_reason
        observation["strict_relation"] = search._relation(
            strict_certificate, str(trial["difficulty"]), parent
        )
        strict_interval = search._finite_interval(strict_certificate)
        observation["strict_lower"] = (
            strict_interval[0] if strict_interval is not None else None
        )
        observation["strict_upper"] = (
            strict_interval[1] if strict_interval is not None else None
        )
        _write_progress(
            progress_path,
            {
                "schema_version": 1,
                "state": "single_axis_bisection",
                "task_id": task_id,
                "attempt": int(attempt),
                "observations": observations,
                "algorithm_results_used": False,
            },
        )
        if accepted is not None:
            return accepted, "accepted"
        strict_relation = str(observation["strict_relation"])
        if strict_reason == "intended_bottleneck_not_active":
            strict_relation = "above"
        elif strict_reason == "full_coverage_not_excluded":
            strict_relation = "above"
        elif strict_reason == "no_safe_partial_route":
            strict_relation = "below"
        elif (
            strict_reason == "incumbent_outside_band"
            and strict_interval is not None
        ):
            band_low, band_high = (
                float(value)
                for value in parent["difficulty_bands"][
                    str(trial["difficulty"])
                ]
            )
            if strict_interval[0] < band_low:
                strict_relation = "below"
            elif strict_interval[0] > band_high:
                strict_relation = "above"
        return None, strict_relation

    anchor_fraction = max(
        0.0,
        min(
            1.0,
            (float(candidate[parameter]) - parameter_low)
            / (parameter_high - parameter_low),
        ),
    )
    # 松弛端点在 24 节点时可能很难给出 MILP 上界；优先探测冻结原值
    # 和三个内点，既保持确定性，也避免在无信息端点浪费最终时限。
    probe_fractions = []
    for fraction in (anchor_fraction, 0.25, 0.125, 0.50, 0.75, 0.875):
        if not any(abs(fraction - known) <= 1e-12 for known in probe_fractions):
            probe_fractions.append(float(fraction))
    relations: Dict[float, str] = {}
    for fraction in probe_fractions:
        accepted, relation = evaluate(fraction)
        if accepted is not None:
            return accepted
        relations[fraction] = relation
        below = sorted(
            value for value, state in relations.items() if state == "below"
        )
        above = sorted(
            value for value, state in relations.items() if state == "above"
        )
        brackets = [
            (left, right)
            for left in below
            for right in above
            if left < right
        ]
        if brackets:
            left, right = min(
                brackets, key=lambda pair: pair[1] - pair[0]
            )
            break
        unknown = sorted(
            value for value, state in relations.items() if state == "unknown"
        )
        boundary_pairs = [
            (left, right)
            for left in below
            for right in unknown
            if left < right
        ]
        if boundary_pairs:
            # “过紧—未知”边界同样有信息：未知侧组合空间更大，
            # 向过紧侧折半可快速找到首次与目标区间相交的预算。
            left, right = min(
                boundary_pairs, key=lambda pair: pair[1] - pair[0]
            )
            break
    else:
        return None
    for _ in range(
        int(
            protocol["pretest_parametric_certificate_search"][
                "bracket_refinement_steps"
            ]
        )
    ):
        middle = (left + right) / 2.0
        accepted, relation = evaluate(middle)
        if accepted is not None:
            return accepted
        if relation == "below":
            left = middle
        elif relation in {"above", "unknown"}:
            right = middle
        else:
            break
    return None


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


def _reuse_audit(
    protocol: Mapping[str, Any],
    source_protocol: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    map_root: Path,
) -> Dict[str, Any]:
    reasons: List[str] = []
    section_hashes = {}
    for key in IDENTITY_SECTIONS:
        source_hash = _canonical(source_protocol.get(key))
        current_hash = _canonical(protocol.get(key))
        section_hashes[key] = {
            "source": source_hash,
            "current": current_hash,
            "equal": source_hash == current_hash,
        }
        if source_hash != current_hash:
            reasons.append(f"input_semantics_changed={key}")
    if registry.get("protocol_hash") != source_protocol.get("protocol_hash"):
        reasons.append("sealed_registry_protocol_mismatch")
    registered = {
        str(row["map_id"]): dict(row) for row in registry.get("maps", ())
    }
    if len(registered) != 24:
        reasons.append("sealed_registry_map_count_not_24")
    ids = set()
    for row in source_rows:
        task_id = str(row.get("id", ""))
        if not task_id or task_id in ids:
            reasons.append(f"duplicate_or_empty_source_id={task_id}")
        ids.add(task_id)
        if multimap._canonical_hash(row, excluded=("task_hash",)) != row.get(
            "task_hash"
        ):
            reasons.append(f"source_task_hash_invalid={task_id}")
        map_record = registered.get(str(row.get("map_id", "")))
        if map_record is None:
            reasons.append(f"source_map_unregistered={task_id}")
            continue
        for field in ("map_hash", "map_file_sha256"):
            if str(row.get(field, "")) != str(map_record.get(field, "")):
                reasons.append(f"source_{field}_mismatch={task_id}")
    for map_record in registered.values():
        path = Path(map_root) / str(map_record["map_file"])
        if not path.is_file() or v32._sha256_file(path) != str(
            map_record["map_file_sha256"]
        ):
            reasons.append(f"sealed_map_file_hash_mismatch={map_record['map_id']}")
    expected = {
        f"synthetic_test__synthetic_test__map_{map_index:03d}__task_{task_index:02d}"
        for map_index in range(24)
        for task_index in range(9)
    }
    missing = sorted(expected - ids)
    if len(source_rows) != 213 or missing != [
        "synthetic_test__synthetic_test__map_023__task_06",
        "synthetic_test__synthetic_test__map_023__task_07",
        "synthetic_test__synthetic_test__map_023__task_08",
    ]:
        reasons.append("source_213_grid_identity_failed")
    return {
        "schema_version": 1,
        "passed": not reasons,
        "algorithm_results_used": False,
        "source_protocol_hash": source_protocol["protocol_hash"],
        "target_protocol_hash": protocol["protocol_hash"],
        "source_record_count": len(source_rows),
        "source_records_sha256": v32._sha256_file(SOURCE_RECORDS),
        "sealed_registry_hash": registry.get("registry_hash"),
        "identity_section_hashes": section_hashes,
        "missing_task_ids": missing,
        "reasons": reasons,
    }


def build_synthetic_tasks(
    protocol_path: Path = DEFAULT_PROTOCOL,
    output_root: Path = DEFAULT_OUTPUT,
    map_root: Path = DEFAULT_MAP_ROOT,
) -> Dict[str, Any]:
    protocol = v32.load_v3_2_protocol(protocol_path)
    if protocol.get("protocol_version") != "multimap_generalization_v3_2_12":
        raise RuntimeError("synthetic repair is frozen to v3.2.12")
    source_protocol = multimap.load_protocol(SOURCE_PROTOCOL)
    source_rows = v32._read_jsonl(SOURCE_RECORDS)
    registry_path = (
        Path(map_root) / "procedural" / "synthetic_test" / "map_registry.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    audit = _reuse_audit(
        protocol, source_protocol, source_rows, registry, map_root
    )
    destination_dir = Path(output_root) / "manifests" / "synthetic_test"
    destination_dir.mkdir(parents=True, exist_ok=True)
    v32._write_json(destination_dir / "reuse_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(
            "sealed synthetic reuse audit failed: "
            + "; ".join(audit["reasons"][:5])
        )
    v32._write_json(
        destination_dir / "repair_design_v2.json",
        {
            "schema_version": 1,
            "protocol_hash": protocol["protocol_hash"],
            "selection_basis": "model_free_milp_certificates_only",
            "precertified_rejected_geometry_attempts": {
                key: list(value)
                for key, value in PRECERTIFIED_REJECTED_ATTEMPTS.items()
            },
            "attempt_0_to_4_rejection_evidence": (
                "deterministic single-axis screens and fixed 60-second final "
                "certificates exhausted each geometry without an accepted "
                "moderate-band lower bound; observed strict incumbents stayed "
                "below 0.70 while looser boundary screens became unresolved"
            ),
            "algorithm_results_used": False,
        },
    )

    provider = multimap.FrozenMapProvider.from_registries(
        map_root, [registry_path]
    )
    parent = json.loads(
        multimap.PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
    )
    records = {str(row["id"]): dict(row) for row in source_rows}
    map_index = 23
    map_record = dict(registry["maps"][map_index])
    bundle = multimap._load_map_bundle(map_root, map_record)
    maximum_attempts = int(
        protocol["pretest_parametric_certificate_search"][
            "fixed_geometry_attempt_count"
        ]
    )
    for task_index in (6, 7, 8):
        task_id = (
            f"synthetic_test__{map_record['map_id']}"
            f"__task_{task_index:02d}"
        )
        if task_id in records:
            continue
        design = multimap._task_design(map_index, task_index)
        minimum, interval = multimap._effective_task_radius_range(
            map_record,
            bundle,
            protocol,
            node_count=int(design["node_count"]),
            difficulty=str(design["difficulty"]),
        )
        accepted = None
        for attempt in range(maximum_attempts):
            if attempt in PRECERTIFIED_REJECTED_ATTEMPTS.get(task_id, ()):
                continue
            try:
                candidate = multimap._task_candidate(
                    map_record,
                    bundle,
                    protocol,
                    parent,
                    split="synthetic_test",
                    map_index=map_index,
                    task_index=task_index,
                    attempt=attempt,
                    master_seed=int(
                        protocol["map_splits"]["synthetic_test"]["seed"]
                    ),
                    geometry_radius_range_m=interval,
                    geometry_minimum_feasible_radius_m=minimum,
                )
            except RuntimeError:
                continue
            # 旧 2000 次失败集中在单约束单元。修复时先释放两个非目标
            # 资源并做目标预算的一维搜索；只有未命中才进入通用三资源搜索。
            # 两条路径共享完全相同的 60 秒最终接受门槛。
            cache = search.ProbeCache()
            if str(candidate["constraint_type"]) != "mixed":
                accepted = _single_axis_binary_search(
                    candidate,
                    provider,
                    parent,
                    protocol,
                    cache,
                    progress_path=destination_dir / "repair_progress.json",
                    task_id=task_id,
                    attempt=attempt,
                )
            else:
                accepted = search.certify_candidate_with_parametric_search(
                    candidate,
                    provider,
                    parent,
                    protocol,
                    cache=cache,
                )
            if accepted is not None:
                break
        if accepted is None:
            raise RuntimeError(
                f"{task_id} exhausted {maximum_attempts} parametric candidates"
            )
        records[task_id] = accepted
        partial = [records[key] for key in sorted(records)]
        (destination_dir / "records.jsonl").write_text(
            v32._jsonl(partial), encoding="utf-8"
        )
        v32._write_json(
            destination_dir / "generation_checkpoint.json",
            {
                "schema_version": 1,
                "state": "running",
                "protocol_hash": protocol["protocol_hash"],
                "reused": 213,
                "generated": len(partial) - 213,
                "completed": len(partial),
                "expected": 216,
                "latest_task_id": task_id,
                "algorithm_results_used": False,
            },
        )

    rows = [records[key] for key in sorted(records)]
    if len(rows) != 216:
        raise RuntimeError("synthetic task grid is not exactly 216 rows")
    records_path = destination_dir / "records.jsonl"
    records_path.write_text(v32._jsonl(rows), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "parent_difficulty_protocol_hash": parent["protocol_hash"],
        "split": "synthetic_test",
        "map_count": 24,
        "tasks_per_map": 9,
        "scenario_count": 216,
        "map_registry_path": str(registry_path.resolve()),
        "map_registry_hash": registry["registry_hash"],
        "map_provider_hash": provider.provider_hash,
        "records_sha256": v32._sha256_file(records_path),
        "reused_record_count": 213,
        "new_record_count": 3,
        "repair_search_order": (
            "single_constraint_axis_first_then_general_parametric; "
            "mixed_general_parametric_only"
        ),
        "precertified_rejected_geometry_attempts": {
            key: list(value)
            for key, value in PRECERTIFIED_REJECTED_ATTEMPTS.items()
        },
        "reuse_audit_sha256": v32._sha256_file(
            destination_dir / "reuse_audit.json"
        ),
        "selection_used_algorithm_results": False,
        "sharded": False,
        "smoke": False,
    }
    manifest["manifest_hash"] = multimap._canonical_hash(
        manifest, excluded=("manifest_hash",)
    )
    v32._write_json(destination_dir / "manifest.json", manifest)
    final_audit = multimap.audit_task_manifest(
        protocol_path,
        map_root,
        destination_dir / "manifest.json",
        expected_map_count=24,
        expected_tasks_per_map=9,
    )
    v32._write_json(destination_dir / "audit.json", final_audit)
    if not final_audit["passed"]:
        raise RuntimeError(
            "synthetic 216-row audit failed: "
            + "; ".join(final_audit["reasons"][:5])
        )
    v32._write_json(
        destination_dir / "generation_checkpoint.json",
        {
            "schema_version": 1,
            "state": "completed",
            "protocol_hash": protocol["protocol_hash"],
            "reused": 213,
            "generated": 3,
            "completed": 216,
            "expected": 216,
            "records_sha256": manifest["records_sha256"],
            "manifest_hash": manifest["manifest_hash"],
            "algorithm_results_used": False,
        },
    )
    return {"manifest": manifest, "audit": final_audit}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="v3.2.12 sealed synthetic reuse and three-task repair"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    args = parser.parse_args(argv)
    result = build_synthetic_tasks(
        args.protocol, args.output_root, args.map_root
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

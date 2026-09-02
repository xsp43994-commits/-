#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Versioned post-result multi-objective analysis; deliberately creates no plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from uav_inspection.analysis import v3_2_14_statistics as legacy


ROOT = WORKSPACE_ROOT
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14"
FINAL_RESULTS = OUTPUT / "formal_evaluation/results/final_results.jsonl"
FINAL_AUDIT = OUTPUT / "formal_evaluation/results/final_audit.json"
MATRIX = OUTPUT / "formal_evaluation/evaluation_matrix.jsonl"
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "manuscript_multiobjective_v1_protocol.json"
)
DESTINATION = OUTPUT / "analysis/manuscript_multiobjective_v1"
# 仅指向冻结分析实际使用的10条历史曲线，不再依赖已清理的旧训练目录。
TRAINING_ROOT = OUTPUT / "analysis/training_trace_inputs_v2"

EXPECTED_ROWS = 21648
EXPECTED_MATRIX_SHA256 = (
    "48a31ee9b58d41a617fff61acb6eba6a2d9a930767d7af15856f70a964686224"
)
EXPECTED_RESULTS_SHA256 = (
    "4b620c21566c2e33c875f6bea2017b741b02a7d30d70aa50add60a6d06214a2c"
)
CORE_MODELS = ("full", "traditional_ppo", "a2c_pointer")
NOMINAL_SCOPES = {
    "synthetic_all": {
        "families": ("synthetic_learning", "synthetic_main_baselines"),
        "models": (
            "full",
            "traditional_ppo",
            "a2c_pointer",
            "nearest_feasible",
            "priority_resource_greedy",
            "aco",
            "ga",
            "sa",
            "milp",
        ),
        "maps": 24,
    },
    "real_all": {
        "families": ("real_learning", "real_baselines"),
        "models": (
            "full",
            "traditional_ppo",
            "a2c_pointer",
            "nearest_feasible",
            "priority_resource_greedy",
            "aco",
            "milp",
        ),
        "maps": 8,
    },
}
INTERNAL_WEIGHTS = {
    "D1": {
        "weighted_coverage": 0.50,
        "high_priority_coverage": 0.25,
        "oracle_attainment": 0.25,
    },
    "D2": {
        "time_efficiency": 0.40,
        "energy_efficiency": 0.35,
        "distance_efficiency": 0.25,
    },
    "D3": {
        "safe_rate": 0.50,
        "return_rate": 0.20,
        "constraint_free_rate": 0.20,
        "soc_utility": 0.10,
    },
    "D4": {
        "mean_retention": 0.40,
        "worst_retention": 0.30,
        "perturbed_safe_rate": 0.20,
        "map_consistency": 0.10,
    },
    "D5": {
        "online_planning_efficiency": 0.70,
        "scaling_efficiency": 0.30,
    },
}
DEFAULT_WEIGHTS = {
    "D1": 0.35,
    "D2": 0.25,
    "D3": 0.20,
    "D4": 0.12,
    "D5": 0.08,
}
WEIGHT_RANGES = {
    "D1": (0.25, 0.50),
    "D2": (0.15, 0.35),
    "D3": (0.15, 0.35),
    "D4": (0.05, 0.20),
    "D5": (0.05, 0.20),
}
SCENARIOS = {
    "balanced_default": DEFAULT_WEIGHTS,
    "mission_first": {"D1": 0.50, "D2": 0.20, "D3": 0.15, "D4": 0.10, "D5": 0.05},
    "safety_first": {"D1": 0.30, "D2": 0.20, "D3": 0.30, "D4": 0.12, "D5": 0.08},
    "realtime_first": {"D1": 0.30, "D2": 0.20, "D3": 0.20, "D4": 0.10, "D5": 0.20},
}
DEADLINE_SENSITIVITY = (1.0, 5.0, 10.0, 30.0, 60.0)
DEFAULT_DEADLINE_S = 10.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def finite_mean(values: Iterable[float], default: float = 0.0) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float(default)


def weighted_geometric(
    values: Mapping[str, float], weights: Mapping[str, float]
) -> float:
    """严格补偿型受限：任一正权重维度为零时，总分为零。"""
    total = sum(float(weights[key]) for key in values)
    if total <= 0:
        raise ValueError("weights must have positive sum")
    log_sum = 0.0
    for key, value in values.items():
        weight = float(weights[key]) / total
        bounded = clip01(float(value))
        if weight > 0 and bounded <= 0:
            return 0.0
        log_sum += weight * math.log(bounded)
    return float(math.exp(log_sum))


def weighted_arithmetic(
    values: Mapping[str, float], weights: Mapping[str, float]
) -> float:
    total = sum(float(weights[key]) for key in values)
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return float(
        sum(clip01(float(value)) * float(weights[key]) for key, value in values.items())
        / total
    )


def online_utility(planning_time_s: float, deadline_s: float) -> float:
    if deadline_s <= 0:
        raise ValueError("deadline must be positive")
    return clip01(
        1.0 - math.log1p(max(0.0, planning_time_s)) / math.log1p(deadline_s)
    )


def renormalize_weights(
    weights: Mapping[str, float], dimensions: Sequence[str]
) -> dict[str, float]:
    total = sum(float(weights[key]) for key in dimensions)
    return {key: float(weights[key]) / total for key in dimensions}


def enumerate_weight_grid(step: float = 0.05) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    dimensions = tuple(WEIGHT_RANGES)
    bounds = {
        key: (
            int(round(WEIGHT_RANGES[key][0] / step)),
            int(round(WEIGHT_RANGES[key][1] / step)),
        )
        for key in dimensions
    }
    output = []
    for values in itertools.product(
        *(range(bounds[key][0], bounds[key][1] + 1) for key in dimensions)
    ):
        if sum(values) == units:
            output.append(
                {key: round(value * step, 10) for key, value in zip(dimensions, values)}
            )
    if not output:
        raise RuntimeError("weight grid is empty")
    return output


def pareto_membership(
    rows: Sequence[Mapping[str, Any]], objectives: Mapping[str, str]
) -> list[bool]:
    result = []
    for index, row in enumerate(rows):
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            weak = True
            strict = False
            for key, direction in objectives.items():
                left = float(row[key])
                right = float(other[key])
                if direction == "max":
                    weak &= right >= left
                    strict |= right > left
                elif direction == "min":
                    weak &= right <= left
                    strict |= right < left
                else:
                    raise ValueError(f"unknown objective direction: {direction}")
            if weak and strict:
                dominated = True
                break
        result.append(not dominated)
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected = protocol["protocol_hash"]
    actual = canonical_hash({key: value for key, value in protocol.items() if key != "protocol_hash"})
    if expected != actual:
        raise RuntimeError("multi-objective protocol hash drift")
    if protocol["implementation_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("multi-objective implementation hash drift")
    if protocol["matrix_sha256"] != EXPECTED_MATRIX_SHA256:
        raise RuntimeError("frozen matrix identity drift")
    return protocol


def _validate_inputs() -> dict[str, Any]:
    audit = json.loads(FINAL_AUDIT.read_text(encoding="utf-8"))
    if not audit.get("passed") or int(audit.get("row_count", -1)) != EXPECTED_ROWS:
        raise RuntimeError("formal evaluation audit is not complete")
    if audit.get("matrix_sha256") != EXPECTED_MATRIX_SHA256:
        raise RuntimeError("formal evaluation matrix hash drift")
    if sha256_file(MATRIX) != EXPECTED_MATRIX_SHA256:
        raise RuntimeError("matrix file hash drift")
    if sha256_file(FINAL_RESULTS) != EXPECTED_RESULTS_SHA256:
        raise RuntimeError("formal result hash drift")
    return audit


def _nominal_task_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    accepted = {
        family
        for scope in NOMINAL_SCOPES.values()
        for family in scope["families"]
    }
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["family"]) in accepted and str(row["condition"]) == "nominal":
            domain = "synthetic" if str(row["family"]).startswith("synthetic") else "real"
            grouped[(domain, str(row["model"]), str(row["map_id"]), str(row["task_id"]))].append(row)

    output: list[dict[str, Any]] = []
    for (domain, model, map_id, task_id), repeats in sorted(grouped.items()):
        safe = [row for row in repeats if bool(row["safe"])]
        component = {
            "weighted_coverage": finite_mean(float(row["weighted_coverage"]) for row in repeats),
            "high_priority_coverage": finite_mean(
                float(row["high_priority_coverage"]) for row in repeats
            ),
            "oracle_attainment": finite_mean(
                clip01(float(row["weighted_coverage"]) / float(row["oracle_upper"]))
                for row in repeats
                if float(row["oracle_upper"]) > 0
            ),
            "time_efficiency": finite_mean(
                (clip01(1.0 - float(row["time_utilization"])) for row in safe),
                default=0.0,
            ),
            "energy_efficiency": finite_mean(
                (clip01(1.0 - float(row["energy_utilization"])) for row in safe),
                default=0.0,
            ),
            "distance_efficiency": finite_mean(
                (clip01(1.0 - float(row["distance_utilization"])) for row in safe),
                default=0.0,
            ),
            "safe_rate": finite_mean(float(row["safe_rate"]) for row in repeats),
            "return_rate": finite_mean(float(row["return_rate"]) for row in repeats),
            "constraint_free_rate": 1.0
            - finite_mean(float(row["violation_rate"]) for row in repeats),
            "soc_utility": finite_mean(
                clip01(float(row["min_remaining_soc"])) for row in repeats
            ),
        }
        output.append(
            {
                "domain": domain,
                "model": model,
                "map_id": map_id,
                "task_id": task_id,
                "node_count": int(repeats[0]["node_count"]),
                "repeat_count": len(repeats),
                **component,
                "D1_mission_effectiveness": weighted_arithmetic(
                    {key: component[key] for key in INTERNAL_WEIGHTS["D1"]},
                    INTERNAL_WEIGHTS["D1"],
                ),
                "D2_resource_efficiency": weighted_arithmetic(
                    {key: component[key] for key in INTERNAL_WEIGHTS["D2"]},
                    INTERNAL_WEIGHTS["D2"],
                ),
                "D3_safety_reliability": weighted_geometric(
                    {key: component[key] for key in INTERNAL_WEIGHTS["D3"]},
                    INTERNAL_WEIGHTS["D3"],
                ),
                "mean_safe_time_s": finite_mean(
                    (float(row["time_s"]) for row in safe), default=float("nan")
                ),
                "mean_safe_energy_wh": finite_mean(
                    (float(row["energy_wh"]) for row in safe), default=float("nan")
                ),
                "mean_safe_distance_m": finite_mean(
                    (float(row["distance_m"]) for row in safe), default=float("nan")
                ),
                "mean_planning_time_s": finite_mean(
                    float(row["planning_time_s"]) for row in repeats
                ),
            }
        )
    return output


def _nominal_map_rows(
    task_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    deadline_s: float = DEFAULT_DEADLINE_S,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    raw_grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in task_rows:
        grouped[(str(row["domain"]), str(row["model"]), str(row["map_id"]))].append(row)
    for row in raw_rows:
        family = str(row["family"])
        if family in {"synthetic_learning", "synthetic_main_baselines", "real_learning", "real_baselines"}:
            domain = "synthetic" if family.startswith("synthetic") else "real"
            raw_grouped[(domain, str(row["model"]), str(row["map_id"]))].append(row)

    output = []
    component_keys = tuple(
        dict.fromkeys(
            list(INTERNAL_WEIGHTS["D1"])
            + list(INTERNAL_WEIGHTS["D2"])
            + list(INTERNAL_WEIGHTS["D3"])
        )
    )
    for key, tasks in sorted(grouped.items()):
        domain, model, map_id = key
        components = {name: finite_mean(float(row[name]) for row in tasks) for name in component_keys}
        raw = raw_grouped[key]
        p95 = float(np.percentile([float(row["planning_time_s"]) for row in raw], 95))
        by_size = defaultdict(list)
        for row in raw:
            by_size[int(row["node_count"])].append(float(row["planning_time_s"]))
        median16 = float(np.median(by_size[16]))
        median24 = float(np.median(by_size[24]))
        ratio = median24 / median16 if median16 > 0 else float("inf")
        scaling = clip01(1.0 / max(1.0, ratio))
        online = online_utility(p95, deadline_s)
        output.append(
            {
                "domain": domain,
                "model": model,
                "map_id": map_id,
                "task_count": len(tasks),
                **components,
                "D1_mission_effectiveness": weighted_arithmetic(
                    {name: components[name] for name in INTERNAL_WEIGHTS["D1"]},
                    INTERNAL_WEIGHTS["D1"],
                ),
                "D2_resource_efficiency": weighted_arithmetic(
                    {name: components[name] for name in INTERNAL_WEIGHTS["D2"]},
                    INTERNAL_WEIGHTS["D2"],
                ),
                "D3_safety_reliability": weighted_geometric(
                    {name: components[name] for name in INTERNAL_WEIGHTS["D3"]},
                    INTERNAL_WEIGHTS["D3"],
                ),
                "D5_online_deployability": weighted_arithmetic(
                    {
                        "online_planning_efficiency": online,
                        "scaling_efficiency": scaling,
                    },
                    INTERNAL_WEIGHTS["D5"],
                ),
                "planning_time_p95_s": p95,
                "planning_time_median_16_s": median16,
                "planning_time_median_24_s": median24,
                "planning_scaling_ratio_24_to_16": ratio,
                "online_planning_efficiency": online,
                "scaling_efficiency": scaling,
                "mean_safe_time_s": finite_mean(float(row["mean_safe_time_s"]) for row in tasks),
                "mean_safe_energy_wh": finite_mean(float(row["mean_safe_energy_wh"]) for row in tasks),
                "mean_safe_distance_m": finite_mean(float(row["mean_safe_distance_m"]) for row in tasks),
            }
        )
    return output


def _robustness_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nominal = {
        (str(row["task_id"]), str(row["model"]), int(row["repeat_seed"])): float(
            row["safe_weighted_coverage"]
        )
        for row in rows
        if str(row["family"]) in {"real_learning", "real_baselines"}
    }
    perturbed = [
        row
        for row in rows
        if str(row["family"]) in {"known_domain_shift", "hidden_model_perception_mismatch"}
    ]
    condition_group: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in perturbed:
        condition_group[
            (str(row["model"]), str(row["map_id"]), str(row["family"]), str(row["condition"]))
        ].append(row)
    condition_rows = []
    expected_conditions = {
        ("known_domain_shift", "wind"),
        ("known_domain_shift", "power_model"),
        ("hidden_model_perception_mismatch", "wind"),
        ("hidden_model_perception_mismatch", "power_model"),
        ("hidden_model_perception_mismatch", "dem_error"),
        ("hidden_model_perception_mismatch", "localization"),
    }
    for key, values in sorted(condition_group.items()):
        model, map_id, family, condition = key
        drops = []
        for row in values:
            identity = (str(row["task_id"]), model, int(row["repeat_seed"]))
            if identity not in nominal:
                raise RuntimeError(f"missing nominal robustness pair: {identity}")
            drops.append(max(0.0, nominal[identity] - float(row["safe_weighted_coverage"])))
        condition_rows.append(
            {
                "model": model,
                "map_id": map_id,
                "family": family,
                "condition": condition,
                "repeat_count": len(values),
                "retention": clip01(1.0 - finite_mean(drops)),
                "perturbed_safe_rate": finite_mean(float(row["safe_rate"]) for row in values),
                "perturbed_safe_weighted_coverage": finite_mean(
                    float(row["safe_weighted_coverage"]) for row in values
                ),
            }
        )
    algorithm_rows = []
    for model in CORE_MODELS:
        selected = [row for row in condition_rows if row["model"] == model]
        present = {(row["family"], row["condition"]) for row in selected}
        maps = {row["map_id"] for row in selected}
        if present != expected_conditions or len(maps) != 8:
            raise RuntimeError(f"incomplete robustness grid for {model}")
        by_condition = defaultdict(list)
        by_map = defaultdict(list)
        for row in selected:
            by_condition[(row["family"], row["condition"])].append(float(row["retention"]))
            by_map[row["map_id"]].append(float(row["perturbed_safe_weighted_coverage"]))
        mean_retention = finite_mean(float(row["retention"]) for row in selected)
        worst_retention = min(finite_mean(values) for values in by_condition.values())
        safe_rate = finite_mean(float(row["perturbed_safe_rate"]) for row in selected)
        map_values = [finite_mean(values) for values in by_map.values()]
        map_consistency = clip01(1.0 - float(np.std(map_values, ddof=1)))
        components = {
            "mean_retention": mean_retention,
            "worst_retention": worst_retention,
            "perturbed_safe_rate": safe_rate,
            "map_consistency": map_consistency,
        }
        algorithm_rows.append(
            {
                "model": model,
                **components,
                "D4_robustness": weighted_arithmetic(components, INTERNAL_WEIGHTS["D4"]),
            }
        )
    return condition_rows, algorithm_rows


def _mechanism_robustness_rows(
    condition_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """按实际可用条件比较，绝不为未运行的扰动补值。"""
    scopes = {
        "known_domain_shift": {
            "family": "known_domain_shift",
            "conditions": {"wind", "power_model"},
            "models": {
                "full",
                "traditional_ppo",
                "a2c_pointer",
                "no_domain_randomization",
                "priority_resource_greedy",
            },
        },
        "hidden_model_perception_mismatch": {
            "family": "hidden_model_perception_mismatch",
            "conditions": {"wind", "power_model", "dem_error", "localization"},
            "models": {
                "full",
                "traditional_ppo",
                "a2c_pointer",
                "no_domain_randomization",
                "no_return_reserve",
                "priority_resource_greedy",
            },
        },
    }
    output = []
    for scope, config in scopes.items():
        for model in sorted(config["models"]):
            selected = [
                row
                for row in condition_rows
                if row["family"] == config["family"] and row["model"] == model
            ]
            conditions = {str(row["condition"]) for row in selected}
            maps = {str(row["map_id"]) for row in selected}
            if conditions != config["conditions"] or len(maps) != 8:
                raise RuntimeError(f"incomplete specialized robustness grid: {scope}/{model}")
            by_condition = defaultdict(list)
            by_map = defaultdict(list)
            for row in selected:
                by_condition[str(row["condition"])].append(float(row["retention"]))
                by_map[str(row["map_id"])].append(
                    float(row["perturbed_safe_weighted_coverage"])
                )
            components = {
                "mean_retention": finite_mean(float(row["retention"]) for row in selected),
                "worst_retention": min(
                    finite_mean(values) for values in by_condition.values()
                ),
                "perturbed_safe_rate": finite_mean(
                    float(row["perturbed_safe_rate"]) for row in selected
                ),
                "map_consistency": clip01(
                    1.0
                    - float(
                        np.std(
                            [finite_mean(values) for values in by_map.values()],
                            ddof=1,
                        )
                    )
                ),
            }
            output.append(
                {
                    "scope": scope,
                    "model": model,
                    "condition_count": len(conditions),
                    "map_count": len(maps),
                    "missing_conditions_imputed": False,
                    **components,
                    "D4_scope_specific": weighted_arithmetic(
                        components, INTERNAL_WEIGHTS["D4"]
                    ),
                }
            )
    return output


def _dimension_rows(
    map_rows: Sequence[Mapping[str, Any]],
    robustness: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for scope_name, config in NOMINAL_SCOPES.items():
        domain = scope_name.split("_")[0]
        for model in config["models"]:
            selected = [
                row for row in map_rows if row["domain"] == domain and row["model"] == model
            ]
            if len(selected) != int(config["maps"]):
                raise RuntimeError(f"{scope_name}/{model} map grid mismatch: {len(selected)}")
            output.append(
                {
                    "scope": scope_name,
                    "model": model,
                    "map_count": len(selected),
                    "D1": finite_mean(float(row["D1_mission_effectiveness"]) for row in selected),
                    "D2": finite_mean(float(row["D2_resource_efficiency"]) for row in selected),
                    "D3": finite_mean(float(row["D3_safety_reliability"]) for row in selected),
                    "D4": float("nan"),
                    "D5": finite_mean(float(row["D5_online_deployability"]) for row in selected),
                    "mean_safe_time_s": finite_mean(float(row["mean_safe_time_s"]) for row in selected),
                    "mean_safe_energy_wh": finite_mean(float(row["mean_safe_energy_wh"]) for row in selected),
                    "mean_safe_distance_m": finite_mean(float(row["mean_safe_distance_m"]) for row in selected),
                    "planning_time_p95_s": finite_mean(float(row["planning_time_p95_s"]) for row in selected),
                }
            )
    robust_by_model = {str(row["model"]): row for row in robustness}
    for model in CORE_MODELS:
        domains = [
            row
            for row in output
            if row["model"] == model and row["scope"] in {"synthetic_all", "real_all"}
        ]
        if len(domains) != 2:
            raise RuntimeError(f"core domain grid missing for {model}")
        output.append(
            {
                "scope": "core_learning_complete",
                "model": model,
                "map_count": 32,
                "D1": finite_mean(float(row["D1"]) for row in domains),
                "D2": finite_mean(float(row["D2"]) for row in domains),
                "D3": finite_mean(float(row["D3"]) for row in domains),
                "D4": float(robust_by_model[model]["D4_robustness"]),
                "D5": finite_mean(float(row["D5"]) for row in domains),
                "mean_safe_time_s": finite_mean(float(row["mean_safe_time_s"]) for row in domains),
                "mean_safe_energy_wh": finite_mean(float(row["mean_safe_energy_wh"]) for row in domains),
                "mean_safe_distance_m": finite_mean(float(row["mean_safe_distance_m"]) for row in domains),
                "planning_time_p95_s": finite_mean(float(row["planning_time_p95_s"]) for row in domains),
            }
        )
    return output


def _score_rows(
    dimensions: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_rows = []
    grid_rows = []
    grids = enumerate_weight_grid()
    for row in dimensions:
        names = ("D1", "D2", "D3", "D4", "D5") if math.isfinite(float(row["D4"])) else ("D1", "D2", "D3", "D5")
        values = {name: float(row[name]) for name in names}
        for scenario, base_weights in SCENARIOS.items():
            weights = renormalize_weights(base_weights, names)
            for method, function in (("geometric", weighted_geometric), ("arithmetic", weighted_arithmetic)):
                scenario_rows.append(
                    {
                        "scope": row["scope"],
                        "model": row["model"],
                        "scenario": scenario,
                        "aggregation": method,
                        "score": function(values, weights),
                        **{f"weight_{name}": weights.get(name, 0.0) for name in DEFAULT_WEIGHTS},
                    }
                )
        projected: dict[tuple[float, ...], dict[str, float]] = {}
        for base_weights in grids:
            weights = renormalize_weights(base_weights, names)
            identity = tuple(round(weights.get(name, 0.0), 10) for name in DEFAULT_WEIGHTS)
            projected[identity] = weights
        for grid_id, weights in enumerate(projected.values()):
            for method, function in (("geometric", weighted_geometric), ("arithmetic", weighted_arithmetic)):
                grid_rows.append(
                    {
                        "scope": row["scope"],
                        "model": row["model"],
                        "grid_id": grid_id,
                        "aggregation": method,
                        "score": function(values, weights),
                        **{f"weight_{name}": weights.get(name, 0.0) for name in DEFAULT_WEIGHTS},
                    }
                )

    summary = []
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in grid_rows:
        grouped[(str(row["scope"]), str(row["aggregation"]), int(row["grid_id"]))].append(row)
    rank_records: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    first_records: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    score_records: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (scope, method, _), rows in grouped.items():
        ordered = sorted(rows, key=lambda item: (-float(item["score"]), str(item["model"])))
        best = float(ordered[0]["score"])
        for rank, item in enumerate(ordered, start=1):
            item["rank"] = rank
            item["is_first"] = float(abs(float(item["score"]) - best) <= 1e-12)
            key = (scope, method, str(item["model"]))
            rank_records[key].append(float(rank))
            first_records[key].append(float(item["is_first"]))
            score_records[key].append(float(item["score"]))
    for key in sorted(rank_records):
        scope, method, model = key
        values = score_records[key]
        summary.append(
            {
                "scope": scope,
                "aggregation": method,
                "model": model,
                "grid_count": len(values),
                "first_place_share": finite_mean(first_records[key]),
                "mean_rank": finite_mean(rank_records[key]),
                "minimum_score": min(values),
                "maximum_score": max(values),
                "mean_score": finite_mean(values),
            }
        )
    return scenario_rows, grid_rows, summary


def _pareto_rows(dimensions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scope in sorted({str(row["scope"]) for row in dimensions}):
        selected = [dict(row) for row in dimensions if str(row["scope"]) == scope]
        objectives = {
            "D1": "max",
            "mean_safe_time_s": "min",
            "mean_safe_energy_wh": "min",
            "mean_safe_distance_m": "min",
            "planning_time_p95_s": "min",
        }
        if scope == "core_learning_complete":
            objectives["D4"] = "max"
        flags = pareto_membership(selected, objectives)
        for row, flag in zip(selected, flags):
            output.append(
                {
                    "scope": scope,
                    "model": row["model"],
                    "is_pareto_nondominated": flag,
                    "objectives": "|".join(objectives),
                    **{key: row[key] for key in objectives},
                }
            )
    return output


def _training_stability(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = []
    for model in ("full", "a2c_pointer"):
        for seed in range(42, 47):
            path = TRAINING_ROOT / f"formal_{model}_seed{seed}_3000ep/metrics.jsonl"
            metrics = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            tail = [row for row in metrics if float(row["episodes_seen"]) > 2500]
            if not tail or float(metrics[-1]["episodes_seen"]) != 3000:
                raise RuntimeError(f"incomplete training trace: {model}/{seed}")
            nominal = [
                row
                for row in rows
                if str(row["model"]) == model
                and int(row["repeat_seed"]) == seed
                and str(row["family"]) in {"synthetic_learning", "real_learning"}
            ]
            nominal_by_domain = {
                domain: [
                    row
                    for row in nominal
                    if str(row["family"]) == family
                ]
                for domain, family in {
                    "synthetic": "synthetic_learning",
                    "real": "real_learning",
                }.items()
            }
            domain_means = {
                domain: finite_mean(
                    float(row["safe_weighted_coverage"]) for row in values
                )
                for domain, values in nominal_by_domain.items()
            }
            domain_safe_rates = {
                domain: finite_mean(float(row["safe_rate"]) for row in values)
                for domain, values in nominal_by_domain.items()
            }
            output.append(
                {
                    "model": model,
                    "training_seed": seed,
                    "tail_update_count": len(tail),
                    "tail_mean_weighted_coverage": finite_mean(
                        float(row["mean_weighted_coverage"]) for row in tail
                    ),
                    "tail_mean_approx_kl": finite_mean(float(row["approx_kl"]) for row in tail),
                    "tail_mean_ratio_deviation": finite_mean(
                        float(row["ratio_deviation"]) for row in tail
                    ),
                    "tail_mean_gradient_norm_pre_clip": finite_mean(
                        float(row["gradient_norm_pre_clip"]) for row in tail
                    ),
                    "tail_mean_entropy": finite_mean(float(row["entropy"]) for row in tail),
                    "best_episode": float(metrics[-1]["best_episode"]),
                    "environment_interactions": int(metrics[-1]["environment_interactions"]),
                    "formal_synthetic_safe_weighted_coverage": domain_means["synthetic"],
                    "formal_real_safe_weighted_coverage": domain_means["real"],
                    "formal_nominal_safe_weighted_coverage": finite_mean(
                        domain_means.values()
                    ),
                    "formal_synthetic_safe_rate": domain_safe_rates["synthetic"],
                    "formal_real_safe_rate": domain_safe_rates["real"],
                    "formal_nominal_safe_rate": finite_mean(
                        domain_safe_rates.values()
                    ),
                }
            )
    summary = []
    metrics = [
        "tail_mean_weighted_coverage",
        "tail_mean_approx_kl",
        "tail_mean_ratio_deviation",
        "tail_mean_gradient_norm_pre_clip",
        "tail_mean_entropy",
        "best_episode",
        "environment_interactions",
        "formal_synthetic_safe_weighted_coverage",
        "formal_real_safe_weighted_coverage",
        "formal_nominal_safe_weighted_coverage",
        "formal_synthetic_safe_rate",
        "formal_real_safe_rate",
        "formal_nominal_safe_rate",
    ]
    for model in ("full", "a2c_pointer"):
        selected = [row for row in output if row["model"] == model]
        record: dict[str, Any] = {"model": model, "seed_count": len(selected)}
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            record[f"{metric}_mean"] = finite_mean(values)
            record[f"{metric}_sample_sd"] = float(np.std(values, ddof=1))
        summary.append(record)
    return output, summary


def _deadline_rows(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scope, config in NOMINAL_SCOPES.items():
        domain = scope.split("_")[0]
        for model in config["models"]:
            selected = [
                row
                for row in raw_rows
                if str(row["model"]) == model
                and str(row["family"]) in config["families"]
            ]
            p95 = float(np.percentile([float(row["planning_time_s"]) for row in selected], 95))
            for deadline in DEADLINE_SENSITIVITY:
                output.append(
                    {
                        "scope": scope,
                        "domain": domain,
                        "model": model,
                        "deadline_s": deadline,
                        "planning_time_p95_s": p95,
                        "online_planning_efficiency": online_utility(p95, deadline),
                    }
                )
    return output


def _metric_dictionary() -> list[dict[str, Any]]:
    return [
        {"field": "D1", "label": "任务与巡检效果", "direction": "maximize", "definition": "0.50加权覆盖+0.25高优先级覆盖+0.25原始oracle达成率"},
        {"field": "D2", "label": "执行与资源效率", "direction": "maximize", "definition": "安全路线的时间/能量/距离预算余量，权重0.40/0.35/0.25；无安全路线记0"},
        {"field": "D3", "label": "安全与返航可靠性", "direction": "maximize", "definition": "安全率、返航率、无违规率、最低SOC效用的加权几何均值"},
        {"field": "D4", "label": "鲁棒性与跨域保持", "direction": "maximize", "definition": "平均保持、最差扰动保持、扰动安全率、地图一致性，权重0.40/0.30/0.20/0.10"},
        {"field": "D5", "label": "在线部署能力", "direction": "maximize", "definition": "10秒默认截止下P95在线规划效用与24/16节点扩展效率，权重0.70/0.30"},
        {"field": "overall_geometric", "label": "非完全补偿综合分", "direction": "maximize", "definition": "默认五维权重0.35/0.25/0.20/0.12/0.08；名义域删除D4后归一化"},
    ]


def run() -> dict[str, Any]:
    protocol = _validate_protocol()
    audit = _validate_inputs()
    # 复用已审计的路线派生器，但不写入旧分析目录。
    enhanced = legacy._analysis_rows()
    if len(enhanced) != EXPECTED_ROWS:
        raise RuntimeError("enhanced result row count mismatch")
    identities = {
        (
            row["family"],
            row["task_id"],
            row["model"],
            row["repeat_seed"],
            row["condition"],
        )
        for row in enhanced
    }
    if len(identities) != EXPECTED_ROWS:
        raise RuntimeError("duplicate enhanced result identity")

    task_rows = _nominal_task_rows(enhanced)
    map_rows = _nominal_map_rows(task_rows, enhanced)
    condition_rows, robustness_rows = _robustness_rows(enhanced)
    mechanism_robustness_rows = _mechanism_robustness_rows(condition_rows)
    dimension_rows = _dimension_rows(map_rows, robustness_rows)
    scenario_rows, grid_rows, grid_summary = _score_rows(dimension_rows)
    pareto_rows = _pareto_rows(dimension_rows)
    training_seed_rows, training_summary_rows = _training_stability(enhanced)
    deadline_rows = _deadline_rows(enhanced)
    domain_rows = [
        row
        for row in dimension_rows
        if row["scope"] in {"synthetic_all", "real_all"}
        and row["model"] in CORE_MODELS
    ]

    files: dict[str, Sequence[Mapping[str, Any]]] = {
        "metric_dictionary.csv": _metric_dictionary(),
        "nominal_task_metrics.csv": task_rows,
        "nominal_map_dimensions.csv": map_rows,
        "dimension_scores.csv": dimension_rows,
        "robustness_condition_dimensions.csv": condition_rows,
        "robustness_model_dimensions.csv": robustness_rows,
        "mechanism_robustness_summary.csv": mechanism_robustness_rows,
        "scenario_scores.csv": scenario_rows,
        "weight_sensitivity_grid.csv": grid_rows,
        "weight_sensitivity_summary.csv": grid_summary,
        "pareto_membership.csv": pareto_rows,
        "training_stability_seed.csv": training_seed_rows,
        "training_stability_summary.csv": training_summary_rows,
        "online_deadline_sensitivity.csv": deadline_rows,
        "domain_generalization_summary.csv": domain_rows,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    forbidden = [
        path
        for suffix in ("*.png", "*.svg", "*.pdf")
        for path in DESTINATION.rglob(suffix)
    ]
    if forbidden:
        raise RuntimeError(f"plot files are forbidden before plot plan: {forbidden}")
    for name, content in files.items():
        _atomic_csv(DESTINATION / name, content)

    output_hashes = {name: sha256_file(DESTINATION / name) for name in sorted(files)}
    analysis_audit = {
        "schema_version": "manuscript_multiobjective_v1",
        "passed": True,
        "analysis_role": "post_result_manuscript_draft_revision",
        "not_preregistered_confirmatory": True,
        "raw_result_row_count": len(enhanced),
        "raw_unique_identity_count": len(identities),
        "nominal_task_row_count": len(task_rows),
        "nominal_map_row_count": len(map_rows),
        "dimension_row_count": len(dimension_rows),
        "robustness_condition_row_count": len(condition_rows),
        "mechanism_robustness_row_count": len(mechanism_robustness_rows),
        "weight_grid_row_count": len(grid_rows),
        "plots_created": False,
        "missing_robustness_imputed": False,
        "matrix_sha256": sha256_file(MATRIX),
        "final_results_sha256": sha256_file(FINAL_RESULTS),
        "parent_final_audit_hash": canonical_hash(audit),
        "protocol_hash": protocol["protocol_hash"],
        "implementation_sha256": sha256_file(Path(__file__)),
        "output_hashes": output_hashes,
    }
    analysis_audit["audit_hash"] = canonical_hash(analysis_audit)
    _atomic_json(DESTINATION / "multiobjective_audit.json", analysis_audit)
    output_hashes["multiobjective_audit.json"] = sha256_file(
        DESTINATION / "multiobjective_audit.json"
    )
    manifest = {
        "schema_version": "manuscript_multiobjective_v1",
        "state": "ready_for_plot_plan",
        "plots_created": False,
        "plot_files": [],
        "row_count": EXPECTED_ROWS,
        "protocol_hash": protocol["protocol_hash"],
        "matrix_sha256": EXPECTED_MATRIX_SHA256,
        "final_results_sha256": EXPECTED_RESULTS_SHA256,
        "analysis_audit_hash": analysis_audit["audit_hash"],
        "csv_and_audit_hashes": output_hashes,
        "next_step": "freeze_formal_plotting_plan_then_render_figures",
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _atomic_json(DESTINATION / "analysis_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="execute the frozen analysis")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

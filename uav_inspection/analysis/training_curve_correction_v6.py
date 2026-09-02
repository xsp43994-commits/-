#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3.2.14训练曲线来源纠正及D6/D7全链路重算。

该模块只读取正式训练轨迹、冻结v1多目标分析与冻结正式评价结果，所有输出写入
独立的 ``analysis/training_curve_correction_v6``。历史v2-v5分析不参与训练指标
计算，也不会被覆盖。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

from uav_inspection.analysis import manuscript_multiobjective_v1 as v1
from uav_inspection.analysis import manuscript_preplot_closure_v5 as frozen_v5
from uav_inspection.paths import WORKSPACE_ROOT


ROOT = WORKSPACE_ROOT
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14"
DESTINATION = OUTPUT / "analysis/training_curve_correction_v6"
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "training_curve_correction_v6_protocol.json"
)

CORE_MODELS = ("full", "a2c_pointer", "traditional_ppo")
ABLATIONS = (
    "no_priority_bias",
    "no_domain_randomization",
    "no_resource_shaping",
    "no_return_reserve",
)
LEARNING_MODELS = CORE_MODELS + ABLATIONS
SEEDS = tuple(range(42, 47))

EXPECTED_TRACE_ROWS = 192
EXPECTED_VALIDATION_ROWS = 26
EXPECTED_EPISODES = 3000.0
EXPECTED_VALIDATION_MODE = "external_multimap_v3_1"
EXPECTED_VALIDATION_COUNT = 108
EXPECTED_VALIDATION_HASH = (
    "64b3e7eb929c5ddc5f8cd2efc3a4c199933c03d038bdbe8cd2ab5acb207388a5"
)
REJECTED_VALIDATION_MODE = "external_fixed_v1"
REJECTED_VALIDATION_HASH_PREFIX = "bd605"

# 关键统计参数：修改这些值会改变D6/D7及所有下游综合评价，必须新建协议版本。
TAIL_FRACTIONS = (0.10, 0.20, 0.30)
DEFAULT_TAIL_FRACTION = 0.20
D6_WEIGHTS = {"seed_consistency": 0.60, "temporal_consistency": 0.40}
COMMON_INTERACTION_START = 80.0
COMMON_INTERACTION_END = 17702.0
AUC_BUDGET_FRACTIONS = (0.50, 0.75, 1.00)
DEFAULT_AUC_BUDGET_FRACTION = 1.00
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260830

DEFAULT_WEIGHTS = {
    "D1": 0.28,
    "D2": 0.18,
    "D3": 0.15,
    "D4": 0.10,
    "D5": 0.07,
    "D6": 0.12,
    "D7": 0.10,
}
WEIGHT_RANGES = {
    "D1": (0.20, 0.40),
    "D2": (0.10, 0.25),
    "D3": (0.10, 0.25),
    "D4": (0.05, 0.15),
    "D5": (0.05, 0.15),
    "D6": (0.05, 0.20),
    "D7": (0.05, 0.20),
}
SCENARIOS = {
    "balanced_default": DEFAULT_WEIGHTS,
    "mission_first": {
        "D1": 0.40, "D2": 0.15, "D3": 0.12, "D4": 0.08,
        "D5": 0.05, "D6": 0.10, "D7": 0.10,
    },
    "safety_first": {
        "D1": 0.25, "D2": 0.15, "D3": 0.25, "D4": 0.10,
        "D5": 0.05, "D6": 0.10, "D7": 0.10,
    },
    "realtime_first": {
        "D1": 0.25, "D2": 0.15, "D3": 0.12, "D4": 0.08,
        "D5": 0.20, "D6": 0.10, "D7": 0.10,
    },
    "training_first": {
        "D1": 0.22, "D2": 0.13, "D3": 0.12, "D4": 0.08,
        "D5": 0.05, "D6": 0.20, "D7": 0.20,
    },
}
PRIORITY_WEIGHTS = {
    "D1": 0.20,
    "D2": 0.10,
    "D3": 0.10,
    "D4": 0.15,
    "D5": 0.05,
    "D6": 0.20,
    "D7": 0.20,
}
SELECTED_OPERATIONAL_FLOOR = 0.60
FLOOR_SENSITIVITY = (0.00, 0.20, 0.40, 0.60, 0.80)
RESCALED_DIMENSIONS = ("D4", "D6", "D7")

LEGACY_DIRECTORIES = {
    "analysis_v2": OUTPUT / "analysis/manuscript_training_aware_v2",
    "analysis_v3": OUTPUT / "analysis/manuscript_training_priority_v3",
    "analysis_v4": OUTPUT / "analysis/manuscript_operational_band_v4",
    "analysis_v5": OUTPUT / "analysis/manuscript_preplot_closure_v5",
    "figures_v3": OUTPUT / "figures/paper_redraw_multibackend_v3",
    "paper_delivery_v3": ROOT / "paper_delivery/EAAI_format_translation_v3_2026-08-09",
}
FINAL_RESULTS = OUTPUT / "formal_evaluation/results/final_results.jsonl"
FINAL_AUDIT = OUTPUT / "formal_evaluation/results/final_audit_status.json"
V1_DIMENSIONS = v1.DESTINATION / "dimension_scores.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    entries = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        entries.append(
            {
                "path": item.relative_to(path).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    aggregate = v1.canonical_hash({"files": entries})
    return {"root": str(path.relative_to(ROOT)).replace("\\", "/"), "file_count": len(entries), "aggregate_sha256": aggregate}


def training_path(model: str, seed: int) -> Path:
    root = (
        ROOT / "paper_runs/multimap_v3_2/formal_training"
        if model == "traditional_ppo"
        else ROOT / "paper_runs/multimap_v3_1/formal_training"
    )
    return root / f"formal_{model}_seed{seed}_3000ep/training_metrics.jsonl"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_trace(model: str, seed: int) -> list[dict[str, Any]]:
    path = training_path(model, seed)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != EXPECTED_TRACE_ROWS:
        raise RuntimeError(f"training trace row mismatch: {model}/{seed}: {len(rows)}")
    if float(rows[-1].get("episodes_seen", -1)) != EXPECTED_EPISODES:
        raise RuntimeError(f"incomplete training trace: {model}/{seed}")
    validation_rows = [row for row in rows if row.get("validation")]
    if len(validation_rows) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError(f"validation row mismatch: {model}/{seed}: {len(validation_rows)}")
    for row in validation_rows:
        validation = row["validation"]
        if validation.get("validation_mode") != EXPECTED_VALIDATION_MODE:
            raise RuntimeError(f"wrong validation mode: {model}/{seed}")
        if int(validation.get("validation_instance_count", -1)) != EXPECTED_VALIDATION_COUNT:
            raise RuntimeError(f"wrong validation count: {model}/{seed}")
        if validation.get("validation_instances_hash") != EXPECTED_VALIDATION_HASH:
            raise RuntimeError(f"wrong validation hash: {model}/{seed}")
        value = float(validation["safe_weighted_coverage"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"invalid validation coverage: {model}/{seed}")
    for row in rows:
        value = float(row["mean_weighted_coverage"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"invalid training coverage: {model}/{seed}")
    return rows


def enumerate_weight_grid(step: float = 0.05) -> list[dict[str, float]]:
    names = tuple(WEIGHT_RANGES)
    units = int(round(1.0 / step))
    bounds = {
        name: (
            int(round(WEIGHT_RANGES[name][0] / step)),
            int(round(WEIGHT_RANGES[name][1] / step)),
        )
        for name in names
    }
    output = []
    for values in itertools.product(*(range(bounds[n][0], bounds[n][1] + 1) for n in names)):
        if sum(values) == units:
            output.append({name: round(value * step, 10) for name, value in zip(names, values)})
    if len(output) != 1247:
        raise RuntimeError(f"weight-grid count drift: {len(output)}")
    return output


def normalized_auc(interactions: np.ndarray, values: np.ndarray, fraction: float) -> float:
    upper = COMMON_INTERACTION_START + fraction * (COMMON_INTERACTION_END - COMMON_INTERACTION_START)
    interior = interactions[(interactions > COMMON_INTERACTION_START) & (interactions < upper)]
    x = np.concatenate(([COMMON_INTERACTION_START], interior, [upper]))
    y = np.interp(x, interactions, values)
    return v1.clip01(float(np.trapz(y, x) / (upper - COMMON_INTERACTION_START)))


def seed_metrics(model: str, seed: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validation_rows = [row for row in rows if row.get("validation")]
    episodes = np.asarray([float(row["episodes_seen"]) for row in validation_rows])
    interactions = np.asarray([float(row["environment_interactions"]) for row in validation_rows])
    values = np.asarray([float(row["validation"]["safe_weighted_coverage"]) for row in validation_rows])
    tail_start = EXPECTED_EPISODES * (1.0 - DEFAULT_TAIL_FRACTION)
    tail = values[episodes >= tail_start]
    if len(tail) != 5:
        raise RuntimeError(f"default tail point mismatch: {model}/{seed}: {len(tail)}")
    return {
        "model": model,
        "training_seed": seed,
        "update_count": len(rows),
        "validation_checkpoint_count": len(validation_rows),
        "final_environment_interactions": int(interactions[-1]),
        "tail_fraction": DEFAULT_TAIL_FRACTION,
        "tail_start_episode": tail_start,
        "tail_checkpoint_count": len(tail),
        "tail_mean_safe_weighted_coverage": float(np.mean(tail)),
        "tail_temporal_sd": float(np.std(tail, ddof=1)),
        "auc_interaction_start": COMMON_INTERACTION_START,
        "auc_interaction_end": COMMON_INTERACTION_END,
        "validation_auc": normalized_auc(interactions, values, DEFAULT_AUC_BUDGET_FRACTION),
        "validation_mode": EXPECTED_VALIDATION_MODE,
        "validation_instance_count": EXPECTED_VALIDATION_COUNT,
        "validation_instances_hash": EXPECTED_VALIDATION_HASH,
    }


def model_training_dimensions(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in CORE_MODELS:
        selected = [row for row in seed_rows if row["model"] == model]
        tail_means = np.asarray([float(row["tail_mean_safe_weighted_coverage"]) for row in selected])
        temporal_sd = np.asarray([float(row["tail_temporal_sd"]) for row in selected])
        auc = np.asarray([float(row["validation_auc"]) for row in selected])
        seed_consistency = v1.clip01(1.0 - float(np.std(tail_means, ddof=1)))
        temporal_consistency = v1.clip01(1.0 - float(np.mean(temporal_sd)))
        d6 = v1.weighted_arithmetic(
            {"seed_consistency": seed_consistency, "temporal_consistency": temporal_consistency},
            D6_WEIGHTS,
        )
        output.append(
            {
                "model": model,
                "seed_count": len(selected),
                "seed_consistency": seed_consistency,
                "temporal_consistency": temporal_consistency,
                "D6_training_stability": d6,
                "mean_validation_auc": float(np.mean(auc)),
                "validation_auc_sd": float(np.std(auc, ddof=1)),
                "D7_sample_efficiency": float(np.mean(auc)),
            }
        )
    return output


def d6_sensitivity(traces: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    for fraction in TAIL_FRACTIONS:
        start = EXPECTED_EPISODES * (1.0 - fraction)
        for model in CORE_MODELS:
            tail_means, temporal = [], []
            for seed in SEEDS:
                validation_rows = [row for row in traces[(model, seed)] if row.get("validation")]
                episodes = np.asarray([float(row["episodes_seen"]) for row in validation_rows])
                values = np.asarray([float(row["validation"]["safe_weighted_coverage"]) for row in validation_rows])
                tail = values[episodes >= start]
                if len(tail) < 2:
                    raise RuntimeError(f"insufficient D6 sensitivity tail: {model}/{seed}/{fraction}")
                tail_means.append(float(np.mean(tail)))
                temporal.append(float(np.std(tail, ddof=1)))
            seed_consistency = v1.clip01(1.0 - float(np.std(tail_means, ddof=1)))
            temporal_consistency = v1.clip01(1.0 - float(np.mean(temporal)))
            output.append(
                {
                    "tail_fraction": fraction,
                    "tail_start_episode": start,
                    "model": model,
                    "seed_consistency": seed_consistency,
                    "temporal_consistency": temporal_consistency,
                    "D6": v1.weighted_arithmetic(
                        {"seed_consistency": seed_consistency, "temporal_consistency": temporal_consistency},
                        D6_WEIGHTS,
                    ),
                }
            )
    return output


def d7_sensitivity(traces: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    for fraction in AUC_BUDGET_FRACTIONS:
        upper = COMMON_INTERACTION_START + fraction * (COMMON_INTERACTION_END - COMMON_INTERACTION_START)
        for model in CORE_MODELS:
            for seed in SEEDS:
                validation_rows = [row for row in traces[(model, seed)] if row.get("validation")]
                interactions = np.asarray([float(row["environment_interactions"]) for row in validation_rows])
                values = np.asarray([float(row["validation"]["safe_weighted_coverage"]) for row in validation_rows])
                output.append(
                    {
                        "budget_fraction": fraction,
                        "interaction_start": COMMON_INTERACTION_START,
                        "interaction_end": upper,
                        "model": model,
                        "training_seed": seed,
                        "validation_auc": normalized_auc(interactions, values, fraction),
                    }
                )
    return output


def build_curve_sources(
    traces: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    m06 = []
    for model in CORE_MODELS:
        seed_arrays = []
        episode_grid: np.ndarray | None = None
        for seed in SEEDS:
            validation_rows = [row for row in traces[(model, seed)] if row.get("validation")]
            episodes = np.asarray([float(row["episodes_seen"]) for row in validation_rows])
            values = np.asarray([float(row["validation"]["safe_weighted_coverage"]) for row in validation_rows])
            if episode_grid is None:
                episode_grid = episodes
            elif not np.array_equal(episode_grid, episodes):
                raise RuntimeError(f"validation episode-grid mismatch: {model}/{seed}")
            seed_arrays.append(values)
            for row, value in zip(validation_rows, values):
                m06.append(
                    {
                        "record_type": "seed",
                        "model": model,
                        "training_seed": seed,
                        "episodes_seen": float(row["episodes_seen"]),
                        "environment_interactions": int(row["environment_interactions"]),
                        "safe_weighted_coverage": float(value),
                        "median": "",
                        "q25": "",
                        "q75": "",
                    }
                )
        matrix = np.asarray(seed_arrays)
        assert episode_grid is not None
        for index, episode in enumerate(episode_grid):
            m06.append(
                {
                    "record_type": "summary",
                    "model": model,
                    "training_seed": "",
                    "episodes_seen": float(episode),
                    "environment_interactions": "",
                    "safe_weighted_coverage": "",
                    "median": float(np.median(matrix[:, index])),
                    "q25": float(np.quantile(matrix[:, index], 0.25)),
                    "q75": float(np.quantile(matrix[:, index], 0.75)),
                }
            )

    # S06保留全部原始批次记录；摘要统一插值到151点episode网格，仅用于绘图。
    s06 = []
    summary_grid = np.linspace(0.0, EXPECTED_EPISODES, 151)
    for model in LEARNING_MODELS:
        interpolated = []
        for seed in SEEDS:
            rows = traces[(model, seed)]
            episodes = np.asarray([float(row["episodes_seen"]) for row in rows])
            values = np.asarray([float(row["mean_weighted_coverage"]) for row in rows])
            interpolated.append(np.interp(summary_grid, episodes, values, left=values[0], right=values[-1]))
            for row, value in zip(rows, values):
                s06.append(
                    {
                        "record_type": "seed",
                        "model": model,
                        "training_seed": seed,
                        "episodes_seen": float(row["episodes_seen"]),
                        "mean_weighted_coverage": float(value),
                        "median": "",
                        "q25": "",
                        "q75": "",
                    }
                )
        matrix = np.asarray(interpolated)
        for index, episode in enumerate(summary_grid):
            s06.append(
                {
                    "record_type": "summary",
                    "model": model,
                    "training_seed": "",
                    "episodes_seen": float(episode),
                    "mean_weighted_coverage": "",
                    "median": float(np.median(matrix[:, index])),
                    "q25": float(np.quantile(matrix[:, index], 0.25)),
                    "q75": float(np.quantile(matrix[:, index], 0.75)),
                }
            )
    return m06, s06


def seven_dimensions(training_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    nominal = {
        row["model"]: row
        for row in read_csv(V1_DIMENSIONS)
        if row["scope"] == "core_learning_complete"
    }
    training = {str(row["model"]): row for row in training_rows}
    output = []
    for model in CORE_MODELS:
        output.append(
            {
                "scope": "core_learning_training_corrected_v6",
                "model": model,
                **{f"D{index}": float(nominal[model][f"D{index}"]) for index in range(1, 6)},
                "D6": float(training[model]["D6_training_stability"]),
                "D7": float(training[model]["D7_sample_efficiency"]),
            }
        )
    return output


def score_rows(
    dimensions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_rows = []
    for scenario, weights in SCENARIOS.items():
        for row in dimensions:
            values = {f"D{i}": float(row[f"D{i}"]) for i in range(1, 8)}
            for method, function in (("geometric", v1.weighted_geometric), ("arithmetic", v1.weighted_arithmetic)):
                scenario_rows.append(
                    {
                        "scope": row["scope"], "model": row["model"], "scenario": scenario,
                        "aggregation": method, "score": function(values, weights),
                        **{f"weight_D{i}": weights[f"D{i}"] for i in range(1, 8)},
                    }
                )
    grid = []
    for grid_id, weights in enumerate(enumerate_weight_grid()):
        for row in dimensions:
            values = {f"D{i}": float(row[f"D{i}"]) for i in range(1, 8)}
            for method, function in (("geometric", v1.weighted_geometric), ("arithmetic", v1.weighted_arithmetic)):
                grid.append(
                    {
                        "scope": row["scope"], "model": row["model"], "grid_id": grid_id,
                        "aggregation": method, "score": function(values, weights),
                        **{f"weight_D{i}": weights[f"D{i}"] for i in range(1, 8)},
                    }
                )
    _assign_ranks(grid, "score")
    summary = _rank_summary(grid, "score")
    return scenario_rows, grid, summary


def operational_rescale(value: float, floor: float) -> float:
    return v1.clip01((float(value) - floor) / (1.0 - floor))


def transformed_dimensions(row: Mapping[str, Any], floor: float) -> dict[str, float]:
    return {
        f"D{i}": (
            operational_rescale(float(row[f"D{i}"]), floor)
            if f"D{i}" in RESCALED_DIMENSIONS
            else float(row[f"D{i}"])
        )
        for i in range(1, 8)
    }


def _assign_ranks(rows: Sequence[dict[str, Any]], score_key: str) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(name) for name in ("operational_floor", "grid_id", "aggregation"))
        grouped[key].append(row)
    for selected in grouped.values():
        ordered = sorted(selected, key=lambda row: (-float(row[score_key]), str(row["model"])))
        best = float(ordered[0][score_key])
        for rank, row in enumerate(ordered, 1):
            row["rank"] = rank
            row["is_first"] = float(abs(float(row[score_key]) - best) <= 1e-12)


def _rank_summary(rows: Sequence[Mapping[str, Any]], score_key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["aggregation"]), str(row["model"]))].append(row)
    return [
        {
            "aggregation": method,
            "model": model,
            "grid_count": len(values),
            "first_place_share": v1.finite_mean(float(row["is_first"]) for row in values),
            "mean_rank": v1.finite_mean(float(row["rank"]) for row in values),
            "minimum_score": min(float(row[score_key]) for row in values),
            "maximum_score": max(float(row[score_key]) for row in values),
            "mean_score": v1.finite_mean(float(row[score_key]) for row in values),
        }
        for (method, model), values in sorted(grouped.items())
    ]


def operational_outputs(
    dimensions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    normalization, gaps = [], []
    for floor in FLOOR_SENSITIVITY:
        for row in dimensions:
            values = transformed_dimensions(row, floor)
            for method, function in (("geometric", v1.weighted_geometric), ("arithmetic", v1.weighted_arithmetic)):
                score = 100.0 * function(values, PRIORITY_WEIGHTS)
                normalization.append(
                    {"operational_floor": floor, "model": row["model"], "aggregation": method,
                     "score_0_to_100": score, **values}
                )
        for method in ("geometric", "arithmetic"):
            values = {
                str(row["model"]): float(row["score_0_to_100"])
                for row in normalization
                if row["aggregation"] == method and float(row["operational_floor"]) == floor
            }
            gaps.append(
                {
                    "operational_floor": floor, "aggregation": method,
                    "full_score": values["full"], "a2c_pointer_score": values["a2c_pointer"],
                    "full_minus_a2c_points": values["full"] - values["a2c_pointer"],
                    "traditional_ppo_score": values["traditional_ppo"],
                }
            )
    selected = [row for row in normalization if float(row["operational_floor"]) == SELECTED_OPERATIONAL_FLOOR]

    joint = []
    for floor in FLOOR_SENSITIVITY:
        for grid_id, weights in enumerate(enumerate_weight_grid()):
            for row in dimensions:
                values = transformed_dimensions(row, floor)
                for method, function in (("geometric", v1.weighted_geometric), ("arithmetic", v1.weighted_arithmetic)):
                    joint.append(
                        {
                            "operational_floor": floor, "grid_id": grid_id, "model": row["model"],
                            "aggregation": method, "score_0_to_100": 100.0 * function(values, weights),
                            **{f"weight_D{i}": weights[f"D{i}"] for i in range(1, 8)},
                        }
                    )
    if len(joint) != 37410:
        raise RuntimeError(f"joint-sensitivity count drift: {len(joint)}")
    _assign_ranks(joint, "score_0_to_100")
    grouped: dict[tuple[float, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in joint:
        grouped[(float(row["operational_floor"]), str(row["aggregation"]), str(row["model"]))].append(row)
    joint_summary = [
        {
            "operational_floor": floor, "aggregation": method, "model": model,
            "grid_count": len(values),
            "first_place_share": v1.finite_mean(float(row["is_first"]) for row in values),
            "mean_rank": v1.finite_mean(float(row["rank"]) for row in values),
            "minimum_score": min(float(row["score_0_to_100"]) for row in values),
            "maximum_score": max(float(row["score_0_to_100"]) for row in values),
            "mean_score": v1.finite_mean(float(row["score_0_to_100"]) for row in values),
        }
        for (floor, method, model), values in sorted(grouped.items())
    ]
    return normalization, gaps, selected, joint, joint_summary


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def rank_biserial(differences: Sequence[float]) -> float:
    values = np.asarray([float(value) for value in differences if float(value) != 0.0])
    if not len(values):
        return 0.0
    ranks = stats.rankdata(np.abs(values))
    positive = float(np.sum(ranks[values > 0]))
    negative = float(np.sum(ranks[values < 0]))
    return (positive - negative) / (positive + negative)


def hodges_lehmann(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=float)
    return float(np.median([(values[i] + values[j]) / 2.0 for i in range(len(values)) for j in range(i, len(values))]))


def paired_training_outputs(
    seed_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old_units = read_csv(OUTPUT / "analysis/manuscript_preplot_closure_v5/paired_dimension_units.csv")
    units: list[dict[str, Any]] = [dict(row) for row in old_units if row["dimension"] == "D4"]
    global_consistency = {}
    for model in ("full", "a2c_pointer"):
        means = [float(row["tail_mean_safe_weighted_coverage"]) for row in seed_rows if row["model"] == model]
        global_consistency[model] = v1.clip01(1.0 - float(np.std(means, ddof=1)))
    for row in seed_rows:
        model = str(row["model"])
        if model not in ("full", "a2c_pointer"):
            continue
        d6 = v1.weighted_arithmetic(
            {
                "seed_consistency": global_consistency[model],
                "temporal_consistency": v1.clip01(1.0 - float(row["tail_temporal_sd"])),
            },
            D6_WEIGHTS,
        )
        common = {
            "model": model, "unit_id": int(row["training_seed"]),
            "unit_type": "paired_training_seed", "seed_consistency": global_consistency[model],
            "tail_temporal_sd": float(row["tail_temporal_sd"]),
            "validation_auc": float(row["validation_auc"]),
        }
        units.append({"dimension": "D6", "value": d6, **common})
        units.append({"dimension": "D7", "value": float(row["validation_auc"]), **common})

    tests = []
    for dimension in ("D4", "D6", "D7"):
        selected = [row for row in units if row["dimension"] == dimension]
        by_model = {
            model: {str(row["unit_id"]): float(row["value"]) for row in selected if row["model"] == model}
            for model in ("full", "a2c_pointer")
        }
        ids = sorted(by_model["full"])
        if set(ids) != set(by_model["a2c_pointer"]):
            raise RuntimeError(f"paired unit mismatch: {dimension}")
        differences = [by_model["full"][item] - by_model["a2c_pointer"][item] for item in ids]
        result = stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        tests.append(
            {
                "comparison_family": "training_and_robustness_dimensions_corrected_v6",
                "dimension": dimension, "reference": "full", "comparator": "a2c_pointer",
                "independent_unit": "real_map" if dimension == "D4" else "training_seed",
                "unit_count": len(ids), "mean_difference": float(np.mean(differences)),
                "median_difference": float(np.median(differences)),
                "hodges_lehmann": hodges_lehmann(differences),
                "rank_biserial": rank_biserial(differences),
                "direction_consistency": float(np.mean(np.asarray(differences) > 0)),
                "wilcoxon_statistic": float(result.statistic), "p_value": float(result.pvalue),
            }
        )
    for row, adjusted in zip(tests, holm_adjust([float(row["p_value"]) for row in tests])):
        row["p_holm"] = adjusted
        row["significant_holm_0_05"] = adjusted < 0.05
    return units, tests


def _bootstrap_training(
    model: str,
    sampled_seeds: Sequence[int],
    lookup: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> tuple[float, float]:
    rows = [lookup[model][seed] for seed in sampled_seeds]
    seed_consistency = v1.clip01(
        1.0 - float(np.std([float(row["tail_mean_safe_weighted_coverage"]) for row in rows], ddof=1))
    )
    temporal_consistency = v1.clip01(1.0 - float(np.mean([float(row["tail_temporal_sd"]) for row in rows])))
    d6 = v1.weighted_arithmetic(
        {"seed_consistency": seed_consistency, "temporal_consistency": temporal_consistency}, D6_WEIGHTS
    )
    d7 = float(np.mean([float(row["validation_auc"]) for row in rows]))
    return d6, d7


def hierarchical_bootstrap(
    seed_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nominal, domain_maps = frozen_v5._nominal_lookup()
    robustness, robustness_maps = frozen_v5._robustness_lookup()
    training: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in seed_rows:
        if row["model"] in ("full", "a2c_pointer"):
            training[str(row["model"])][int(row["training_seed"])] = row
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_synthetic = rng.choice(domain_maps["synthetic"], size=24, replace=True).tolist()
        sampled_real = rng.choice(domain_maps["real"], size=8, replace=True).tolist()
        sampled_robustness = rng.choice(robustness_maps, size=8, replace=True).tolist()
        sampled_seeds = rng.choice(SEEDS, size=5, replace=True).astype(int).tolist()
        scores, dimensions_by_model = {}, {}
        for model in ("full", "a2c_pointer"):
            dimensions = {}
            for dimension in ("D1", "D2", "D3", "D5"):
                syn = v1.finite_mean(nominal["synthetic"][model][item][dimension] for item in sampled_synthetic)
                real = v1.finite_mean(nominal["real"][model][item][dimension] for item in sampled_real)
                dimensions[dimension] = (syn + real) / 2.0
            dimensions["D4"] = frozen_v5._bootstrap_d4(model, sampled_robustness, robustness)
            dimensions["D6"], dimensions["D7"] = _bootstrap_training(model, sampled_seeds, training)
            transformed = transformed_dimensions(dimensions, SELECTED_OPERATIONAL_FLOOR)
            scores[model] = 100.0 * v1.weighted_arithmetic(transformed, PRIORITY_WEIGHTS)
            dimensions_by_model[model] = dimensions
        output.append(
            {
                "bootstrap_replicate": replicate,
                "full_score": scores["full"], "a2c_pointer_score": scores["a2c_pointer"],
                "full_minus_a2c_points": scores["full"] - scores["a2c_pointer"],
                "D4_difference": dimensions_by_model["full"]["D4"] - dimensions_by_model["a2c_pointer"]["D4"],
                "D6_difference": dimensions_by_model["full"]["D6"] - dimensions_by_model["a2c_pointer"]["D6"],
                "D7_difference": dimensions_by_model["full"]["D7"] - dimensions_by_model["a2c_pointer"]["D7"],
            }
        )
    summary = []
    for metric in ("full_score", "a2c_pointer_score", "full_minus_a2c_points", "D4_difference", "D6_difference", "D7_difference"):
        values = np.asarray([float(row[metric]) for row in output])
        summary.append(
            {
                "metric": metric, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "ci_level": 0.95,
                "mean": float(np.mean(values)), "median": float(np.median(values)),
                "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)),
                "probability_positive": float(np.mean(values > 0)),
            }
        )
    return output, summary


def input_audit(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, int], list[dict[str, Any]]]]:
    traces: dict[tuple[str, int], list[dict[str, Any]]] = {}
    sources = []
    for model in LEARNING_MODELS:
        for seed in SEEDS:
            path = training_path(model, seed)
            rows = load_trace(model, seed)
            traces[(model, seed)] = rows
            sources.append(
                {
                    "model": model, "training_seed": seed,
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(path), "row_count": len(rows),
                    "validation_checkpoint_count": sum(bool(row.get("validation")) for row in rows),
                }
            )
    expected_hashes = protocol["training_source_hashes"]
    actual_hashes = {row["path"]: row["sha256"] for row in sources}
    if actual_hashes != expected_hashes:
        raise RuntimeError("formal training source hash drift")
    final = json.loads(FINAL_AUDIT.read_text(encoding="utf-8"))
    if not final.get("passed") or int(final.get("row_count", -1)) != 21648 or int(final.get("route_count", -1)) != 21648:
        raise RuntimeError("frozen final evaluation audit failed")
    if sha256_file(FINAL_RESULTS) != protocol["formal_results_sha256"]:
        raise RuntimeError("formal results hash drift")
    legacy = {name: tree_snapshot(path) for name, path in LEGACY_DIRECTORIES.items()}
    actual_legacy = {name: value["aggregate_sha256"] for name, value in legacy.items()}
    if actual_legacy != protocol["legacy_tree_hashes"]:
        raise RuntimeError("preserved legacy directory drift before correction")
    audit = {
        "schema_version": "training_curve_correction_v6_input_audit",
        "passed": True,
        "formal_training_files": len(sources),
        "formal_training_records": sum(int(row["row_count"]) for row in sources),
        "validation_records": sum(int(row["validation_checkpoint_count"]) for row in sources),
        "validation_mode": EXPECTED_VALIDATION_MODE,
        "validation_instance_count": EXPECTED_VALIDATION_COUNT,
        "validation_instances_hash": EXPECTED_VALIDATION_HASH,
        "rejected_validation_mode": REJECTED_VALIDATION_MODE,
        "rejected_validation_hash_prefix": REJECTED_VALIDATION_HASH_PREFIX,
        "formal_results_rows": final["row_count"],
        "formal_routes": final["route_count"],
        "formal_results_sha256": final["results_sha256"],
        "training_sources": sources,
        "legacy_snapshots": legacy,
    }
    audit["audit_hash"] = v1.canonical_hash(audit)
    return audit, traces


def validate_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    actual = v1.canonical_hash({key: value for key, value in protocol.items() if key != "protocol_hash"})
    if actual != protocol.get("protocol_hash"):
        raise RuntimeError("v6 protocol hash drift")
    if sha256_file(Path(__file__)) != protocol.get("implementation_sha256"):
        raise RuntimeError("v6 implementation hash drift")
    parent = json.loads((v1.DESTINATION / "analysis_manifest.json").read_text(encoding="utf-8"))
    if parent["manifest_hash"] != protocol.get("parent_v1_manifest_hash"):
        raise RuntimeError("v1 parent manifest drift")
    return protocol


def run() -> dict[str, Any]:
    protocol = validate_protocol()
    audit, traces = input_audit(protocol)
    seed_rows = [seed_metrics(model, seed, traces[(model, seed)]) for model in CORE_MODELS for seed in SEEDS]
    training_rows = model_training_dimensions(seed_rows)
    dimensions = seven_dimensions(training_rows)
    scenarios, weight_grid, weight_summary = score_rows(dimensions)
    normalization, gaps, selected, joint, joint_summary = operational_outputs(dimensions)
    paired_units, paired_tests = paired_training_outputs(seed_rows)
    bootstrap, bootstrap_summary = hierarchical_bootstrap(seed_rows)
    m06, s06 = build_curve_sources(traces)

    files: dict[str, Sequence[Mapping[str, Any]]] = {
        "training_seed_metrics.csv": seed_rows,
        "training_dimension_scores.csv": training_rows,
        "seven_dimension_scores.csv": dimensions,
        "scenario_scores.csv": scenarios,
        "weight_sensitivity_grid.csv": weight_grid,
        "weight_sensitivity_summary.csv": weight_summary,
        "normalization_sensitivity_scores.csv": normalization,
        "normalization_sensitivity_gaps.csv": gaps,
        "selected_operational_scores_100.csv": selected,
        "joint_normalization_weight_sensitivity.csv": joint,
        "joint_sensitivity_summary.csv": joint_summary,
        "paired_dimension_units.csv": paired_units,
        "paired_dimension_tests.csv": paired_tests,
        "hierarchical_bootstrap_distribution.csv": bootstrap,
        "hierarchical_bootstrap_summary.csv": bootstrap_summary,
        "d6_tail_sensitivity.csv": d6_sensitivity(traces),
        "d7_budget_sensitivity.csv": d7_sensitivity(traces),
        "M06_source_data.csv": m06,
        "S06_source_data.csv": s06,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    v1._atomic_json(DESTINATION / "input_audit.json", audit)
    for name, rows in files.items():
        v1._atomic_csv(DESTINATION / name, rows)
    output_hashes = {name: sha256_file(DESTINATION / name) for name in sorted(files)}
    output_hashes["input_audit.json"] = sha256_file(DESTINATION / "input_audit.json")
    manifest = {
        "schema_version": "training_curve_correction_v6",
        "state": "ready_for_corrected_figures",
        "passed": True,
        "plots_created": False,
        "protocol_hash": protocol["protocol_hash"],
        "parent_v1_manifest_hash": protocol["parent_v1_manifest_hash"],
        "formal_results_modified": False,
        "formal_results_rows": 21648,
        "formal_routes": 21648,
        "training_file_count": 35,
        "training_episode_count": 105000,
        "common_interaction_window": [COMMON_INTERACTION_START, COMMON_INTERACTION_END],
        "default_tail_fraction": DEFAULT_TAIL_FRACTION,
        "default_auc_budget_fraction": DEFAULT_AUC_BUDGET_FRACTION,
        "joint_sensitivity_rows": len(joint),
        "output_hashes": output_hashes,
    }
    manifest["manifest_hash"] = v1.canonical_hash(manifest)
    v1._atomic_json(DESTINATION / "analysis_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("audit-inputs", "all"), default="all")
    args = parser.parse_args(argv)
    protocol = validate_protocol()
    if args.command == "audit-inputs":
        audit, _ = input_audit(protocol)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seven-dimension training-aware manuscript analysis; creates no plots."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import manuscript_multiobjective_v1 as v1


ROOT = Path(__file__).resolve().parent
V1_DESTINATION = v1.DESTINATION
V1_MANIFEST = V1_DESTINATION / "analysis_manifest.json"
V1_DIMENSIONS = V1_DESTINATION / "dimension_scores.csv"
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "manuscript_training_aware_v2_protocol.json"
)
DESTINATION = v1.OUTPUT / "analysis/manuscript_training_aware_v2"
TRAINING_TRACE_ARCHIVE = v1.OUTPUT / "analysis/training_trace_inputs_v2"
CORE_MODELS = ("full", "traditional_ppo", "a2c_pointer")
TAIL_START_EPISODE = 2500.0
CONVERGENCE_THRESHOLD = 0.90
CONVERGENCE_WINDOW_UPDATES = 5
INTERACTION_BUDGET = 50000.0
TRAINING_DIMENSION_WEIGHTS = {
    "D6": {"seed_consistency": 0.60, "temporal_consistency": 0.40},
    "D7": {"learning_curve_auc": 0.70, "threshold_efficiency": 0.30},
}
# 训练指标合计22%，仍让任务、安全和资源表现占主要权重。
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


def training_path(model: str, seed: int) -> Path:
    if model == "traditional_ppo":
        return (
            ROOT
            / "paper_runs/multimap_v3_2/formal_training/"
            f"formal_traditional_ppo_seed{seed}_3000ep/training_metrics.jsonl"
        )
    # D6/D7冻结分析曾读取第一轮目录中的这10条曲线；清理时仅迁移其原始字节，
    # 不改数值、不换成多地图训练曲线，以保持既有分析和协议哈希可复算。
    return TRAINING_TRACE_ARCHIVE / f"formal_{model}_seed{seed}_3000ep/metrics.jsonl"


def training_source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for model in CORE_MODELS:
        for seed in range(42, 47):
            path = training_path(model, seed)
            if model == "traditional_ppo":
                logical_path = str(path.relative_to(ROOT)).replace("\\", "/")
            else:
                # 保留冻结协议中的逻辑身份；实际文件已迁入只读分析输入档案。
                logical_path = f"paper_runs/training/formal_{model}_seed{seed}_3000ep/metrics.jsonl"
            output[logical_path] = v1.sha256_file(path)
    return output


def enumerate_weight_grid(step: float = 0.05) -> list[dict[str, float]]:
    dimensions = tuple(WEIGHT_RANGES)
    unit_total = int(round(1.0 / step))
    bounds = {
        name: (
            int(round(WEIGHT_RANGES[name][0] / step)),
            int(round(WEIGHT_RANGES[name][1] / step)),
        )
        for name in dimensions
    }
    output = []
    for values in itertools.product(
        *(range(bounds[name][0], bounds[name][1] + 1) for name in dimensions)
    ):
        if sum(values) == unit_total:
            output.append(
                {name: round(value * step, 10) for name, value in zip(dimensions, values)}
            )
    if not output:
        raise RuntimeError("seven-dimension weight grid is empty")
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_trace(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {
        "episodes_seen", "environment_interactions", "mean_weighted_coverage"
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"incompatible training trace: {path}")
    if float(rows[-1]["episodes_seen"]) != 3000.0:
        raise RuntimeError(f"incomplete training trace: {path}")
    return rows


def _seed_training_metrics(model: str, seed: int) -> dict[str, Any]:
    rows = _load_trace(training_path(model, seed))
    episodes = np.asarray([float(row["episodes_seen"]) for row in rows])
    interactions = np.asarray(
        [float(row["environment_interactions"]) for row in rows]
    )
    performance = np.asarray(
        [v1.clip01(float(row["mean_weighted_coverage"])) for row in rows]
    )
    # 固定起点、固定阈值和固定窗口，避免按模型结果选择收敛定义。
    auc = float(
        np.trapz(
            np.concatenate(([performance[0]], performance)),
            np.concatenate(([0.0], episodes)),
        )
        / 3000.0
    )
    window = np.ones(CONVERGENCE_WINDOW_UPDATES) / CONVERGENCE_WINDOW_UPDATES
    rolling = np.convolve(performance, window, mode="valid")
    hits = np.flatnonzero(rolling >= CONVERGENCE_THRESHOLD)
    if len(hits):
        index = int(hits[0] + CONVERGENCE_WINDOW_UPDATES - 1)
        convergence_episode = float(episodes[index])
        convergence_interactions = float(interactions[index])
        threshold_efficiency = v1.clip01(
            1.0 - convergence_interactions / INTERACTION_BUDGET
        )
    else:
        convergence_episode = float("nan")
        convergence_interactions = float("nan")
        threshold_efficiency = 0.0
    tail = performance[episodes > TAIL_START_EPISODE]
    if len(tail) < 2:
        raise RuntimeError(f"insufficient tail updates: {model}/{seed}")
    return {
        "model": model,
        "training_seed": seed,
        "update_count": len(rows),
        "final_environment_interactions": int(interactions[-1]),
        "tail_update_count": len(tail),
        "tail_mean_weighted_coverage": float(np.mean(tail)),
        "tail_temporal_sd": float(np.std(tail, ddof=1)),
        "learning_curve_auc": v1.clip01(auc),
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "convergence_window_updates": CONVERGENCE_WINDOW_UPDATES,
        "convergence_episode": convergence_episode,
        "convergence_environment_interactions": convergence_interactions,
        "threshold_efficiency": threshold_efficiency,
    }


def training_dimensions() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows = [
        _seed_training_metrics(model, seed)
        for model in CORE_MODELS
        for seed in range(42, 47)
    ]
    model_rows = []
    for model in CORE_MODELS:
        selected = [row for row in seed_rows if row["model"] == model]
        tail_means = [float(row["tail_mean_weighted_coverage"]) for row in selected]
        seed_consistency = v1.clip01(1.0 - float(np.std(tail_means, ddof=1)))
        temporal_consistency = v1.clip01(
            1.0 - v1.finite_mean(float(row["tail_temporal_sd"]) for row in selected)
        )
        auc = v1.finite_mean(float(row["learning_curve_auc"]) for row in selected)
        threshold = v1.finite_mean(
            float(row["threshold_efficiency"]) for row in selected
        )
        model_rows.append(
            {
                "model": model,
                "seed_count": len(selected),
                "seed_consistency": seed_consistency,
                "temporal_consistency": temporal_consistency,
                "D6_training_stability": v1.weighted_arithmetic(
                    {
                        "seed_consistency": seed_consistency,
                        "temporal_consistency": temporal_consistency,
                    },
                    TRAINING_DIMENSION_WEIGHTS["D6"],
                ),
                "mean_learning_curve_auc": auc,
                "mean_threshold_efficiency": threshold,
                "D7_sample_efficiency": v1.weighted_arithmetic(
                    {
                        "learning_curve_auc": auc,
                        "threshold_efficiency": threshold,
                    },
                    TRAINING_DIMENSION_WEIGHTS["D7"],
                ),
            }
        )
    return seed_rows, model_rows


def seven_dimension_rows(
    training_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    nominal = {
        row["model"]: row
        for row in _read_csv(V1_DIMENSIONS)
        if row["scope"] == "core_learning_complete"
    }
    training = {str(row["model"]): row for row in training_rows}
    output = []
    for model in CORE_MODELS:
        if model not in nominal or model not in training:
            raise RuntimeError(f"missing seven-dimension input for {model}")
        output.append(
            {
                "scope": "core_learning_training_aware",
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
            values = {f"D{index}": float(row[f"D{index}"]) for index in range(1, 8)}
            for method, function in (
                ("geometric", v1.weighted_geometric),
                ("arithmetic", v1.weighted_arithmetic),
            ):
                scenario_rows.append(
                    {
                        "scope": row["scope"], "model": row["model"],
                        "scenario": scenario, "aggregation": method,
                        "score": function(values, weights),
                        **{f"weight_D{index}": weights[f"D{index}"] for index in range(1, 8)},
                    }
                )
    grid_rows = []
    for grid_id, weights in enumerate(enumerate_weight_grid()):
        for row in dimensions:
            values = {f"D{index}": float(row[f"D{index}"]) for index in range(1, 8)}
            for method, function in (
                ("geometric", v1.weighted_geometric),
                ("arithmetic", v1.weighted_arithmetic),
            ):
                grid_rows.append(
                    {
                        "scope": row["scope"], "model": row["model"],
                        "grid_id": grid_id, "aggregation": method,
                        "score": function(values, weights),
                        **{f"weight_D{index}": weights[f"D{index}"] for index in range(1, 8)},
                    }
                )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in grid_rows:
        grouped[(str(row["aggregation"]), int(row["grid_id"]))].append(row)
    records: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"rank": [], "first": [], "score": []}
    )
    for (method, _), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: (-float(row["score"]), str(row["model"])))
        best = float(ordered[0]["score"])
        for rank, row in enumerate(ordered, 1):
            row["rank"] = rank
            row["is_first"] = float(abs(float(row["score"]) - best) <= 1e-12)
            record = records[(method, str(row["model"]))]
            record["rank"].append(float(rank))
            record["first"].append(float(row["is_first"]))
            record["score"].append(float(row["score"]))
    summary = []
    for (method, model), record in sorted(records.items()):
        summary.append(
            {
                "scope": "core_learning_training_aware",
                "aggregation": method,
                "model": model,
                "grid_count": len(record["score"]),
                "first_place_share": v1.finite_mean(record["first"]),
                "mean_rank": v1.finite_mean(record["rank"]),
                "minimum_score": min(record["score"]),
                "maximum_score": max(record["score"]),
                "mean_score": v1.finite_mean(record["score"]),
            }
        )
    return scenario_rows, grid_rows, summary


def _gap_audit(
    scenario_rows: Sequence[Mapping[str, Any]],
    grid_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def gaps(rows: Sequence[Mapping[str, Any]]) -> list[float]:
        grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for row in rows:
            identity = (
                str(row["aggregation"]),
                str(row.get("scenario", row.get("grid_id"))),
            )
            grouped[identity][str(row["model"])] = float(row["score"])
        return [
            values["full"] - values["a2c_pointer"]
            for values in grouped.values()
            if {"full", "a2c_pointer"}.issubset(values)
        ]
    scenario_gaps = gaps(scenario_rows)
    grid_gaps = gaps(grid_rows)
    return {
        "requested_guaranteed_margin": 0.4,
        "guarantee_not_enforced": True,
        "reason": "weights are fixed independently of desired model ordering",
        "scenario_gap_min": min(scenario_gaps),
        "scenario_gap_max": max(scenario_gaps),
        "sensitivity_grid_gap_min": min(grid_gaps),
        "sensitivity_grid_gap_max": max(grid_gaps),
    }


def _validate_inputs() -> dict[str, Any]:
    v1_manifest = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    if v1_manifest["state"] != "ready_for_plot_plan" or v1_manifest["plots_created"]:
        raise RuntimeError("v1 parent analysis is not frozen pre-plot")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    actual_protocol_hash = v1.canonical_hash(
        {key: value for key, value in protocol.items() if key != "protocol_hash"}
    )
    if protocol["protocol_hash"] != actual_protocol_hash:
        raise RuntimeError("v2 protocol hash drift")
    if protocol["implementation_sha256"] != v1.sha256_file(Path(__file__)):
        raise RuntimeError("v2 implementation hash drift")
    if protocol["training_source_hashes"] != training_source_hashes():
        raise RuntimeError("training trace hash drift")
    if protocol["parent_v1_manifest_hash"] != v1_manifest["manifest_hash"]:
        raise RuntimeError("v1 parent manifest drift")
    return protocol


def run() -> dict[str, Any]:
    protocol = _validate_inputs()
    seed_rows, training_rows = training_dimensions()
    dimensions = seven_dimension_rows(training_rows)
    scenarios, grid, summary = score_rows(dimensions)
    gap_audit = _gap_audit(scenarios, grid)
    files = {
        "training_seed_metrics.csv": seed_rows,
        "training_dimension_scores.csv": training_rows,
        "seven_dimension_scores.csv": dimensions,
        "scenario_scores.csv": scenarios,
        "weight_sensitivity_grid.csv": grid,
        "weight_sensitivity_summary.csv": summary,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.svg", "*.pdf"):
        if list(DESTINATION.rglob(pattern)):
            raise RuntimeError("plot files are forbidden before plot planning")
    for name, rows in files.items():
        v1._atomic_csv(DESTINATION / name, rows)
    v1._atomic_json(DESTINATION / "pairwise_gap_audit.json", gap_audit)
    hashes = {
        name: v1.sha256_file(DESTINATION / name)
        for name in [*files, "pairwise_gap_audit.json"]
    }
    audit = {
        "schema_version": "manuscript_training_aware_v2",
        "passed": True,
        "analysis_role": "post_result_training_aware_manuscript_extension",
        "not_preregistered_confirmatory": True,
        "model_count": len(dimensions),
        "training_seed_row_count": len(seed_rows),
        "weight_grid_vector_count": len(enumerate_weight_grid()),
        "weight_grid_result_count": len(grid),
        "plots_created": False,
        "desired_winner_constraint_used": False,
        "protocol_hash": protocol["protocol_hash"],
        "parent_v1_manifest_hash": json.loads(
            V1_MANIFEST.read_text(encoding="utf-8")
        )["manifest_hash"],
        "output_hashes": hashes,
    }
    audit["audit_hash"] = v1.canonical_hash(audit)
    v1._atomic_json(DESTINATION / "training_aware_audit.json", audit)
    hashes["training_aware_audit.json"] = v1.sha256_file(
        DESTINATION / "training_aware_audit.json"
    )
    manifest = {
        "schema_version": "manuscript_training_aware_v2",
        "state": "ready_for_plot_plan",
        "plots_created": False,
        "plot_files": [],
        "protocol_hash": protocol["protocol_hash"],
        "parent_v1_manifest_hash": audit["parent_v1_manifest_hash"],
        "audit_hash": audit["audit_hash"],
        "output_hashes": hashes,
        "next_step": "freeze_revised_formal_plotting_plan_then_render_figures",
    }
    manifest["manifest_hash"] = v1.canonical_hash(manifest)
    v1._atomic_json(DESTINATION / "analysis_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

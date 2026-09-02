#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Close all statistical and sensitivity gates before formal plotting."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

import manuscript_multiobjective_v1 as v1
import manuscript_operational_band_v4 as v4
import manuscript_training_aware_v2 as v2
import manuscript_training_priority_v3 as v3


ROOT = Path(__file__).resolve().parent
V1 = v1.DESTINATION
V2 = v2.DESTINATION
V3 = v3.DESTINATION
V4 = v4.DESTINATION
SOURCE_MANIFESTS = {
    "v1": V1 / "analysis_manifest.json",
    "v2": V2 / "analysis_manifest.json",
    "v3": V3 / "analysis_manifest.json",
    "v4": V4 / "analysis_manifest.json",
}
NOMINAL_MAPS = V1 / "nominal_map_dimensions.csv"
ROBUSTNESS_CONDITIONS = V1 / "robustness_condition_dimensions.csv"
TRAINING_SEEDS = V2 / "training_seed_metrics.csv"
SEVEN_DIMENSIONS = V2 / "seven_dimension_scores.csv"
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "manuscript_preplot_closure_v5_protocol.json"
)
DESTINATION = v1.OUTPUT / "analysis/manuscript_preplot_closure_v5"
CORE_PAIR = ("full", "a2c_pointer")
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260731
CI_LEVEL = 0.95


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _source_manifest_hashes() -> dict[str, str]:
    return {
        name: json.loads(path.read_text(encoding="utf-8"))["manifest_hash"]
        for name, path in SOURCE_MANIFESTS.items()
    }


def _validate() -> dict[str, Any]:
    for name, path in SOURCE_MANIFESTS.items():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["state"] != "ready_for_plot_plan" or value["plots_created"]:
            raise RuntimeError(f"parent {name} is not frozen pre-plot")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    actual = v1.canonical_hash(
        {key: value for key, value in protocol.items() if key != "protocol_hash"}
    )
    if protocol["protocol_hash"] != actual:
        raise RuntimeError("v5 protocol hash drift")
    if protocol["implementation_sha256"] != v1.sha256_file(Path(__file__)):
        raise RuntimeError("v5 implementation hash drift")
    if protocol["source_manifest_hashes"] != _source_manifest_hashes():
        raise RuntimeError("v1-v4 source manifest drift")
    return protocol


def joint_sensitivity_rows() -> list[dict[str, Any]]:
    dimensions = _csv(SEVEN_DIMENSIONS)
    output = []
    for floor in v4.FLOOR_SENSITIVITY:
        for grid_id, weights in enumerate(v2.enumerate_weight_grid()):
            for row in dimensions:
                values = v4.transformed_dimensions(row, floor)
                for method, function in (
                    ("geometric", v1.weighted_geometric),
                    ("arithmetic", v1.weighted_arithmetic),
                ):
                    score = 100.0 * function(values, weights)
                    output.append(
                        {
                            "operational_floor": floor,
                            "grid_id": grid_id,
                            "model": row["model"],
                            "aggregation": method,
                            "score_0_to_100": score,
                            **{f"weight_D{index}": weights[f"D{index}"] for index in range(1, 8)},
                        }
                    )
    grouped: dict[tuple[float, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        grouped[
            (
                float(row["operational_floor"]),
                int(row["grid_id"]),
                str(row["aggregation"]),
            )
        ].append(row)
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: (-float(row["score_0_to_100"]), str(row["model"])))
        best = float(ordered[0]["score_0_to_100"])
        for rank, row in enumerate(ordered, 1):
            row["rank"] = rank
            row["is_first"] = float(abs(float(row["score_0_to_100"]) - best) <= 1e-12)
    return output


def joint_sensitivity_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                float(row["operational_floor"]),
                str(row["aggregation"]),
                str(row["model"]),
            )
        ].append(row)
    output = []
    for (floor, method, model), values in sorted(grouped.items()):
        scores = [float(row["score_0_to_100"]) for row in values]
        output.append(
            {
                "operational_floor": floor,
                "aggregation": method,
                "model": model,
                "grid_count": len(values),
                "first_place_share": v1.finite_mean(float(row["is_first"]) for row in values),
                "mean_rank": v1.finite_mean(float(row["rank"]) for row in values),
                "minimum_score": min(scores),
                "maximum_score": max(scores),
                "mean_score": v1.finite_mean(scores),
            }
        )
    return output


def _robustness_map_units() -> list[dict[str, Any]]:
    rows = [
        row
        for row in _csv(ROBUSTNESS_CONDITIONS)
        if row["model"] in CORE_PAIR
    ]
    global_consistency = {}
    for model in CORE_PAIR:
        selected = [row for row in rows if row["model"] == model]
        by_map: dict[str, list[float]] = defaultdict(list)
        for row in selected:
            by_map[row["map_id"]].append(float(row["perturbed_safe_weighted_coverage"]))
        global_consistency[model] = v1.clip01(
            1.0
            - float(
                np.std(
                    [v1.finite_mean(values) for values in by_map.values()],
                    ddof=1,
                )
            )
        )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["map_id"])].append(row)
    output = []
    for (model, map_id), values in sorted(grouped.items()):
        if len(values) != 6:
            raise RuntimeError(f"robustness condition count mismatch: {model}/{map_id}")
        retention = [float(row["retention"]) for row in values]
        safe = [float(row["perturbed_safe_rate"]) for row in values]
        components = {
            "mean_retention": v1.finite_mean(retention),
            "worst_retention": min(retention),
            "perturbed_safe_rate": v1.finite_mean(safe),
            "map_consistency": global_consistency[model],
        }
        output.append(
            {
                "dimension": "D4",
                "model": model,
                "unit_id": map_id,
                "unit_type": "real_map",
                "value": v1.weighted_arithmetic(components, v1.INTERNAL_WEIGHTS["D4"]),
                **components,
            }
        )
    return output


def _training_units() -> list[dict[str, Any]]:
    rows = [row for row in _csv(TRAINING_SEEDS) if row["model"] in CORE_PAIR]
    seed_consistency = {}
    for model in CORE_PAIR:
        means = [
            float(row["tail_mean_weighted_coverage"])
            for row in rows
            if row["model"] == model
        ]
        seed_consistency[model] = v1.clip01(1.0 - float(np.std(means, ddof=1)))
    output = []
    for row in rows:
        model = row["model"]
        d6 = v1.weighted_arithmetic(
            {
                "seed_consistency": seed_consistency[model],
                "temporal_consistency": v1.clip01(1.0 - float(row["tail_temporal_sd"])),
            },
            v2.TRAINING_DIMENSION_WEIGHTS["D6"],
        )
        d7 = v1.weighted_arithmetic(
            {
                "learning_curve_auc": float(row["learning_curve_auc"]),
                "threshold_efficiency": float(row["threshold_efficiency"]),
            },
            v2.TRAINING_DIMENSION_WEIGHTS["D7"],
        )
        for dimension, value in (("D6", d6), ("D7", d7)):
            output.append(
                {
                    "dimension": dimension,
                    "model": model,
                    "unit_id": row["training_seed"],
                    "unit_type": "paired_training_seed",
                    "value": value,
                    "seed_consistency": seed_consistency[model],
                    "tail_temporal_sd": float(row["tail_temporal_sd"]),
                    "learning_curve_auc": float(row["learning_curve_auc"]),
                    "threshold_efficiency": float(row["threshold_efficiency"]),
                }
            )
    return output


def paired_dimension_units() -> list[dict[str, Any]]:
    return _robustness_map_units() + _training_units()


def rank_biserial_paired(differences: Sequence[float]) -> float:
    values = np.asarray([float(value) for value in differences if float(value) != 0.0])
    if len(values) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(values))
    positive = float(np.sum(ranks[values > 0]))
    negative = float(np.sum(ranks[values < 0]))
    return (positive - negative) / (positive + negative)


def hodges_lehmann_paired(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=float)
    walsh = [
        (values[left] + values[right]) / 2.0
        for left in range(len(values))
        for right in range(left, len(values))
    ]
    return float(np.median(walsh))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_dimension_tests(
    units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw = []
    for dimension in ("D4", "D6", "D7"):
        selected = [row for row in units if row["dimension"] == dimension]
        by_model = {
            model: {str(row["unit_id"]): float(row["value"]) for row in selected if row["model"] == model}
            for model in CORE_PAIR
        }
        if set(by_model["full"]) != set(by_model["a2c_pointer"]):
            raise RuntimeError(f"paired unit mismatch for {dimension}")
        ids = sorted(by_model["full"])
        differences = [
            by_model["full"][unit] - by_model["a2c_pointer"][unit]
            for unit in ids
        ]
        result = stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        raw.append(
            {
                "comparison_family": "training_and_robustness_dimensions",
                "dimension": dimension,
                "reference": "full",
                "comparator": "a2c_pointer",
                "independent_unit": "real_map" if dimension == "D4" else "training_seed",
                "unit_count": len(ids),
                "mean_difference": v1.finite_mean(differences),
                "median_difference": float(np.median(differences)),
                "hodges_lehmann": hodges_lehmann_paired(differences),
                "rank_biserial": rank_biserial_paired(differences),
                "direction_consistency": v1.finite_mean(float(value > 0) for value in differences),
                "wilcoxon_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    adjusted = holm_adjust([float(row["p_value"]) for row in raw])
    for row, value in zip(raw, adjusted):
        row["p_holm"] = value
        row["significant_holm_0_05"] = value < 0.05
    return raw


def _nominal_lookup() -> tuple[dict[str, dict[str, dict[str, dict[str, float]]]], dict[str, list[str]]]:
    lookup: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    maps: dict[str, set[str]] = defaultdict(set)
    for row in _csv(NOMINAL_MAPS):
        if row["model"] not in CORE_PAIR:
            continue
        domain = row["domain"]
        model = row["model"]
        map_id = row["map_id"]
        maps[domain].add(map_id)
        lookup[domain][model][map_id] = {
            "D1": float(row["D1_mission_effectiveness"]),
            "D2": float(row["D2_resource_efficiency"]),
            "D3": float(row["D3_safety_reliability"]),
            "D5": float(row["D5_online_deployability"]),
        }
    frozen_maps = {domain: sorted(values) for domain, values in maps.items()}
    if len(frozen_maps.get("synthetic", [])) != 24 or len(frozen_maps.get("real", [])) != 8:
        raise RuntimeError("nominal independent map grid mismatch")
    return lookup, frozen_maps


def _robustness_lookup() -> tuple[dict[str, dict[str, list[dict[str, str]]]], list[str]]:
    lookup: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in _csv(ROBUSTNESS_CONDITIONS):
        if row["model"] in CORE_PAIR:
            lookup[row["model"]][row["map_id"]].append(row)
    maps = sorted(lookup["full"])
    if len(maps) != 8 or set(maps) != set(lookup["a2c_pointer"]):
        raise RuntimeError("robustness independent map grid mismatch")
    return lookup, maps


def _training_lookup() -> tuple[dict[str, dict[int, dict[str, str]]], list[int]]:
    lookup: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in _csv(TRAINING_SEEDS):
        if row["model"] in CORE_PAIR:
            lookup[row["model"]][int(row["training_seed"])] = row
    seeds = sorted(lookup["full"])
    if seeds != [42, 43, 44, 45, 46] or set(seeds) != set(lookup["a2c_pointer"]):
        raise RuntimeError("paired training seed grid mismatch")
    return lookup, seeds


def _bootstrap_d4(
    model: str,
    sampled_maps: Sequence[str],
    lookup: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]],
) -> float:
    selected = [row for map_id in sampled_maps for row in lookup[model][map_id]]
    by_condition: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_map_occurrence = []
    offset = 0
    for map_id in sampled_maps:
        values = lookup[model][map_id]
        by_map_occurrence.append(
            v1.finite_mean(float(row["perturbed_safe_weighted_coverage"]) for row in values)
        )
        for row in values:
            by_condition[(row["family"], row["condition"])].append(float(row["retention"]))
        offset += 1
    components = {
        "mean_retention": v1.finite_mean(float(row["retention"]) for row in selected),
        "worst_retention": min(v1.finite_mean(values) for values in by_condition.values()),
        "perturbed_safe_rate": v1.finite_mean(float(row["perturbed_safe_rate"]) for row in selected),
        "map_consistency": v1.clip01(1.0 - float(np.std(by_map_occurrence, ddof=1))),
    }
    return v1.weighted_arithmetic(components, v1.INTERNAL_WEIGHTS["D4"])


def _bootstrap_training(
    model: str,
    sampled_seeds: Sequence[int],
    lookup: Mapping[str, Mapping[int, Mapping[str, str]]],
) -> tuple[float, float]:
    rows = [lookup[model][seed] for seed in sampled_seeds]
    seed_consistency = v1.clip01(
        1.0
        - float(
            np.std(
                [float(row["tail_mean_weighted_coverage"]) for row in rows],
                ddof=1,
            )
        )
    )
    temporal_consistency = v1.clip01(
        1.0 - v1.finite_mean(float(row["tail_temporal_sd"]) for row in rows)
    )
    d6 = v1.weighted_arithmetic(
        {
            "seed_consistency": seed_consistency,
            "temporal_consistency": temporal_consistency,
        },
        v2.TRAINING_DIMENSION_WEIGHTS["D6"],
    )
    d7 = v1.weighted_arithmetic(
        {
            "learning_curve_auc": v1.finite_mean(float(row["learning_curve_auc"]) for row in rows),
            "threshold_efficiency": v1.finite_mean(float(row["threshold_efficiency"]) for row in rows),
        },
        v2.TRAINING_DIMENSION_WEIGHTS["D7"],
    )
    return d6, d7


def hierarchical_bootstrap() -> list[dict[str, Any]]:
    nominal, domain_maps = _nominal_lookup()
    robustness, robustness_maps = _robustness_lookup()
    training, seeds = _training_lookup()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_synthetic = rng.choice(domain_maps["synthetic"], size=24, replace=True).tolist()
        sampled_real = rng.choice(domain_maps["real"], size=8, replace=True).tolist()
        sampled_robustness = rng.choice(robustness_maps, size=8, replace=True).tolist()
        sampled_seeds = rng.choice(seeds, size=5, replace=True).astype(int).tolist()
        scores = {}
        dimensions_by_model = {}
        for model in CORE_PAIR:
            dimensions = {}
            for dimension in ("D1", "D2", "D3", "D5"):
                synthetic_mean = v1.finite_mean(
                    nominal["synthetic"][model][map_id][dimension]
                    for map_id in sampled_synthetic
                )
                real_mean = v1.finite_mean(
                    nominal["real"][model][map_id][dimension]
                    for map_id in sampled_real
                )
                dimensions[dimension] = (synthetic_mean + real_mean) / 2.0
            dimensions["D4"] = _bootstrap_d4(model, sampled_robustness, robustness)
            dimensions["D6"], dimensions["D7"] = _bootstrap_training(
                model, sampled_seeds, training
            )
            transformed = {
                name: (
                    v4.operational_rescale(value, v4.SELECTED_OPERATIONAL_FLOOR)
                    if name in v4.RESCALED_DIMENSIONS
                    else value
                )
                for name, value in dimensions.items()
            }
            scores[model] = 100.0 * v1.weighted_arithmetic(
                transformed, v3.PRIORITY_WEIGHTS
            )
            dimensions_by_model[model] = dimensions
        output.append(
            {
                "bootstrap_replicate": replicate,
                "full_score": scores["full"],
                "a2c_pointer_score": scores["a2c_pointer"],
                "full_minus_a2c_points": scores["full"] - scores["a2c_pointer"],
                "D4_difference": dimensions_by_model["full"]["D4"] - dimensions_by_model["a2c_pointer"]["D4"],
                "D6_difference": dimensions_by_model["full"]["D6"] - dimensions_by_model["a2c_pointer"]["D6"],
                "D7_difference": dimensions_by_model["full"]["D7"] - dimensions_by_model["a2c_pointer"]["D7"],
            }
        )
    return output


def bootstrap_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    alpha = (1.0 - CI_LEVEL) / 2.0
    output = []
    for metric in (
        "full_score",
        "a2c_pointer_score",
        "full_minus_a2c_points",
        "D4_difference",
        "D6_difference",
        "D7_difference",
    ):
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        output.append(
            {
                "metric": metric,
                "bootstrap_replicates": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "ci_level": CI_LEVEL,
                "ci_low": float(np.quantile(values, alpha)),
                "ci_high": float(np.quantile(values, 1.0 - alpha)),
                "probability_positive": float(np.mean(values > 0.0)),
            }
        )
    return output


def run() -> dict[str, Any]:
    protocol = _validate()
    joint = joint_sensitivity_rows()
    joint_summary = joint_sensitivity_summary(joint)
    units = paired_dimension_units()
    tests = paired_dimension_tests(units)
    bootstrap = hierarchical_bootstrap()
    bootstrap_stats = bootstrap_summary(bootstrap)
    files = {
        "joint_normalization_weight_sensitivity.csv": joint,
        "joint_sensitivity_summary.csv": joint_summary,
        "paired_dimension_units.csv": units,
        "paired_dimension_tests.csv": tests,
        "hierarchical_bootstrap_distribution.csv": bootstrap,
        "hierarchical_bootstrap_summary.csv": bootstrap_stats,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.svg", "*.pdf"):
        if list(DESTINATION.rglob(pattern)):
            raise RuntimeError("plot files are forbidden before plot planning")
    for name, rows in files.items():
        v1._atomic_csv(DESTINATION / name, rows)
    hashes = {name: v1.sha256_file(DESTINATION / name) for name in files}
    audit = {
        "schema_version": "manuscript_preplot_closure_v5",
        "passed": True,
        "joint_sensitivity_result_count": len(joint),
        "joint_sensitivity_summary_count": len(joint_summary),
        "paired_dimension_unit_count": len(units),
        "paired_test_count": len(tests),
        "bootstrap_result_count": len(bootstrap),
        "bootstrap_summary_count": len(bootstrap_stats),
        "source_manifest_hashes": _source_manifest_hashes(),
        "formal_result_row_count": 21648,
        "formal_result_sha256": v1.EXPECTED_RESULTS_SHA256,
        "plots_created": False,
        "protocol_hash": protocol["protocol_hash"],
        "output_hashes": hashes,
    }
    audit["audit_hash"] = v1.canonical_hash(audit)
    v1._atomic_json(DESTINATION / "preplot_closure_audit.json", audit)
    hashes["preplot_closure_audit.json"] = v1.sha256_file(
        DESTINATION / "preplot_closure_audit.json"
    )
    manifest = {
        "schema_version": "manuscript_preplot_closure_v5",
        "state": "ready_for_formal_plot_plan",
        "plots_created": False,
        "plot_files": [],
        "protocol_hash": protocol["protocol_hash"],
        "audit_hash": audit["audit_hash"],
        "source_manifest_hashes": audit["source_manifest_hashes"],
        "output_hashes": hashes,
        "next_step": "user_supplied_formal_plot_plan",
    }
    manifest["manifest_hash"] = v1.canonical_hash(manifest)
    v1._atomic_json(DESTINATION / "analysis_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

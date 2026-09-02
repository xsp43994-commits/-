#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Map-level preregistered statistics for v3.2.14; deliberately creates no plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from scipy import stats

from uav_inspection.experiments import paper_v3_2_experiments as v32
from uav_inspection.evaluation import v3_2_14_evaluation_smoke as smoke


ROOT = WORKSPACE_ROOT
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14"
FORMAL = OUTPUT / "formal_evaluation"
RESULTS = FORMAL / "results"
FINAL_RESULTS = RESULTS / "final_results.jsonl"
FINAL_AUDIT = RESULTS / "final_audit.json"
MATRIX = FORMAL / "evaluation_matrix.jsonl"
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/protocol.json"
)
ANALYSIS_PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "analysis_protocol.json"
)
ANALYSIS_ERRATUM = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/"
    "analysis_implementation_erratum.json"
)
DESTINATION = OUTPUT / "analysis/pre_plot_statistics"
SYNTHETIC_TASKS = (
    OUTPUT / "manifests/synthetic_test/records.jsonl"
)
REAL_TASKS = (
    FORMAL / "real_tasks_parallel/records.jsonl"
)


FAMILIES: Dict[str, Dict[str, Any]] = {
    "synthetic_main_algorithms": {
        "sources": ("synthetic_learning", "synthetic_main_baselines"),
        "algorithms": (
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
        "conditions": ("nominal",),
        "independent_maps": 24,
        "role": "confirmatory",
    },
    "synthetic_ablations": {
        "sources": ("synthetic_learning",),
        "algorithms": (
            "full",
            "no_priority_bias",
            "no_domain_randomization",
            "no_resource_shaping",
            "no_return_reserve",
        ),
        "conditions": ("nominal",),
        "independent_maps": 24,
        "role": "confirmatory",
    },
    "real_main_algorithms": {
        "sources": ("real_learning", "real_baselines"),
        "algorithms": (
            "full",
            "traditional_ppo",
            "a2c_pointer",
            "nearest_feasible",
            "priority_resource_greedy",
            "aco",
            "milp",
        ),
        "conditions": ("nominal",),
        "independent_maps": 8,
        "role": "confirmatory",
    },
    "real_ablations": {
        "sources": ("real_learning",),
        "algorithms": (
            "full",
            "no_priority_bias",
            "no_domain_randomization",
            "no_resource_shaping",
            "no_return_reserve",
        ),
        "conditions": ("nominal",),
        "independent_maps": 8,
        "role": "confirmatory",
    },
    "known_domain_shift": {
        "sources": ("known_domain_shift",),
        "algorithms": (
            "full",
            "traditional_ppo",
            "a2c_pointer",
            "no_domain_randomization",
            "priority_resource_greedy",
        ),
        "conditions": ("power_model", "wind"),
        "independent_maps": 8,
        "role": "confirmatory",
    },
    "hidden_model_perception_mismatch": {
        "sources": ("hidden_model_perception_mismatch",),
        "algorithms": (
            "full",
            "traditional_ppo",
            "a2c_pointer",
            "no_domain_randomization",
            "no_return_reserve",
            "priority_resource_greedy",
        ),
        "conditions": (
            "dem_error",
            "localization",
            "power_model",
            "wind",
        ),
        "independent_maps": 8,
        "role": "confirmatory",
    },
}


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["family"]),
        str(row["task_id"]),
        str(row["model"]),
        (
            int(row["training_seed"])
            if row.get("training_seed") is not None
            else None
        ),
        (
            int(row["planner_seed"])
            if row.get("planner_seed") is not None
            else None
        ),
        str(row["condition"]),
    )


def _atomic_csv(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if value is None
                        or (
                            isinstance(value, (float, np.floating))
                            and not math.isfinite(float(value))
                        )
                        else value
                    )
                    for key, value in row.items()
                }
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.ones(count, dtype=np.float64)
    running = 0.0
    for offset, raw_index in enumerate(order):
        candidate = min(
            1.0, (count - offset) * float(p_values[int(raw_index)])
        )
        running = max(running, candidate)
        adjusted[int(raw_index)] = running
    return adjusted.tolist()


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[np.abs(differences) > 1e-12]
    if not nonzero.size:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / max(positive + negative, 1e-12)


def hodges_lehmann(differences: np.ndarray) -> float:
    walsh = (
        differences[:, None] + differences[None, :]
    ) / 2.0
    return float(
        np.median(walsh[np.triu_indices(len(differences))])
    )


def interquartile_mean(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if not len(ordered):
        return float("nan")
    lower = int(math.floor(0.25 * len(ordered)))
    upper = int(math.ceil(0.75 * len(ordered)))
    return float(np.mean(ordered[lower:upper]))


def _route_directories() -> Iterable[Path]:
    for family in ("synthetic_learning", "real_learning"):
        shards = RESULTS / family / "shards"
        for shard in sorted(path for path in shards.iterdir() if path.is_dir()):
            yield shard / "routes"
    for family in (
        "synthetic_main_baselines",
        "synthetic_supplementary",
        "real_baselines",
        "known_domain_shift",
        "hidden_model_perception_mismatch",
    ):
        jobs = RESULTS / family / "jobs"
        for job in sorted(path for path in jobs.iterdir() if path.is_dir()):
            yield job / "routes"


def _visit_order(route: Mapping[str, Any]) -> list[int]:
    candidates = (
        route.get("visit_order_locked"),
        route.get("detail", {}).get("visit_order"),
        route.get("detail", {}).get("metrics", {}).get("visited_order"),
        route.get("result", {}).get("visit_order"),
        route.get("observation_plan", {}).get("visit_order"),
        route.get("observation_plan", {})
        .get("metrics", {})
        .get("visited_order"),
    )
    for candidate in candidates:
        if candidate is not None:
            return [int(value) for value in candidate]
    raise RuntimeError("route has no visit order")


def _observation_metrics(
    route: Mapping[str, Any],
) -> Mapping[str, Any]:
    for candidate in (
        route.get("detail", {}).get("metrics"),
        route.get("result", {}).get("metrics"),
        route.get("observation_plan", {}).get("metrics"),
    ):
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _constraint_violation_summary(
    observation_metrics: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], int]:
    raw = observation_metrics.get("constraint_violations", ())
    if isinstance(raw, (list, tuple)):
        violations = list(raw)
        inferred_count = len(violations)
    elif isinstance(raw, (int, np.integer)) and not isinstance(raw, bool):
        # 传统规划器只保存违规次数；详细约束名称不可据此反推。
        violations = []
        inferred_count = int(raw)
    else:
        raise TypeError(
            "constraint_violations must be a list/tuple or integer count"
        )
    count = int(
        observation_metrics.get(
            "constraint_violation_count", inferred_count
        )
    )
    return violations, count


def _route_derived(
    task_rows: Mapping[str, Mapping[str, Any]],
) -> Dict[tuple[Any, ...], Dict[str, Any]]:
    derived: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for directory in _route_directories():
        if not directory.is_dir():
            raise RuntimeError(f"missing route directory {directory}")
        for path in directory.glob("*.json"):
            route = json.loads(path.read_text(encoding="utf-8"))
            matrix_row = route["matrix_row"]
            key = _key(matrix_row)
            if key in derived:
                raise RuntimeError(f"duplicate route analysis key {key}")
            task = task_rows[str(matrix_row["task_id"])]
            priorities = np.asarray(task["priorities"], dtype=np.float64)
            visited = {
                index
                for index in _visit_order(route)
                if 0 <= index < len(priorities)
            }
            priority_coverage: Dict[str, float] = {}
            for label, value in (("low", 1.0), ("medium", 2.0), ("high", 3.0)):
                indices = np.flatnonzero(priorities == value)
                priority_coverage[f"{label}_priority_coverage"] = (
                    float(
                        sum(int(index) in visited for index in indices)
                        / len(indices)
                    )
                    if len(indices)
                    else float("nan")
                )
            observation_metrics = _observation_metrics(route)
            violations, count = _constraint_violation_summary(
                observation_metrics
            )
            failed_constraints = sorted(
                {
                    str(name)
                    for violation in violations
                    for name in violation.get("failed_constraints", ())
                }
            )
            derived[key] = {
                **priority_coverage,
                "dangerous_action_proposal_count": count,
                "dangerous_action_proposal_rate": float(count > 0),
                "environment_interception_rate": float(count > 0),
                "failed_constraints": "|".join(failed_constraints),
            }
    if len(derived) != 21648:
        raise RuntimeError(
            f"route-derived row count {len(derived)} != 21648"
        )
    return derived


def _analysis_rows() -> list[Dict[str, Any]]:
    final_audit = json.loads(FINAL_AUDIT.read_text(encoding="utf-8"))
    if not (
        final_audit.get("passed")
        and int(final_audit.get("row_count", -1)) == 21648
        and int(final_audit.get("route_count", -1)) == 21648
    ):
        raise RuntimeError("final evaluation audit has not passed")
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    analysis_protocol = json.loads(
        ANALYSIS_PROTOCOL.read_text(encoding="utf-8")
    )
    expected_hash = analysis_protocol["analysis_protocol_hash"]
    actual_hash = smoke._canonical_hash(
        {
            key: value
            for key, value in analysis_protocol.items()
            if key != "analysis_protocol_hash"
        }
    )
    if expected_hash != actual_hash:
        raise RuntimeError("analysis protocol hash drift")
    current_implementation = v32._sha256_file(Path(__file__))
    implementation_valid = (
        analysis_protocol["implementation_sha256"]
        == current_implementation
    )
    if not implementation_valid and ANALYSIS_ERRATUM.is_file():
        erratum = json.loads(
            ANALYSIS_ERRATUM.read_text(encoding="utf-8")
        )
        implementation_valid = (
            erratum["erratum_hash"]
            == smoke._canonical_hash(
                {
                    key: value
                    for key, value in erratum.items()
                    if key != "erratum_hash"
                }
            )
            and erratum["parent_analysis_protocol_hash"]
            == expected_hash
            and erratum["original_implementation_sha256"]
            == analysis_protocol["implementation_sha256"]
            and erratum["corrected_implementation_sha256"]
            == current_implementation
            and erratum.get("statistical_rules_changed") is False
            and erratum.get("algorithm_scores_used_for_fix") is False
        )
    if (
        analysis_protocol["parent_protocol_hash"]
        != protocol["protocol_hash"]
        or analysis_protocol["matrix_sha256"]
        != final_audit["matrix_sha256"]
        or not implementation_valid
    ):
        raise RuntimeError("analysis implementation identity mismatch")

    tasks = {
        str(row["id"]): row
        for row in (
            v32._read_jsonl(SYNTHETIC_TASKS)
            + v32._read_jsonl(REAL_TASKS)
        )
    }
    routes = _route_derived(tasks)
    rows = v32._read_jsonl(FINAL_RESULTS)
    nominal: Dict[tuple[str, str, int], float] = {}
    for row in rows:
        if str(row["family"]) not in ("real_learning", "real_baselines"):
            continue
        seed = (
            row["training_seed"]
            if row.get("training_seed") is not None
            else row["planner_seed"]
        )
        nominal[
            (str(row["task_id"]), str(row["model"]), int(seed))
        ] = float(row["safe_weighted_coverage"])

    enhanced: list[Dict[str, Any]] = []
    for row in rows:
        key = _key(row)
        task = tasks[str(row["task_id"])]
        seed = (
            row["training_seed"]
            if row.get("training_seed") is not None
            else row["planner_seed"]
        )
        safe = bool(row["safe"])
        oracle_lower = float(row["oracle_lower"])
        oracle_upper = float(row["oracle_upper"])
        primary = float(row["safe_weighted_coverage"])
        nominal_value = nominal.get(
            (str(row["task_id"]), str(row["model"]), int(seed))
        )
        enhanced.append(
            {
                **row,
                **routes[key],
                "repeat_seed": int(seed),
                "safe_rate": float(safe),
                "return_rate": float(bool(row["returned"])),
                "violation_rate": float(
                    bool(row["energy_violation"])
                    or bool(row["distance_violation"])
                    or bool(row["time_violation"])
                    or bool(row["dynamics_violation"])
                ),
                "stranded_rate": float(
                    str(row["termination_reason"]) == "stranded"
                ),
                "safe_energy_utilization": (
                    float(row["energy_utilization"])
                    if safe
                    else float("nan")
                ),
                "safe_distance_utilization": (
                    float(row["distance_utilization"])
                    if safe
                    else float("nan")
                ),
                "safe_time_utilization": (
                    float(row["time_utilization"])
                    if safe
                    else float("nan")
                ),
                "oracle_attainment_lower": (
                    primary / oracle_upper
                    if oracle_upper > 0
                    else float("nan")
                ),
                "oracle_attainment_upper": (
                    min(1.0, primary / oracle_lower)
                    if oracle_lower > 0
                    else float("nan")
                ),
                "oracle_regret_lower": max(0.0, oracle_lower - primary),
                "oracle_regret_upper": max(0.0, oracle_upper - primary),
                "nominal_safe_weighted_coverage": nominal_value,
                "robustness_drop": (
                    float(nominal_value - primary)
                    if nominal_value is not None
                    else float("nan")
                ),
                "task_priority_layout": str(task["priority_layout"]),
            }
        )
    return enhanced


Nested = Dict[
    str,
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
]


def _nested(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    metric: str,
) -> Nested:
    algorithms = tuple(config["algorithms"])
    sources = set(config["sources"])
    conditions = set(config["conditions"])
    grouped: MutableMapping[
        tuple[str, str, str, str], list[float]
    ] = defaultdict(list)
    for row in rows:
        if (
            str(row["family"]) not in sources
            or str(row["model"]) not in algorithms
            or str(row["condition"]) not in conditions
        ):
            continue
        value = float(row[metric])
        if math.isfinite(value):
            grouped[
                (
                    str(row["model"]),
                    str(row["map_id"]),
                    str(row["task_id"]),
                    str(row["condition"]),
                )
            ].append(value)
    output: Nested = {algorithm: {} for algorithm in algorithms}
    for (algorithm, map_id, task_id, condition), values in grouped.items():
        output[algorithm].setdefault(map_id, {}).setdefault(
            task_id, {}
        )[condition] = np.asarray(values, dtype=np.float64)
    return output


def _validate_grid(
    nested: Nested, config: Mapping[str, Any]
) -> tuple[list[str], Dict[str, list[str]]]:
    algorithms = tuple(config["algorithms"])
    reference = nested["full"]
    maps = sorted(reference)
    tasks = {
        map_id: sorted(reference[map_id])
        for map_id in maps
    }
    conditions = set(config["conditions"])
    if len(maps) != int(config["independent_maps"]):
        raise RuntimeError("independent map count mismatch")
    for algorithm in algorithms:
        if set(nested[algorithm]) != set(maps):
            raise RuntimeError(f"{algorithm} map grid mismatch")
        for map_id in maps:
            if set(nested[algorithm][map_id]) != set(tasks[map_id]):
                raise RuntimeError(f"{algorithm}/{map_id} task grid mismatch")
            for task_id in tasks[map_id]:
                if (
                    set(nested[algorithm][map_id][task_id])
                    != conditions
                ):
                    raise RuntimeError(
                        f"{algorithm}/{task_id} condition grid mismatch"
                    )
    return maps, tasks


def _map_means(
    nested: Nested,
    config: Mapping[str, Any],
) -> tuple[list[str], Dict[str, np.ndarray]]:
    maps, tasks = _validate_grid(nested, config)
    conditions = tuple(config["conditions"])
    means: Dict[str, np.ndarray] = {}
    for algorithm in config["algorithms"]:
        values = []
        for map_id in maps:
            task_values = []
            for task_id in tasks[map_id]:
                task_values.append(
                    np.mean(
                        [
                            np.mean(
                                nested[algorithm][map_id][task_id][condition]
                            )
                            for condition in conditions
                        ]
                    )
                )
            values.append(float(np.mean(task_values)))
        means[str(algorithm)] = np.asarray(values, dtype=np.float64)
    return maps, means


def _bootstrap_interval(
    reference: Mapping[str, Dict[str, Dict[str, np.ndarray]]],
    comparator: Mapping[str, Dict[str, Dict[str, np.ndarray]]],
    maps: Sequence[str],
    tasks: Mapping[str, Sequence[str]],
    conditions: Sequence[str],
    samples: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    map_estimates = np.empty((samples, len(maps)), dtype=np.float64)
    for map_column, map_id in enumerate(maps):
        task_ids = list(tasks[map_id])
        inner = np.empty((samples, len(task_ids)), dtype=np.float64)
        for task_column, task_id in enumerate(task_ids):
            difference = np.zeros(samples, dtype=np.float64)
            for condition in conditions:
                ref = reference[map_id][task_id][condition]
                cmp = comparator[map_id][task_id][condition]
                ref_indices = rng.integers(
                    0, len(ref), size=(samples, len(ref))
                )
                cmp_indices = rng.integers(
                    0, len(cmp), size=(samples, len(cmp))
                )
                difference += (
                    ref[ref_indices].mean(axis=1)
                    - cmp[cmp_indices].mean(axis=1)
                )
            inner[:, task_column] = difference / len(conditions)
        selected_tasks = rng.integers(
            0, len(task_ids), size=(samples, len(task_ids))
        )
        row_index = np.arange(samples)[:, None]
        map_estimates[:, map_column] = inner[
            row_index, selected_tasks
        ].mean(axis=1)
    selected_maps = rng.integers(
        0, len(maps), size=(samples, len(maps))
    )
    row_index = np.arange(samples)[:, None]
    estimates = map_estimates[row_index, selected_maps].mean(axis=1)
    return (
        float(np.quantile(estimates, alpha / 2)),
        float(np.quantile(estimates, 1 - alpha / 2)),
    )


def _family_statistics(
    family: str,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    analysis_protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    metric = "safe_weighted_coverage"
    nested = _nested(rows, config, metric)
    maps, means = _map_means(nested, config)
    _, tasks = _validate_grid(nested, config)
    algorithms = tuple(config["algorithms"])
    matrix = np.vstack([means[algorithm] for algorithm in algorithms])
    if np.any(np.ptp(matrix, axis=0) > 0):
        result = stats.friedmanchisquare(
            *[means[algorithm] for algorithm in algorithms]
        )
        friedman_stat = float(result.statistic)
        friedman_p = float(result.pvalue)
    else:
        friedman_stat, friedman_p = 0.0, 1.0

    pairwise = []
    raw_p = []
    samples = int(analysis_protocol["bootstrap"]["samples"])
    alpha = float(analysis_protocol["alpha"])
    base_seed = int(analysis_protocol["bootstrap"]["seed"])
    for comparator in algorithms:
        if comparator == "full":
            continue
        differences = means["full"] - means[comparator]
        if np.allclose(differences, 0):
            statistic, p_value = 0.0, 1.0
        else:
            result = stats.wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
            )
            statistic, p_value = (
                float(result.statistic),
                float(result.pvalue),
            )
        seed_material = (
            f"{base_seed}|{family}|{comparator}".encode("utf-8")
        )
        pair_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8], "little"
        )
        low, high = _bootstrap_interval(
            nested["full"],
            nested[comparator],
            maps,
            tasks,
            tuple(config["conditions"]),
            samples,
            alpha,
            np.random.default_rng(pair_seed),
        )
        pairwise.append(
            {
                "statistical_family": family,
                "reference": "full",
                "comparator": comparator,
                "metric": metric,
                "map_count": len(maps),
                "mean_difference": float(np.mean(differences)),
                "median_difference": float(np.median(differences)),
                "hodges_lehmann": hodges_lehmann(differences),
                "rank_biserial": rank_biserial(differences),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
        raw_p.append(p_value)
    for record, adjusted in zip(pairwise, holm_adjust(raw_p)):
        record["p_holm"] = float(adjusted)
        record["significant_holm"] = bool(adjusted < alpha)
    summaries = []
    map_rows = []
    for algorithm in algorithms:
        values = means[algorithm]
        summaries.append(
            {
                "statistical_family": family,
                "algorithm": algorithm,
                "metric": metric,
                "map_count": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "iqm_exploratory": interquartile_mean(values),
            }
        )
        map_rows.extend(
            {
                "statistical_family": family,
                "algorithm": algorithm,
                "map_id": map_id,
                "safe_weighted_coverage": float(value),
            }
            for map_id, value in zip(maps, values)
        )
    return {
        "omnibus": {
            "statistical_family": family,
            "metric": metric,
            "test": "friedman",
            "map_count": len(maps),
            "statistic": friedman_stat,
            "p_value": friedman_p,
        },
        "pairwise": pairwise,
        "summaries": summaries,
        "map_rows": map_rows,
    }


def _condition_and_interaction_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    condition_rows = []
    for family in (
        "known_domain_shift",
        "hidden_model_perception_mismatch",
    ):
        selected = [row for row in rows if row["family"] == family]
        for (model, condition), group in _group(
            selected, ("model", "condition")
        ).items():
            values = np.asarray(
                [float(row["safe_weighted_coverage"]) for row in group]
            )
            drops = np.asarray(
                [
                    float(row["robustness_drop"])
                    for row in group
                    if math.isfinite(float(row["robustness_drop"]))
                ]
            )
            condition_rows.append(
                {
                    "family": family,
                    "algorithm": model,
                    "condition": condition,
                    "run_count": len(values),
                    "safe_weighted_coverage_mean": float(np.mean(values)),
                    "safe_weighted_coverage_median": float(np.median(values)),
                    "robustness_drop_mean": (
                        float(np.mean(drops))
                        if len(drops)
                        else float("nan")
                    ),
                    "robustness_drop_median": (
                        float(np.median(drops))
                        if len(drops)
                        else float("nan")
                    ),
                }
            )
    interaction_rows = []
    factors = (
        "node_count",
        "difficulty",
        "constraint_type",
        "priority_layout",
    )
    for family, config in FAMILIES.items():
        selected = [
            row
            for row in rows
            if row["family"] in set(config["sources"])
            and row["model"] in set(config["algorithms"])
            and row["condition"] in set(config["conditions"])
        ]
        for factor in factors:
            for (model, level), group in _group(
                selected, ("model", factor)
            ).items():
                values = np.asarray(
                    [
                        float(row["safe_weighted_coverage"])
                        for row in group
                    ]
                )
                interaction_rows.append(
                    {
                        "statistical_family": family,
                        "algorithm": model,
                        "factor": factor,
                        "level": level,
                        "role": "exploratory",
                        "run_count": len(values),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                    }
                )
    return condition_rows, interaction_rows


def _group(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> Dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: Dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return grouped


def _descriptive_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    metrics = (
        "safe_weighted_coverage",
        "safe_rate",
        "return_rate",
        "violation_rate",
        "stranded_rate",
        "coverage",
        "weighted_coverage",
        "visited_count",
        "high_priority_coverage",
        "medium_priority_coverage",
        "low_priority_coverage",
        "oracle_attainment_lower",
        "oracle_attainment_upper",
        "oracle_regret_lower",
        "oracle_regret_upper",
        "safe_energy_utilization",
        "safe_distance_utilization",
        "safe_time_utilization",
        "dangerous_action_proposal_rate",
        "environment_interception_rate",
        "planning_time_s",
    )
    output = []
    for (family, model), group in _group(
        rows, ("family", "model")
    ).items():
        for metric in metrics:
            values = np.asarray(
                [
                    float(row[metric])
                    for row in group
                    if math.isfinite(float(row[metric]))
                ],
                dtype=np.float64,
            )
            output.append(
                {
                    "result_family": family,
                    "algorithm": model,
                    "metric": metric,
                    "valid_run_count": len(values),
                    "mean": (
                        float(np.mean(values))
                        if len(values)
                        else float("nan")
                    ),
                    "median": (
                        float(np.median(values))
                        if len(values)
                        else float("nan")
                    ),
                    "q25": (
                        float(np.quantile(values, 0.25))
                        if len(values)
                        else float("nan")
                    ),
                    "q75": (
                        float(np.quantile(values, 0.75))
                        if len(values)
                        else float("nan")
                    ),
                }
            )
    return output


def run() -> Dict[str, Any]:
    analysis_protocol = json.loads(
        ANALYSIS_PROTOCOL.read_text(encoding="utf-8")
    )
    rows = _analysis_rows()
    omnibus = []
    pairwise = []
    summaries = []
    map_rows = []
    for family, config in FAMILIES.items():
        result = _family_statistics(
            family, config, rows, analysis_protocol
        )
        omnibus.append(result["omnibus"])
        pairwise.extend(result["pairwise"])
        summaries.extend(result["summaries"])
        map_rows.extend(result["map_rows"])
    condition_rows, interaction_rows = (
        _condition_and_interaction_summaries(rows)
    )
    descriptive = _descriptive_summary(rows)

    DESTINATION.mkdir(parents=True, exist_ok=True)
    _atomic_csv(DESTINATION / "confirmatory_omnibus.csv", omnibus)
    _atomic_csv(DESTINATION / "confirmatory_pairwise.csv", pairwise)
    _atomic_csv(DESTINATION / "map_level_primary.csv", map_rows)
    _atomic_csv(DESTINATION / "algorithm_primary_summary.csv", summaries)
    _atomic_csv(DESTINATION / "descriptive_metrics.csv", descriptive)
    _atomic_csv(
        DESTINATION / "robustness_condition_summary.csv", condition_rows
    )
    _atomic_csv(
        DESTINATION / "exploratory_interactions.csv", interaction_rows
    )
    plot_columns = (
        "family",
        "map_id",
        "road_index",
        "task_id",
        "task_index",
        "model",
        "training_seed",
        "planner_seed",
        "condition",
        "node_count",
        "difficulty",
        "constraint_type",
        "priority_layout",
        "safe_weighted_coverage",
        "safe_rate",
        "return_rate",
        "violation_rate",
        "stranded_rate",
        "coverage",
        "weighted_coverage",
        "visited_count",
        "high_priority_coverage",
        "medium_priority_coverage",
        "low_priority_coverage",
        "oracle_attainment_lower",
        "oracle_attainment_upper",
        "oracle_regret_lower",
        "oracle_regret_upper",
        "safe_energy_utilization",
        "safe_distance_utilization",
        "safe_time_utilization",
        "dangerous_action_proposal_rate",
        "environment_interception_rate",
        "robustness_drop",
        "planning_time_s",
        "termination_reason",
    )
    _atomic_csv(
        DESTINATION / "frozen_plot_input.csv",
        [
            {column: row.get(column) for column in plot_columns}
            for row in rows
        ],
    )
    files = sorted(
        path
        for path in DESTINATION.iterdir()
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "state": "ready_for_plotting",
        "plots_created": False,
        "row_count": len(rows),
        "statistical_families": list(FAMILIES),
        "confirmatory_metric": "safe_weighted_coverage",
        "analysis_protocol_hash": analysis_protocol[
            "analysis_protocol_hash"
        ],
        "final_results_sha256": v32._sha256_file(FINAL_RESULTS),
        "files": {
            path.name: v32._sha256_file(path) for path in files
        },
    }
    manifest["manifest_hash"] = smoke._canonical_hash(manifest)
    smoke._atomic_json(
        DESTINATION / "analysis_manifest.json", manifest
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

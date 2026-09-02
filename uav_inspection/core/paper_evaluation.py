#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPO+Pointer论文级统计与出版图表入口。

本模块只读取实验运行器保存的原始结果，不从轨迹反推覆盖率，不重算能耗，
也不生成依赖样本集合的人工综合分。统计单位始终是测试场景。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np

# 论文批处理在无桌面会话和自动测试中运行，固定无界面后端避免弹窗或Tk崩溃。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 保留SVG/PDF中的文字对象，便于投稿前在矢量编辑器中微调。
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

try:
    from scipy import stats
except ImportError as exc:  # pragma: no cover - doctor命令会先检查依赖
    raise RuntimeError("论文统计需要scipy，请在Deeplearning-gpu环境中运行。") from exc


SCHEMA_VERSION = 1

REQUIRED_COLUMNS = (
    "scenario_id",
    "scenario_hash",
    "manifest_hash",
    "split",
    "algorithm",
    "returned",
    "energy_violation",
    "distance_violation",
    "time_violation",
    "weighted_coverage",
    "coverage",
    "energy_wh",
    "distance_m",
    "time_s",
    "min_remaining_soc",
    "planning_time_s",
)

NUMERIC_COLUMNS = (
    "weighted_coverage",
    "coverage",
    "energy_wh",
    "distance_m",
    "time_s",
    "min_remaining_soc",
    "planning_time_s",
)

OPTIONAL_NUMERIC_COLUMNS = (
    "training_seed",
    "planner_seed",
    "replicate_id",
    "evaluations",
    "optimality_gap",
    "node_count",
    "power_scale",
    "visited_count",
    "low_priority_coverage",
    "medium_priority_coverage",
    "high_priority_coverage",
    "energy_utilization",
    "distance_utilization",
    "time_utilization",
    "solver_dual_bound",
)

OPTIONAL_COUNT_COLUMNS = ("visited_count",)
OPTIONAL_COVERAGE_COLUMNS = (
    "low_priority_coverage",
    "medium_priority_coverage",
    "high_priority_coverage",
)
OPTIONAL_UTILIZATION_COLUMNS = (
    "energy_utilization",
    "distance_utilization",
    "time_utilization",
)

AUXILIARY_RUN_METRICS = (
    "weighted_coverage",
    "coverage",
    "visited_count",
    *OPTIONAL_COVERAGE_COLUMNS,
    "planning_time_s",
    "evaluations",
    "solver_dual_bound",
)

SAFE_RESOURCE_METRICS = (
    "energy_wh",
    "distance_m",
    "time_s",
    "min_remaining_soc",
    *OPTIONAL_UTILIZATION_COLUMNS,
)

RUN_ID_COLUMNS = ("training_seed", "planner_seed", "replicate_id")

BOOL_COLUMNS = (
    "returned",
    "energy_violation",
    "distance_violation",
    "time_violation",
)

OPTIONAL_BOOL_COLUMNS = ("dynamics_violation",)
TRISTATE_BOOL_COLUMNS = ("optimality_certified",)

METHOD_FAMILY_MEMBERS = {
    "learning": (
        "full",
        "ppo_mlp",
        "a2c_pointer",
        "no_priority_bias",
        "no_domain_randomization",
        "no_resource_shaping",
        "no_return_reserve",
    ),
    "greedy": ("nearest_feasible", "priority_resource_greedy", "greedy"),
    "metaheuristic": ("aco", "ga", "sa", "pso"),
    "exact_or_search": ("milp_orienteering", "exact_pareto_dp", "a_star"),
}

# 同一算法族使用同色系；算法无论出现在哪张图中都保持固定颜色。
METHOD_FAMILY_PALETTE = {
    "learning": (
        "#355C8A",
        "#5F82AD",
        "#7D9CC0",
        "#9BB4D0",
        "#B4C7DC",
        "#8190B3",
        "#A8A4C5",
    ),
    "greedy": ("#B96520", "#DF9246", "#E8B778"),
    "metaheuristic": ("#236F70", "#4A9290", "#72B0AA", "#9CCBC3"),
    "exact_or_search": ("#6A4C93", "#947BB3", "#B9A8CF"),
    "other": ("#4D4D4D", "#7A7A7A", "#A6A6A6"),
}

FIGURE_DESCRIPTIONS = {
    "primary_safe_weighted_coverage": "唯一确认性主指标的场景级分布",
    "safety_rate_ci": "各方法的场景等权安全率及95%置信区间",
    "priority_coverage_profile": "高、中、低优先级巡检点覆盖率",
    "constraint_violation_rates": "能量、距离、时间和动力学违规率",
    "planning_time_median_iqr": "在线规划时间的场景级中位数与四分位距",
    "safe_resource_profile": "仅安全路线的资源消耗与预算利用率",
    "termination_reason_distribution": "场景等权的任务终止原因分布",
    "representative_routes": "按冻结规则预注册场景与算法身份的代表路线",
    "coverage_energy_pareto": "安全路线能耗与安全加权覆盖率的权衡",
    "node_count_generalization": "同生成机制下的规模泛化",
    "power_sensitivity": "功率参数敏感性",
}


@dataclass(frozen=True)
class EvaluationConfig:
    """论文统计参数集中配置，避免图表和检验各自使用魔法数字。"""

    reference_algorithm: str = "full"
    primary_split: str = "id_test"
    primary_power_scale: float = 1.0
    alpha: float = 0.05
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    figure_dpi: int = 600
    figure_formats: Tuple[str, ...] = ("pdf", "svg")
    generate_figures: bool = True
    statistics_enabled: bool = True
    analysis_role: str = "preregistered_family"
    included_algorithms: Optional[Tuple[str, ...]] = None
    expected_manifest_hash: Optional[str] = None
    expected_scenario_hash: Optional[str] = None

    def validate(self) -> None:
        if not self.reference_algorithm.strip():
            raise ValueError("reference_algorithm不能为空。")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha必须位于(0, 1)。")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples必须大于0。")
        if not math.isfinite(self.primary_power_scale) or self.primary_power_scale <= 0.0:
            raise ValueError("primary_power_scale必须是有限正数。")
        normalized_formats = {str(item).lower().lstrip(".") for item in self.figure_formats}
        unsupported = normalized_formats - {"pdf", "svg", "png", "tif", "tiff"}
        if unsupported:
            raise ValueError(f"不支持的图表格式: {sorted(unsupported)}")
        if self.figure_dpi <= 0:
            raise ValueError("figure_dpi必须大于0。")
        if self.analysis_role not in {
            "preregistered_family",
            "secondary_descriptive",
            "development",
        }:
            raise ValueError("analysis_role不是受支持的分析角色。")
        if self.included_algorithms is not None:
            normalized_algorithms = tuple(
                str(item).strip() for item in self.included_algorithms
            )
            if not normalized_algorithms or any(not item for item in normalized_algorithms):
                raise ValueError("included_algorithms必须包含至少一个非空算法名。")
            if len(set(normalized_algorithms)) != len(normalized_algorithms):
                raise ValueError("included_algorithms不能包含重复算法名。")
            if self.reference_algorithm not in normalized_algorithms:
                raise ValueError("included_algorithms必须包含reference_algorithm。")
        for name in ("expected_manifest_hash", "expected_scenario_hash"):
            value = getattr(self, name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{name}不能是空字符串。")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    """把不可估计的非有限统计量写成JSON null，而不是非标准NaN。"""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coerce_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{name}必须是布尔值，当前为{value!r}。")


def _read_json_records(path: Path) -> List[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("records", "results", "rows"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"{path}必须包含结果数组，或records/results/rows字段。")


def load_result_records(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """读取并严格校验CSV/JSON/JSONL长表。"""

    records: List[Mapping[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                records.extend(csv.DictReader(stream))
        elif path.suffix.lower() in {".json", ".jsonl"}:
            records.extend(_read_json_records(path))
        else:
            raise ValueError(f"不支持的结果格式: {path.suffix}")

    validated: List[Dict[str, Any]] = []
    for row_index, source in enumerate(records):
        missing = [name for name in REQUIRED_COLUMNS if name not in source]
        if missing:
            raise ValueError(f"第{row_index}行缺少字段: {', '.join(missing)}")
        row = dict(source)
        row["scenario_id"] = str(row["scenario_id"]).strip()
        row["scenario_hash"] = str(row["scenario_hash"]).strip()
        row["manifest_hash"] = str(row["manifest_hash"]).strip()
        row["split"] = str(row["split"]).strip()
        row["algorithm"] = str(row["algorithm"]).strip()
        if any(
            not row[name]
            for name in (
                "scenario_id",
                "scenario_hash",
                "manifest_hash",
                "split",
                "algorithm",
            )
        ):
            raise ValueError(
                f"第{row_index}行的场景、清单、划分和算法标识均不能为空。"
            )
        for name in BOOL_COLUMNS:
            row[name] = _coerce_bool(row[name], name)
        for name in OPTIONAL_BOOL_COLUMNS:
            row[name] = (
                False
                if row.get(name) in (None, "")
                else _coerce_bool(row[name], name)
            )
        for name in TRISTATE_BOOL_COLUMNS:
            row[name] = (
                None
                if row.get(name) in (None, "")
                else _coerce_bool(row[name], name)
            )
        row["termination_reason"] = str(
            row.get("termination_reason", "unknown")
        ).strip() or "unknown"
        for name in NUMERIC_COLUMNS:
            row[name] = float(row[name])
            if not math.isfinite(row[name]):
                raise ValueError(f"第{row_index}行{name}不是有限数。")
        for name in OPTIONAL_NUMERIC_COLUMNS:
            value = row.get(name)
            if value not in (None, ""):
                row[name] = float(value)
                if not math.isfinite(row[name]):
                    raise ValueError(f"第{row_index}行{name}不是有限数。")
            else:
                row[name] = None
        for name in RUN_ID_COLUMNS:
            value = row[name]
            if value is not None:
                if not float(value).is_integer():
                    raise ValueError(f"第{row_index}行{name}必须是整数。")
                row[name] = int(value)
        if all(row[name] is None for name in RUN_ID_COLUMNS):
            raise ValueError(
                f"第{row_index}行必须提供training_seed、planner_seed或replicate_id。"
            )
        if row["node_count"] is not None:
            if not float(row["node_count"]).is_integer() or row["node_count"] <= 0:
                raise ValueError(f"第{row_index}行node_count必须是正整数。")
            row["node_count"] = int(row["node_count"])
        if row["power_scale"] is None:
            # 未标注功率倍率只代表名义1.0，不能匹配任意敏感性条件。
            row["power_scale"] = 1.0
        if row["power_scale"] <= 0.0:
            raise ValueError(f"第{row_index}行power_scale必须大于0。")
        if row["optimality_gap"] is not None and row["optimality_gap"] < 0.0:
            raise ValueError(f"第{row_index}行optimality_gap不能为负。")
        for name in ("coverage", "weighted_coverage"):
            if not -1e-9 <= row[name] <= 1.0 + 1e-9:
                raise ValueError(f"第{row_index}行{name}必须位于[0, 1]。")
        for name in OPTIONAL_COVERAGE_COLUMNS:
            value = row[name]
            if value is not None and not -1e-9 <= value <= 1.0 + 1e-9:
                raise ValueError(f"第{row_index}行{name}必须位于[0, 1]。")
        for name in OPTIONAL_COUNT_COLUMNS:
            value = row[name]
            if value is not None:
                if not float(value).is_integer() or value < 0.0:
                    raise ValueError(f"第{row_index}行{name}必须是非负整数。")
                row[name] = int(value)
        for name in OPTIONAL_UTILIZATION_COLUMNS:
            value = row[name]
            # 利用率可高于1，用于如实保留超预算运行，但不能为负。
            if value is not None and value < 0.0:
                raise ValueError(f"第{row_index}行{name}不能为负。")
        for name in ("energy_wh", "distance_m", "time_s", "planning_time_s"):
            if row[name] < 0.0:
                raise ValueError(f"第{row_index}行{name}不能为负。")
        row["safe"] = bool(
            row["returned"]
            and not row["energy_violation"]
            and not row["distance_violation"]
            and not row["time_violation"]
            and not row["dynamics_violation"]
        )
        # 论文主指标：不安全路线不能依靠高覆盖掩盖任务失败。
        row["safe_weighted_coverage"] = (
            float(row["weighted_coverage"]) if row["safe"] else 0.0
        )
        row["safe_coverage"] = float(row["coverage"]) if row["safe"] else 0.0
        validated.append(row)
    if not validated:
        raise ValueError("结果文件中没有可评估记录。")

    seen_runs: Dict[Tuple[Any, ...], int] = {}
    scenario_hashes: MutableMapping[Tuple[Any, ...], set] = defaultdict(set)
    manifest_hashes: MutableMapping[str, set] = defaultdict(set)
    for row_index, row in enumerate(validated):
        run_key = (
            row["split"],
            row["scenario_id"],
            row["scenario_hash"],
            row["algorithm"],
            row["node_count"],
            row["power_scale"],
            row["training_seed"],
            row["planner_seed"],
            row["replicate_id"],
        )
        if run_key in seen_runs:
            raise ValueError(
                f"第{row_index}行与第{seen_runs[run_key]}行具有重复运行身份。"
            )
        seen_runs[run_key] = row_index
        scenario_key = (
            row["split"],
            row["scenario_id"],
            row["node_count"],
            row["power_scale"],
        )
        scenario_hashes[scenario_key].add(row["scenario_hash"])
        manifest_hashes[row["split"]].add(row["manifest_hash"])
    inconsistent = {
        key: sorted(values)
        for key, values in scenario_hashes.items()
        if len(values) != 1
    }
    if inconsistent:
        raise ValueError(f"相同场景身份出现不一致scenario_hash：{inconsistent}")
    bad_manifests = {
        split: sorted(values)
        for split, values in manifest_hashes.items()
        if len(values) != 1
    }
    if bad_manifests:
        raise ValueError(f"同一split混入多个manifest_hash：{bad_manifests}")
    return validated


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(array)) if array.size else float("nan")


def _summary(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {
            name: float("nan")
            for name in ("mean", "sd", "median", "q1", "q3", "iqr")
        }
    q1 = float(np.quantile(array, 0.25))
    q3 = float(np.quantile(array, 0.75))
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def _mean_t_interval(
    values: Sequence[float], alpha: float, *, lower: float = -math.inf, upper: float = math.inf
) -> Tuple[float, float]:
    """以场景为统计单位的均值区间，不按重复次数给场景加权。"""

    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return float("nan"), float("nan")
    mean = float(np.mean(array))
    if array.size == 1:
        return max(lower, mean), min(upper, mean)
    sem = float(stats.sem(array))
    half = float(stats.t.ppf(1.0 - alpha / 2.0, array.size - 1)) * sem
    return max(lower, mean - half), min(upper, mean + half)


def aggregate_results(
    rows: Sequence[Mapping[str, Any]], alpha: float
) -> List[Dict[str, Any]]:
    groups: MutableMapping[
        Tuple[str, str, Optional[float], Optional[float]], List[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        node_count = row.get("node_count")
        power_scale = row.get("power_scale")
        groups[
            (
                str(row["split"]),
                str(row["algorithm"]),
                None if node_count in (None, "") else float(node_count),
                None if power_scale in (None, "") else float(power_scale),
            )
        ].append(row)

    output: List[Dict[str, Any]] = []
    for (split, algorithm, node_count, power_scale), group in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        by_scenario: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in group:
            by_scenario[str(row["scenario_id"])].append(row)
        scenario_rows: List[Dict[str, Any]] = []
        for scenario_id, repeats in sorted(by_scenario.items()):
            safe_repeats = [row for row in repeats if bool(row["safe"])]
            scenario_record: Dict[str, Any] = {
                "scenario_id": scenario_id,
                "safe_fraction": len(safe_repeats) / len(repeats),
                "repeat_count": len(repeats),
                "safe_repeat_count": len(safe_repeats),
            }
            for metric in ("safe_weighted_coverage", "safe_coverage"):
                scenario_record[metric] = _mean(
                    float(row[metric]) for row in repeats
                )
            for metric in AUXILIARY_RUN_METRICS:
                observations = [
                    float(row[metric])
                    for row in repeats
                    if row.get(metric) not in (None, "")
                ]
                if observations:
                    scenario_record[metric] = _mean(observations)
            for metric in (
                "returned",
                "energy_violation",
                "distance_violation",
                "time_violation",
                "dynamics_violation",
            ):
                scenario_record[metric] = _mean(
                    float(bool(row.get(metric, False))) for row in repeats
                )
            for metric in SAFE_RESOURCE_METRICS:
                observations = [
                    float(row[metric])
                    for row in safe_repeats
                    if row.get(metric) not in (None, "")
                ]
                if observations:
                    scenario_record[metric] = _mean(observations)
            scenario_rows.append(scenario_record)

        scenario_safe_fractions = [
            float(row["safe_fraction"]) for row in scenario_rows
        ]
        safe_rate = _mean(scenario_safe_fractions)
        safe_low, safe_high = _mean_t_interval(
            scenario_safe_fractions, alpha, lower=0.0, upper=1.0
        )
        safe_repeat_count = sum(int(row["safe_repeat_count"]) for row in scenario_rows)
        repeat_count = sum(int(row["repeat_count"]) for row in scenario_rows)
        record: Dict[str, Any] = {
            "split": split,
            "algorithm": algorithm,
            "n_runs": len(group),
            "n_scenarios": len(scenario_rows),
            "safe_repeat_count": safe_repeat_count,
            "repeat_count": repeat_count,
            "fully_safe_scenario_count": sum(
                math.isclose(value, 1.0, abs_tol=1e-12)
                for value in scenario_safe_fractions
            ),
            "safe_rate": safe_rate,
            "safe_rate_ci_low": safe_low,
            "safe_rate_ci_high": safe_high,
            "mean_within_scenario_safe_fraction": safe_rate,
            "statistical_unit": "scenario",
            "safety_rule": "mean_repeat_safety_with_equal_scenario_weight",
        }
        certification_rows = [
            row for row in group if row.get("optimality_certified") is not None
        ]
        if certification_rows:
            certified_count = sum(
                bool(row["optimality_certified"]) for row in certification_rows
            )
            record.update(
                {
                    "optimality_certification_eligible_runs": len(
                        certification_rows
                    ),
                    "optimality_certified_runs": certified_count,
                    "optimality_certification_rate": certified_count
                    / len(certification_rows),
                }
            )
        gap_values = [
            float(row["optimality_gap"])
            for row in group
            if row.get("optimality_gap") is not None
        ]
        if gap_values:
            for key, value in _summary(gap_values).items():
                record[f"optimality_gap_{key}"] = value
        if node_count is not None:
            record["node_count"] = int(node_count)
        if power_scale is not None:
            record["power_scale"] = power_scale
        for metric in (
            "safe_weighted_coverage",
            "safe_coverage",
            *AUXILIARY_RUN_METRICS,
        ):
            metric_values = [
                float(row[metric])
                for row in scenario_rows
                if row.get(metric) not in (None, "")
                and math.isfinite(float(row[metric]))
            ]
            if not metric_values:
                continue
            for key, value in _summary(metric_values).items():
                record[f"{metric}_{key}"] = value
            record[f"{metric}_n_scenarios"] = len(metric_values)
        for metric in (
            "returned",
            "energy_violation",
            "distance_violation",
            "time_violation",
            "dynamics_violation",
        ):
            record[f"{metric}_rate"] = _mean(
                float(row[metric]) for row in scenario_rows
            )
        # 资源效率只在安全路线中比较，避免失败路线的截断消耗产生虚假优势。
        for metric in SAFE_RESOURCE_METRICS:
            metric_values = [
                float(row[metric])
                for row in scenario_rows
                if row.get(metric) not in (None, "")
                and math.isfinite(float(row[metric]))
            ]
            if not metric_values:
                continue
            for key, value in _summary(metric_values).items():
                record[f"safe_{metric}_{key}"] = value
            record[f"safe_{metric}_n_scenarios"] = len(metric_values)
        output.append(record)
    return output


def _scenario_metric(
    rows: Sequence[Mapping[str, Any]],
    split: str,
    metric: str,
    power_scale: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    values: MutableMapping[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        row_power = row.get("power_scale")
        normalized_power = 1.0 if row_power in (None, "") else float(row_power)
        matching_power = (
            power_scale is None
            or math.isclose(
                normalized_power, power_scale, rel_tol=0.0, abs_tol=1e-12
            )
        )
        metric_value = row.get(metric)
        if (
            str(row["split"]) == split
            and matching_power
            and metric_value not in (None, "")
        ):
            values[(str(row["algorithm"]), str(row["scenario_id"]))].append(
                float(metric_value)
            )
    result: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (algorithm, scenario_id), observations in values.items():
        result[algorithm][scenario_id] = float(np.mean(observations))
    return dict(result)


def _scenario_repeat_metric(
    rows: Sequence[Mapping[str, Any]],
    split: str,
    metric: str,
    power_scale: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    values: MutableMapping[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        row_power = row.get("power_scale")
        normalized_power = 1.0 if row_power in (None, "") else float(row_power)
        if str(row["split"]) != split or not math.isclose(
            normalized_power, power_scale, rel_tol=0.0, abs_tol=1e-12
        ):
            continue
        values[(str(row["algorithm"]), str(row["scenario_id"]))].append(
            float(row[metric])
        )
    result: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for (algorithm, scenario_id), observations in values.items():
        result[algorithm][scenario_id] = np.asarray(observations, dtype=np.float64)
    return dict(result)


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.ones(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[int(index)]))
        running = max(running, candidate)
        adjusted[int(index)] = running
    return adjusted.tolist()


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[np.abs(differences) > 1e-12]
    if not nonzero.size:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / max(positive + negative, 1e-12)


def _hodges_lehmann(differences: np.ndarray) -> float:
    if not differences.size:
        return float("nan")
    walsh = (differences[:, None] + differences[None, :]) / 2.0
    return float(np.median(walsh[np.triu_indices(len(differences))]))


def _hierarchical_paired_bootstrap_interval(
    reference: Mapping[str, np.ndarray],
    comparator: Mapping[str, np.ndarray],
    scenario_ids: Sequence[str],
    samples: int,
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """外层配对重采样场景，内层分别重采样各算法在该场景的随机重复。"""

    if not scenario_ids:
        return float("nan"), float("nan")
    estimates = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected = rng.integers(0, len(scenario_ids), size=len(scenario_ids))
        differences = []
        for selected_index in selected:
            scenario_id = scenario_ids[int(selected_index)]
            reference_values = reference[scenario_id]
            comparator_values = comparator[scenario_id]
            reference_draw = reference_values[
                rng.integers(0, len(reference_values), size=len(reference_values))
            ]
            comparator_draw = comparator_values[
                rng.integers(0, len(comparator_values), size=len(comparator_values))
            ]
            differences.append(
                float(np.mean(reference_draw) - np.mean(comparator_draw))
            )
        estimates[sample_index] = float(np.mean(differences))
    return (
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


def statistical_comparison(
    rows: Sequence[Mapping[str, Any]], config: EvaluationConfig
) -> Dict[str, Any]:
    config.validate()
    all_scenario_repeats = _scenario_repeat_metric(
        rows,
        config.primary_split,
        "safe_weighted_coverage",
        config.primary_power_scale,
    )
    if config.included_algorithms is None:
        scenario_repeats = all_scenario_repeats
    else:
        requested = tuple(str(item).strip() for item in config.included_algorithms)
        missing_algorithms = sorted(set(requested) - set(all_scenario_repeats))
        if missing_algorithms:
            raise ValueError(
                "预注册统计族中的算法在主测试条件下不存在："
                f"{missing_algorithms}"
            )
        # 先限定统计族，再做网格完整性检查和Holm校正，避免补充算法污染主族。
        scenario_repeats = {
            algorithm: all_scenario_repeats[algorithm] for algorithm in requested
        }
    scenario_values = {
        algorithm: {
            scenario_id: float(np.mean(observations))
            for scenario_id, observations in by_scenario.items()
        }
        for algorithm, by_scenario in scenario_repeats.items()
    }
    algorithms = sorted(scenario_values)
    if config.reference_algorithm not in scenario_values:
        raise ValueError(
            f"主测试集不存在参考算法 {config.reference_algorithm!r}，"
            f"可用算法为{algorithms}。"
        )

    reference_ids = set(scenario_values[config.reference_algorithm])
    incomplete = {
        algorithm: {
            "missing": sorted(reference_ids - set(scenario_values[algorithm])),
            "extra": sorted(set(scenario_values[algorithm]) - reference_ids),
        }
        for algorithm in algorithms
        if set(scenario_values[algorithm]) != reference_ids
    }
    if incomplete:
        raise ValueError(
            "主要比较的algorithm×scenario网格不完整，禁止静默取交集："
            f"{incomplete}"
        )
    common_ids = sorted(reference_ids)
    omnibus: Dict[str, Any] = {
        "test": "friedman",
        "scenario_count": len(common_ids),
        "algorithms": algorithms,
        "statistic": None,
        "p_value": None,
    }
    if len(algorithms) >= 3 and len(common_ids) >= 2:
        arrays = [
            np.asarray([scenario_values[a][sid] for sid in common_ids], dtype=np.float64)
            for a in algorithms
        ]
        matrix = np.vstack(arrays)
        if np.any(np.ptp(matrix, axis=0) > 0.0):
            result = stats.friedmanchisquare(*arrays)
            omnibus.update(statistic=float(result.statistic), p_value=float(result.pvalue))

    rng = np.random.default_rng(config.bootstrap_seed)
    pairwise: List[Dict[str, Any]] = []
    raw_p: List[float] = []
    reference = scenario_values[config.reference_algorithm]
    for comparator in algorithms:
        if comparator == config.reference_algorithm:
            continue
        paired_ids = sorted(set(reference) & set(scenario_values[comparator]))
        differences = np.asarray(
            [reference[sid] - scenario_values[comparator][sid] for sid in paired_ids],
            dtype=np.float64,
        )
        if not differences.size or np.allclose(differences, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            test = stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        ci_low, ci_high = _hierarchical_paired_bootstrap_interval(
            scenario_repeats[config.reference_algorithm],
            scenario_repeats[comparator],
            paired_ids,
            config.bootstrap_samples,
            config.alpha,
            rng,
        )
        pairwise.append(
            {
                "reference": config.reference_algorithm,
                "comparator": comparator,
                "scenario_count": len(paired_ids),
                "mean_difference": float(np.mean(differences)) if differences.size else float("nan"),
                "median_difference": float(np.median(differences)) if differences.size else float("nan"),
                "hodges_lehmann": _hodges_lehmann(differences),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_method": "paired_scenario_then_independent_within_scenario_repeat",
                "reference_repeat_count_min": min(
                    len(scenario_repeats[config.reference_algorithm][sid])
                    for sid in paired_ids
                ) if paired_ids else 0,
                "reference_repeat_count_max": max(
                    len(scenario_repeats[config.reference_algorithm][sid])
                    for sid in paired_ids
                ) if paired_ids else 0,
                "comparator_repeat_count_min": min(
                    len(scenario_repeats[comparator][sid]) for sid in paired_ids
                ) if paired_ids else 0,
                "comparator_repeat_count_max": max(
                    len(scenario_repeats[comparator][sid]) for sid in paired_ids
                ) if paired_ids else 0,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial": _rank_biserial(differences),
            }
        )
        raw_p.append(p_value)
    for record, adjusted in zip(pairwise, _holm_adjust(raw_p)):
        record["p_holm"] = float(adjusted)
        record["significant_holm"] = bool(adjusted < config.alpha)

    return {
        "primary_metric": "safe_weighted_coverage",
        "primary_split": config.primary_split,
        "reference_algorithm": config.reference_algorithm,
        "included_algorithms": algorithms,
        "omnibus": omnibus,
        "pairwise": pairwise,
        "bootstrap_unit": "scenario_outer_repeat_inner",
        "repeat_counts_by_algorithm": {
            algorithm: {
                "min_per_scenario": min(len(values) for values in by_scenario.values()),
                "max_per_scenario": max(len(values) for values in by_scenario.values()),
                "total": sum(len(values) for values in by_scenario.values()),
            }
            for algorithm, by_scenario in scenario_repeats.items()
        },
        "repeat_identity": "training_seed_or_planner_seed_with_optional_replicate_id",
        "interpretation": (
            "正的差值表示参考算法更高；Wilcoxon以场景均值配对，置信区间外层重采样场景、"
            "内层分别重采样各算法在该场景的训练或规划重复。"
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    ""
                    if isinstance(value, (float, np.floating))
                    and not math.isfinite(float(value))
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
            ],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def _method_family(algorithm: str) -> str:
    normalized = str(algorithm).strip().lower()
    for family, members in METHOD_FAMILY_MEMBERS.items():
        if normalized in members:
            return family
    if normalized.startswith("no_") or "pointer" in normalized or "ppo" in normalized:
        return "learning"
    if "greedy" in normalized:
        return "greedy"
    if normalized in {"aco", "ga", "sa", "pso"}:
        return "metaheuristic"
    if any(token in normalized for token in ("milp", "exact", "a_star")):
        return "exact_or_search"
    return "other"


def _method_color(algorithm: str) -> str:
    family = _method_family(algorithm)
    palette = METHOD_FAMILY_PALETTE[family]
    members = METHOD_FAMILY_MEMBERS.get(family, ())
    normalized = str(algorithm).strip().lower()
    if normalized in members:
        return palette[members.index(normalized) % len(palette)]
    stable_index = sum(ord(character) for character in normalized) % len(palette)
    return palette[stable_index]


def _method_color_map(algorithms: Sequence[str]) -> Dict[str, str]:
    return {str(algorithm): _method_color(str(algorithm)) for algorithm in algorithms}


def _save_figure(
    fig: plt.Figure,
    directory: Path,
    stem: str,
    config: EvaluationConfig,
) -> List[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for raw_suffix in config.figure_formats:
        suffix = str(raw_suffix).lower().lstrip(".")
        path = directory / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=config.figure_dpi,
            bbox_inches="tight",
        )
        saved.append(path)
    plt.close(fig)
    return saved


def _write_figure_manifest(
    figures_dir: Path,
    stems: Sequence[str],
    algorithms: Sequence[str],
    config: EvaluationConfig,
) -> None:
    files: List[Dict[str, Any]] = []
    for stem in stems:
        for raw_suffix in config.figure_formats:
            suffix = str(raw_suffix).lower().lstrip(".")
            path = figures_dir / f"{stem}.{suffix}"
            exists = path.is_file()
            size = path.stat().st_size if exists else 0
            file_record: Dict[str, Any] = {
                "figure": stem,
                "path": path.name,
                "format": suffix,
                "exists": exists,
                "nonempty": size > 0,
                "size_bytes": size,
            }
            if exists and suffix == "svg":
                svg_text = path.read_text(encoding="utf-8", errors="replace")
                file_record["editable_text_nodes_present"] = "<text" in svg_text
            elif suffix == "pdf":
                file_record["editable_text_configured"] = True
            elif suffix in {"tif", "tiff"}:
                file_record["raster_dpi"] = config.figure_dpi
            files.append(file_record)

    manifest = {
        "schema_version": 1,
        "figure_contract": {
            "core_conclusion": (
                "以安全加权覆盖率进行唯一确认性比较，并用优先级覆盖、违规率、"
                "安全资源消耗和计算时间解释方法差异。"
            ),
            "archetype": "quantitative_grid",
            "backend": "Python/matplotlib",
            "target_output": "double_column_publication_figure",
            "analysis_role": config.analysis_role,
            "primary_metric": "safe_weighted_coverage",
            "confirmatory_metrics": (
                ["safe_weighted_coverage"] if config.statistics_enabled else []
            ),
            "statistics_unit": "scenario_after_within_scenario_repeat_aggregation",
            "resource_policy": "safe_routes_only",
            "forbidden_composites": ["radar_score", "subjective_weighted_score"],
            "evidence_hierarchy": {
                "hero": "primary_safe_weighted_coverage",
                "explanation": [
                    "priority_coverage_profile",
                    "constraint_violation_rates",
                    "safe_resource_profile",
                    "termination_reason_distribution",
                    "coverage_energy_pareto",
                ],
                "robustness": ["node_count_generalization", "power_sensitivity"],
            },
            "reviewer_risks": [
                "不把不安全路线的低资源消耗解释为效率优势",
                "不把辅助覆盖指标升级为额外确认性主指标",
                "不把超时精确算法结果误标为最优解",
            ],
        },
        "figures": [
            {
                "stem": stem,
                "description": FIGURE_DESCRIPTIONS.get(stem, stem),
            }
            for stem in stems
        ],
        "method_visual_identity": {
            algorithm: {
                "family": _method_family(algorithm),
                "color": _method_color(algorithm),
            }
            for algorithm in algorithms
        },
        "export": {
            "formats": [str(item).lower().lstrip(".") for item in config.figure_formats],
            "dpi": config.figure_dpi,
            "svg_text_as_text": True,
            "pdf_fonttype": 42,
        },
        "qa": {
            "all_files_present_and_nonempty": bool(files)
            and all(item["exists"] and item["nonempty"] for item in files),
            "all_svg_files_have_text_nodes": all(
                item.get("editable_text_nodes_present", True) for item in files
            ),
            "radar_or_composite_score_generated": False,
            "files": files,
        },
    }
    _write_json(figures_dir / "figure_manifest.json", manifest)


def _algorithm_order(rows: Sequence[Mapping[str, Any]], reference: str) -> List[str]:
    algorithms = sorted({str(row["algorithm"]) for row in rows})
    if reference in algorithms:
        algorithms.remove(reference)
        algorithms.insert(0, reference)
    return algorithms


def _safe_float_name(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("路线文件名中的功率倍率必须是有限数。")
    text = format(number, ".17g")
    return "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in text
    ).strip("._")


def generate_figures(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, config: EvaluationConfig
) -> List[str]:
    """生成最小论文主图集；不存在相应字段的扩展图会自动跳过。"""

    _set_publication_style()
    figures_dir = output_dir / "figures"
    figure_rows = [
        row
        for row in rows
        if config.included_algorithms is None
        or str(row["algorithm"]) in set(config.included_algorithms)
    ]
    test_rows = [
        row
        for row in figure_rows
        if str(row["split"]) == config.primary_split
        and math.isclose(
            1.0
            if row.get("power_scale") in (None, "")
            else float(row["power_scale"]),
            config.primary_power_scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    algorithms = _algorithm_order(test_rows, config.reference_algorithm)
    all_algorithms = _algorithm_order(figure_rows, config.reference_algorithm)
    colors = _method_color_map(all_algorithms)
    created: List[str] = []

    scenario_values = _scenario_metric(
        test_rows,
        config.primary_split,
        "safe_weighted_coverage",
        config.primary_power_scale,
    )
    values = [list(scenario_values.get(algorithm, {}).values()) for algorithm in algorithms]
    if algorithms and all(values):
        fig, ax = plt.subplots(figsize=(max(7.0, 1.25 * len(algorithms)), 4.8))
        boxes = ax.boxplot(values, labels=algorithms, patch_artist=True, showfliers=False)
        for patch, algorithm in zip(boxes["boxes"], algorithms):
            patch.set_facecolor(colors[algorithm])
            patch.set_alpha(0.55)
        ax.set_ylabel("Safe weighted coverage")
        ax.set_xlabel("Method")
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis="x", rotation=25)
        _save_figure(fig, figures_dir, "primary_safe_weighted_coverage", config)
        created.append("primary_safe_weighted_coverage")

    if algorithms:
        rates, lows, highs = [], [], []
        for algorithm in algorithms:
            group = [row for row in test_rows if str(row["algorithm"]) == algorithm]
            by_scenario: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
            for row in group:
                by_scenario[str(row["scenario_id"])].append(row)
            scenario_safe_fractions = [
                _mean(float(bool(row["safe"])) for row in repeats)
                for repeats in by_scenario.values()
            ]
            rate = _mean(scenario_safe_fractions)
            low, high = _mean_t_interval(
                scenario_safe_fractions,
                config.alpha,
                lower=0.0,
                upper=1.0,
            )
            rates.append(rate)
            lows.append(rate - low)
            highs.append(high - rate)
        fig, ax = plt.subplots(figsize=(max(7.0, 1.25 * len(algorithms)), 4.8))
        positions = np.arange(len(algorithms))
        ax.bar(positions, rates, color=[colors[a] for a in algorithms], alpha=0.78)
        ax.errorbar(positions, rates, yerr=[lows, highs], fmt="none", ecolor="black", capsize=4)
        ax.set_xticks(positions, algorithms, rotation=25, ha="right")
        ax.set_ylabel("Safety rate (scenario mean, 95% CI)")
        ax.set_ylim(0.0, 1.05)
        _save_figure(fig, figures_dir, "safety_rate_ci", config)
        created.append("safety_rate_ci")

    priority_specs = [
        ("low_priority_coverage", "Low priority"),
        ("medium_priority_coverage", "Medium priority"),
        ("high_priority_coverage", "High priority"),
    ]
    priority_specs = [
        (field, title)
        for field, title in priority_specs
        if any(row.get(field) not in (None, "") for row in test_rows)
    ]
    if algorithms and priority_specs:
        fig, axes = plt.subplots(
            1,
            len(priority_specs),
            figsize=(max(7.2, 3.2 * len(priority_specs)), 4.4),
            sharey=True,
            squeeze=False,
        )
        positions = np.arange(len(algorithms))
        for axis_index, (field, title) in enumerate(priority_specs):
            ax = axes[0, axis_index]
            scenario_metric = _scenario_metric(
                test_rows,
                config.primary_split,
                field,
                config.primary_power_scale,
            )
            for position, algorithm in enumerate(algorithms):
                observations = list(scenario_metric.get(algorithm, {}).values())
                if not observations:
                    ax.text(
                        0.02,
                        position,
                        "NA",
                        ha="left",
                        va="center",
                        fontsize=6,
                        color="#767676",
                        rotation=90,
                    )
                    continue
                mean = _mean(observations)
                low, high = _mean_t_interval(
                    observations, config.alpha, lower=0.0, upper=1.0
                )
                ax.scatter(
                    mean,
                    position,
                    s=28,
                    color=colors[algorithm],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
                ax.errorbar(
                    mean,
                    position,
                    xerr=[[mean - low], [high - mean]],
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.8,
                    capsize=2.5,
                )
            ax.set_title(title)
            ax.set_yticks(positions)
            ax.set_yticklabels(algorithms)
            ax.tick_params(axis="y", labelleft=axis_index == 0)
            ax.set_xlim(0.0, 1.05)
            ax.set_xlabel("Scenario mean (95% CI)")
            if axis_index == 0:
                ax.set_ylabel("Method")
        axes[0, 0].invert_yaxis()
        _save_figure(fig, figures_dir, "priority_coverage_profile", config)
        created.append("priority_coverage_profile")

    violation_specs = (
        ("energy_violation", "Energy violation"),
        ("distance_violation", "Distance violation"),
        ("time_violation", "Time violation"),
        ("dynamics_violation", "Dynamics violation"),
    )
    if algorithms:
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6), sharey=True)
        positions = np.arange(len(algorithms))
        for axis_index, (ax, (field, title)) in enumerate(zip(axes.flat, violation_specs)):
            scenario_metric = _scenario_metric(
                test_rows,
                config.primary_split,
                field,
                config.primary_power_scale,
            )
            for position, algorithm in enumerate(algorithms):
                observations = list(scenario_metric.get(algorithm, {}).values())
                if not observations:
                    continue
                rate = _mean(observations)
                low, high = _mean_t_interval(
                    observations, config.alpha, lower=0.0, upper=1.0
                )
                ax.scatter(
                    rate,
                    position,
                    s=28,
                    color=colors[algorithm],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
                ax.errorbar(
                    rate,
                    position,
                    xerr=[[rate - low], [high - rate]],
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.8,
                    capsize=2.5,
                )
            ax.set_title(title)
            ax.set_yticks(positions)
            ax.set_yticklabels(algorithms)
            ax.tick_params(axis="y", labelleft=axis_index % 2 == 0)
            ax.set_xlim(0.0, 1.05)
            ax.set_xlabel("Scenario mean (95% CI)" if axis_index >= 2 else "")
        axes[0, 0].set_ylabel("Method")
        axes[1, 0].set_ylabel("Method")
        axes[0, 0].invert_yaxis()
        _save_figure(fig, figures_dir, "constraint_violation_rates", config)
        created.append("constraint_violation_rates")

    planning_values = _scenario_metric(
        test_rows,
        config.primary_split,
        "planning_time_s",
        config.primary_power_scale,
    )
    if algorithms and all(planning_values.get(algorithm) for algorithm in algorithms):
        medians, q1_values, q3_values = [], [], []
        for algorithm in algorithms:
            summary = _summary(planning_values[algorithm].values())
            medians.append(summary["median"])
            q1_values.append(summary["q1"])
            q3_values.append(summary["q3"])
        fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(algorithms)), 4.5))
        positions = np.arange(len(algorithms))
        ax.scatter(
            positions,
            medians,
            s=32,
            color=[colors[algorithm] for algorithm in algorithms],
            zorder=3,
        )
        ax.errorbar(
            positions,
            medians,
            yerr=[
                np.asarray(medians) - np.asarray(q1_values),
                np.asarray(q3_values) - np.asarray(medians),
            ],
            fmt="none",
            ecolor="#4D4D4D",
            elinewidth=1.2,
            capsize=3,
        )
        if min(medians) > 0.0 and max(medians) / min(medians) >= 100.0:
            ax.set_yscale("log")
        ax.set_xticks(positions, algorithms, rotation=35, ha="right")
        ax.set_ylabel("Online planning time (s; median and IQR)")
        _save_figure(fig, figures_dir, "planning_time_median_iqr", config)
        created.append("planning_time_median_iqr")

    resource_specs = (
        ("energy_wh", "Energy", "Wh"),
        ("distance_m", "Distance", "m"),
        ("time_s", "Mission time", "s"),
        ("energy_utilization", "Energy utilization", "fraction"),
        ("distance_utilization", "Distance utilization", "fraction"),
        ("time_utilization", "Time utilization", "fraction"),
    )
    available_resources = [
        (field, title, unit)
        for field, title, unit in resource_specs
        if any(bool(row["safe"]) and row.get(field) not in (None, "") for row in test_rows)
    ]
    if algorithms and available_resources:
        resource_rows = 1 if len(available_resources) <= 3 else 2
        fig, axes = plt.subplots(
            resource_rows,
            3,
            figsize=(12.0, max(3.8 * resource_rows, 0.75 * len(algorithms))),
            squeeze=False, sharey=True,
        )
        positions = np.arange(len(algorithms))
        for axis_index, ax in enumerate(axes.flat):
            if axis_index >= len(available_resources):
                ax.set_visible(False)
                continue
            field, title, unit = available_resources[axis_index]
            for position, algorithm in enumerate(algorithms):
                by_scenario: MutableMapping[str, List[float]] = defaultdict(list)
                for row in test_rows:
                    if (
                        str(row["algorithm"]) == algorithm
                        and bool(row["safe"])
                        and row.get(field) not in (None, "")
                    ):
                        by_scenario[str(row["scenario_id"])].append(float(row[field]))
                observations = [
                    _mean(repeats) for repeats in by_scenario.values() if repeats
                ]
                if not observations:
                    ax.text(0.02, position, "NA", ha="left", va="center", fontsize=6)
                    continue
                summary = _summary(observations)
                median = summary["median"]
                ax.scatter(
                    median,
                    position,
                    s=28,
                    color=colors[algorithm],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
                ax.errorbar(
                    median,
                    position,
                    xerr=[
                        [median - summary["q1"]],
                        [summary["q3"] - median],
                    ],
                    fmt="none",
                    ecolor="#4D4D4D",
                    elinewidth=0.8,
                    capsize=2.5,
                )
            ax.set_title(title)
            ax.set_yticks(positions)
            ax.set_yticklabels(algorithms)
            ax.tick_params(axis="y", labelleft=axis_index % 3 == 0)
            ax.set_xlabel(f"{unit}; median and IQR")
            if axis_index % 3 == 0:
                ax.set_ylabel("Method")
        axes[0, 0].invert_yaxis()
        _save_figure(fig, figures_dir, "safe_resource_profile", config)
        created.append("safe_resource_profile")

    termination_reasons = sorted(
        {str(row["termination_reason"]) for row in test_rows},
        key=lambda reason: (
            -sum(str(row["termination_reason"]) == reason for row in test_rows),
            reason,
        ),
    )
    if algorithms and termination_reasons:
        fractions = np.zeros((len(termination_reasons), len(algorithms)), dtype=float)
        for algorithm_index, algorithm in enumerate(algorithms):
            by_scenario: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
            for row in test_rows:
                if str(row["algorithm"]) == algorithm:
                    by_scenario[str(row["scenario_id"])].append(row)
            for reason_index, reason in enumerate(termination_reasons):
                scenario_fractions = [
                    _mean(
                        float(str(row["termination_reason"]) == reason)
                        for row in repeats
                    )
                    for repeats in by_scenario.values()
                    if repeats
                ]
                fractions[reason_index, algorithm_index] = _mean(scenario_fractions)
        fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(algorithms)), 4.8))
        bottoms = np.zeros(len(algorithms), dtype=float)
        reason_colors = plt.get_cmap("tab20")(
            np.linspace(0.05, 0.95, max(1, len(termination_reasons)))
        )
        for reason_index, reason in enumerate(termination_reasons):
            ax.bar(
                np.arange(len(algorithms)),
                fractions[reason_index],
                bottom=bottoms,
                label=reason,
                color=reason_colors[reason_index],
                width=0.78,
            )
            bottoms += fractions[reason_index]
        ax.set_xticks(np.arange(len(algorithms)), algorithms, rotation=35, ha="right")
        ax.set_ylabel("Scenario-equal fraction")
        ax.set_ylim(0.0, 1.02)
        ax.legend(
            frameon=False,
            fontsize=7,
            ncol=min(3, len(termination_reasons)),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
        )
        _save_figure(fig, figures_dir, "termination_reason_distribution", config)
        created.append("termination_reason_distribution")

    pareto = []
    for algorithm in algorithms:
        group = [row for row in test_rows if str(row["algorithm"]) == algorithm]
        by_scenario: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in group:
            by_scenario[str(row["scenario_id"])].append(row)
        scenario_energy = []
        scenario_coverage = []
        for repeats in by_scenario.values():
            safe_repeats = [row for row in repeats if bool(row["safe"])]
            if safe_repeats:
                scenario_energy.append(
                    _mean(float(row["energy_wh"]) for row in safe_repeats)
                )
            scenario_coverage.append(
                _mean(float(row["safe_weighted_coverage"]) for row in repeats)
            )
        if scenario_energy:
            pareto.append(
                (
                    algorithm,
                    _mean(scenario_energy),
                    _mean(scenario_coverage),
                )
            )
    if pareto:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for algorithm, energy, coverage in pareto:
            ax.scatter(energy, coverage, s=64, color=colors[algorithm], label=algorithm)
            ax.annotate(algorithm, (energy, coverage), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Safe-route energy (Wh; lower is better)")
        ax.set_ylabel("Safe weighted coverage (higher is better)")
        ax.set_ylim(-0.02, 1.02)
        _save_figure(fig, figures_dir, "coverage_energy_pareto", config)
        created.append("coverage_energy_pareto")

    for field, stem, xlabel in (
        ("node_count", "node_count_generalization", "Inspection-node count"),
        ("power_scale", "power_sensitivity", "Power multiplier"),
    ):
        available = [row for row in figure_rows if row.get(field) not in (None, "")]
        if field == "node_count":
            available = [
                row for row in available if str(row["split"]).startswith("scale_")
            ]
        else:
            available = [
                row for row in available if str(row["split"]) == config.primary_split
            ]
        x_values = sorted({float(row[field]) for row in available})
        if len(x_values) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        for algorithm in _algorithm_order(available, config.reference_algorithm):
            algorithm_rows = [row for row in available if str(row["algorithm"]) == algorithm]
            y_values = []
            for value in x_values:
                by_scenario: MutableMapping[str, List[float]] = defaultdict(list)
                for row in algorithm_rows:
                    if float(row[field]) == value:
                        by_scenario[str(row["scenario_id"])].append(
                            float(row["safe_weighted_coverage"])
                        )
                y_values.append(
                    _mean(_mean(repeats) for repeats in by_scenario.values())
                )
            ax.plot(x_values, y_values, marker="o", linewidth=1.8, label=algorithm, color=colors.get(algorithm))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Safe weighted coverage")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        _save_figure(fig, figures_dir, stem, config)
        created.append(stem)
    _write_figure_manifest(figures_dir, created, all_algorithms, config)
    return created


def generate_representative_route_figure(
    input_paths: Sequence[Path],
    output_dir: Path,
    representative: Mapping[str, Any],
    config: EvaluationConfig,
    *,
    existing_stems: Sequence[str] = (),
    all_algorithms: Sequence[str] = (),
) -> str:
    """只按冻结场景和种子身份提取路线，禁止事后挑选“好看”样例。"""

    scenario_id = str(representative["scenario_id"])
    displays = [dict(item) for item in representative["display_algorithms"]]
    candidates: Dict[Tuple[str, int], Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]] = {}
    for raw_path in input_paths:
        root = Path(raw_path)
        if not root.is_dir() or not (root / "run_config.json").is_file():
            continue
        run_config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
        immutable = dict(run_config.get("immutable") or {})
        for display in displays:
            algorithm = str(display["algorithm"])
            seed_field = "training_seed" if "training_seed" in display else "planner_seed"
            seed = int(display[seed_field])
            if run_config.get("kind") == "learning_evaluation":
                if (
                    algorithm != str(immutable.get("variant"))
                    or seed_field != "training_seed"
                    or seed != int(immutable.get("training_seed", -1))
                ):
                    continue
                route_path = root / "routes" / (
                    f"{scenario_id}__power{_safe_float_name(config.primary_power_scale)}.json"
                )
            elif run_config.get("kind") == "traditional_baselines":
                if algorithm not in set(immutable.get("algorithms", ())) or seed_field != "planner_seed":
                    continue
                route_path = root / "routes" / (
                    f"{scenario_id}__{algorithm}__seed{seed}"
                    f"__power{_safe_float_name(config.primary_power_scale)}.json"
                )
            else:
                continue
            if not route_path.is_file():
                continue
            payload = json.loads(route_path.read_text(encoding="utf-8"))
            row = dict(payload.get("row") or {})
            if (
                str(row.get("scenario_id")) != scenario_id
                or str(row.get("algorithm")) != algorithm
                or int(row.get(seed_field, -1)) != seed
            ):
                raise ValueError(f"代表路线身份与冻结规则不一致：{route_path}")
            detail = payload.get("detail") if "detail" in payload else payload.get("result")
            if not isinstance(detail, Mapping):
                raise ValueError(f"代表路线文件缺少detail/result：{route_path}")
            flight_path = np.asarray(detail.get("flight_path", ()), dtype=float)
            if flight_path.ndim != 2 or flight_path.shape[0] < 2 or flight_path.shape[1] != 3:
                raise ValueError(f"代表路线flight_path必须为N×3且N>=2：{route_path}")
            key = (algorithm, seed)
            if key in candidates:
                raise ValueError(f"检测到重复代表路线身份：{key}")
            candidates[key] = (row, flight_path, dict(payload.get("record") or {}))

    expected = {
        (str(item["algorithm"]), int(item.get("training_seed", item.get("planner_seed"))))
        for item in displays
    }
    if set(candidates) != expected:
        raise ValueError(
            "代表路线矩阵不完整："
            f"missing={sorted(expected - set(candidates))}, unexpected={sorted(set(candidates) - expected)}"
        )

    _set_publication_style()
    fig = plt.figure(figsize=(11.0, 8.3))
    for panel_index, display in enumerate(displays, start=1):
        algorithm = str(display["algorithm"])
        seed = int(display.get("training_seed", display.get("planner_seed")))
        row, flight_path, record = candidates[(algorithm, seed)]
        ax = fig.add_subplot(2, 2, panel_index, projection="3d")
        points = np.asarray(record.get("inspection_points_xyz", ()), dtype=float)
        priorities = np.asarray(record.get("priorities", ()), dtype=int)
        if points.ndim == 2 and points.shape[1] == 3 and priorities.size == points.shape[0]:
            priority_colors = np.asarray(["#9A9A9A", "#E69F00", "#C33C54"])
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=priority_colors[np.clip(priorities, 1, 3) - 1],
                s=22,
                depthshade=False,
            )
        ax.plot(
            flight_path[:, 0], flight_path[:, 1], flight_path[:, 2],
            color=_method_color(algorithm), linewidth=1.8,
        )
        ax.scatter(
            [flight_path[0, 0]], [flight_path[0, 1]], [flight_path[0, 2]],
            marker="*", s=75, color="#111111", depthshade=False,
        )
        ax.set_title(
            f"{algorithm} (seed {seed})\n"
            f"safe weighted coverage={float(row['safe_weighted_coverage']):.3f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=28, azim=-58)
    fig.suptitle(
        f"Pre-registered representative scenario: {scenario_id}", y=0.98, fontsize=11
    )
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color="#9A9A9A", label="Low priority"),
            Line2D([0], [0], marker="o", linestyle="", color="#E69F00", label="Medium priority"),
            Line2D([0], [0], marker="o", linestyle="", color="#C33C54", label="High priority"),
            Line2D([0], [0], marker="*", linestyle="", color="#111111", markersize=9, label="Start / return"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(bottom=0.12, top=0.90, hspace=0.30, wspace=0.16)
    figures_dir = output_dir / "figures"
    stem = "representative_routes"
    _save_figure(fig, figures_dir, stem, config)
    stems = [*existing_stems, stem]
    identities = list(dict.fromkeys([*all_algorithms, *(item["algorithm"] for item in displays)]))
    _write_figure_manifest(figures_dir, stems, identities, config)
    return stem


def run_analysis(
    input_paths: Sequence[Path], output_dir: Path, config: EvaluationConfig
) -> Dict[str, Any]:
    config.validate()
    rows = load_result_records(input_paths)
    if config.expected_manifest_hash is not None:
        unexpected = sorted(
            {
                str(row["manifest_hash"])
                for row in rows
                if str(row["manifest_hash"]) != str(config.expected_manifest_hash)
            }
        )
        if unexpected:
            raise ValueError(
                "结果manifest_hash与冻结清单不一致："
                f"expected={config.expected_manifest_hash!r}, found={unexpected}"
            )
    if config.expected_scenario_hash is not None:
        unexpected = sorted(
            {
                str(row["scenario_hash"])
                for row in rows
                if str(row["scenario_hash"]) != str(config.expected_scenario_hash)
            }
        )
        if unexpected:
            raise ValueError(
                "结果scenario_hash与冻结基础场景不一致："
                f"expected={config.expected_scenario_hash!r}, found={unexpected}"
            )
    summary = aggregate_results(rows, config.alpha)
    statistics = (
        statistical_comparison(rows, config)
        if config.statistics_enabled
        else {
            "schema_version": SCHEMA_VERSION,
            "status": "secondary_descriptive_only",
            "primary_metric": "safe_weighted_coverage",
            "pairwise": [],
            "omnibus": None,
            "multiplicity": None,
            "interpretation": (
                "该二级实验只报告预注册稳健性曲线和描述统计，不新增确认性检验族。"
            ),
        }
    )
    figures = generate_figures(rows, output_dir, config) if config.generate_figures else []

    _write_csv(output_dir / "results_validated.csv", rows)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "pairwise_primary.csv", statistics["pairwise"])
    _write_json(output_dir / "statistics.json", statistics)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "input_files": [str(Path(path).resolve()) for path in input_paths],
        "record_count": len(rows),
        "scenario_count": len({str(row["scenario_id"]) for row in rows}),
        "algorithms": sorted({str(row["algorithm"]) for row in rows}),
        "splits": sorted({str(row["split"]) for row in rows}),
        "manifest_hashes": sorted(
            {
                str(row["manifest_hash"])
                for row in rows
                if row.get("manifest_hash") not in (None, "")
            }
        ),
        "scenario_hashes": sorted(
            {
                str(row["scenario_hash"])
                for row in rows
                if row.get("scenario_hash") not in (None, "")
            }
        ),
        "unsafe_count": sum(not bool(row["safe"]) for row in rows),
        "dynamics_violation_count": sum(
            bool(row["dynamics_violation"]) for row in rows
        ),
        "termination_reason_counts": dict(
            sorted(Counter(str(row["termination_reason"]) for row in rows).items())
        ),
        "primary_metric": "safe_weighted_coverage",
        "confirmatory_metrics": (
            ["safe_weighted_coverage"] if config.statistics_enabled else []
        ),
        "analysis_role": config.analysis_role,
        "optional_formal_field_counts": {
            field: sum(row.get(field) not in (None, "") for row in rows)
            for field in (
                "visited_count",
                *OPTIONAL_COVERAGE_COLUMNS,
                *OPTIONAL_UTILIZATION_COLUMNS,
                "solver_dual_bound",
            )
        },
        "resource_summary_population": "safe_routes_only",
        "statistical_unit": "scenario_after_within_scenario_repeat_aggregation",
        "generated_figures": figures,
        "figure_contract_manifest": (
            "figures/figure_manifest.json" if config.generate_figures else None
        ),
        "config": asdict(config),
        "warnings": (
            []
            if len({str(row["algorithm"]) for row in rows}) >= 2
            else ["只有一个算法，无法形成论文比较结论。"]
        ),
    }
    _write_json(output_dir / "data_audit.json", audit)
    return audit


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO+Pointer论文级统计与出版图表")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="原始CSV/JSON/JSONL长表。")
    parser.add_argument("--output", type=Path, required=True, help="统计与图表输出目录。")
    parser.add_argument("--reference", default="full", help="配对比较参考算法。")
    parser.add_argument("--split", default="id_test", help="主要统计测试集名称。")
    parser.add_argument("--primary-power-scale", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=10_000, help="bootstrap重采样次数。")
    parser.add_argument("--seed", type=int, default=2026, help="统计重采样种子。")
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg"], help="图表格式。")
    parser.add_argument("--figure-dpi", type=int, default=600, help="TIFF/PNG等栅格图分辨率。")
    parser.add_argument("--no-figures", action="store_true", help="只生成统计表，不绘图。")
    parser.add_argument(
        "--descriptive-only",
        action="store_true",
        help="二级实验只生成描述统计和稳健性图，不运行新的确认性检验族。",
    )
    parser.add_argument(
        "--include-algorithms",
        nargs="+",
        default=None,
        help="预注册统计族算法名；筛选发生在网格检查与Holm校正之前。",
    )
    parser.add_argument("--expected-manifest-hash", default=None)
    parser.add_argument("--expected-scenario-hash", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = EvaluationConfig(
        reference_algorithm=args.reference,
        primary_split=args.split,
        primary_power_scale=args.primary_power_scale,
        bootstrap_samples=args.bootstrap,
        bootstrap_seed=args.seed,
        figure_dpi=args.figure_dpi,
        figure_formats=tuple(args.formats),
        generate_figures=not args.no_figures,
        statistics_enabled=not args.descriptive_only,
        analysis_role=(
            "secondary_descriptive" if args.descriptive_only else "development"
        ),
        included_algorithms=(
            None
            if args.include_algorithms is None
            else tuple(args.include_algorithms)
        ),
        expected_manifest_hash=args.expected_manifest_hash,
        expected_scenario_hash=args.expected_scenario_hash,
    )
    audit = run_analysis(args.input, args.output, config)
    print(
        f"论文统计完成: {audit['record_count']}条记录, "
        f"{len(audit['algorithms'])}个算法, 输出={Path(args.output).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

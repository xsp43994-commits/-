"""论文正式测试协议冻结与原始结果审计。

本模块刻意把两条路径分开：协议构建只读取场景清单、训练元数据、检查点字节和
代码/环境身份；只有 :func:`audit_result_runs` 会读取正式评估结果。这样可在查看
``id_test`` 成绩前先封存实验方案，避免结果反向影响协议。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_NAME = "frozen_test_v1"
PROTOCOL_FILENAME = "protocol.json"
RESULT_SCHEMA_VERSION = 2

LEARNING_VARIANTS: Tuple[str, ...] = (
    "full",
    "ppo_mlp",
    "a2c_pointer",
    "no_priority_bias",
    "no_domain_randomization",
    "no_resource_shaping",
    "no_return_reserve",
)
CORE_LEARNING_VARIANTS: Tuple[str, ...] = ("full", "ppo_mlp", "a2c_pointer")
ABLATION_VARIANTS: Tuple[str, ...] = (
    "no_priority_bias",
    "no_domain_randomization",
    "no_resource_shaping",
    "no_return_reserve",
)
TRAINING_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)
PLANNER_SEEDS: Tuple[int, ...] = tuple(range(42, 52))
DETERMINISTIC_PLANNER_SEED = 42

MAIN_BASELINES: Tuple[str, ...] = (
    "nearest_feasible",
    "priority_resource_greedy",
    "aco",
    "ga",
    "sa",
    "milp_orienteering",
)
SUPPLEMENTARY_BASELINES: Tuple[str, ...] = (
    "a_star",
    "pso",
    "exact_pareto_dp",
)
STOCHASTIC_BASELINES = frozenset({"aco", "ga", "sa", "pso"})
TIME_LIMITED_BASELINES = frozenset(
    {"milp_orienteering", "a_star", "exact_pareto_dp"}
)
DETERMINISTIC_BASELINES = frozenset(
    (set(MAIN_BASELINES) | set(SUPPLEMENTARY_BASELINES)) - STOCHASTIC_BASELINES
)

# 这些数量来自冻结清单设计，不得根据正式测试结果增删场景。
SPLIT_COUNTS: Dict[str, int] = {
    "validation": 64,
    "id_test": 100,
    "stress_test": 100,
    "scale_8": 25,
    "scale_12": 25,
    "scale_20": 25,
    "scale_24": 25,
}

REPRESENTATIVE_FIELDS: Tuple[str, ...] = (
    "initial_soc",
    "distance_budget_scale",
    "time_budget_scale",
    "wind_scale",
    "wind_rotation_deg",
    "wind_vertical_bias_mps",
)

FORMAL_METRIC_FIELDS: Tuple[str, ...] = (
    "returned",
    "energy_violation",
    "distance_violation",
    "time_violation",
    "dynamics_violation",
    "termination_reason",
    "weighted_coverage",
    "safe_weighted_coverage",
    "coverage",
    "safe_coverage",
    "visited_count",
    "low_priority_coverage",
    "medium_priority_coverage",
    "high_priority_coverage",
    "energy_wh",
    "distance_m",
    "time_s",
    "energy_budget_wh",
    "distance_budget_m",
    "time_budget_s",
    "energy_utilization",
    "distance_utilization",
    "time_utilization",
    "min_remaining_soc",
    "planning_time_s",
    "evaluations",
    "optimality_gap",
    "solver_dual_bound",
    "solver_status",
    "optimality_certified",
)

PROVENANCE_FIELDS: Tuple[str, ...] = (
    "schema_version",
    "scenario_id",
    "split",
    "algorithm",
    "variant",
    "training_seed",
    "planner_seed",
    "replicate_id",
    "checkpoint_hash",
    "scenario_hash",
    "manifest_hash",
    "node_count",
    "power_scale",
    "simulation_only",
    "protocol_hash",
)

DEFAULT_CODE_FILES: Tuple[str, ...] = (
    "final_python_ppo_pointer.py",
    "ppo_training_scenario.py",
    "paper_experiments.py",
    "paper_evaluation.py",
    "paper_protocol.py",
)


class ProtocolError(ValueError):
    """协议身份、冻结资产或结果审计不满足要求。"""


ManifestInput = Union[
    Path,
    str,
    Tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    Mapping[str, Any],
]
ResultInput = Union[Path, str, Mapping[str, Any], Sequence[Mapping[str, Any]]]


def _strict_json_loads(text: str, *, location: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ProtocolError(f"{location}包含非有限JSON常量：{value}")

    try:
        return json.loads(text, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"{location}不是有效JSON：{exc}") from exc


def _read_json(path: Path) -> Dict[str, Any]:
    payload = _strict_json_loads(path.read_text(encoding="utf-8"), location=str(path))
    if not isinstance(payload, dict):
        raise ProtocolError(f"{path}必须是JSON对象。")
    return payload


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Union[Path, str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_text(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_canonical_json(dict(record)) + "\n" for record in records)


def selected_records_hash(records: Sequence[Mapping[str, Any]]) -> str:
    """返回与实验编排器一致的场景子集SHA-256。"""

    return _sha256_bytes(_manifest_text(records).encode("utf-8"))


def _manifest_hash(metadata: Mapping[str, Any], records_hash: str) -> str:
    identity = dict(metadata)
    identity.pop("manifest_hash", None)
    # 兼容paper_experiments.prepare的既有清单算法：元数据使用json.dumps默认
    # ensure_ascii=True，而instances.jsonl本身使用ensure_ascii=False。
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes((canonical + records_hash).encode("utf-8"))


def _coerce_manifest(
    manifest: ManifestInput,
    *,
    require_frozen_counts: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[Path]]:
    root: Optional[Path] = None
    if isinstance(manifest, (str, Path)):
        candidate = Path(manifest)
        root = candidate if candidate.is_dir() else candidate.parent
        metadata_path = root / "manifest.json"
        if candidate.is_file() and candidate.name == "manifest.json":
            metadata_path = candidate
        if not metadata_path.is_file():
            raise FileNotFoundError(f"缺少冻结清单元数据：{metadata_path}")
        metadata = _read_json(metadata_path)
        records_path = root / str(metadata.get("records_file", "instances.jsonl"))
        if not records_path.is_file():
            raise FileNotFoundError(f"缺少冻结清单记录：{records_path}")
        raw_records = records_path.read_text(encoding="utf-8")
        records = []
        for line_number, line in enumerate(raw_records.splitlines(), start=1):
            if not line.strip():
                continue
            item = _strict_json_loads(
                line, location=f"{records_path}第{line_number}行"
            )
            if not isinstance(item, dict):
                raise ProtocolError(f"{records_path}第{line_number}行必须是JSON对象。")
            records.append(item)
        actual_file_hash = sha256_file(records_path)
        if actual_file_hash != str(metadata.get("records_sha256", "")):
            raise ProtocolError("冻结清单instances.jsonl的SHA-256与manifest.json不一致。")
    elif isinstance(manifest, tuple) and len(manifest) == 2:
        metadata = dict(manifest[0])
        records = [dict(item) for item in manifest[1]]
    elif isinstance(manifest, Mapping):
        if "metadata" in manifest and "records" in manifest:
            metadata = dict(manifest["metadata"])
            records = [dict(item) for item in manifest["records"]]
        else:
            metadata = dict(manifest)
            embedded = metadata.pop("records", None)
            if embedded is None:
                raise ProtocolError("内存清单必须同时提供metadata与records。")
            records = [dict(item) for item in embedded]
    else:
        raise TypeError("manifest必须是清单目录、(metadata, records)或映射对象。")

    if int(metadata.get("schema_version", -1)) != 1:
        raise ProtocolError("只支持schema_version=1的冻结场景清单。")
    records_hash = selected_records_hash(records)
    if records_hash != str(metadata.get("records_sha256", "")):
        raise ProtocolError("规范化清单记录哈希与records_sha256不一致。")
    if _manifest_hash(metadata, records_hash) != str(metadata.get("manifest_hash", "")):
        raise ProtocolError("manifest_hash校验失败。")

    ids = [str(item.get("id", "")) for item in records]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ProtocolError("清单场景ID不能为空或重复。")
    actual_counts = dict(Counter(str(item.get("split", "")) for item in records))
    declared_counts = {
        str(key): int(value) for key, value in dict(metadata.get("split_counts") or {}).items()
    }
    if actual_counts != declared_counts:
        raise ProtocolError(
            f"清单split数量与元数据不一致：actual={actual_counts}, declared={declared_counts}"
        )
    if require_frozen_counts and declared_counts != SPLIT_COUNTS:
        raise ProtocolError(
            f"frozen_test_v1要求固定split数量：{SPLIT_COUNTS}，实际为{declared_counts}。"
        )
    return metadata, records, root


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"{field}必须是数值，不能是布尔值。")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field}必须是有限数。") from exc
    if not math.isfinite(number):
        raise ProtocolError(f"{field}必须是有限数。")
    return number


def select_representative_scenario(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """按六字段z标准化后到分量中位数的距离选择代表性``id_test``。

    标准差使用总体标准差；若某字段没有变异，该字段距离固定为零。完全并列时按
    场景ID字典序选择，保证规则在任何机器上得到相同结果。
    """

    candidates = [dict(item) for item in records if item.get("split") == "id_test"]
    if not candidates:
        raise ProtocolError("清单中没有id_test场景，无法预注册代表路线。")
    candidates.sort(key=lambda item: str(item.get("id", "")))
    matrix: List[List[float]] = []
    for record in candidates:
        if not str(record.get("id", "")):
            raise ProtocolError("代表场景候选缺少id。")
        matrix.append(
            [
                _finite_number(record.get(field), field=f"{record['id']}.{field}")
                for field in REPRESENTATIVE_FIELDS
            ]
        )

    columns = list(zip(*matrix))
    medians: List[float] = []
    scales: List[float] = []
    for column in columns:
        ordered = sorted(column)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        mean = sum(ordered) / len(ordered)
        variance = sum((value - mean) ** 2 for value in ordered) / len(ordered)
        medians.append(median)
        scales.append(math.sqrt(variance))

    ranked: List[Tuple[float, str, int]] = []
    for index, (record, values) in enumerate(zip(candidates, matrix)):
        squared = sum(
            0.0 if scale == 0.0 else ((value - median) / scale) ** 2
            for value, median, scale in zip(values, medians, scales)
        )
        ranked.append((math.sqrt(squared), str(record["id"]), index))
    distance, scenario_id, _ = min(ranked, key=lambda item: (item[0], item[1]))
    return {
        "scenario_id": scenario_id,
        "selection_split": "id_test",
        "fields": list(REPRESENTATIVE_FIELDS),
        "standardization": "population_standard_deviation",
        "center": "componentwise_median",
        "distance": "euclidean_z_score_distance",
        "tie_break": "lexicographically_smallest_scenario_id",
        "selected_distance": distance,
        "field_medians": dict(zip(REPRESENTATIVE_FIELDS, medians)),
        "field_standard_deviations": dict(zip(REPRESENTATIVE_FIELDS, scales)),
        "display_algorithms": [
            {"algorithm": "full", "training_seed": 42},
            {"algorithm": "priority_resource_greedy", "planner_seed": 42},
            {"algorithm": "aco", "planner_seed": 42},
            {"algorithm": "milp_orienteering", "planner_seed": 42},
        ],
    }


def _portable_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _code_fingerprints(
    repo_root: Path,
    code_files: Optional[Sequence[Union[Path, str]]],
) -> List[Dict[str, Any]]:
    if code_files is None:
        selected = [repo_root / value for value in DEFAULT_CODE_FILES]
        package = repo_root / "python_classical_algs"
        if not package.is_dir():
            raise FileNotFoundError(f"缺少传统算法包：{package}")
        selected.extend(sorted(package.glob("*.py"), key=lambda item: item.name))
    else:
        selected = [
            Path(value) if Path(value).is_absolute() else repo_root / Path(value)
            for value in code_files
        ]
    unique: Dict[str, Path] = {}
    for path in selected:
        if not path.is_file():
            raise FileNotFoundError(f"协议代码指纹文件不存在：{path}")
        unique[_portable_path(path, repo_root)] = path
    return [
        {
            "path": name,
            "sha256": sha256_file(unique[name]),
            "size_bytes": unique[name].stat().st_size,
        }
        for name in sorted(unique)
    ]


def collect_environment_metadata() -> Dict[str, Any]:
    """收集不会包含当前时间的可复现实验环境身份。"""

    packages: Dict[str, Optional[str]] = {}
    for name in ("numpy", "scipy", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    gpus: List[Dict[str, str]] = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                if not line.strip():
                    continue
                name, separator, driver = line.partition(",")
                gpus.append(
                    {
                        "name": name.strip(),
                        "driver_version": driver.strip() if separator else "",
                    }
                )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        pass

    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "packages": packages,
        "gpus": gpus,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }


def _checkpoint_grid(
    training_root: Path,
    *,
    manifest_hash: str,
    scenario_hash: str,
    repo_root: Path,
) -> List[Dict[str, Any]]:
    checkpoints: List[Dict[str, Any]] = []
    seen_pairs = set()
    for variant in LEARNING_VARIANTS:
        for seed in TRAINING_SEEDS:
            run_dir = training_root / f"formal_{variant}_seed{seed}_3000ep"
            if not run_dir.is_dir():
                raise FileNotFoundError(f"缺少正式训练目录：{run_dir}")
            status = _read_json(run_dir / "status.json")
            config = _read_json(run_dir / "run_config.json")
            verification = _read_json(run_dir / "checkpoint_verification.json")
            checkpoint = run_dir / "best_safe.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"缺少best_safe检查点：{checkpoint}")

            if status.get("state") != "completed":
                raise ProtocolError(f"{run_dir.name}训练状态不是completed。")
            if status.get("best_safe_available") is not True:
                raise ProtocolError(f"{run_dir.name}未确认best_safe_available。")
            if status.get("checkpoint_verification_passed") is not True:
                raise ProtocolError(f"{run_dir.name}状态未确认检查点验证通过。")
            if int(status.get("episodes_seen", -1)) != 3000 or int(
                status.get("target_episodes", -1)
            ) != 3000:
                raise ProtocolError(f"{run_dir.name}不是完整3000回合正式训练。")
            if Path(str(status.get("best_checkpoint", ""))).name != "best_safe.pt":
                raise ProtocolError(f"{run_dir.name}的best_checkpoint不是best_safe.pt。")

            training_cfg = dict(config.get("training_config") or {})
            required_config = {
                "kind": (config.get("kind"), "learning_training"),
                "variant": (config.get("variant"), variant),
                "training_seed": (config.get("training_seed"), seed),
                "target_episodes": (config.get("target_episodes"), 3000),
                "scenario_hash": (config.get("scenario_hash"), scenario_hash),
                "manifest_hash": (config.get("manifest_hash"), manifest_hash),
                "training_config.experiment_variant": (
                    training_cfg.get("experiment_variant"),
                    variant,
                ),
                "training_config.seed": (training_cfg.get("seed"), seed),
                "training_config.experiment_stage": (
                    training_cfg.get("experiment_stage"),
                    "formal",
                ),
                "training_config.max_episodes": (
                    training_cfg.get("max_episodes"),
                    3000,
                ),
                "training_config.scenario_hash": (
                    training_cfg.get("scenario_hash"),
                    scenario_hash,
                ),
                "training_config.paper_manifest_hash": (
                    training_cfg.get("paper_manifest_hash"),
                    manifest_hash,
                ),
            }
            mismatches = [
                name for name, (actual, expected) in required_config.items() if actual != expected
            ]
            if mismatches:
                raise ProtocolError(
                    f"{run_dir.name}训练配置身份不一致：{', '.join(mismatches)}"
                )

            if verification.get("passed") is not True:
                raise ProtocolError(f"{run_dir.name}的checkpoint_verification.passed不是true。")
            if verification.get("safe_checkpoint_available") is not True:
                raise ProtocolError(f"{run_dir.name}未验证安全检查点存在。")
            expected_simulation_only = variant == "no_return_reserve"
            if verification.get("simulation_only") is not expected_simulation_only:
                raise ProtocolError(
                    f"{run_dir.name}的simulation_only标签与消融定义不一致。"
                )
            matches = [
                item
                for item in verification.get("checkpoints", [])
                if isinstance(item, Mapping)
                and item.get("file") == "best_safe.pt"
                and item.get("checkpoint_kind") == "best_safe"
            ]
            if len(matches) != 1:
                raise ProtocolError(f"{run_dir.name}必须且只能有一条best_safe验证记录。")
            verified = dict(matches[0])
            if verified.get("safe") is not True or verified.get(
                "deterministic_reproducible"
            ) is not True:
                raise ProtocolError(f"{run_dir.name}的best_safe未通过安全且确定性重放验证。")
            actual_hash = sha256_file(checkpoint)
            if str(verified.get("sha256", "")) != actual_hash:
                raise ProtocolError(f"{run_dir.name}的best_safe SHA-256验证失败。")

            pair = (variant, seed)
            if pair in seen_pairs:
                raise ProtocolError(f"重复训练身份：{pair}")
            seen_pairs.add(pair)
            checkpoints.append(
                {
                    "variant": variant,
                    "training_seed": seed,
                    "checkpoint_kind": "best_safe",
                    "path": _portable_path(checkpoint, repo_root),
                    "run_dir": _portable_path(run_dir, repo_root),
                    "sha256": actual_hash,
                    "size_bytes": checkpoint.stat().st_size,
                    "scenario_hash": scenario_hash,
                    "manifest_hash": manifest_hash,
                    "status_state": "completed",
                    "target_episodes": 3000,
                    "verification_passed": True,
                    "safe": True,
                    "deterministic_reproducible": True,
                    "simulation_only": expected_simulation_only,
                }
            )
    expected = {(variant, seed) for variant in LEARNING_VARIANTS for seed in TRAINING_SEEDS}
    if seen_pairs != expected or len(checkpoints) != 35:
        raise ProtocolError("正式检查点必须恰好覆盖7个变体×5个训练种子。")
    return checkpoints


def _metric_definitions() -> Dict[str, Any]:
    row_metrics = {
        "safe_weighted_coverage": {
            "role": "sole_primary",
            "direction": "higher",
            "definition": "weighted_coverage if safe else 0",
        },
        "weighted_coverage": {
            "role": "coverage_auxiliary",
            "direction": "higher",
            "definition": "sum of visited priority weights / sum of all priority weights",
        },
        "coverage": {
            "role": "coverage_auxiliary",
            "direction": "higher",
            "definition": "visited_count / node_count",
        },
        "safe_coverage": {
            "role": "coverage_auxiliary",
            "direction": "higher",
            "definition": "coverage if safe else 0",
        },
        "visited_count": {"role": "coverage_auxiliary", "direction": "higher"},
        "low_priority_coverage": {"role": "priority_coverage", "direction": "higher"},
        "medium_priority_coverage": {
            "role": "priority_coverage",
            "direction": "higher",
        },
        "high_priority_coverage": {"role": "priority_coverage", "direction": "higher"},
        "returned": {"role": "safety", "direction": "higher"},
        "energy_violation": {"role": "safety", "direction": "lower"},
        "distance_violation": {"role": "safety", "direction": "lower"},
        "time_violation": {"role": "safety", "direction": "lower"},
        "dynamics_violation": {"role": "safety", "direction": "lower"},
        "termination_reason": {"role": "safety_diagnostic", "direction": None},
        "min_remaining_soc": {"role": "safety_resource", "direction": "higher"},
        "energy_wh": {
            "role": "resource_safe_routes_only",
            "direction": "lower",
            "unit": "Wh",
        },
        "distance_m": {
            "role": "resource_safe_routes_only",
            "direction": "lower",
            "unit": "m",
        },
        "time_s": {
            "role": "resource_safe_routes_only",
            "direction": "lower",
            "unit": "s",
        },
        "energy_budget_wh": {"role": "resource_budget", "unit": "Wh"},
        "distance_budget_m": {"role": "resource_budget", "unit": "m"},
        "time_budget_s": {"role": "resource_budget", "unit": "s"},
        "energy_utilization": {"role": "resource_safe_routes_only", "direction": "lower"},
        "distance_utilization": {
            "role": "resource_safe_routes_only",
            "direction": "lower",
        },
        "time_utilization": {"role": "resource_safe_routes_only", "direction": "lower"},
        "planning_time_s": {"role": "online_compute", "direction": "lower", "unit": "s"},
        "evaluations": {"role": "search_compute", "direction": "lower"},
        "optimality_gap": {"role": "exact_reference", "direction": "lower"},
        "solver_dual_bound": {"role": "exact_reference", "direction": None},
        "solver_status": {"role": "exact_reference", "direction": None},
        "optimality_certified": {"role": "exact_reference", "direction": "higher"},
    }
    return {
        "primary_metric": "safe_weighted_coverage",
        "safe_definition": (
            "returned and not energy_violation and not distance_violation "
            "and not time_violation and not dynamics_violation"
        ),
        "row_metrics": row_metrics,
        "aggregate_metrics": {
            "safety_rate": "scenario-equal mean of safe indicator",
            "return_rate": "scenario-equal mean of returned",
            "violation_rates": "scenario-equal rates for each violation flag",
            "termination_distribution": "counts and rates; failed rows are retained",
            "runtime_summary": "median and interquartile range",
        },
        "resource_population": "safe routes only",
        "internal_only": ["objective"],
        "forbidden_rankings": ["manual_composite_score", "radar_total", "subjective_weight_rank"],
    }


def _statistics_families() -> Dict[str, Any]:
    common = {
        "unit": "scenario",
        "within_scenario_reduction": "mean over training/planner seeds before testing",
        "omnibus": "Friedman",
        "pairwise": "paired Wilcoxon signed-rank versus full",
        "multiplicity": "Holm correction within this family",
        "effect_sizes": ["rank_biserial", "Hodges_Lehmann"],
        "confidence_interval": "10000-replicate hierarchical bootstrap",
    }
    return {
        "main": {
            **common,
            "members": ["full", "ppo_mlp", "a2c_pointer", *MAIN_BASELINES],
            "reference": "full",
            "comparisons": ["ppo_mlp", "a2c_pointer", *MAIN_BASELINES],
            "status": "confirmatory",
        },
        "ablation": {
            **common,
            "members": ["full", *ABLATION_VARIANTS],
            "reference": "full",
            "comparisons": list(ABLATION_VARIANTS),
            "status": "confirmatory",
        },
        "supplementary": {
            **common,
            "members": ["full", *SUPPLEMENTARY_BASELINES],
            "reference": "full",
            "comparisons": list(SUPPLEMENTARY_BASELINES),
            "status": "exploratory",
        },
    }


def compute_protocol_hash(protocol: Mapping[str, Any]) -> str:
    """计算协议总哈希；顶层``protocol_hash``字段本身不参与计算。"""

    identity = dict(protocol)
    identity.pop("protocol_hash", None)
    return _sha256_bytes(_canonical_json(identity).encode("utf-8"))


def build_frozen_protocol(
    training_root: Union[Path, str],
    manifest: ManifestInput,
    *,
    repo_root: Optional[Union[Path, str]] = None,
    code_files: Optional[Sequence[Union[Path, str]]] = None,
    environment: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """构建``frozen_test_v1``，不读取任何评估结果或测试成绩。"""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    metadata, records, manifest_root = _coerce_manifest(
        manifest, require_frozen_counts=True
    )
    manifest_hash = str(metadata["manifest_hash"])
    scenario_hash = str(metadata.get("base_scenario_hash", ""))
    if len(manifest_hash) != 64 or len(scenario_hash) != 64:
        raise ProtocolError("manifest_hash和base_scenario_hash必须是64位SHA-256。")

    checkpoints = _checkpoint_grid(
        Path(training_root),
        manifest_hash=manifest_hash,
        scenario_hash=scenario_hash,
        repo_root=root,
    )
    code_identity = _code_fingerprints(root, code_files)
    environment_identity = dict(environment or collect_environment_metadata())
    environment_hash = _sha256_bytes(_canonical_json(environment_identity).encode("utf-8"))

    protocol: Dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "immutable": True,
        "manifest": {
            "source": None
            if manifest_root is None
            else _portable_path(manifest_root, root),
            "manifest_hash": manifest_hash,
            "base_scenario_hash": scenario_hash,
            "records_sha256": str(metadata["records_sha256"]),
            "split_counts": dict(SPLIT_COUNTS),
            "manifest_seed": int(metadata.get("manifest_seed", -1)),
        },
        "checkpoints": checkpoints,
        "algorithms": {
            "learning_variants": list(LEARNING_VARIANTS),
            "core_learning_variants": list(CORE_LEARNING_VARIANTS),
            "ablation_variants": list(ABLATION_VARIANTS),
            "main_baselines": list(MAIN_BASELINES),
            "supplementary_baselines": list(SUPPLEMENTARY_BASELINES),
            "stochastic_baselines": sorted(STOCHASTIC_BASELINES),
            "deterministic_baselines": sorted(DETERMINISTIC_BASELINES),
        },
        "seeds": {
            "training": list(TRAINING_SEEDS),
            "stochastic_planner": list(PLANNER_SEEDS),
            "deterministic_planner_identity": DETERMINISTIC_PLANNER_SEED,
        },
        "budgets": {
            "learning": {
                "decode_mode": "deterministic",
                "replicates": "five independently trained checkpoints",
            },
            "stochastic_search": {
                "algorithms": sorted(STOCHASTIC_BASELINES),
                "planner_seeds": list(PLANNER_SEEDS),
                "max_evaluations": 50_000,
                "time_limit_s": None,
                "rule": "complete the evaluation budget; no common 60-second truncation",
            },
            "time_limited_exact_or_graph": {
                "algorithms": sorted(TIME_LIMITED_BASELINES),
                "max_evaluations": None,
                "time_limit_s": 60.0,
            },
            "deterministic_greedy": {
                "algorithms": ["nearest_feasible", "priority_resource_greedy"],
                "replicates": 1,
                "planner_seed_identity": DETERMINISTIC_PLANNER_SEED,
            },
            "certification": {
                "milp_orienteering": "certified only when HiGHS reports optimal",
                "a_star": "certified only after complete search",
                "exact_pareto_dp": "certified only after complete search",
                "timeout": "retain feasible result and real gap/bound; never claim optimality",
            },
        },
        "statistics_families": _statistics_families(),
        "metrics": _metric_definitions(),
        "secondary_experiments": {
            "stress_test": {
                "scope": "full",
                "splits": ["stress_test"],
                "learning_variants": list(LEARNING_VARIANTS),
                "baselines": list(MAIN_BASELINES),
                "power_scales": [1.0],
                "expected_rows": 6800,
            },
            "scale_generalization": {
                "scope": "core",
                "splits": ["scale_8", "scale_12", "scale_20", "scale_24"],
                "learning_variants": list(CORE_LEARNING_VARIANTS),
                "baselines": list(MAIN_BASELINES),
                "power_scales": [1.0],
                "expected_rows": 4800,
                "fine_tuning": False,
            },
            "power_sensitivity": {
                "scope": "core",
                "splits": ["id_test"],
                "learning_variants": list(CORE_LEARNING_VARIANTS),
                "baselines": list(MAIN_BASELINES),
                "power_scales": [0.8, 0.9, 1.0, 1.1, 1.2],
                "new_power_scales": [0.8, 0.9, 1.1, 1.2],
                "reuse_nominal_power_results": True,
                "expected_new_rows": 19_200,
            },
        },
        "primary_id_test_counts": {
            "learning": 3500,
            "main_baselines": 3300,
            "supplementary_baselines": 1200,
            "total": 8000,
        },
        "representative_scenario": select_representative_scenario(records),
        "timing_protocol": {
            "execution": "single process with no concurrent heavy workload",
            "learning_inference": (
                "warm up once and synchronize CUDA before/after timing; exclude model loading "
                "and disk I/O"
            ),
            "traditional_algorithms": (
                "include evaluator/context initialization, search, and final unified route replay"
            ),
            "offline_training_hours": 26.95,
            "offline_training_reporting": "report separately from online planning time",
            "runtime_summary": "median and IQR",
        },
        "claim_boundaries": {
            "training_budget": (
                "same 3000-episode budget and update schedule; observed environment interactions "
                "range from 46844 to 50072 and are not identical"
            ),
            "supported_mask_claim": "return-aware multi-resource feasibility mask",
            "no_return_reserve_label": "simulation safety ablation",
            "supported_domains": [
                "current road/DEM simulation",
                "registered stress conditions",
                "same-generator scale generalization without fine-tuning",
            ],
            "unsupported_claims": [
                "cross-map generalization",
                "real-flight safety certification",
                "measured real-world power",
                "independent energy-mask contribution",
                "independent reachability-mask contribution",
            ],
            "failure_policy": "retain all failed, unsafe, stranded, and constraint-violating rows",
            "bug_policy": (
                "invalidate and rerun the complete affected algorithm-by-scenario matrix; never "
                "rerun only unfavorable outcomes"
            ),
        },
        "code_fingerprints": code_identity,
        "environment": environment_identity,
        "environment_sha256": environment_hash,
    }
    protocol["protocol_hash"] = compute_protocol_hash(protocol)
    _validate_protocol(protocol)
    return protocol


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if int(protocol.get("schema_version", -1)) != PROTOCOL_SCHEMA_VERSION:
        raise ProtocolError("不支持的正式测试协议版本。")
    if protocol.get("protocol_name") != PROTOCOL_NAME or protocol.get("immutable") is not True:
        raise ProtocolError("协议不是不可变的frozen_test_v1。")
    if str(protocol.get("protocol_hash", "")) != compute_protocol_hash(protocol):
        raise ProtocolError("protocol_hash校验失败，协议可能已被修改。")
    algorithms = dict(protocol.get("algorithms") or {})
    if tuple(algorithms.get("learning_variants", ())) != LEARNING_VARIANTS:
        raise ProtocolError("学习变体集合偏离frozen_test_v1。")
    if tuple(algorithms.get("main_baselines", ())) != MAIN_BASELINES:
        raise ProtocolError("主基线集合偏离frozen_test_v1。")
    if tuple(algorithms.get("supplementary_baselines", ())) != SUPPLEMENTARY_BASELINES:
        raise ProtocolError("补充基线集合偏离frozen_test_v1。")
    if tuple(dict(protocol.get("seeds") or {}).get("training", ())) != TRAINING_SEEDS:
        raise ProtocolError("训练种子偏离42--46。")
    if tuple(dict(protocol.get("seeds") or {}).get("stochastic_planner", ())) != PLANNER_SEEDS:
        raise ProtocolError("随机规划种子偏离42--51。")
    if dict(dict(protocol.get("manifest") or {}).get("split_counts") or {}) != SPLIT_COUNTS:
        raise ProtocolError("协议split数量偏离frozen_test_v1。")
    checkpoints = list(protocol.get("checkpoints") or [])
    identities = {
        (item.get("variant"), item.get("training_seed"))
        for item in checkpoints
        if isinstance(item, Mapping)
    }
    expected = {(variant, seed) for variant in LEARNING_VARIANTS for seed in TRAINING_SEEDS}
    if len(checkpoints) != 35 or identities != expected:
        raise ProtocolError("协议必须固定完整的35个best_safe检查点。")
    if dict(protocol.get("metrics") or {}).get("primary_metric") != "safe_weighted_coverage":
        raise ProtocolError("唯一主指标必须是safe_weighted_coverage。")
    environment_hash = _sha256_bytes(
        _canonical_json(dict(protocol.get("environment") or {})).encode("utf-8")
    )
    if str(protocol.get("environment_sha256", "")) != environment_hash:
        raise ProtocolError("environment_sha256与冻结环境元数据不一致。")


def write_frozen_protocol(
    protocol: Mapping[str, Any], destination: Union[Path, str]
) -> Path:
    """原子写入协议；同一位置只允许幂等读取，不允许覆盖不同内容。"""

    _validate_protocol(protocol)
    target = Path(destination)
    if target.suffix.lower() != ".json":
        target = target / PROTOCOL_FILENAME
    if target.exists():
        existing = load_frozen_protocol(target)
        if _canonical_json(existing) != _canonical_json(dict(protocol)):
            raise FileExistsError(f"冻结协议已存在且内容不同，拒绝覆盖：{target}")
        return target
    if target.parent.exists() and not target.parent.is_dir():
        raise FileExistsError(f"协议父路径不是目录：{target.parent}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(protocol), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_frozen_protocol(path: Union[Path, str]) -> Dict[str, Any]:
    target = Path(path)
    if target.is_dir() or target.suffix.lower() != ".json":
        target = target / PROTOCOL_FILENAME
    protocol = _read_json(target)
    _validate_protocol(protocol)
    return protocol


def verify_protocol_assets(
    protocol: Union[Mapping[str, Any], Path, str],
    repo_root: Optional[Union[Path, str]] = None,
    verify_checkpoints: bool = True,
    verify_code: bool = True,
) -> Dict[str, Any]:
    """在正式运行前重算冻结文件身份，阻止协议生成后的静默漂移。"""

    frozen = load_frozen_protocol(protocol) if isinstance(protocol, (str, Path)) else dict(protocol)
    _validate_protocol(frozen)
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent

    verified_code: List[str] = []
    if verify_code:
        for item in frozen.get("code_fingerprints", []):
            stored_path = Path(str(item.get("path", "")))
            path = stored_path if stored_path.is_absolute() else root / stored_path
            if not path.is_file():
                raise FileNotFoundError(f"冻结代码文件不存在：{path}")
            if path.stat().st_size != int(item.get("size_bytes", -1)) or sha256_file(
                path
            ) != str(item.get("sha256", "")):
                raise ProtocolError(f"冻结后代码文件发生漂移：{path}")
            verified_code.append(_portable_path(path, root))

    verified_checkpoints: List[str] = []
    if verify_checkpoints:
        checkpoints = list(frozen.get("checkpoints") or [])
        if len(checkpoints) != 35:
            raise ProtocolError("正式执行前必须能核验全部35个冻结检查点。")
        for item in checkpoints:
            stored_path = Path(str(item.get("path", "")))
            path = stored_path if stored_path.is_absolute() else root / stored_path
            if path.name != "best_safe.pt" or not path.is_file():
                raise FileNotFoundError(f"冻结best_safe检查点不存在：{path}")
            if path.stat().st_size != int(item.get("size_bytes", -1)) or sha256_file(
                path
            ) != str(item.get("sha256", "")):
                raise ProtocolError(f"冻结后检查点发生漂移：{path}")
            verified_checkpoints.append(_portable_path(path, root))

    return {
        "passed": True,
        "protocol_hash": str(frozen["protocol_hash"]),
        "code_verification": "passed" if verify_code else "skipped",
        "checkpoint_verification": "passed" if verify_checkpoints else "skipped",
        "verified_code_files": verified_code,
        "verified_checkpoints": verified_checkpoints,
    }


def freeze_protocol(
    training_root: Union[Path, str],
    manifest: ManifestInput,
    destination: Union[Path, str],
    **kwargs: Any,
) -> Path:
    """构建并不可覆盖地写入正式协议的便利入口。"""

    return write_frozen_protocol(
        build_frozen_protocol(training_root, manifest, **kwargs), destination
    )


def _canonical_float(value: Any) -> str:
    number = _finite_number(value, field="power_scale")
    return format(number, ".17g")


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in value).strip("._")


def _normalize_power_scales(power_scales: Iterable[Any]) -> Tuple[float, ...]:
    if isinstance(power_scales, str):
        values: Iterable[Any] = [item.strip() for item in power_scales.split(",") if item.strip()]
    else:
        values = power_scales
    result: List[float] = []
    keys = set()
    for value in values:
        number = _finite_number(value, field="power_scale")
        if number <= 0.0:
            raise ProtocolError("power_scale必须大于0。")
        key = _canonical_float(number)
        if key in keys:
            raise ProtocolError("power_scales不能重复。")
        keys.add(key)
        result.append(number)
    if not result:
        raise ProtocolError("power_scales不能为空。")
    return tuple(result)


def _family_algorithms(protocol: Mapping[str, Any], family: str) -> Tuple[str, ...]:
    algorithms = dict(protocol["algorithms"])
    named = {
        "learning": tuple(algorithms["learning_variants"]),
        "main_baselines": tuple(algorithms["main_baselines"]),
        "supplementary_baselines": tuple(algorithms["supplementary_baselines"]),
        "main": (
            "full",
            "ppo_mlp",
            "a2c_pointer",
            *tuple(algorithms["main_baselines"]),
        ),
        "primary_comparison": (
            "full",
            "ppo_mlp",
            "a2c_pointer",
            *tuple(algorithms["main_baselines"]),
        ),
        "ablation": ("full", *tuple(algorithms["ablation_variants"])),
        "supplementary": (
            "full",
            *tuple(algorithms["supplementary_baselines"]),
        ),
        "supplementary_exploratory": (
            "full",
            *tuple(algorithms["supplementary_baselines"]),
        ),
        "full_secondary": (
            *tuple(algorithms["learning_variants"]),
            *tuple(algorithms["main_baselines"]),
        ),
        "core_secondary": (
            *tuple(algorithms["core_learning_variants"]),
            *tuple(algorithms["main_baselines"]),
        ),
        "all": (
            *tuple(algorithms["learning_variants"]),
            *tuple(algorithms["main_baselines"]),
            *tuple(algorithms["supplementary_baselines"]),
        ),
    }
    all_algorithms = set(named["all"])
    if family in all_algorithms:
        return (family,)
    if family not in named:
        raise ProtocolError(f"未知审计family：{family}")
    return named[family]


def _selected_manifest_records(
    records: Sequence[Mapping[str, Any]], split: str
) -> List[Dict[str, Any]]:
    if split == "all":
        selected = [dict(item) for item in records]
    elif split == "scale":
        selected = [dict(item) for item in records if str(item.get("split", "")).startswith("scale_")]
    else:
        selected = [dict(item) for item in records if item.get("split") == split]
    if not selected:
        raise ProtocolError(f"清单中没有split={split!r}的场景。")
    return selected


def _seed_values(protocol: Mapping[str, Any], algorithm: str) -> Tuple[str, Tuple[int, ...]]:
    if algorithm in LEARNING_VARIANTS:
        return "training_seed", tuple(int(value) for value in protocol["seeds"]["training"])
    if algorithm in STOCHASTIC_BASELINES:
        return "planner_seed", tuple(
            int(value) for value in protocol["seeds"]["stochastic_planner"]
        )
    return "planner_seed", (
        int(protocol["seeds"]["deterministic_planner_identity"]),
    )


def _load_rows_file(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"结果文件不存在：{path}")
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = _strict_json_loads(line, location=f"{path}第{line_number}行")
        if not isinstance(item, dict):
            raise ProtocolError(f"{path}第{line_number}行必须是JSON对象。")
        rows.append(item)
    return rows


def _validate_result_directory(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    selected_records: Sequence[Mapping[str, Any]],
    requested_split: str,
) -> None:
    status = _read_json(path / "status.json")
    config = _read_json(path / "run_config.json")
    if status.get("state") != "completed":
        raise ProtocolError(f"结果目录尚未completed：{path}")
    if int(status.get("completed", -1)) != len(rows) or int(status.get("total", -1)) != len(rows):
        raise ProtocolError(f"{path}的status完成数/总数与results.jsonl行数不一致。")
    immutable = dict(config.get("immutable") or {})
    if config.get("kind") not in {"learning_evaluation", "traditional_baselines"}:
        raise ProtocolError(f"{path}的run_config.kind不是正式评估类型。")
    expected_identity = {
        "protocol_hash": str(protocol["protocol_hash"]),
        "manifest_hash": str(manifest_metadata["manifest_hash"]),
        "scenario_hash": str(manifest_metadata["base_scenario_hash"]),
        "selected_records_sha256": selected_records_hash(selected_records),
        "record_count": len(selected_records),
    }
    for field, expected in expected_identity.items():
        actual = immutable.get(field, config.get(field))
        if actual != expected:
            raise ProtocolError(
                f"{path}的run_config.{field}不一致：actual={actual!r}, expected={expected!r}"
            )
    configured_split = str(immutable.get("split", ""))
    if configured_split != requested_split:
        raise ProtocolError(
            f"{path}的run_config.split={configured_split!r}，预期{requested_split!r}。"
        )
    configured_powers = _normalize_power_scales(immutable.get("power_scales", ()))
    row_powers = tuple(
        float(value)
        for value in sorted(
            {_canonical_float(row.get("power_scale")) for row in rows},
            key=float,
        )
    )
    if {_canonical_float(value) for value in configured_powers} != {
        _canonical_float(value) for value in row_powers
    }:
        raise ProtocolError(f"{path}的run_config.power_scales与结果行不一致。")

    row_algorithms = {str(row.get("algorithm", "")) for row in rows}
    if config["kind"] == "learning_evaluation":
        if len(row_algorithms) != 1:
            raise ProtocolError(f"{path}的单个学习评估目录混入多个算法。")
        algorithm = next(iter(row_algorithms))
        seeds = {row.get("training_seed") for row in rows}
        if len(seeds) != 1:
            raise ProtocolError(f"{path}的单个学习评估目录混入多个训练种子。")
        seed = int(next(iter(seeds)))
        checkpoint_hashes = {str(row.get("checkpoint_hash", "")) for row in rows}
        expected = {
            str(item["sha256"])
            for item in protocol["checkpoints"]
            if item["variant"] == algorithm and int(item["training_seed"]) == seed
        }
        if immutable.get("variant") != algorithm or int(
            immutable.get("training_seed", -1)
        ) != seed:
            raise ProtocolError(f"{path}的run_config学习身份与结果行不一致。")
        if immutable.get("checkpoint_kind") != "best_safe":
            raise ProtocolError(f"{path}不是best_safe正式评估。")
        if {str(immutable.get("checkpoint_sha256", ""))} != expected or checkpoint_hashes != expected:
            raise ProtocolError(f"{path}的run_config检查点哈希与冻结协议不一致。")
    else:
        configured_algorithms = {str(value) for value in immutable.get("algorithms", ())}
        if configured_algorithms != row_algorithms:
            raise ProtocolError(f"{path}的run_config.algorithms与结果行不一致。")
        configured_seeds = {int(value) for value in immutable.get("planner_seeds", ())}
        used_seeds = {int(row["planner_seed"]) for row in rows}
        if not used_seeds.issubset(configured_seeds):
            raise ProtocolError(f"{path}使用了run_config以外的planner_seed。")
        configured_budgets = immutable.get("algorithm_budgets")
        if configured_budgets is not None:
            if not isinstance(configured_budgets, Mapping):
                raise ProtocolError(f"{path}的algorithm_budgets必须是映射。")
            for algorithm in row_algorithms:
                budget = dict(configured_budgets.get(algorithm) or {})
                if algorithm in STOCHASTIC_BASELINES:
                    if budget.get("max_evaluations") != 50_000 or budget.get(
                        "time_limit_s"
                    ) not in (None, ""):
                        raise ProtocolError(
                            f"{path}中{algorithm}必须完成50000次评价且无统一60秒截断。"
                        )
                elif algorithm in TIME_LIMITED_BASELINES:
                    if budget.get("max_evaluations") not in (None, "") or float(
                        budget.get("time_limit_s", -1.0)
                    ) != 60.0:
                        raise ProtocolError(f"{path}中{algorithm}必须使用60秒限时且无评价上限。")
        else:
            stochastic = row_algorithms & STOCHASTIC_BASELINES
            limited = row_algorithms & TIME_LIMITED_BASELINES
            if stochastic and limited:
                raise ProtocolError(f"{path}混合预算算法时必须记录algorithm_budgets。")
            if stochastic and (
                int(immutable.get("max_evaluations", -1)) != 50_000
                or immutable.get("time_limit_s") not in (None, "")
            ):
                raise ProtocolError(f"{path}的随机搜索预算偏离50000次且无统一时限。")
            if limited and (
                immutable.get("max_evaluations") not in (None, "")
                or float(immutable.get("time_limit_s", -1.0)) != 60.0
            ):
                raise ProtocolError(f"{path}的限时算法预算必须是60秒且无评价上限。")


def _validate_formal_row(row: Mapping[str, Any], *, location: str) -> None:
    missing = [field for field in (*FORMAL_METRIC_FIELDS, *PROVENANCE_FIELDS) if field not in row]
    if missing:
        raise ProtocolError(f"{location}缺少正式字段：{', '.join(missing)}")
    if int(row["schema_version"]) != RESULT_SCHEMA_VERSION:
        raise ProtocolError(
            f"{location}.schema_version必须为{RESULT_SCHEMA_VERSION}。"
        )
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ProtocolError(f"{location}.{key}包含NaN或Inf。")
    for field in (
        "returned",
        "energy_violation",
        "distance_violation",
        "time_violation",
        "dynamics_violation",
        "simulation_only",
    ):
        if not isinstance(row[field], bool):
            raise ProtocolError(f"{location}.{field}必须是布尔值。")
    if row["optimality_certified"] is not None and not isinstance(
        row["optimality_certified"], bool
    ):
        raise ProtocolError(f"{location}.optimality_certified必须是布尔值或null。")
    if not isinstance(row["termination_reason"], str) or not row["termination_reason"]:
        raise ProtocolError(f"{location}.termination_reason不能为空。")

    optional_numeric = {
        "low_priority_coverage",
        "medium_priority_coverage",
        "high_priority_coverage",
        "energy_budget_wh",
        "distance_budget_m",
        "time_budget_s",
        "energy_utilization",
        "distance_utilization",
        "time_utilization",
        "evaluations",
        "optimality_gap",
        "solver_dual_bound",
    }
    numeric_fields = {
        "weighted_coverage",
        "safe_weighted_coverage",
        "coverage",
        "safe_coverage",
        "visited_count",
        "energy_wh",
        "distance_m",
        "time_s",
        "min_remaining_soc",
        "planning_time_s",
        "replicate_id",
        "node_count",
        "power_scale",
        *optional_numeric,
    }
    for field in numeric_fields:
        if row[field] is None and field in optional_numeric:
            continue
        _finite_number(row[field], field=f"{location}.{field}")
    for field in (
        "weighted_coverage",
        "safe_weighted_coverage",
        "coverage",
        "safe_coverage",
        "low_priority_coverage",
        "medium_priority_coverage",
        "high_priority_coverage",
    ):
        if row[field] is None:
            continue
        value = float(row[field])
        if value < -1e-9 or value > 1.0 + 1e-9:
            raise ProtocolError(f"{location}.{field}必须位于[0, 1]。")
    for value_field, budget_field, utilization_field in (
        ("energy_wh", "energy_budget_wh", "energy_utilization"),
        ("distance_m", "distance_budget_m", "distance_utilization"),
        ("time_s", "time_budget_s", "time_utilization"),
    ):
        if row[budget_field] is None or row[utilization_field] is None:
            raise ProtocolError(
                f"{location}必须落盘{budget_field}和{utilization_field}，不得由图表反推。"
            )
        value = float(row[value_field])
        budget = float(row[budget_field])
        utilization = float(row[utilization_field])
        if value < 0.0 or budget <= 0.0 or utilization < 0.0:
            raise ProtocolError(f"{location}的资源值/预算/利用率范围无效。")
        if not math.isclose(utilization, value / budget, rel_tol=1e-9, abs_tol=1e-12):
            raise ProtocolError(f"{location}.{utilization_field}与统一评价器预算不一致。")
    if float(row["planning_time_s"]) < 0.0:
        raise ProtocolError(f"{location}.planning_time_s不能为负。")
    safe = bool(row["returned"]) and not any(
        bool(row[field])
        for field in (
            "energy_violation",
            "distance_violation",
            "time_violation",
            "dynamics_violation",
        )
    )
    expected_weighted = float(row["weighted_coverage"]) if safe else 0.0
    expected_coverage = float(row["coverage"]) if safe else 0.0
    if not math.isclose(float(row["safe_weighted_coverage"]), expected_weighted, abs_tol=1e-12):
        raise ProtocolError(f"{location}.safe_weighted_coverage不符合冻结安全定义。")
    if not math.isclose(float(row["safe_coverage"]), expected_coverage, abs_tol=1e-12):
        raise ProtocolError(f"{location}.safe_coverage不符合冻结安全定义。")


def _route_filename(row: Mapping[str, Any]) -> str:
    power = _safe_name(_canonical_float(row["power_scale"]))
    if str(row["algorithm"]) in LEARNING_VARIANTS:
        return f"{row['scenario_id']}__power{power}.json"
    return (
        f"{row['scenario_id']}__{row['algorithm']}__seed{int(row['planner_seed'])}"
        f"__power{power}.json"
    )


def audit_result_runs(
    protocol: Union[Mapping[str, Any], Path, str],
    manifest: ManifestInput,
    inputs: Union[ResultInput, Sequence[ResultInput]],
    family: str,
    split: str,
    power_scales: Iterable[Any] = (1.0,),
) -> Dict[str, Any]:
    """严格审计一组正式原始长表，成功时返回可落盘的摘要。

    ``family``可使用``learning``、``main_baselines``、``supplementary_baselines``、
    三个统计族、``full_secondary``/``core_secondary``，也可直接传单个算法名。
    """

    frozen = load_frozen_protocol(protocol) if isinstance(protocol, (str, Path)) else dict(protocol)
    _validate_protocol(frozen)
    metadata, records, _ = _coerce_manifest(manifest, require_frozen_counts=True)
    identity = dict(frozen["manifest"])
    for field, manifest_field in (
        ("manifest_hash", "manifest_hash"),
        ("base_scenario_hash", "base_scenario_hash"),
        ("records_sha256", "records_sha256"),
    ):
        if str(identity.get(field, "")) != str(metadata.get(manifest_field, "")):
            raise ProtocolError(f"审计清单的{manifest_field}与冻结协议不一致。")
    selected_records = _selected_manifest_records(records, split)
    records_by_id = {str(record["id"]): record for record in selected_records}
    algorithms = _family_algorithms(frozen, str(family))
    powers = _normalize_power_scales(power_scales)
    power_keys = {_canonical_float(value) for value in powers}

    raw_inputs: List[Any]
    if isinstance(inputs, (str, Path, Mapping)):
        raw_inputs = [inputs]
    else:
        raw_inputs = list(inputs)
    if not raw_inputs:
        raise ProtocolError("inputs不能为空。")

    sourced_rows: List[Tuple[Dict[str, Any], Optional[Path], str]] = []
    result_directories: List[Path] = []
    for input_index, item in enumerate(raw_inputs):
        if isinstance(item, Mapping):
            sourced_rows.append((dict(item), None, f"inputs[{input_index}]"))
            continue
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, Path)):
            for row_index, row in enumerate(item):
                if not isinstance(row, Mapping):
                    raise TypeError("内存结果序列只能包含映射对象。")
                sourced_rows.append(
                    (dict(row), None, f"inputs[{input_index}][{row_index}]")
                )
            continue
        path = Path(item)
        if path.is_dir():
            rows = _load_rows_file(path / "results.jsonl")
            _validate_result_directory(
                path,
                rows,
                frozen,
                metadata,
                selected_records,
                split,
            )
            result_directories.append(path)
            sourced_rows.extend(
                (row, path, f"{path / 'results.jsonl'}第{index}行")
                for index, row in enumerate(rows, start=1)
            )
        else:
            rows = _load_rows_file(path)
            sourced_rows.extend(
                (row, None, f"{path}第{index}行")
                for index, row in enumerate(rows, start=1)
            )

    expected_checkpoint_hash = {
        (str(item["variant"]), int(item["training_seed"])): str(item["sha256"])
        for item in frozen["checkpoints"]
    }
    expected_keys = set()
    for scenario_id in records_by_id:
        for algorithm in algorithms:
            seed_field, seeds = _seed_values(frozen, algorithm)
            del seed_field
            for seed in seeds:
                for power in powers:
                    expected_keys.add(
                        (scenario_id, algorithm, int(seed), _canonical_float(power))
                    )

    actual_keys = set()
    for row_index, (row, source_root, location) in enumerate(sourced_rows, start=1):
        _validate_formal_row(row, location=location)
        scenario_id = str(row["scenario_id"])
        algorithm = str(row["algorithm"])
        if scenario_id not in records_by_id:
            raise ProtocolError(f"{location}含协议外场景ID：{scenario_id}")
        record = records_by_id[scenario_id]
        if algorithm not in algorithms:
            raise ProtocolError(f"{location}含family外算法：{algorithm}")
        if str(row["split"]) != str(record["split"]):
            raise ProtocolError(f"{location}.split与清单场景不一致。")
        power_key = _canonical_float(row["power_scale"])
        if power_key not in power_keys:
            raise ProtocolError(f"{location}含协议外power_scale：{row['power_scale']}")
        if int(row["replicate_id"]) != int(record["replicate_id"]):
            raise ProtocolError(f"{location}.replicate_id与清单不一致。")
        if int(row["node_count"]) != int(record["node_count"]):
            raise ProtocolError(f"{location}.node_count与清单不一致。")
        visited_value = float(row["visited_count"])
        if not visited_value.is_integer() or not 0 <= int(visited_value) <= int(
            record["node_count"]
        ):
            raise ProtocolError(f"{location}.visited_count不是有效节点计数。")
        expected_coverage = int(visited_value) / int(record["node_count"])
        if not math.isclose(
            float(row["coverage"]), expected_coverage, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ProtocolError(f"{location}.coverage与visited_count/node_count不一致。")
        priority_fields = {
            1: "low_priority_coverage",
            2: "medium_priority_coverage",
            3: "high_priority_coverage",
        }
        priorities = [int(value) for value in record.get("priorities", [])]
        for level, field in priority_fields.items():
            if level in priorities and row[field] is None:
                raise ProtocolError(f"{location}.{field}缺失，无法审计分优先级覆盖。")
        if str(row["scenario_hash"]) != str(metadata["base_scenario_hash"]):
            raise ProtocolError(f"{location}.scenario_hash与冻结协议不一致。")
        if str(row["manifest_hash"]) != str(metadata["manifest_hash"]):
            raise ProtocolError(f"{location}.manifest_hash与冻结协议不一致。")
        if str(row["protocol_hash"]) != str(frozen["protocol_hash"]):
            raise ProtocolError(f"{location}.protocol_hash与冻结协议不一致。")

        seed_field, allowed_seeds = _seed_values(frozen, algorithm)
        raw_seed = row[seed_field]
        if raw_seed is None or isinstance(raw_seed, bool):
            raise ProtocolError(f"{location}.{seed_field}不能为空。")
        seed = int(raw_seed)
        if seed not in allowed_seeds:
            raise ProtocolError(f"{location}.{seed_field}不在冻结种子集合中。")
        if algorithm in LEARNING_VARIANTS:
            if str(row["variant"]) != algorithm or row["planner_seed"] not in (None, ""):
                raise ProtocolError(f"{location}的学习模型variant/planner_seed身份错误。")
            if bool(row["simulation_only"]) != (algorithm == "no_return_reserve"):
                raise ProtocolError(f"{location}.simulation_only与学习变体定义不一致。")
            expected_hash = expected_checkpoint_hash[(algorithm, seed)]
            if str(row["checkpoint_hash"]) != expected_hash:
                raise ProtocolError(f"{location}.checkpoint_hash与冻结检查点不一致。")
        else:
            if row["training_seed"] not in (None, "") or str(row["variant"]) not in ("", algorithm):
                raise ProtocolError(f"{location}的传统算法training_seed/variant身份错误。")
            if str(row["checkpoint_hash"]) != "":
                raise ProtocolError(f"{location}的传统算法不得携带checkpoint_hash。")
            if row["simulation_only"] is not False:
                raise ProtocolError(f"{location}的传统算法不得标记simulation_only。")

        key = (scenario_id, algorithm, seed, power_key)
        if key in actual_keys:
            raise ProtocolError(f"检测到重复正式结果键：{key}")
        actual_keys.add(key)
        if source_root is not None:
            route = source_root / "routes" / _route_filename(row)
            if not route.is_file():
                raise ProtocolError(f"正式结果缺少对应路线文件：{route}")

    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys
    if missing or unexpected:
        missing_sample = sorted(missing)[:5]
        unexpected_sample = sorted(unexpected)[:5]
        raise ProtocolError(
            "正式结果矩阵不完整："
            f"expected={len(expected_keys)}, actual={len(actual_keys)}, "
            f"missing_sample={missing_sample}, unexpected_sample={unexpected_sample}"
        )

    return {
        "schema_version": 1,
        "passed": True,
        "protocol_hash": str(frozen["protocol_hash"]),
        "manifest_hash": str(metadata["manifest_hash"]),
        "family": str(family),
        "split": str(split),
        "power_scales": list(powers),
        "algorithms": list(algorithms),
        "scenario_count": len(records_by_id),
        "row_count": len(actual_keys),
        "expected_row_count": len(expected_keys),
        "result_directories": [str(path.resolve()) for path in result_directories],
        "checks": [
            "exact_scenario_ids",
            "algorithm_and_seed_repeats",
            "checkpoint_manifest_scenario_protocol_hashes",
            "completed_status_and_run_config",
            "no_duplicate_or_nonfinite_rows",
            "formal_metric_fields",
            "route_files",
        ],
    }


__all__ = [
    "ABLATION_VARIANTS",
    "CORE_LEARNING_VARIANTS",
    "DETERMINISTIC_BASELINES",
    "FORMAL_METRIC_FIELDS",
    "LEARNING_VARIANTS",
    "MAIN_BASELINES",
    "PLANNER_SEEDS",
    "PROTOCOL_NAME",
    "PROTOCOL_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "ProtocolError",
    "REPRESENTATIVE_FIELDS",
    "SPLIT_COUNTS",
    "STOCHASTIC_BASELINES",
    "SUPPLEMENTARY_BASELINES",
    "TRAINING_SEEDS",
    "audit_result_runs",
    "build_frozen_protocol",
    "collect_environment_metadata",
    "compute_protocol_hash",
    "freeze_protocol",
    "load_frozen_protocol",
    "selected_records_hash",
    "select_representative_scenario",
    "sha256_file",
    "verify_protocol_assets",
    "write_frozen_protocol",
]

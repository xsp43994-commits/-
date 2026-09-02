"""论文级实验编排入口。

本模块只负责可复现实验的目录、清单、训练调用和原始长表落盘；统计检验与
论文图由 :mod:`paper_evaluation` 完成。所有重量级模块均懒加载，导入本文件
不会启动训练、创建日志或初始化 CUDA。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
PAPER_RUNS_ROOT = ROOT / "paper_runs"
DEFAULT_SCENARIO = ROOT / "scenario_data" / "mountain_road_16pt.npz"
DEFAULT_MANIFEST = PAPER_RUNS_ROOT / "manifests" / "frozen_v1"
DEFAULT_PROTOCOL = PAPER_RUNS_ROOT / "protocols" / "frozen_test_v1"
DEFAULT_MANIFEST_SEED = 20260720
MANIFEST_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 2

LEARNING_VARIANTS = (
    "full",
    "ppo_mlp",
    "a2c_pointer",
    "no_priority_bias",
    "no_domain_randomization",
    "no_resource_shaping",
    "no_return_reserve",
)
BASELINE_ALGORITHMS = (
    "nearest_feasible",
    "priority_resource_greedy",
    "aco",
    "ga",
    "sa",
    "milp_orienteering",
    "a_star",
    "pso",
    "exact_pareto_dp",
)
STOCHASTIC_BASELINES = frozenset({"aco", "ga", "sa", "pso"})
LONG_TABLE_FIELDS = (
    "scenario_id",
    "split",
    "algorithm",
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
    "variant",
    "training_seed",
    "planner_seed",
    "replicate_id",
    "checkpoint_hash",
    "scenario_hash",
    "manifest_hash",
    "evaluations",
    "optimality_gap",
    "solver_dual_bound",
    "solver_status",
    "optimality_certified",
    "node_count",
    "power_scale",
    "simulation_only",
    "protocol_hash",
)
SPLIT_COUNTS = {
    "validation": 64,
    "id_test": 100,
    "stress_test": 100,
    "scale_8": 25,
    "scale_12": 25,
    "scale_20": 25,
    "scale_24": 25,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(f"无法JSON序列化: {type(value).__name__}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
    )


class AtomicJsonlWriter:
    """每次整文件原子替换，避免断电留下半行JSON。"""

    def __init__(self, path: Path, *, resume: bool = False):
        self.path = path
        self.lines: List[str] = []
        if resume and path.exists():
            self.lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]

    def append(self, record: Mapping[str, Any]) -> None:
        self.lines.append(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
                allow_nan=False,
            )
        )
        _atomic_write_text(self.path, "\n".join(self.lines) + "\n")

    def records(self) -> List[Dict[str, Any]]:
        """解析当前原子快照；续跑前拒绝损坏或非对象JSON行。"""

        records: List[Dict[str, Any]] = []
        for line_number, line in enumerate(self.lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{self.path}第{line_number}行不是有效JSON，拒绝续跑。"
                ) from exc
            if not isinstance(record, dict):
                raise TypeError(f"{self.path}第{line_number}行必须是JSON对象。")
            records.append(record)
        return records


class DurableResultJsonlWriter:
    """面向数万结果行的追加写入器，并在续跑时修复唯一的尾部残行。"""

    def __init__(
        self,
        path: Path,
        *,
        resume: bool = False,
        repair_trailing: bool = True,
    ):
        self.path = path
        self._records: List[Dict[str, Any]] = []
        if resume and path.exists():
            self._recover_and_load(repair_trailing=repair_trailing)

    @staticmethod
    def _decode_record(line: bytes, *, location: str) -> Dict[str, Any]:
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{location}不是有效UTF-8 JSON对象。") from exc
        if not isinstance(record, dict):
            raise TypeError(f"{location}必须是JSON对象。")
        return record

    def _recover_and_load(self, *, repair_trailing: bool) -> None:
        payload = self.path.read_bytes()
        last_newline = payload.rfind(b"\n")
        complete_prefix = payload[: last_newline + 1] if last_newline >= 0 else b""
        trailing = payload[last_newline + 1 :] if last_newline >= 0 else payload
        for line_number, line in enumerate(complete_prefix.splitlines(), start=1):
            if line.strip():
                self._records.append(
                    self._decode_record(
                        line, location=f"{self.path}第{line_number}行"
                    )
                )
        if not trailing.strip():
            return
        try:
            trailing_record = self._decode_record(
                trailing, location=f"{self.path}末行"
            )
        except ValueError:
            if not repair_trailing:
                raise ValueError(
                    f"{self.path}存在损坏的尾部残行；dry-run严格只读，"
                    "请使用真实--resume-existing执行有证据保留的修复。"
                )
            # 只允许修复最后一行；残片另存证据后原子截回最后一个完整换行。
            quarantine = self.path.with_name(
                f"{self.path.name}.trailing_partial.{int(time.time())}.{os.getpid()}"
            )
            _atomic_write_bytes(quarantine, trailing)
            _atomic_write_bytes(self.path, complete_prefix)
            return
        self._records.append(trailing_record)
        if not repair_trailing:
            return
        with self.path.open("ab") as stream:
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def append(self, record: Mapping[str, Any]) -> None:
        line = (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
                allow_nan=False,
            )
            + "\n"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self._records.append(dict(record))

    def records(self) -> List[Dict[str, Any]]:
        return [dict(record) for record in self._records]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip()).strip("._")
    if not name:
        raise ValueError("运行名称不能为空。")
    return name


def _reserve_directory(path: Path, *, dry_run: bool) -> Path:
    if path.exists():
        raise FileExistsError(
            f"目录已存在，程序不会覆盖或删除：{path}。若是训练中断，请使用 resume。"
        )
    if not dry_run:
        path.mkdir(parents=True, exist_ok=False)
    return path


def _snapshot_scenario_files(source: Path, run_dir: Path) -> Path:
    """复制NPZ/JSON场景对，保证源目录变化后运行证据仍自包含。"""

    source = source.resolve()
    prefix = source.with_suffix("") if source.suffix.lower() in {".npz", ".json"} else source
    copied: Dict[str, Path] = {}
    for suffix in (".npz", ".json"):
        candidate = prefix.with_suffix(suffix)
        if candidate.exists():
            destination = run_dir / f"scenario_snapshot{suffix}"
            shutil.copy2(candidate, destination)
            copied[suffix] = destination
    if not copied:
        raise FileNotFoundError(f"场景快照源不存在：{source}")
    preferred_suffix = source.suffix.lower()
    return copied.get(preferred_suffix, copied.get(".npz", next(iter(copied.values()))))


def _save_learning_curve(returns: Sequence[float], destination: Path) -> None:
    """用纯SVG保存回报与50回合移动均值，避免训练依赖GUI绘图库。"""

    values = np.asarray(list(returns), dtype=np.float64)
    if not values.size:
        return
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("episode_returns含NaN/Inf，拒绝生成学习曲线。")
    width, height = 960.0, 520.0
    left, right, top, bottom = 72.0, 28.0, 48.0, 62.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    low, high = float(values.min()), float(values.max())
    if np.isclose(low, high):
        low -= 0.5
        high += 0.5

    def polyline(series: np.ndarray, start_episode: int) -> str:
        denominator = max(values.size - 1, 1)
        coordinates = []
        for offset, value in enumerate(series):
            episode_index = start_episode - 1 + offset
            x = left + plot_width * episode_index / denominator
            y = top + plot_height * (high - float(value)) / (high - low)
            coordinates.append(f"{x:.2f},{y:.2f}")
        return " ".join(coordinates)

    raw_line = polyline(values, 1)
    window = min(50, values.size)
    moving_element = ""
    if window >= 2:
        moving = np.convolve(values, np.ones(window) / window, mode="valid")
        moving_element = (
            f'<polyline points="{polyline(moving, window)}" fill="none" '
            'stroke="#D55E00" stroke-width="2.5"/>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="520" viewBox="0 0 960 520">
<rect width="100%" height="100%" fill="white"/>
<text x="480" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Learning return</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>
<polyline points="{raw_line}" fill="none" stroke="#0072B2" stroke-opacity="0.45" stroke-width="1.2"/>
{moving_element}
<text x="480" y="502" text-anchor="middle" font-family="sans-serif" font-size="14">Episode</text>
<text x="20" y="260" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 20 260)">Return</text>
<text x="64" y="53" text-anchor="end" font-family="monospace" font-size="12">{high:.3f}</text>
<text x="64" y="458" text-anchor="end" font-family="monospace" font-size="12">{low:.3f}</text>
</svg>
'''
    _atomic_write_text(destination, svg)


def _scenario_module() -> Any:
    return importlib.import_module("ppo_training_scenario")


def _ppo_module() -> Any:
    return importlib.import_module("final_python_ppo_pointer")


def _load_scenario(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"找不到固定场景：{path}")
    return _scenario_module().load_training_scenario(path)


def _seed_for(master_seed: int, scenario_hash: str, split: str, index: int) -> int:
    text = f"{master_seed}|{scenario_hash}|{split}|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")


def _domain_parameters(split: str, rng: np.random.Generator) -> Dict[str, float]:
    if split == "stress_test":
        return {
            "initial_soc": float(rng.uniform(0.75, 0.90)),
            "distance_budget_scale": float(rng.uniform(0.75, 0.90)),
            "time_budget_scale": float(rng.uniform(0.75, 0.90)),
            "wind_scale": float(rng.uniform(1.00, 1.20)),
            "wind_rotation_deg": float(rng.uniform(-15.0, 15.0)),
            "wind_vertical_bias_mps": float(rng.uniform(-1.0, 1.0)),
            "power_scale": 1.0,
        }
    return {
        "initial_soc": float(rng.uniform(0.80, 1.00)),
        "distance_budget_scale": float(rng.uniform(0.85, 1.00)),
        "time_budget_scale": float(rng.uniform(0.85, 1.00)),
        "wind_scale": float(rng.uniform(0.80, 1.20)),
        "wind_rotation_deg": float(rng.uniform(-15.0, 15.0)),
        "wind_vertical_bias_mps": float(rng.uniform(-1.0, 1.0)),
        "power_scale": 1.0,
    }


def _road_arms(scenario: Any) -> List[np.ndarray]:
    start = np.asarray(scenario.start_pos, dtype=np.float64)
    arms: List[np.ndarray] = []
    for road in (scenario.road_1, scenario.road_2):
        road_arr = np.asarray(road, dtype=np.float64)
        index = int(np.argmin(np.linalg.norm(road_arr[:, :2] - start[:2], axis=1)))
        for path in (road_arr[: index + 1][::-1], road_arr[index:]):
            if np.linalg.norm(path[0, :2] - start[:2]) > 1e-7:
                path = np.vstack([start, path])
            else:
                path = path.copy()
                path[0] = start
            arms.append(path)
    return arms


def _interpolate_path(path: np.ndarray, distances_m: np.ndarray, scale: float) -> np.ndarray:
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1) * scale)]
    )
    usable = np.minimum(distances_m, cumulative[-1])
    return np.column_stack(
        [np.interp(usable, cumulative, path[:, dimension]) for dimension in range(3)]
    )


def _priority_counts(node_count: int) -> Tuple[int, int, int]:
    raw = np.asarray([5.0, 6.0, 5.0]) * node_count / 16.0
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts), kind="stable")[: node_count - int(counts.sum())]:
        counts[index] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _scale_layout(scenario: Any, node_count: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if node_count % 4 != 0:
        raise ValueError("规模泛化节点数必须能被四条道路分支均分。")
    per_arm = node_count // 4
    scale = float(scenario.coordinate_scale_m_per_unit)
    points: List[np.ndarray] = []
    risks: List[float] = []
    base_arm = np.asarray(scenario.point_arm_ids)
    base_dist = np.asarray(scenario.point_along_arm_distances_m, dtype=float)
    base_risk = np.asarray(scenario.risk_scores, dtype=float)
    for arm_id, path in enumerate(_road_arms(scenario)):
        interval = 800.0 / per_arm
        distances = (np.arange(per_arm) + rng.uniform(0.30, 0.80, per_arm)) * interval
        points.extend(_interpolate_path(path, distances, scale))
        mask = base_arm == arm_id
        order = np.argsort(base_dist[mask])
        risks.extend(
            np.interp(distances, base_dist[mask][order], base_risk[mask][order]).tolist()
        )
    point_array = np.asarray(points, dtype=np.float32)
    risk_array = np.asarray(risks)
    high, medium, _low = _priority_counts(node_count)
    priorities = np.ones(node_count, dtype=np.int32)
    order = np.argsort(-risk_array, kind="stable")
    priorities[order[:high]] = 3
    priorities[order[high : high + medium]] = 2
    return point_array, priorities


def build_manifest_records(scenario: Any, master_seed: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    scenario_hash = str(scenario.scenario_hash)
    for split, count in SPLIT_COUNTS.items():
        node_count = int(split.split("_")[-1]) if split.startswith("scale_") else 16
        for index in range(count):
            instance_seed = _seed_for(master_seed, scenario_hash, split, index)
            rng = np.random.default_rng(instance_seed)
            if split.startswith("scale_"):
                points, priorities = _scale_layout(scenario, node_count, rng)
                service_times = np.full(node_count, 20.0, dtype=np.float32)
            else:
                points = np.asarray(scenario.inspection_points, dtype=np.float32)
                priorities = np.asarray(scenario.priorities, dtype=np.int32)
                service_times = np.asarray(scenario.service_times_s, dtype=np.float32)
            record: Dict[str, Any] = {
                "id": f"{split}_{index:03d}",
                "split": split,
                "replicate_id": index,
                "instance_seed": instance_seed,
                "node_count": node_count,
                "inspection_points_xyz": points.tolist(),
                "priorities": priorities.tolist(),
                "service_times_s": service_times.tolist(),
            }
            record.update(_domain_parameters(split, rng))
            records.append(record)
    return records


def _manifest_text(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
            allow_nan=False,
        )
        + "\n"
        for record in records
    )


def prepare_manifest(
    scenario_file: Path,
    manifest_root: Path,
    *,
    manifest_seed: int = DEFAULT_MANIFEST_SEED,
    dry_run: bool = False,
) -> Dict[str, Any]:
    scenario = _load_scenario(scenario_file)
    records = build_manifest_records(scenario, manifest_seed)
    records_text = _manifest_text(records)
    records_hash = hashlib.sha256(records_text.encode("utf-8")).hexdigest()
    metadata: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_by": "paper_experiments.prepare",
        "base_scenario_file": str(scenario_file.resolve()),
        "base_scenario_hash": str(scenario.scenario_hash),
        "manifest_seed": int(manifest_seed),
        "split_counts": dict(SPLIT_COUNTS),
        "records_file": "instances.jsonl",
        "records_sha256": records_hash,
        "training_seed_namespace": "model seeds 42-46; evaluation instances are never sampled as training episodes",
        "selection_integration_status": "external_fixed_v1",
    }
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    metadata["manifest_hash"] = hashlib.sha256((canonical + records_hash).encode("utf-8")).hexdigest()
    if dry_run:
        return metadata
    if manifest_root.exists():
        existing_meta = manifest_root / "manifest.json"
        existing_records = manifest_root / "instances.jsonl"
        if not existing_meta.exists() or not existing_records.exists():
            raise FileExistsError(f"清单目录已存在但不完整，拒绝覆盖：{manifest_root}")
        old = json.loads(existing_meta.read_text(encoding="utf-8"))
        if old.get("manifest_hash") != metadata["manifest_hash"] or _sha256_file(existing_records) != records_hash:
            raise FileExistsError(f"清单目录已存在且内容不同，拒绝覆盖：{manifest_root}")
        return old
    manifest_root.mkdir(parents=True, exist_ok=False)
    _atomic_write_text(manifest_root / "instances.jsonl", records_text)
    _write_json(manifest_root / "manifest.json", metadata)
    return metadata


def load_manifest(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Path]:
    root = path if path.is_dir() else path.parent
    metadata_path = root / "manifest.json"
    records_path = root / "instances.jsonl"
    if not metadata_path.exists() or not records_path.exists():
        raise FileNotFoundError(f"清单必须包含manifest.json和instances.jsonl：{root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("不支持的论文场景清单版本。")
    if _sha256_file(records_path) != metadata.get("records_sha256"):
        raise ValueError("论文场景清单哈希校验失败。")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != sum(int(value) for value in metadata["split_counts"].values()):
        raise ValueError("论文场景清单记录数量与元数据不一致。")
    return metadata, records, root


def _select_records(records: Sequence[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    if split == "all":
        return list(records)
    if split == "scale":
        return [record for record in records if record["split"].startswith("scale_")]
    selected = [record for record in records if record["split"] == split]
    if not selected:
        raise ValueError(f"清单中没有split={split!r}。")
    return selected


def _parse_power_scales(value: str) -> List[float]:
    """解析功率倍率，并拒绝非正数、NaN/Inf和重复值。"""

    scales: List[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        scale = float(item)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("--power-scales中的每个倍率都必须是有限正数。")
        if any(np.isclose(scale, existing, rtol=0.0, atol=1e-12) for existing in scales):
            raise ValueError("--power-scales不能包含重复倍率。")
        scales.append(scale)
    if not scales:
        raise ValueError("--power-scales至少需要一个倍率。")
    return scales


def _canonical_float(value: Any) -> str:
    """为续跑键和文件名提供稳定、无低精度舍入冲突的浮点表示。"""

    number = float(value)
    if not np.isfinite(number):
        raise ValueError("结果键中的浮点数必须有限。")
    return format(number, ".17g")


def _selected_records_hash(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_manifest_text(records).encode("utf-8")).hexdigest()


def _protocol_for_run(
    args: argparse.Namespace, manifest_metadata: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """加载并核对冻结协议；批处理可复用已完成资产校验的协议对象。"""

    cached = getattr(args, "_protocol_payload", None)
    protocol_path = getattr(args, "protocol", None)
    if cached is None and protocol_path is None:
        return None
    module = importlib.import_module("paper_protocol")
    protocol = (
        dict(cached)
        if cached is not None
        else module.load_frozen_protocol(Path(protocol_path))
    )
    identity = dict(protocol.get("manifest") or {})
    expected = {
        "manifest_hash": str(manifest_metadata["manifest_hash"]),
        "base_scenario_hash": str(manifest_metadata["base_scenario_hash"]),
        "records_sha256": str(manifest_metadata["records_sha256"]),
    }
    for field, value in expected.items():
        if str(identity.get(field, "")) != value:
            raise ValueError(f"冻结协议的{field}与评估清单不一致。")
    if not bool(getattr(args, "_protocol_assets_verified", False)):
        verifier = getattr(module, "verify_protocol_assets", None)
        if not callable(verifier):
            raise RuntimeError("paper_protocol缺少verify_protocol_assets资产校验入口。")
        verifier(protocol, repo_root=ROOT)
    return protocol


def _protocol_checkpoint(
    protocol: Mapping[str, Any], *, variant: str, training_seed: int
) -> Dict[str, Any]:
    matches = [
        dict(item)
        for item in protocol.get("checkpoints", ())
        if str(item.get("variant")) == str(variant)
        and int(item.get("training_seed", -1)) == int(training_seed)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"冻结协议中检查点身份必须唯一：variant={variant}, seed={training_seed}。"
        )
    return matches[0]


def _validate_checkpoint_provenance(
    payload: Mapping[str, Any],
    *,
    scenario_hash: str,
    manifest_hash: str,
) -> Dict[str, Any]:
    """论文评估只接受明确绑定当前场景和冻结清单的检查点。"""

    cfg = dict(payload.get("cfg") or {})
    stored_scenario_hash = str(cfg.get("scenario_hash", ""))
    stored_manifest_hash = str(cfg.get("paper_manifest_hash", ""))
    if not stored_scenario_hash:
        raise ValueError("检查点缺少scenario_hash，不能进入论文评估。")
    if stored_scenario_hash != str(scenario_hash):
        raise ValueError(
            "检查点场景哈希与冻结清单不一致："
            f"checkpoint={stored_scenario_hash}, expected={scenario_hash}。"
        )
    if not stored_manifest_hash:
        raise ValueError("检查点缺少paper_manifest_hash，不能进入论文评估。")
    if stored_manifest_hash != str(manifest_hash):
        raise ValueError(
            "检查点论文清单哈希不一致："
            f"checkpoint={stored_manifest_hash}, expected={manifest_hash}。"
        )
    return cfg


def _prepare_resumable_result_run(
    run_dir: Path,
    run_config: Mapping[str, Any],
    *,
    resume_existing: bool,
    dry_run: bool,
) -> None:
    """创建结果目录，或严格核对已有目录的不可变科学配置。"""

    config_path = run_dir / "run_config.json"
    if resume_existing:
        if not run_dir.is_dir() or not config_path.is_file():
            raise FileNotFoundError(
                f"--resume-existing要求已有目录和run_config.json：{run_dir}"
            )
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        if stored.get("schema_version") != run_config.get("schema_version"):
            raise ValueError("已有结果目录的run_config版本不兼容。")
        if stored.get("kind") != run_config.get("kind"):
            raise ValueError("已有结果目录的任务类型与当前命令不一致。")
        stored_immutable = dict(stored.get("immutable") or {})
        expected_immutable = dict(run_config.get("immutable") or {})
        mismatch_keys = sorted(
            key
            for key in set(stored_immutable) | set(expected_immutable)
            if stored_immutable.get(key) != expected_immutable.get(key)
        )
        if mismatch_keys:
            raise ValueError(
                "已有结果目录的不可变配置与当前命令不一致："
                + ", ".join(mismatch_keys)
            )
        return
    _reserve_directory(run_dir, dry_run=dry_run)
    if not dry_run:
        _write_json(config_path, run_config)


def _index_completed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_builder: Callable[[Mapping[str, Any]], Tuple[str, ...]],
    planned_keys: Set[Tuple[str, ...]],
    expected_provenance: Mapping[str, Any],
) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    """把JSONL行作为完成标记，并拒绝重复、越界或来源漂移的记录。"""

    completed: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for index, raw_row in enumerate(rows, start=1):
        row = dict(raw_row)
        for field, expected in expected_provenance.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"results.jsonl第{index}行的{field}与run_config不一致。"
                )
        key = key_builder(row)
        if key not in planned_keys:
            raise ValueError(f"results.jsonl第{index}行不属于当前不可变任务清单。")
        if key in completed:
            raise ValueError(f"results.jsonl包含重复完成键：{key!r}。")
        completed[key] = row
    return completed


def _training_cfg(
    ppo: Any,
    scenario: Any,
    run_dir: Path,
    args: argparse.Namespace,
    manifest_metadata: Mapping[str, Any],
    validation_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    # 只传场景事实和实验选择，让resolve_config自行补齐默认值并锁定变体结构。
    # 若先复制full默认配置，ppo_mlp会被误判为企图把其结构改回Pointer。
    cfg = copy.deepcopy(dict(scenario.as_training_inputs().get("cfg", {})))
    cfg.update(
        {
            "max_episodes": int(args.episodes),
            "seed": int(args.seed),
            "checkpoint_dir": str(run_dir),
            "experiment_variant": str(args.variant),
            "experiment_stage": str(args.stage),
            "run_id": run_dir.name,
            "scenario_hash": str(scenario.scenario_hash),
            "validation_scenarios": 64,
            "paper_manifest_hash": str(manifest_metadata["manifest_hash"]),
            "paper_validation_integration": "external_fixed_v1",
        }
    )
    # dry-run和run_config也必须展示变体锁定后的真实参数，而不是默认Pointer占位值。
    resolved = ppo.resolve_config(cfg) if hasattr(ppo, "resolve_config") else cfg
    normalized, digest = ppo.normalize_validation_instances(
        validation_records,
        np.asarray(scenario.inspection_points),
        np.asarray(scenario.priorities),
        resolved,
    )
    resolved.update(
        {
            "validation_instances": normalized,
            "validation_instances_hash": digest,
            "validation_scenarios": len(normalized),
            "validation_mode": "external_fixed_v1",
        }
    )
    return resolved


def _confirm_formal(args: argparse.Namespace) -> None:
    if int(args.episodes) < 3000 or bool(args.yes) or bool(args.dry_run):
        return
    if not sys.stdin.isatty():
        raise RuntimeError("3000回合正式训练需要交互确认；自动化调用请显式添加--yes。")
    answer = input(f"确认从头训练 {args.variant} / seed={args.seed} / {args.episodes}回合？[y/N] ")
    if answer.strip().lower() not in {"y", "yes"}:
        raise RuntimeError("用户取消正式训练。")


def _status_payload(state: str, **fields: Any) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "state": state,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **fields,
    }


def _verify_training_checkpoints(
    ppo: Any,
    scenario: Any,
    run_dir: Path,
    requested_device: str,
    expected_validation_hash: str,
) -> Dict[str, Any]:
    """独立重载检查点并重复推断，证明落盘文件可复现且标签真实。"""

    map_location = (
        getattr(ppo, "device", "cpu")
        if requested_device == "auto"
        else requested_device
    )
    reports: List[Dict[str, Any]] = []
    run_is_simulation_only: Optional[bool] = None
    for name, expected_kind in (
        ("latest.pt", "latest"),
        ("best_safe.pt", "best_safe"),
        ("best_candidate.pt", "best_candidate"),
    ):
        checkpoint = run_dir / name
        if not checkpoint.exists():
            continue
        model, payload = ppo.load_checkpoint(checkpoint, map_location=map_location)
        actual_kind = str(payload.get("checkpoint_kind", ""))
        if actual_kind != expected_kind:
            raise RuntimeError(f"{name}的checkpoint_kind={actual_kind!r}与文件名不符。")
        cfg = dict(payload["cfg"])
        simulation_only = bool(cfg.get("simulation_only", False))
        if run_is_simulation_only is None:
            run_is_simulation_only = simulation_only
        elif run_is_simulation_only != simulation_only:
            raise RuntimeError("同一训练目录中的检查点simulation_only标记不一致。")
        stored_validation_hash = str(cfg.get("validation_instances_hash", ""))
        if stored_validation_hash and stored_validation_hash != expected_validation_hash:
            raise RuntimeError(f"{name}使用了不同的冻结验证清单。")

        details = []
        for _ in range(2):
            details.append(
                ppo.plan_with_policy_improved(
                    model,
                    scenario.start_pos,
                    scenario.inspection_points,
                    scenario.priorities,
                    scenario.terrain,
                    cfg,
                    scenario.wind_data,
                    return_details=True,
                    decode_mode="deterministic",
                )
            )
        first_path = np.asarray(details[0]["path"], dtype=np.float64)
        second_path = np.asarray(details[1]["path"], dtype=np.float64)
        reproducible = bool(
            first_path.shape == second_path.shape
            and np.all(np.isfinite(first_path))
            and np.allclose(first_path, second_path, rtol=0.0, atol=1e-7)
        )
        first_metrics = dict(details[0].get("metrics") or {})
        second_metrics = dict(details[1].get("metrics") or {})
        first_reason = details[0].get(
            "termination_reason", first_metrics.get("termination_reason")
        )
        second_reason = details[1].get(
            "termination_reason", second_metrics.get("termination_reason")
        )
        reproducible = reproducible and (
            details[0].get("visit_order") == details[1].get("visit_order")
            and first_reason == second_reason
        )
        if not reproducible:
            raise RuntimeError(f"{name}两次确定性推断结果不一致。")
        safe = bool(first_metrics.get("returned")) and not any(
            bool(first_metrics.get(field, False))
            for field in (
                "energy_violation",
                "distance_violation",
                "time_violation",
                "dynamics_violation",
            )
        )
        if expected_kind == "best_safe" and not safe:
            raise RuntimeError("best_safe.pt独立重载后不是安全返航路线。")
        rng_state = dict(payload.get("rng_state") or {})
        training_state = dict(payload.get("training_state") or {})
        training_summary = dict(payload.get("training_summary") or {})
        returns = list(payload.get("returns") or [])
        episodes_seen = int(
            training_state.get(
                "episodes_seen", training_summary.get("episodes_seen", len(returns))
            )
        )
        resumable = bool(
            payload.get("optimizer_state_dict") is not None
            and {"python", "numpy_global", "torch", "training_generator"}.issubset(
                rng_state
            )
            and rng_state.get("training_generator") is not None
            and episodes_seen == len(returns)
        )
        if expected_kind == "latest" and not resumable:
            raise RuntimeError("latest.pt缺少优化器、完整RNG或一致的回合历史，不能安全续训。")
        reports.append(
            {
                "file": name,
                "sha256": _sha256_file(checkpoint),
                "checkpoint_kind": actual_kind or expected_kind,
                "deterministic_reproducible": reproducible,
                "resumable": resumable if expected_kind == "latest" else None,
                "safe": safe,
                "termination_reason": first_reason,
            }
        )
        del model
    if not any(item["file"] == "latest.pt" for item in reports):
        raise RuntimeError("训练目录缺少可恢复的latest.pt。")
    safe_checkpoint_available = any(
        item["file"] == "best_safe.pt" and bool(item["safe"]) for item in reports
    )
    if not bool(run_is_simulation_only) and not safe_checkpoint_available:
        raise RuntimeError("完整硬约束训练必须包含独立重载后安全的best_safe.pt。")
    return {
        "passed": True,
        "latest_reproducible_and_resumable": True,
        "safe_checkpoint_available": safe_checkpoint_available,
        "simulation_only": bool(run_is_simulation_only),
        "checkpoints": reports,
    }


def _run_training(args: argparse.Namespace, *, resume: bool) -> Path:
    ppo = _ppo_module()
    scenario = _load_scenario(args.scenario_file)
    if resume:
        run_dir = args.run_dir.resolve()
        config_path = run_dir / "run_config.json"
        latest = run_dir / "latest.pt"
        if not config_path.exists() or not latest.exists():
            raise FileNotFoundError("续训目录必须同时包含run_config.json和latest.pt。")
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest_snapshot = run_dir / "manifest_snapshot.json"
        validation_snapshot = run_dir / "validation_instances.jsonl"
        if not manifest_snapshot.exists() or not validation_snapshot.exists():
            raise FileNotFoundError("续训目录缺少冻结验证清单快照，拒绝重新抽样。")
        manifest_metadata = json.loads(manifest_snapshot.read_text(encoding="utf-8"))
        validation_records = [
            json.loads(line)
            for line in validation_snapshot.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if str(scenario.scenario_hash) != str(manifest_metadata["base_scenario_hash"]):
            raise ValueError("续训场景哈希与运行目录中的冻结清单不一致。")
        if len(validation_records) != SPLIT_COUNTS["validation"]:
            raise ValueError("续训快照必须恰好包含64个冻结validation实例。")
        if str(run_config.get("manifest_hash", "")) != str(
            manifest_metadata.get("manifest_hash", "")
        ):
            raise ValueError("续训目录中的manifest哈希不一致。")
        resume_model, resume_payload = ppo.load_checkpoint(latest, map_location="cpu")
        if str(resume_payload.get("checkpoint_kind", "")) != "latest":
            raise ValueError("续训目录中的latest.pt未标记为可恢复检查点。")
        resume_state = dict(resume_payload.get("training_state") or {})
        resume_summary = dict(resume_payload.get("training_summary") or {})
        resume_returns = list(resume_payload.get("returns") or [])
        process_start_episodes = int(
            resume_state.get(
                "episodes_seen",
                resume_summary.get("episodes_seen", len(resume_returns)),
            )
        )
        del resume_model
        cfg = dict(run_config["training_config"])
        cfg["max_episodes"] = int(args.episodes)
        # checkpoint_dir是运行时落盘位置；目录移动后必须跟随当前--run-dir。
        cfg["checkpoint_dir"] = str(run_dir)
        if int(args.episodes) <= 0:
            raise ValueError("resume的累计目标回合数必须大于0。")
        run_config["target_episodes"] = int(args.episodes)
        run_config["training_config"] = cfg
        resume_from: Optional[Path] = latest
        metrics_writer = AtomicJsonlWriter(run_dir / "metrics.jsonl", resume=True)
    else:
        _confirm_formal(args)
        manifest_metadata, manifest_records, manifest_root = load_manifest(args.manifest)
        if str(scenario.scenario_hash) != str(manifest_metadata["base_scenario_hash"]):
            raise ValueError("训练场景哈希与冻结论文清单不一致。")
        validation_records = _select_records(manifest_records, "validation")
        if len(validation_records) != SPLIT_COUNTS["validation"]:
            raise ValueError("正式选模必须恰好使用64个冻结validation实例。")
        name = args.run_name or f"{args.stage}_{args.variant}_seed{args.seed}_{args.episodes}ep"
        run_dir = PAPER_RUNS_ROOT / "training" / _safe_name(name)
        _reserve_directory(run_dir, dry_run=args.dry_run)
        cfg = _training_cfg(
            ppo, scenario, run_dir, args, manifest_metadata, validation_records
        )
        run_config = {
            "schema_version": 1,
            "kind": "learning_training",
            "variant": args.variant,
            "training_seed": int(args.seed),
            "target_episodes": int(args.episodes),
            "scenario_file": str(args.scenario_file.resolve()),
            "scenario_snapshot_file": (
                "scenario_snapshot.json"
                if args.scenario_file.suffix.lower() == ".json"
                else "scenario_snapshot.npz"
            ),
            "scenario_hash": str(scenario.scenario_hash),
            "manifest_hash": str(manifest_metadata["manifest_hash"]),
            "manifest_source": str(manifest_root.resolve()),
            "validation_instances_hash": str(cfg["validation_instances_hash"]),
            "validation_instance_count": len(validation_records),
            "device": args.device,
            "training_config": cfg,
        }
        resume_from = None
        process_start_episodes = 0
        metrics_writer = AtomicJsonlWriter(run_dir / "metrics.jsonl")
    if args.dry_run:
        preview = copy.deepcopy(run_config)
        preview_cfg = dict(preview.get("training_config") or {})
        if preview_cfg.get("validation_instances") is not None:
            preview_cfg["validation_instances"] = (
                f"<{len(preview_cfg['validation_instances'])} frozen instances; "
                f"hash={preview_cfg.get('validation_instances_hash', '')}>"
            )
        preview["training_config"] = preview_cfg
        print(
            json.dumps(
                {
                    "action": "resume" if resume else "train",
                    "run_dir": str(run_dir),
                    "config": preview,
                },
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
        )
        return run_dir
    run_started = time.perf_counter()
    if not resume:
        snapshot_path = _snapshot_scenario_files(args.scenario_file, run_dir)
        run_config["scenario_snapshot_file"] = snapshot_path.name
        _write_json(run_dir / "manifest_snapshot.json", manifest_metadata)
        _atomic_write_text(
            run_dir / "validation_instances.jsonl", _manifest_text(validation_records)
        )
    _write_json(run_dir / "run_config.json", run_config)
    _write_json(
        run_dir / "status.json",
        _status_payload("running", target_episodes=args.episodes, process_id=os.getpid()),
    )

    def callback(record: Mapping[str, Any]) -> None:
        metrics_writer.append(record)
        elapsed = time.perf_counter() - run_started
        episodes_seen = int(record.get("episodes_seen") or 0)
        episodes_completed_this_process = max(
            0, episodes_seen - process_start_episodes
        )
        eta = (
            elapsed
            / episodes_completed_this_process
            * max(0, int(args.episodes) - episodes_seen)
            if episodes_completed_this_process > 0
            else None
        )
        _write_json(
            run_dir / "status.json",
            _status_payload(
                "running",
                target_episodes=args.episodes,
                latest_update=record.get("update"),
                episodes_seen=episodes_seen,
                session_start_episodes=process_start_episodes,
                session_completed_episodes=episodes_completed_this_process,
                elapsed_s=elapsed,
                estimated_remaining_s=eta,
                process_id=os.getpid(),
                latest_metrics=dict(record),
            ),
        )

    try:
        if hasattr(ppo, "setup_logging"):
            ppo.setup_logging(run_dir / "logs")
        model, returns = ppo.train_policy_improved(
            scenario.start_pos,
            scenario.inspection_points,
            scenario.priorities,
            scenario.terrain,
            cfg,
            scenario.wind_data,
            resume_from=resume_from,
            metrics_callback=callback,
            target_device=args.device,
            validation_instances=validation_records,
        )
        del model
        checkpoint_verification = _verify_training_checkpoints(
            ppo,
            scenario,
            run_dir,
            args.device,
            str(cfg["validation_instances_hash"]),
        )
        _write_json(
            run_dir / "checkpoint_verification.json", checkpoint_verification
        )
        best_safe_path = run_dir / "best_safe.pt"
        best_candidate_path = run_dir / "best_candidate.pt"
        selected_checkpoint = (
            best_safe_path if best_safe_path.exists() else best_candidate_path
        )
        if not selected_checkpoint.exists():
            raise RuntimeError("训练结束后既没有best_safe.pt，也没有best_candidate.pt。")
        _write_json(run_dir / "episode_returns.json", {"returns": [float(v) for v in returns]})
        _save_learning_curve(returns, run_dir / "learning_curve.svg")
        _write_json(
            run_dir / "status.json",
            _status_payload(
                "completed",
                target_episodes=args.episodes,
                episodes_seen=len(returns),
                latest_checkpoint=str(run_dir / "latest.pt"),
                best_checkpoint=str(selected_checkpoint),
                best_safe_available=best_safe_path.exists(),
                checkpoint_verification_passed=bool(
                    checkpoint_verification["passed"]
                ),
                wall_time_s=time.perf_counter() - run_started,
            ),
        )
    except Exception as exc:
        _write_json(run_dir / "status.json", _status_payload("failed", error=f"{type(exc).__name__}: {exc}"))
        raise
    return run_dir


def _transform_wind(
    wind_data: Optional[Mapping[str, Any]], record: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """复用PPO内核的冻结风场变换，防止训练选模和独立测试语义漂移。"""

    return _ppo_module().transform_wind_for_domain_instance(wind_data, record)


def _instance_cfg(base_cfg: Mapping[str, Any], scenario: Any, record: Mapping[str, Any], power_scale: float) -> Dict[str, Any]:
    ppo = _ppo_module()
    nominal_cfg = copy.deepcopy(dict(base_cfg))
    nominal_cfg.update(dict(scenario.as_training_inputs().get("cfg", {})))
    condition = dict(record)
    condition["power_scale"] = float(power_scale)
    cfg, _ = ppo.apply_frozen_domain_instance(
        nominal_cfg, scenario.wind_data, condition
    )
    cfg["seed"] = int(record["instance_seed"])
    return cfg


def _optional_finite_float(value: Any, *, field: str) -> Optional[float]:
    if value in (None, ""):
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{field}必须是有限数或空值。")
    return number


def _priority_coverages(
    record: Mapping[str, Any], metrics: Mapping[str, Any]
) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
    priorities = np.asarray(record.get("priorities", []), dtype=np.int64).reshape(-1)
    raw_order = metrics.get("visited_order", metrics.get("visit_order"))
    if raw_order is None:
        count = metrics.get("visited_count")
        return (None if count in (None, "") else int(count), None, None, None)
    order = tuple(int(value) for value in raw_order)
    if len(set(order)) != len(order) or any(
        value < 0 or value >= priorities.size for value in order
    ):
        raise ValueError("原始访问索引含重复或越界节点，拒绝生成正式指标。")
    visited = set(order)
    coverage_by_priority: Dict[int, Optional[float]] = {}
    for level in (1, 2, 3):
        members = [index for index, value in enumerate(priorities) if value == level]
        coverage_by_priority[level] = (
            None
            if not members
            else sum(index in visited for index in members) / len(members)
        )
    return (
        len(order),
        coverage_by_priority[1],
        coverage_by_priority[2],
        coverage_by_priority[3],
    )


def _long_row(
    *,
    record: Mapping[str, Any],
    algorithm: str,
    metrics: Mapping[str, Any],
    planning_time_s: float,
    checkpoint_hash: str = "",
    training_seed: Optional[int] = None,
    planner_seed: Optional[int] = None,
    variant: str = "",
    evaluations: Optional[int] = None,
    optimality_gap: Optional[float] = None,
    solver_dual_bound: Optional[float] = None,
    solver_status: str = "",
    optimality_certified: Optional[bool] = None,
    simulation_only: bool = False,
    protocol_hash: str = "",
    scenario_hash: str,
    manifest_hash: str,
    power_scale: float,
) -> Dict[str, Any]:
    returned = bool(metrics.get("returned", False))
    energy_violation = bool(
        metrics.get("energy_violation", float(metrics.get("energy_utilization", 0.0)) > 1.0 + 1e-9)
    )
    distance_violation = bool(
        metrics.get("distance_violation", float(metrics.get("distance_utilization", 0.0)) > 1.0 + 1e-9)
    )
    time_violation = bool(
        metrics.get("time_violation", float(metrics.get("time_utilization", 0.0)) > 1.0 + 1e-9)
    )
    dynamics_violation = bool(metrics.get("dynamics_violation", False))
    termination_reason = str(metrics.get("termination_reason", ""))
    # 与统计管线保持一致：返航且资源/动力学均无违规才计入安全覆盖。
    safe = returned and not (
        energy_violation or distance_violation or time_violation or dynamics_violation
    )
    coverage = float(metrics.get("coverage", 0.0))
    weighted_coverage = float(metrics.get("weighted_coverage", 0.0))
    visited_count, low_coverage, medium_coverage, high_coverage = _priority_coverages(
        record, metrics
    )
    energy_wh = float(
        metrics.get("energy_wh", metrics.get("total_energy_consumed", 0.0))
    )
    distance_m = float(
        metrics.get("distance_m", metrics.get("total_distance", 0.0))
    )
    time_s = float(metrics.get("time_s", metrics.get("total_time_s", 0.0)))
    energy_budget_wh = _optional_finite_float(
        metrics.get("energy_budget_wh"), field="energy_budget_wh"
    )
    distance_budget_m = _optional_finite_float(
        metrics.get("distance_budget_m"), field="distance_budget_m"
    )
    time_budget_s = _optional_finite_float(
        metrics.get("time_budget_s"), field="time_budget_s"
    )

    def utilization(name: str, value: float, budget: Optional[float]) -> Optional[float]:
        raw = _optional_finite_float(metrics.get(name), field=name)
        if raw is not None:
            return raw
        return None if budget is None or budget <= 0.0 else value / budget

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "scenario_id": record["id"],
        "split": record["split"],
        "algorithm": algorithm,
        "returned": returned,
        "energy_violation": energy_violation,
        "distance_violation": distance_violation,
        "time_violation": time_violation,
        "dynamics_violation": dynamics_violation,
        "termination_reason": termination_reason,
        "weighted_coverage": weighted_coverage,
        "safe_weighted_coverage": weighted_coverage if safe else 0.0,
        "coverage": coverage,
        "safe_coverage": coverage if safe else 0.0,
        "visited_count": visited_count,
        "low_priority_coverage": low_coverage,
        "medium_priority_coverage": medium_coverage,
        "high_priority_coverage": high_coverage,
        "energy_wh": energy_wh,
        "distance_m": distance_m,
        "time_s": time_s,
        "energy_budget_wh": energy_budget_wh,
        "distance_budget_m": distance_budget_m,
        "time_budget_s": time_budget_s,
        "energy_utilization": utilization(
            "energy_utilization", energy_wh, energy_budget_wh
        ),
        "distance_utilization": utilization(
            "distance_utilization", distance_m, distance_budget_m
        ),
        "time_utilization": utilization("time_utilization", time_s, time_budget_s),
        "min_remaining_soc": float(metrics.get("min_remaining_soc", 0.0)),
        "planning_time_s": float(planning_time_s),
        "variant": variant,
        "training_seed": training_seed,
        "planner_seed": planner_seed,
        "replicate_id": int(record["replicate_id"]),
        "checkpoint_hash": checkpoint_hash,
        "scenario_hash": scenario_hash,
        "manifest_hash": manifest_hash,
        "evaluations": evaluations,
        "optimality_gap": optimality_gap,
        "solver_dual_bound": solver_dual_bound,
        "solver_status": str(solver_status),
        "optimality_certified": optimality_certified,
        "node_count": int(record["node_count"]),
        "power_scale": float(power_scale),
        "simulation_only": bool(simulation_only),
        "protocol_hash": str(protocol_hash),
    }


def _write_long_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=sorted(set(LONG_TABLE_FIELDS) | {"schema_version"}),
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in writer.fieldnames})
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _evaluate(args: argparse.Namespace) -> Path:
    metadata, records, manifest_root = load_manifest(args.manifest)
    records = _select_records(records, args.split)
    protocol = _protocol_for_run(args, metadata)
    protocol_hash = "" if protocol is None else str(protocol["protocol_hash"])
    ppo = _ppo_module()
    scenario = _load_scenario(args.scenario_file)
    if str(scenario.scenario_hash) != str(metadata["base_scenario_hash"]):
        raise ValueError("场景哈希与冻结清单不一致。")
    checkpoint = args.checkpoint.resolve()
    checkpoint_hash = _sha256_file(checkpoint)
    power_scales = _parse_power_scales(args.power_scales)
    map_location = getattr(ppo, "device", "cpu") if args.device == "auto" else args.device
    model, payload = ppo.load_checkpoint(checkpoint, map_location=map_location)
    base_cfg = _validate_checkpoint_provenance(
        payload,
        scenario_hash=str(scenario.scenario_hash),
        manifest_hash=str(metadata["manifest_hash"]),
    )
    variant = str(base_cfg.get("experiment_variant", "full"))
    training_seed = int(payload.get("seed", base_cfg.get("seed", -1)))
    if protocol is not None:
        frozen_checkpoint = _protocol_checkpoint(
            protocol, variant=variant, training_seed=training_seed
        )
        if checkpoint_hash != str(frozen_checkpoint["sha256"]):
            raise ValueError("评估检查点SHA-256与冻结协议不一致。")
        if str(payload.get("checkpoint_kind", "")) != "best_safe":
            raise ValueError("正式协议评估只允许best_safe检查点。")
    default_name = f"eval_{checkpoint.parent.name}_{checkpoint.stem}_{args.split}"
    run_dir = PAPER_RUNS_ROOT / "evaluation" / _safe_name(
        args.run_name or default_name
    )
    tasks = [
        (power_scale, record)
        for power_scale in power_scales
        for record in records
    ]
    planned_keys = {
        (str(record["id"]), _canonical_float(power_scale))
        for power_scale, record in tasks
    }
    if len(planned_keys) != len(tasks):
        raise ValueError("评估任务清单包含重复完成键。")
    run_config = {
        "schema_version": 2,
        "kind": "learning_evaluation",
        "immutable": {
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_kind": str(payload.get("checkpoint_kind", "")),
            "variant": variant,
            "training_seed": training_seed,
            "scenario_hash": str(scenario.scenario_hash),
            "manifest_hash": str(metadata["manifest_hash"]),
            "protocol_hash": protocol_hash,
            "selected_records_sha256": _selected_records_hash(records),
            "split": str(args.split),
            "record_count": len(records),
            "power_scales": power_scales,
            "device": str(map_location),
        },
        "runtime": {
            "checkpoint_file": str(checkpoint),
            "scenario_file": str(args.scenario_file.resolve()),
            "manifest_source": str(manifest_root.resolve()),
        },
    }
    _prepare_resumable_result_run(
        run_dir,
        run_config,
        resume_existing=bool(args.resume_existing),
        dry_run=bool(args.dry_run),
    )
    writer = DurableResultJsonlWriter(
        run_dir / "results.jsonl",
        resume=bool(args.resume_existing),
        repair_trailing=not bool(args.dry_run),
    )
    rows = writer.records()
    completed = _index_completed_rows(
        rows,
        key_builder=lambda row: (
            str(row["scenario_id"]),
            _canonical_float(row["power_scale"]),
        ),
        planned_keys=planned_keys,
        expected_provenance={
            "algorithm": variant,
            "variant": variant,
            "training_seed": training_seed,
            "checkpoint_hash": checkpoint_hash,
            "scenario_hash": str(scenario.scenario_hash),
            "manifest_hash": str(metadata["manifest_hash"]),
            "protocol_hash": protocol_hash,
        },
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "action": "resume_evaluate" if args.resume_existing else "evaluate",
                    "instances": len(tasks),
                    "completed": len(completed),
                    "remaining": len(tasks) - len(completed),
                    "protocol_hash": protocol_hash,
                    "run_dir": str(run_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return run_dir
    _write_json(
        run_dir / "status.json",
        _status_payload("running", completed=len(completed), total=len(tasks)),
    )
    torch_module = getattr(ppo, "torch", None)

    def synchronize_cuda() -> None:
        device_type = getattr(map_location, "type", str(map_location).split(":")[0])
        if (
            str(device_type) == "cuda"
            and torch_module is not None
            and torch_module.cuda.is_available()
        ):
            torch_module.cuda.synchronize()

    remaining_tasks = [
        (power_scale, record)
        for power_scale, record in tasks
        if (str(record["id"]), _canonical_float(power_scale)) not in completed
    ]
    try:
        if remaining_tasks:
            # 正式计时前只为本次尚未完成的首项预热，避免重复任务污染计时。
            first_power_scale, first_record = remaining_tasks[0]
            first_cfg = _instance_cfg(
                base_cfg, scenario, first_record, first_power_scale
            )
            ppo.plan_with_policy_improved(
                model,
                scenario.start_pos,
                np.asarray(first_record["inspection_points_xyz"], dtype=np.float32),
                np.asarray(first_record["priorities"], dtype=np.float32),
                scenario.terrain,
                first_cfg,
                _transform_wind(scenario.wind_data, first_record),
                return_details=True,
                decode_mode="deterministic",
            )
            synchronize_cuda()
        for power_scale, record in remaining_tasks:
            cfg = _instance_cfg(base_cfg, scenario, record, power_scale)
            wind_data = _transform_wind(scenario.wind_data, record)
            synchronize_cuda()
            started = time.perf_counter()
            detail = ppo.plan_with_policy_improved(
                model,
                scenario.start_pos,
                np.asarray(record["inspection_points_xyz"], dtype=np.float32),
                np.asarray(record["priorities"], dtype=np.float32),
                scenario.terrain,
                cfg,
                wind_data,
                return_details=True,
                decode_mode="deterministic",
            )
            synchronize_cuda()
            elapsed = time.perf_counter() - started
            metrics = dict(detail.get("metrics", detail))
            row = _long_row(
                record=record,
                algorithm=variant,
                metrics=metrics,
                planning_time_s=elapsed,
                checkpoint_hash=checkpoint_hash,
                training_seed=training_seed,
                variant=variant,
                simulation_only=bool(base_cfg.get("simulation_only", False)),
                scenario_hash=str(scenario.scenario_hash),
                manifest_hash=str(metadata["manifest_hash"]),
                protocol_hash=protocol_hash,
                power_scale=power_scale,
            )
            route_name = (
                f"{record['id']}__power{_safe_name(_canonical_float(power_scale))}.json"
            )
            # 路线先落盘，JSONL完成标记后落盘；断电恢复不会跳过缺路线的任务。
            _write_json(run_dir / "routes" / route_name, {"record": record, "detail": detail, "row": row})
            writer.append(row)
            rows.append(row)
            key = (str(record["id"]), _canonical_float(power_scale))
            completed[key] = row
            _write_json(
                run_dir / "status.json",
                _status_payload("running", completed=len(completed), total=len(tasks)),
            )
        _write_long_csv(run_dir / "results.csv", rows)
        _write_json(
            run_dir / "status.json",
            _status_payload("completed", completed=len(rows), total=len(tasks)),
        )
    except Exception as exc:
        _write_json(
            run_dir / "status.json",
            _status_payload(
                "failed",
                completed=len(completed),
                total=len(tasks),
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise
    return run_dir


def _baseline_metrics(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = raw.get("metrics", raw)
    if not isinstance(metrics, Mapping):
        raise TypeError("传统算法PlanningResult.metrics必须是映射。")
    return metrics


def _baseline_plan(
    args: argparse.Namespace, package: Any
) -> Tuple[List[str], List[int], Dict[str, Dict[str, Optional[float]]], Set[str]]:
    """从统一规划器注册表解析算法、随机种子和逐算法停止预算。"""

    requested = str(getattr(args, "algorithms", "") or "").strip()
    profile = str(getattr(args, "profile", "main")).strip().lower()
    if requested:
        algorithms = [value.strip() for value in requested.split(",") if value.strip()]
        selection_mode = "custom"
    else:
        if profile == "custom":
            raise ValueError("--profile custom必须同时提供--algorithms。")
        planner_names = getattr(package, "planner_names", None)
        if not callable(planner_names):
            raise RuntimeError("传统算法包缺少planner_names统一注册表入口。")
        algorithms = list(planner_names(profile))
        selection_mode = profile
    if not algorithms or len(set(algorithms)) != len(algorithms):
        raise ValueError("--algorithms必须非空且不能包含重复算法。")

    available = set(getattr(package, "PLANNERS", {})) or set(BASELINE_ALGORITHMS)
    unknown = sorted(set(algorithms) - available)
    if unknown:
        raise ValueError("未知传统算法：" + ", ".join(unknown))

    seeds = [
        int(value.strip())
        for value in str(args.planner_seeds).split(",")
        if value.strip()
    ]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--planner-seeds必须非空且不能包含重复种子。")

    specs = getattr(package, "PLANNER_SPECS", {})
    deterministic = set(getattr(package, "DETERMINISTIC_PLANNERS", ()))
    if not deterministic:
        deterministic = set(algorithms) - set(STOCHASTIC_BASELINES)
    budgets: Dict[str, Dict[str, Optional[float]]] = {}
    for algorithm in algorithms:
        spec = specs.get(algorithm) if isinstance(specs, Mapping) else None
        max_evaluations = getattr(spec, "max_evaluations", None)
        time_limit_s = getattr(spec, "time_limit_s", None)
        # 手工算法清单可用于开发验收并显式覆盖预算；正式profile始终采用冻结注册值。
        if selection_mode == "custom":
            if args.max_evaluations is not None:
                max_evaluations = int(args.max_evaluations)
            if args.time_limit_s is not None:
                time_limit_s = float(args.time_limit_s)
        budgets[algorithm] = {
            "max_evaluations": (
                None if max_evaluations is None else int(max_evaluations)
            ),
            "time_limit_s": None if time_limit_s is None else float(time_limit_s),
        }
    return algorithms, seeds, budgets, deterministic


def _baselines(args: argparse.Namespace) -> Path:
    metadata, records, manifest_root = load_manifest(args.manifest)
    records = _select_records(records, args.split)
    protocol = _protocol_for_run(args, metadata)
    protocol_hash = "" if protocol is None else str(protocol["protocol_hash"])
    package = importlib.import_module("python_classical_algs")
    algorithms, seeds, algorithm_budgets, deterministic = _baseline_plan(
        args, package
    )
    if protocol is not None:
        frozen_algorithms = dict(protocol["algorithms"])
        profile_members = {
            "main": list(frozen_algorithms["main_baselines"]),
            "supplementary": list(frozen_algorithms["supplementary_baselines"]),
            "all": [
                *frozen_algorithms["main_baselines"],
                *frozen_algorithms["supplementary_baselines"],
            ],
        }
        if args.algorithms:
            raise ValueError("绑定冻结协议时必须使用--profile，不能手工改写--algorithms。")
        if args.profile not in profile_members or algorithms != profile_members[args.profile]:
            raise ValueError("传统算法profile与冻结协议成员不一致。")
        frozen_seeds = [int(value) for value in protocol["seeds"]["stochastic_planner"]]
        if seeds != frozen_seeds:
            raise ValueError("正式传统算法planner seeds必须严格为42--51。")
    power_scales = _parse_power_scales(args.power_scales)
    tasks = [
        (power_scale, record, algorithm, planner_seed)
        for power_scale in power_scales
        for record in records
        for algorithm in algorithms
        for planner_seed in (
            seeds if algorithm not in deterministic else seeds[:1]
        )
    ]
    total = len(tasks)
    planned_keys = {
        (
            str(record["id"]),
            str(algorithm),
            str(int(planner_seed)),
            _canonical_float(power_scale),
        )
        for power_scale, record, algorithm, planner_seed in tasks
    }
    if len(planned_keys) != total:
        raise ValueError("传统算法任务清单包含重复完成键。")
    run_dir = PAPER_RUNS_ROOT / "baselines" / _safe_name(args.run_name or f"baselines_{args.split}")
    common = importlib.import_module("python_classical_algs.common")
    scenario = _load_scenario(args.scenario_file)
    if str(scenario.scenario_hash) != str(metadata["base_scenario_hash"]):
        raise ValueError("场景哈希与冻结清单不一致。")
    ppo = _ppo_module()
    run_config = {
        "schema_version": 2,
        "kind": "traditional_baselines",
        "immutable": {
            "scenario_hash": str(metadata["base_scenario_hash"]),
            "manifest_hash": str(metadata["manifest_hash"]),
            "protocol_hash": protocol_hash,
            "selected_records_sha256": _selected_records_hash(records),
            "split": str(args.split),
            "record_count": len(records),
            "profile": str(args.profile),
            "algorithms": algorithms,
            "planner_seeds": seeds,
            "power_scales": power_scales,
            "algorithm_budgets": algorithm_budgets,
        },
        "runtime": {
            "scenario_file": str(args.scenario_file.resolve()),
            "manifest_source": str(manifest_root.resolve()),
        },
    }
    _prepare_resumable_result_run(
        run_dir,
        run_config,
        resume_existing=bool(args.resume_existing),
        dry_run=bool(args.dry_run),
    )
    writer = DurableResultJsonlWriter(
        run_dir / "results.jsonl",
        resume=bool(args.resume_existing),
        repair_trailing=not bool(args.dry_run),
    )
    rows = writer.records()
    completed = _index_completed_rows(
        rows,
        key_builder=lambda row: (
            str(row["scenario_id"]),
            str(row["algorithm"]),
            str(int(row["planner_seed"])),
            _canonical_float(row["power_scale"]),
        ),
        planned_keys=planned_keys,
        expected_provenance={
            "scenario_hash": str(metadata["base_scenario_hash"]),
            "manifest_hash": str(metadata["manifest_hash"]),
            "protocol_hash": protocol_hash,
        },
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "action": "resume_baselines" if args.resume_existing else "baselines",
                    "planned_runs": total,
                    "completed": len(completed),
                    "remaining": total - len(completed),
                    "profile": str(args.profile),
                    "algorithms": algorithms,
                    "algorithm_budgets": algorithm_budgets,
                    "protocol_hash": protocol_hash,
                    "run_dir": str(run_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return run_dir
    _write_json(
        run_dir / "status.json",
        _status_payload("running", completed=len(completed), total=total),
    )
    try:
        for power_scale in power_scales:
            for record in records:
                group_tasks = [
                    (algorithm, planner_seed)
                    for task_power, task_record, algorithm, planner_seed in tasks
                    if task_power == power_scale and task_record["id"] == record["id"]
                    and (
                        str(record["id"]),
                        str(algorithm),
                        str(int(planner_seed)),
                        _canonical_float(power_scale),
                    )
                    not in completed
                ]
                if not group_tasks:
                    continue
            # build_context统一应用功率倍率；这里不能提前把四项功率再乘一次。
                cfg = _instance_cfg(ppo.DEFAULT_CONFIG, scenario, record, 1.0)
                cfg.update(
                    {
                        "id": record["id"],
                        "node_count": record["node_count"],
                        "inspection_points_xyz": record["inspection_points_xyz"],
                        "priorities": record["priorities"],
                        "service_times_s": record["service_times_s"],
                        "wind_data": _transform_wind(scenario.wind_data, record),
                        "power_scale": power_scale,
                    }
                )
                for algorithm, planner_seed in group_tasks:
                    # 传统算法在线计时包含统一评价器/上下文构建、搜索和最终路线回放。
                    started = time.perf_counter()
                    problem = common.build_context(args.scenario_file, cfg=cfg)
                    result = package.run_planner(
                        algorithm,
                        problem,
                        seed=planner_seed,
                        budget=algorithm_budgets[algorithm],
                    )
                    elapsed = time.perf_counter() - started
                    raw = result.as_dict() if hasattr(result, "as_dict") else dict(result)
                    metrics = _baseline_metrics(raw)
                    result_metadata = dict(raw.get("metadata") or {})
                    certified = result_metadata.get("optimality_certified")
                    gap = raw.get("optimality_gap", result_metadata.get("optimality_gap"))
                    dual_bound = result_metadata.get(
                        "objective_dual_bound", result_metadata.get("mip_dual_bound")
                    )
                    solver_status = result_metadata.get(
                        "solver_status", result_metadata.get("status", "")
                    )
                    if algorithm == "exact_pareto_dp" and bool(certified):
                        gap = 0.0
                    row = _long_row(
                        record=record,
                        algorithm=algorithm,
                        metrics=metrics,
                        planning_time_s=elapsed,
                        planner_seed=planner_seed,
                        # 跨算法长表统一使用冻结清单的基础场景哈希；派生问题哈希保留在route result中。
                        scenario_hash=str(metadata["base_scenario_hash"]),
                        manifest_hash=str(metadata["manifest_hash"]),
                        protocol_hash=protocol_hash,
                        power_scale=power_scale,
                        evaluations=raw.get("evaluations"),
                        optimality_gap=gap,
                        solver_dual_bound=dual_bound,
                        solver_status=str(solver_status),
                        optimality_certified=(
                            None if certified is None else bool(certified)
                        ),
                    )
                    route_name = (
                        f"{record['id']}__{algorithm}__seed{planner_seed}"
                        f"__power{_safe_name(_canonical_float(power_scale))}.json"
                    )
                    _write_json(
                        run_dir / "routes" / route_name,
                        {"record": record, "result": raw, "row": row},
                    )
                    writer.append(row)
                    rows.append(row)
                    key = (
                        str(record["id"]),
                        str(algorithm),
                        str(int(planner_seed)),
                        _canonical_float(power_scale),
                    )
                    completed[key] = row
                    _write_json(
                        run_dir / "status.json",
                        _status_payload("running", completed=len(completed), total=total),
                    )
        _write_long_csv(run_dir / "results.csv", rows)
        _write_json(
            run_dir / "status.json",
            _status_payload("completed", completed=len(rows), total=total),
        )
    except Exception as exc:
        _write_json(
            run_dir / "status.json",
            _status_payload(
                "failed",
                completed=len(completed),
                total=total,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise
    return run_dir


def _freeze_protocol(args: argparse.Namespace) -> Path:
    module = importlib.import_module("paper_protocol")
    protocol = module.build_frozen_protocol(
        args.training_root,
        args.manifest,
        repo_root=ROOT,
    )
    target = Path(args.output)
    protocol_file = target if target.suffix.lower() == ".json" else target / "protocol.json"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "action": "freeze_protocol",
                    "dry_run": True,
                    "output": str(protocol_file),
                    "protocol_hash": protocol["protocol_hash"],
                    "checkpoint_count": len(protocol["checkpoints"]),
                    "primary_id_test_counts": protocol["primary_id_test_counts"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return protocol_file
    return module.write_frozen_protocol(protocol, target)


def _selected_learning_variants(
    value: Optional[str], protocol: Mapping[str, Any]
) -> List[str]:
    algorithms = dict(protocol["algorithms"])
    all_variants = list(algorithms["learning_variants"])
    raw = "all" if value is None else str(value).strip()
    aliases = {
        "all": all_variants,
        "core": list(algorithms["core_learning_variants"]),
        "ablation": ["full", *algorithms["ablation_variants"]],
    }
    selected = aliases.get(raw)
    if selected is None:
        selected = [item.strip() for item in raw.split(",") if item.strip()]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("--variants必须非空且不能重复。")
    unknown = sorted(set(selected) - set(all_variants))
    if unknown:
        raise ValueError("--variants含冻结协议外模型：" + ", ".join(unknown))
    return [variant for variant in all_variants if variant in set(selected)]


def _checkpoint_path_from_protocol(item: Mapping[str, Any]) -> Path:
    path = Path(str(item["path"]))
    return path if path.is_absolute() else ROOT / path


def _evaluate_batch(args: argparse.Namespace) -> Dict[str, Any]:
    metadata, records, _manifest_root = load_manifest(args.manifest)
    protocol = _protocol_for_run(args, metadata)
    if protocol is None:
        raise ValueError("evaluate-batch必须绑定--protocol。")
    variants = _selected_learning_variants(args.variants, protocol)
    power_scales = _parse_power_scales(args.power_scales)
    selected_records = _select_records(records, args.split)
    checkpoints = [
        dict(item)
        for item in protocol["checkpoints"]
        if str(item["variant"]) in set(variants)
    ]
    expected_checkpoint_count = len(variants) * len(protocol["seeds"]["training"])
    if len(checkpoints) != expected_checkpoint_count:
        raise ValueError("冻结协议的学习检查点矩阵不完整。")
    checkpoints.sort(
        key=lambda item: (
            variants.index(str(item["variant"])),
            int(item["training_seed"]),
        )
    )
    power_tag = ""
    if power_scales != [1.0]:
        power_tag = "__power_" + "_".join(
            _safe_name(_canonical_float(value)) for value in power_scales
        )
    prefix = _safe_name(args.run_name or str(protocol["protocol_name"]))
    planned: List[Tuple[Dict[str, Any], Path, Path]] = []
    for item in checkpoints:
        run_name = _safe_name(
            f"{prefix}__{item['variant']}__seed{item['training_seed']}"
            f"__{args.split}{power_tag}"
        )
        run_dir = PAPER_RUNS_ROOT / "evaluation" / run_name
        planned.append((item, _checkpoint_path_from_protocol(item), run_dir))
    if not args.resume_existing:
        occupied = [str(run_dir) for _item, _checkpoint, run_dir in planned if run_dir.exists()]
        if occupied:
            raise FileExistsError(
                "批量评估目标已存在；拒绝在执行一半后才发现冲突：" + ", ".join(occupied[:5])
            )

    summary = {
        "action": "evaluate_batch",
        "dry_run": bool(args.dry_run),
        "protocol_hash": str(protocol["protocol_hash"]),
        "variants": variants,
        "training_seeds": list(protocol["seeds"]["training"]),
        "split": str(args.split),
        "power_scales": power_scales,
        "scenario_count": len(selected_records),
        "checkpoint_count": len(checkpoints),
        "planned_rows": len(checkpoints) * len(selected_records) * len(power_scales),
        "run_directories": [str(run_dir) for _item, _checkpoint, run_dir in planned],
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    completed_dirs: List[str] = []
    for item, checkpoint, run_dir in planned:
        child_args = argparse.Namespace(
            manifest=Path(args.manifest),
            split=str(args.split),
            scenario_file=Path(args.scenario_file),
            checkpoint=checkpoint,
            power_scales=",".join(_canonical_float(value) for value in power_scales),
            device=str(args.device),
            run_name=run_dir.name,
            resume_existing=bool(args.resume_existing),
            dry_run=False,
            protocol=Path(args.protocol),
            _protocol_payload=protocol,
            _protocol_assets_verified=True,
        )
        actual = _evaluate(child_args)
        completed_dirs.append(str(actual))
        print(
            f"批量评估完成 {item['variant']} seed={item['training_seed']}: {actual}",
            flush=True,
        )
    summary["completed_run_directories"] = completed_dirs
    return summary


def _audit(args: argparse.Namespace) -> Dict[str, Any]:
    module = importlib.import_module("paper_protocol")
    report = module.audit_result_runs(
        args.protocol,
        args.manifest,
        args.inputs,
        family=args.family,
        split=args.split,
        power_scales=_parse_power_scales(args.power_scales),
    )
    if args.output is not None and args.run_name is not None:
        raise ValueError("audit的--output与--run-name不能同时使用。")
    output_value = args.output
    if output_value is None and args.run_name is not None:
        output_value = PAPER_RUNS_ROOT / "audits" / f"{_safe_name(args.run_name)}.json"
    if output_value is not None and not args.dry_run:
        output = Path(output_value)
        if output.exists():
            raise FileExistsError(f"审计报告已存在，拒绝覆盖：{output}")
        _write_json(output, report)
        report = {**report, "report_file": str(output)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _analysis_inputs(inputs: Sequence[Path]) -> List[Path]:
    """审计保留结果目录语义，统计器则读取目录内唯一的原始JSONL长表。"""

    resolved: List[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            result_file = path / "results.jsonl"
            if not result_file.is_file():
                raise FileNotFoundError(f"结果目录缺少results.jsonl：{path}")
            resolved.append(result_file)
        else:
            resolved.append(path)
    return resolved


def _aggregate(args: argparse.Namespace) -> Path:
    output_dir = PAPER_RUNS_ROOT / "analysis" / _safe_name(args.run_name)
    if output_dir.exists():
        raise FileExistsError(f"分析目录已存在，程序不会覆盖：{output_dir}")
    audit_power_scales = _parse_power_scales(args.power_scales)
    if args.dry_run:
        print(json.dumps({"action": "aggregate", "inputs": [str(path) for path in args.inputs], "family": args.family, "power_scales": audit_power_scales, "formats": args.formats, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
        return output_dir
    module = importlib.import_module("paper_evaluation")
    config_type = getattr(module, "EvaluationConfig")
    manifest_metadata = None
    formal_audit = None
    included_algorithms = None
    frozen_protocol = None
    statistics_enabled = True
    analysis_role = "preregistered_family"
    if args.manifest is not None:
        manifest_metadata, _records, _root = load_manifest(args.manifest)
    if args.protocol is not None:
        if args.manifest is None:
            raise ValueError("绑定--protocol进行统计时必须同时提供--manifest。")
        protocol_module = importlib.import_module("paper_protocol")
        protocol = protocol_module.load_frozen_protocol(args.protocol)
        frozen_protocol = protocol
        verifier = getattr(protocol_module, "verify_protocol_assets", None)
        if not callable(verifier):
            raise RuntimeError("paper_protocol缺少verify_protocol_assets资产校验入口。")
        verifier(protocol, repo_root=ROOT)
        formal_audit = protocol_module.audit_result_runs(
            protocol,
            args.manifest,
            args.inputs,
            family=args.family,
            split=args.primary_split,
            power_scales=audit_power_scales,
        )
        family = dict(protocol["statistics_families"]).get(args.family)
        if isinstance(family, Mapping):
            if str(args.reference_algorithm) != str(family["reference"]):
                raise ValueError("--reference-algorithm偏离该预注册统计族。")
            included_algorithms = tuple(str(value) for value in family["members"])
        elif args.family in {"full_secondary", "core_secondary"}:
            included_algorithms = tuple(str(value) for value in formal_audit["algorithms"])
            statistics_enabled = False
            analysis_role = "secondary_descriptive"
        else:
            raise ValueError("--family不是预注册统计族或二级描述实验族。")
    config = config_type(
        reference_algorithm=args.reference_algorithm,
        primary_split=args.primary_split,
        primary_power_scale=args.primary_power_scale,
        bootstrap_samples=args.bootstrap_samples,
        figure_dpi=args.figure_dpi,
        figure_formats=tuple(args.formats),
        expected_manifest_hash=(
            None if manifest_metadata is None else str(manifest_metadata["manifest_hash"])
        ),
        expected_scenario_hash=(
            None
            if manifest_metadata is None
            else str(manifest_metadata["base_scenario_hash"])
        ),
        statistics_enabled=statistics_enabled,
        analysis_role=analysis_role,
        included_algorithms=included_algorithms,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.with_name(f".{output_dir.name}.{os.getpid()}.tmp")
    if temporary_dir.exists():
        raise FileExistsError(f"分析临时目录已存在：{temporary_dir}")
    try:
        # 全部审计、统计和绘图先在临时目录完成；失败不会占用正式run-name。
        analysis_inputs = _analysis_inputs(args.inputs)
        result = module.run_analysis(analysis_inputs, temporary_dir, config)
        if (
            frozen_protocol is not None
            and args.family == "main"
            and args.primary_split == "id_test"
            and math.isclose(args.primary_power_scale, 1.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            route_stem = module.generate_representative_route_figure(
                args.inputs,
                temporary_dir,
                frozen_protocol["representative_scenario"],
                config,
                existing_stems=result.get("generated_figures", ()),
                all_algorithms=result.get("algorithms", ()),
            )
            result["generated_figures"] = [
                *result.get("generated_figures", ()),
                route_stem,
            ]
            result["representative_scenario"] = frozen_protocol[
                "representative_scenario"
            ]
            _write_json(temporary_dir / "data_audit.json", result)
        _write_json(
            temporary_dir / "orchestration_result.json",
            {
                "result": result or {},
                "inputs": args.inputs,
                "analysis_inputs": analysis_inputs,
                "manifest": manifest_metadata,
                "protocol_audit": formal_audit,
                "statistics_family": args.family if args.protocol is not None else None,
            },
        )
        os.replace(temporary_dir, output_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return output_dir


def _doctor(args: argparse.Namespace) -> int:
    checks: List[Dict[str, Any]] = []
    checks.append({"name": "python", "ok": sys.version_info >= (3, 9), "detail": sys.version.split()[0]})
    try:
        ppo = _ppo_module()
        torch = ppo.torch
        cuda = bool(torch.cuda.is_available())
        checks.append({"name": "torch", "ok": True, "detail": str(torch.__version__)})
        checks.append({"name": "cuda", "ok": cuda, "detail": torch.cuda.get_device_name(0) if cuda else "不可用"})
        available_variants = set(getattr(ppo, "EXPERIMENT_VARIANTS", {}))
        missing_variants = sorted(set(LEARNING_VARIANTS) - available_variants)
        checks.append(
            {
                "name": "learning_variants",
                "ok": not missing_variants,
                "detail": {"available": sorted(available_variants), "missing": missing_variants},
            }
        )
    except Exception as exc:
        checks.append({"name": "torch", "ok": False, "detail": str(exc)})
    try:
        scenario = _load_scenario(args.scenario_file)
        checks.append({"name": "scenario", "ok": True, "detail": str(scenario.scenario_hash)})
    except Exception as exc:
        checks.append({"name": "scenario", "ok": False, "detail": str(exc)})
    free_gb = shutil.disk_usage(ROOT).free / (1024**3)
    checks.append({"name": "disk_free_gb", "ok": free_gb >= args.min_free_gb, "detail": round(free_gb, 2)})
    checks.append({"name": "paper_runs_isolated", "ok": PAPER_RUNS_ROOT.resolve() != (ROOT / "training_runs").resolve(), "detail": str(PAPER_RUNS_ROOT)})
    try:
        package = importlib.import_module("python_classical_algs")
        registry_ok = (
            hasattr(package, "run_planner")
            and hasattr(package, "planner_names")
            and hasattr(package, "PLANNER_SPECS")
            and set(package.PLANNER_SPECS) == set(BASELINE_ALGORITHMS)
        )
        checks.append({"name": "classical_adapter", "ok": registry_ok, "detail": sorted(getattr(package, "PLANNER_SPECS", {}))})
    except Exception as exc:
        checks.append({"name": "classical_adapter", "ok": False, "detail": str(exc)})
    try:
        scipy_optimize = importlib.import_module("scipy.optimize")
        checks.append(
            {
                "name": "scipy_highs_milp",
                "ok": callable(getattr(scipy_optimize, "milp", None)),
                "detail": "scipy.optimize.milp",
            }
        )
    except Exception as exc:
        checks.append({"name": "scipy_highs_milp", "ok": False, "detail": str(exc)})
    try:
        evaluation = importlib.import_module("paper_evaluation")
        evaluation_ok = hasattr(evaluation, "EvaluationConfig") and hasattr(evaluation, "run_analysis")
        checks.append({"name": "evaluation_adapter", "ok": evaluation_ok, "detail": "paper_evaluation.run_analysis"})
    except Exception as exc:
        checks.append({"name": "evaluation_adapter", "ok": False, "detail": str(exc)})
    try:
        protocol = importlib.import_module("paper_protocol")
        required = (
            "build_frozen_protocol",
            "write_frozen_protocol",
            "load_frozen_protocol",
            "verify_protocol_assets",
            "audit_result_runs",
        )
        checks.append(
            {
                "name": "protocol_adapter",
                "ok": all(callable(getattr(protocol, name, None)) for name in required),
                "detail": list(required),
            }
        )
    except Exception as exc:
        checks.append({"name": "protocol_adapter", "ok": False, "detail": str(exc)})
    payload = {
        "ok": all(bool(item["ok"]) for item in checks),
        "checks": checks,
        "platform": platform.platform(),
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0 if payload["ok"] else 1


def _status(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    status_path = run_dir / "status.json"
    if not status_path.exists():
        raise FileNotFoundError(f"找不到状态文件：{status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            status["latest_metrics"] = json.loads(lines[-1])
            status["metric_records"] = len(lines)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO+Pointer论文实验编排器（正式训练由用户亲自执行）")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: str, *, scenario: bool = True) -> argparse.ArgumentParser:
        child = sub.add_parser(command)
        child.add_argument("--dry-run", action="store_true", help="只展示计划，不创建目录、不训练。")
        if scenario:
            child.add_argument("--scenario-file", type=Path, default=DEFAULT_SCENARIO)
        return child

    doctor = common("doctor")
    doctor.add_argument("--min-free-gb", type=float, default=5.0)
    prepare = common("prepare")
    prepare.add_argument("--manifest-seed", type=int, default=DEFAULT_MANIFEST_SEED)
    prepare.add_argument("--manifest-name", default="frozen_v1")

    freeze_protocol = common("freeze-protocol", scenario=False)
    freeze_protocol.add_argument(
        "--training-root", type=Path, default=PAPER_RUNS_ROOT / "training"
    )
    freeze_protocol.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    freeze_protocol.add_argument("--output", type=Path, default=DEFAULT_PROTOCOL)

    for command in ("smoke", "train"):
        child = common(command)
        child.add_argument("--variant", choices=LEARNING_VARIANTS, default="full")
        child.add_argument("--seed", type=int, default=42)
        child.add_argument("--episodes", type=int, required=command == "train", default=4 if command == "smoke" else None)
        child.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
        child.add_argument("--run-name")
        child.add_argument(
            "--manifest",
            type=Path,
            default=DEFAULT_MANIFEST,
            help="prepare生成的冻结论文清单目录；训练选模固定使用其中64条validation。",
        )
        child.add_argument("--yes", action="store_true")
        child.set_defaults(stage="smoke" if command == "smoke" else "formal")

    resume = common("resume")
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--episodes", type=int, required=True, help="累计目标回合数，不是追加回合数。")
    resume.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")

    evaluate = common("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument(
        "--protocol",
        type=Path,
        help="正式评估必须绑定的frozen_test_v1；省略时仅用于开发验收。",
    )
    evaluate.add_argument("--split", choices=("validation", "id_test", "stress_test", "scale", "all"), default="id_test")
    evaluate.add_argument("--power-scales", default="1.0")
    evaluate.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    evaluate.add_argument("--run-name")
    evaluate.add_argument(
        "--resume-existing",
        action="store_true",
        help="核对run_config后续跑同名目录，并跳过results.jsonl中的已完成键。",
    )

    evaluate_batch = common("evaluate-batch")
    evaluate_batch.add_argument("--protocol", type=Path, required=True)
    evaluate_batch.add_argument("--manifest", type=Path, required=True)
    evaluate_batch.add_argument(
        "--split",
        choices=("validation", "id_test", "stress_test", "scale", "all"),
        default="id_test",
    )
    evaluate_batch.add_argument(
        "--variants",
        default="all",
        help="all、core、ablation或逗号分隔的冻结模型变体。",
    )
    evaluate_batch.add_argument("--power-scales", default="1.0")
    evaluate_batch.add_argument(
        "--device", choices=("cpu", "cuda", "auto"), default="auto"
    )
    evaluate_batch.add_argument("--run-name", help="35个顺序子任务共用的名称前缀。")
    evaluate_batch.add_argument("--resume-existing", action="store_true")

    baselines = common("baselines")
    baselines.add_argument("--manifest", type=Path, required=True)
    baselines.add_argument(
        "--protocol",
        type=Path,
        help="正式基线必须绑定的frozen_test_v1；省略时仅用于开发验收。",
    )
    baselines.add_argument("--split", choices=("validation", "id_test", "stress_test", "scale", "all"), default="id_test")
    baselines.add_argument(
        "--profile",
        choices=("main", "supplementary", "all", "custom"),
        default="main",
        help="注册表算法组；正式主实验使用main，补充实验使用supplementary。",
    )
    baselines.add_argument(
        "--algorithms",
        help="仅用于开发验收的显式算法清单；提供后按custom处理。",
    )
    baselines.add_argument("--planner-seeds", default="42,43,44,45,46,47,48,49,50,51")
    baselines.add_argument(
        "--power-scales",
        default="1.0",
        help="功率代理倍率，逗号分隔；默认只跑名义1.0。",
    )
    baselines.add_argument(
        "--max-evaluations",
        type=int,
        help="仅覆盖显式--algorithms清单的评价次数；正式profile采用注册表预算。",
    )
    baselines.add_argument(
        "--time-limit-s",
        type=float,
        help="仅覆盖显式--algorithms清单的限时；正式profile采用注册表预算。",
    )
    baselines.add_argument("--run-name")
    baselines.add_argument(
        "--resume-existing",
        action="store_true",
        help="核对run_config后续跑同名目录，并跳过results.jsonl中的已完成键。",
    )

    aggregate = common("aggregate", scenario=False)
    aggregate.add_argument("--inputs", type=Path, nargs="+", required=True)
    aggregate.add_argument("--manifest", type=Path)
    aggregate.add_argument("--protocol", type=Path)
    aggregate.add_argument(
        "--family",
        choices=(
            "main",
            "ablation",
            "supplementary",
            "full_secondary",
            "core_secondary",
        ),
        default="main",
    )
    aggregate.add_argument("--run-name", default="paper_analysis")
    aggregate.add_argument("--reference-algorithm", default="full")
    aggregate.add_argument("--primary-split", default="id_test")
    aggregate.add_argument("--primary-power-scale", type=float, default=1.0)
    aggregate.add_argument(
        "--power-scales",
        default="1.0",
        help="审计输入中应完整覆盖的倍率集合；二级功率分析可传五个倍率。",
    )
    aggregate.add_argument("--bootstrap-samples", type=int, default=10_000)
    aggregate.add_argument("--figure-dpi", type=int, default=600)
    aggregate.add_argument(
        "--formats", nargs="+", default=["pdf", "svg", "tiff"]
    )
    audit = common("audit", scenario=False)
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--inputs", type=Path, nargs="+", required=True)
    audit.add_argument("--family", required=True)
    audit.add_argument(
        "--split",
        choices=("validation", "id_test", "stress_test", "scale", "all"),
        required=True,
    )
    audit.add_argument("--power-scales", default="1.0")
    audit.add_argument("--output", type=Path)
    audit.add_argument("--run-name", help="写入paper_runs/audits/<run-name>.json。")
    status = common("status", scenario=False)
    status.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "prepare":
            root = PAPER_RUNS_ROOT / "manifests" / _safe_name(args.manifest_name)
            metadata = prepare_manifest(args.scenario_file, root, manifest_seed=args.manifest_seed, dry_run=args.dry_run)
            print(json.dumps({"manifest_dir": str(root), **metadata}, ensure_ascii=False, indent=2))
        elif args.command == "freeze-protocol":
            protocol_file = _freeze_protocol(args)
            if not args.dry_run:
                print(f"完成：{protocol_file}")
        elif args.command == "smoke":
            if not 1 <= args.episodes <= 4:
                parser.error("smoke只允许1-4回合。")
            print(f"完成：{_run_training(args, resume=False)}")
        elif args.command == "train":
            if args.episodes <= 0:
                parser.error("--episodes必须大于0。")
            args.stage = "pilot" if args.episodes < 3000 else "formal"
            print(f"完成：{_run_training(args, resume=False)}")
        elif args.command == "resume":
            config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
            snapshot_name = config.get("scenario_snapshot_file")
            snapshot_path = (
                args.run_dir / str(snapshot_name) if snapshot_name else None
            )
            args.scenario_file = (
                snapshot_path
                if snapshot_path is not None and snapshot_path.exists()
                else Path(config["scenario_file"])
            )
            args.variant = config["variant"]
            args.seed = int(config["training_seed"])
            args.stage = str(config["training_config"].get("experiment_stage", "formal"))
            args.run_name = args.run_dir.name
            args.yes = True
            print(f"完成：{_run_training(args, resume=True)}")
        elif args.command == "evaluate":
            print(f"完成：{_evaluate(args)}")
        elif args.command == "evaluate-batch":
            batch_result = _evaluate_batch(args)
            if not args.dry_run:
                print(json.dumps(batch_result, ensure_ascii=False, indent=2))
        elif args.command == "baselines":
            print(f"完成：{_baselines(args)}")
        elif args.command == "aggregate":
            print(f"完成：{_aggregate(args)}")
        elif args.command == "audit":
            _audit(args)
        elif args.command == "status":
            return _status(args)
        return 0
    except Exception as exc:
        print(f"执行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

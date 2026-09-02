#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""传统/优化基线共用的公平任务接口。

本模块不维护第二套飞行物理模型。所有边代价、拍摄悬停代价、三资源硬约束与
返航动作均复用 :mod:`final_python_ppo_pointer` 的 v2 环境。搜索算法只产生一个
无重复巡检点前缀；前缀内任一不可行动作都会使候选无效，不做截断或修路。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from uav_inspection.core import final_python_ppo_pointer as ppo
from uav_inspection.core.ppo_training_scenario import load_training_scenario


EPS = 1e-9
DEFAULT_SCENARIO_FILE = (
    Path(__file__).resolve().parents[1] / "scenario_data" / "mountain_road_16pt.npz"
)


@dataclass(frozen=True)
class PlannerBudget:
    """一次在线规划允许消耗的候选评价次数和墙钟时间。"""

    max_evaluations: Optional[int] = 50_000
    time_limit_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_evaluations is not None and int(self.max_evaluations) <= 0:
            raise ValueError("max_evaluations 必须为正整数或 None。")
        if self.time_limit_s is not None and float(self.time_limit_s) <= 0.0:
            raise ValueError("time_limit_s 必须为正数或 None。")


# 论文复现实验的关键参数集中于此；快速测试应显式传入更小的配置。
@dataclass(frozen=True)
class ACOConfig:
    ants: int = 40
    iterations: int = 1250
    alpha: float = 1.0
    beta: float = 2.0
    evaporation: float = 0.20
    deposit_scale: float = 1.0
    stop_weight: float = 0.08
    initial_pheromone: float = 1.0


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 80
    generations: int = 625
    tournament_size: int = 3
    crossover_rate: float = 0.90
    mutation_rate: float = 0.20
    elite_count: int = 2


@dataclass(frozen=True)
class SAConfig:
    iterations: int = 50_000
    initial_temperature: float = 0.20
    final_temperature: float = 1e-4
    restart_interval: int = 5_000


@dataclass(frozen=True)
class PSOConfig:
    swarm_size: int = 60
    iterations: int = 834
    inertia: float = 0.72
    cognitive: float = 1.49
    social: float = 1.49
    max_velocity: float = 0.25


@dataclass(frozen=True)
class AStarConfig:
    dominance_tolerance: float = 1e-9


@dataclass(frozen=True)
class ParetoDPConfig:
    dominance_tolerance: float = 1e-9
    # None 表示不截断Pareto标签，搜索完全结束时才可声明精确最优。
    max_labels_per_state: Optional[int] = None


@dataclass
class ProblemInstance:
    """所有规划器共享的一份冻结任务实例。"""

    start_pos: np.ndarray
    points: np.ndarray
    priorities: np.ndarray
    terrain: np.ndarray
    cfg: Mapping[str, Any]
    wind_data: Optional[Mapping[str, Any]]
    scenario_hash: str
    name: str = "mountain_road_16pt"

    def __post_init__(self) -> None:
        self.start_pos = np.asarray(self.start_pos, dtype=np.float32).reshape(-1).copy()
        self.points = np.asarray(self.points, dtype=np.float32).copy()
        self.priorities = np.asarray(self.priorities, dtype=np.float32).reshape(-1).copy()
        self.terrain = np.asarray(self.terrain, dtype=np.float32).copy()
        self.cfg = ppo.resolve_config(dict(self.cfg))
        self.wind_data = _copy_wind(self.wind_data)
        if self.points.ndim != 2 or self.points.shape[1] not in (2, 3):
            raise ValueError("points 必须是 [N,2] 或 [N,3] 数组。")
        if self.points.shape[0] != self.priorities.size:
            raise ValueError("points 与 priorities 数量不一致。")
        if self.points.shape[0] == 0:
            raise ValueError("至少需要一个巡检点。")
        if not self.scenario_hash:
            self.scenario_hash = _problem_hash(self)

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])


@dataclass
class PlanningResult:
    """统一规划输出；可以无损写入逐运行JSON或论文长表。"""

    algorithm: str
    visit_order: Tuple[int, ...]
    path: np.ndarray
    flight_path: np.ndarray
    segments: Tuple[Mapping[str, Any], ...]
    metrics: Dict[str, Any]
    runtime_s: float
    evaluations: int
    seed: int
    scenario_hash: str
    status: str = "ok"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "visit_order": list(self.visit_order),
            "path": np.asarray(self.path, dtype=np.float64).tolist(),
            "flight_path": np.asarray(self.flight_path, dtype=np.float64).tolist(),
            "segments": [_jsonable(dict(item)) for item in self.segments],
            "metrics": _jsonable(self.metrics),
            "runtime_s": float(self.runtime_s),
            "evaluations": int(self.evaluations),
            "seed": int(self.seed),
            "scenario_hash": self.scenario_hash,
            "status": self.status,
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True)
class RouteLabel:
    """不含最终返航边的搜索标签。"""

    order: Tuple[int, ...]
    mask: int
    last: int
    energy_wh: float
    distance_m: float
    time_s: float


@dataclass(frozen=True)
class RouteEvaluation:
    order: Tuple[int, ...]
    objective: float
    energy_wh: float
    distance_m: float
    time_s: float
    coverage: float
    weighted_coverage: float
    returned: bool
    termination_reason: str


class SearchController:
    """让所有迭代算法使用同一种评价次数与墙钟预算语义。"""

    def __init__(self, budget: Optional[PlannerBudget] = None) -> None:
        self.budget = budget or PlannerBudget()
        self.started_at = time.perf_counter()
        self.evaluations = 0

    @property
    def elapsed_s(self) -> float:
        return float(time.perf_counter() - self.started_at)

    @property
    def exhausted(self) -> bool:
        if (
            self.budget.max_evaluations is not None
            and self.evaluations >= int(self.budget.max_evaluations)
        ):
            return True
        return bool(
            self.budget.time_limit_s is not None
            and self.elapsed_s >= float(self.budget.time_limit_s)
        )

    def consume(self, count: int = 1) -> bool:
        if count <= 0:
            raise ValueError("评价次数增量必须为正。")
        if self.exhausted:
            return False
        if (
            self.budget.max_evaluations is not None
            and self.evaluations + int(count) > int(self.budget.max_evaluations)
        ):
            return False
        self.evaluations += int(count)
        return True


class MissionEvaluator:
    """预计算PPO v2航段，并对大量候选做等价的快速合法性判断。"""

    def __init__(self, problem: ProblemInstance) -> None:
        self.problem = problem
        self.template = ppo.build_episode(
            problem.start_pos,
            problem.points,
            problem.terrain,
            problem.cfg,
            problem.wind_data,
            randomize=False,
        )
        self.n = problem.point_count
        self.depot = self.n
        self._segments: Dict[Tuple[int, int], ppo.SegmentEstimate] = {}
        self._precompute_segments()
        self.service_times_s = np.asarray(self.template["service_times_s"], dtype=np.float64)
        factor = float(self.template["cfg"]["resource_safety_factor"])
        hover = float(self.template["cfg"]["hover_power_w"])
        self.service_energy_wh = hover * self.service_times_s / 3600.0 * factor
        self.energy_budget_wh = float(self.template["energy_budget_wh"])
        self.distance_budget_m = float(self.template["max_route_distance"])
        self.time_budget_s = float(self.template["max_mission_time_s"])
        self._priority_nonnegative = np.clip(problem.priorities.astype(np.float64), 0.0, None)
        self._total_priority = float(np.sum(self._priority_nonnegative))

    def _precompute_segments(self) -> None:
        for target in range(self.n):
            self._segments[(self.depot, target)] = ppo._get_segment(
                self.template,
                self.depot,
                target,
                is_takeoff=True,
                is_landing=False,
            )
            self._segments[(target, self.depot)] = ppo._get_segment(
                self.template,
                target,
                self.depot,
                is_takeoff=False,
                is_landing=True,
            )
        self._segments[(self.depot, self.depot)] = ppo._get_segment(
            self.template,
            self.depot,
            self.depot,
            is_takeoff=False,
            is_landing=True,
        )
        for source in range(self.n):
            for target in range(self.n):
                if source != target:
                    self._segments[(source, target)] = ppo._get_segment(
                        self.template,
                        source,
                        target,
                        is_takeoff=False,
                        is_landing=False,
                    )

    def start_label(self) -> RouteLabel:
        return RouteLabel((), 0, self.depot, 0.0, 0.0, 0.0)

    def try_append(self, label: RouteLabel, node: int) -> Optional[RouteLabel]:
        """仅在访问后仍能满足三类返航预算时生成新标签。"""

        node = int(node)
        if node < 0 or node >= self.n or label.mask & (1 << node):
            return None
        outgoing = self._segments[(label.last, node)]
        returning = self._segments[(node, self.depot)]
        if not outgoing.feasible or not returning.feasible:
            return None
        energy = label.energy_wh + outgoing.energy_wh + float(self.service_energy_wh[node])
        distance = label.distance_m + outgoing.distance_m
        duration = label.time_s + outgoing.time_s + float(self.service_times_s[node])
        if energy + returning.energy_wh > self.energy_budget_wh + 1e-6:
            return None
        if distance + returning.distance_m > self.distance_budget_m + 1e-6:
            return None
        if duration + returning.time_s > self.time_budget_s + 1e-6:
            return None
        return RouteLabel(
            label.order + (node,),
            label.mask | (1 << node),
            node,
            float(energy),
            float(distance),
            float(duration),
        )

    def finish_label(self, label: RouteLabel) -> RouteEvaluation:
        returning = self._segments[(label.last, self.depot)]
        returned = bool(returning.feasible)
        energy = label.energy_wh + returning.energy_wh
        distance = label.distance_m + returning.distance_m
        duration = label.time_s + returning.time_s
        returned = returned and (
            energy <= self.energy_budget_wh + 1e-6
            and distance <= self.distance_budget_m + 1e-6
            and duration <= self.time_budget_s + 1e-6
        )
        coverage = len(label.order) / max(self.n, 1)
        priority_sum = float(sum(self._priority_nonnegative[i] for i in label.order))
        weighted = (
            priority_sum / self._total_priority
            if self._total_priority > EPS
            else coverage
        )
        objective = self._objective(weighted, coverage, energy, distance, duration)
        if not returned:
            objective = -math.inf
        return RouteEvaluation(
            order=label.order,
            objective=float(objective),
            energy_wh=float(energy),
            distance_m=float(distance),
            time_s=float(duration),
            coverage=float(coverage),
            weighted_coverage=float(weighted),
            returned=returned,
            termination_reason=(
                "returned_full" if len(label.order) == self.n else "returned_partial"
            ) if returned else "constraint_failure",
        )

    def evaluate_order(
        self, order: Sequence[int], *, prefix_length: Optional[int] = None
    ) -> RouteEvaluation:
        """评价显式前缀；前缀内任一非法动作都会使整个候选不可行。"""

        raw = tuple(int(value) for value in order)
        if any(value < 0 or value >= self.n for value in raw):
            raise ValueError("候选序列含越界巡检点。")
        if len(set(raw)) != len(raw):
            raise ValueError("候选序列含重复巡检点；算法必须从表示层保证唯一性。")
        limit = len(raw) if prefix_length is None else int(prefix_length)
        if limit < 0 or limit > len(raw):
            raise ValueError("prefix_length 超出候选序列范围。")
        label = self.start_label()
        for node in raw[:limit]:
            appended = self.try_append(label, node)
            if appended is None:
                # 不把非法长路线静默截断成较短合法路线，否则等价于算法外修路。
                return RouteEvaluation(
                    order=tuple(raw[:limit]),
                    objective=-math.inf,
                    energy_wh=float(label.energy_wh),
                    distance_m=float(label.distance_m),
                    time_s=float(label.time_s),
                    coverage=0.0,
                    weighted_coverage=0.0,
                    returned=False,
                    termination_reason="infeasible_candidate",
                )
            label = appended
        return self.finish_label(label)

    def marginal_task_gain(self, node: int) -> float:
        weights = self.template["cfg"]["reward_weights"]
        priority_gain = (
            float(self._priority_nonnegative[int(node)]) / self._total_priority
            if self._total_priority > EPS
            else 1.0 / max(self.n, 1)
        )
        return (
            float(weights["priority"]) * priority_gain
            + float(weights["coverage"]) / max(self.n, 1)
        )

    def _objective(
        self,
        weighted_coverage: float,
        coverage: float,
        energy_wh: float,
        distance_m: float,
        time_s: float,
    ) -> float:
        weights = self.template["cfg"]["reward_weights"]
        return (
            float(weights["priority"]) * weighted_coverage
            + float(weights["coverage"]) * coverage
            - float(weights["energy"]) * energy_wh / max(self.energy_budget_wh, EPS)
            - float(weights["distance"]) * distance_m / max(self.distance_budget_m, EPS)
            - float(weights["time"]) * time_s / max(self.time_budget_s, EPS)
        )

    def build_result(
        self,
        algorithm: str,
        evaluation: RouteEvaluation,
        controller: SearchController,
        seed: int,
        *,
        status: str = "ok",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PlanningResult:
        """通过真实PPO环境重放最终序列，防止快速搜索器与环境语义漂移。"""

        state = ppo.build_episode(
            self.problem.start_pos,
            self.problem.points,
            self.problem.terrain,
            self.problem.cfg,
            self.problem.wind_data,
            randomize=False,
        )
        for node in evaluation.order:
            state, _, done = ppo.step_env_improved(
                state,
                node,
                self.problem.points,
                self.problem.priorities,
                self.problem.terrain,
                self.problem.cfg,
                self.problem.wind_data,
            )
            if done:
                raise AssertionError("巡检前缀在显式返航前意外终止。")
        state, _, done = ppo.step_env_improved(
            state,
            self.n,
            self.problem.points,
            self.problem.priorities,
            self.problem.terrain,
            self.problem.cfg,
            self.problem.wind_data,
        )
        if not done:
            raise AssertionError("最终返航动作没有终止任务。")
        metrics = ppo._episode_metrics(state, self.problem.priorities)
        replay_objective = self._objective(
            float(metrics["weighted_coverage"]),
            float(metrics["coverage"]),
            float(metrics["energy_wh"]),
            float(metrics["distance_m"]),
            float(metrics["time_s"]),
        )
        metrics.update(
            {
                "objective": float(replay_objective),
                "safe_weighted_coverage": (
                    float(metrics["weighted_coverage"]) if metrics["returned"] else 0.0
                ),
                "energy_budget_wh": self.energy_budget_wh,
                "distance_budget_m": self.distance_budget_m,
                "time_budget_s": self.time_budget_s,
                # 传统适配器保留旧版数值字段，同时单独保留可审计的记录数组。
                "constraint_violation_records": copy.deepcopy(
                    metrics.get("constraint_violations", [])
                ),
                "constraint_violations": 0,
                "constraint_violation_count": 0,
            }
        )
        if not math.isclose(replay_objective, evaluation.objective, abs_tol=2e-6):
            raise AssertionError("快速候选评价与PPO环境重放的目标值不一致。")
        serial_segments = []
        for raw in state["executed_segments"]:
            item = dict(raw)
            item["mean_wind_mps"] = np.asarray(item["mean_wind_mps"]).tolist()
            item["flight_path"] = np.asarray(item["flight_path"]).tolist()
            serial_segments.append(item)
        result_metadata = {
            "planner_budget": asdict(controller.budget),
            "problem_name": self.problem.name,
            "problem_point_count": self.n,
            "return_to_start": bool(self.problem.cfg["return_to_start"]),
        }
        result_metadata.update(dict(metadata or {}))
        return PlanningResult(
            algorithm=str(algorithm),
            visit_order=tuple(int(i) for i in state["visited"]),
            path=np.asarray(state["path_history"], dtype=np.float32),
            flight_path=np.asarray(state["flight_path"], dtype=np.float32),
            segments=tuple(serial_segments),
            metrics=metrics,
            runtime_s=controller.elapsed_s,
            evaluations=controller.evaluations,
            seed=int(seed),
            scenario_hash=self.problem.scenario_hash,
            status=status,
            metadata=result_metadata,
        )


def build_context(
    scenario_file: Optional[Union[str, Path]] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> ProblemInstance:
    """从已持久化的NPZ/JSON场景建立统一问题实例。"""

    path = Path(scenario_file) if scenario_file is not None else DEFAULT_SCENARIO_FILE
    scenario = load_training_scenario(path)
    inputs = scenario.as_training_inputs()
    raw = dict(cfg or {})
    points = np.asarray(
        raw.pop("inspection_points_xyz", raw.pop("inspection_points", inputs["points"])),
        dtype=np.float32,
    )
    priorities = np.asarray(raw.pop("priorities", inputs["priorities"]), dtype=np.float32)
    service_times = np.asarray(
        raw.pop("service_times_s", inputs["service_times_s"]), dtype=np.float32
    )
    wind_data = raw.pop("wind_data", inputs["wind_data"])
    power_scale = float(raw.pop("power_scale", 1.0))
    if power_scale <= 0.0 or not np.isfinite(power_scale):
        raise ValueError("power_scale 必须是有限正数。")

    # manifest身份字段进入输出长表，不应被误当成PPO物理参数。
    identity_fields = {
        key: raw.pop(key)
        for key in ("id", "split", "instance_seed", "node_count")
        if key in raw
    }
    if "node_count" in identity_fields and int(identity_fields["node_count"]) != int(points.shape[0]):
        raise ValueError("manifest 的 node_count 与 inspection_points_xyz 数量不一致。")
    merged_cfg = dict(raw)
    merged_cfg.update(
        {
            "coordinate_scale_m_per_unit": float(inputs["coordinate_scale_m_per_unit"]),
            "service_times_s": service_times,
            "point_z_mode": str(inputs["cfg"]["point_z_mode"]),
            "terrain_clearance_m": float(inputs["cfg"]["terrain_clearance_m"]),
            "return_to_start": True,
        }
    )
    if not math.isclose(power_scale, 1.0, abs_tol=1e-12):
        for field_name in (
            "hover_power_w",
            "cruise_power_w",
            "climb_power_w",
            "descent_power_w",
        ):
            base_value = float(merged_cfg.get(field_name, ppo.DEFAULT_CONFIG[field_name]))
            merged_cfg[field_name] = base_value * power_scale

    derived = bool(cfg) or points.shape != np.asarray(inputs["points"]).shape or not np.array_equal(
        points, np.asarray(inputs["points"], dtype=np.float32)
    )
    return ProblemInstance(
        start_pos=inputs["start_pos"],
        points=points,
        priorities=priorities,
        terrain=inputs["terrain"],
        cfg=merged_cfg,
        wind_data=wind_data,
        scenario_hash="" if derived else scenario.scenario_hash,
        name=str(identity_fields.get("id", Path(path).stem)),
    )


def make_problem(
    start_pos: Sequence[float],
    points: np.ndarray,
    priorities: Sequence[float],
    terrain: np.ndarray,
    cfg: Optional[Mapping[str, Any]] = None,
    wind_data: Optional[Mapping[str, Any]] = None,
    *,
    name: str = "in_memory",
    scenario_hash: str = "",
) -> ProblemInstance:
    """供单元测试和冻结派生场景使用的内存构造器。"""

    return ProblemInstance(
        np.asarray(start_pos),
        np.asarray(points),
        np.asarray(priorities),
        np.asarray(terrain),
        dict(cfg or {}),
        wind_data,
        scenario_hash,
        name,
    )


def save_result(result: PlanningResult, path: Union[str, Path]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def save_path(path: Sequence[Sequence[float]], output_file: Union[str, Path]) -> Path:
    """保留旧调度器所需的窄兼容接口。"""

    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, np.asarray(path, dtype=np.float32))
    return target


def dominates(a: RouteLabel, b: RouteLabel, tolerance: float = 1e-9) -> bool:
    resources_a = (a.energy_wh, a.distance_m, a.time_s)
    resources_b = (b.energy_wh, b.distance_m, b.time_s)
    no_worse = all(x <= y + tolerance for x, y in zip(resources_a, resources_b))
    strictly_better = any(x < y - tolerance for x, y in zip(resources_a, resources_b))
    return bool(no_worse and strictly_better)


def _copy_wind(wind_data: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if wind_data is None:
        return None
    result: Dict[str, Any] = {}
    for key, value in wind_data.items():
        result[str(key)] = np.asarray(value).copy() if isinstance(value, (np.ndarray, list, tuple)) else value
    return result


def _problem_hash(problem: ProblemInstance) -> str:
    digest = hashlib.sha256()
    for value in (problem.start_pos, problem.points, problem.priorities, problem.terrain):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    digest.update(
        json.dumps(
            _jsonable(dict(problem.cfg)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    if problem.wind_data is not None:
        for key in sorted(problem.wind_data):
            digest.update(str(key).encode("utf-8"))
            value = problem.wind_data[key]
            if isinstance(value, np.ndarray):
                array = np.ascontiguousarray(value)
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(str(array.shape).encode("ascii"))
                digest.update(array.tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


__all__ = [
    "ACOConfig",
    "AStarConfig",
    "DEFAULT_SCENARIO_FILE",
    "GAConfig",
    "MissionEvaluator",
    "PSOConfig",
    "ParetoDPConfig",
    "PlannerBudget",
    "PlanningResult",
    "ProblemInstance",
    "RouteEvaluation",
    "RouteLabel",
    "SAConfig",
    "SearchController",
    "build_context",
    "dominates",
    "make_problem",
    "save_path",
    "save_result",
]

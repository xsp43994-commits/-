#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文级传统/优化规划基线的统一入口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .a_star import AStarPathPlanning, plan_a_star, plan_exact_pareto_dp
from .aco import AntColonyOptimization, plan_aco
from .common import PlannerBudget, PlanningResult, ProblemInstance
from .ga import GeneticAlgorithm, plan_ga
from .greedy import plan_nearest_feasible, plan_priority_resource_greedy
from .milp import (
    MILPConfig,
    plan_milp_orienteering,
    solve_resource_threshold_milp,
)
from .pso import ParticleSwarmOptimization, plan_pso
from .sa import SimulatedAnnealing, plan_sa


PLANNERS: Dict[str, Callable[..., PlanningResult]] = {
    "nearest_feasible": plan_nearest_feasible,
    "priority_resource_greedy": plan_priority_resource_greedy,
    "milp_orienteering": plan_milp_orienteering,
    "a_star": plan_a_star,
    "aco": plan_aco,
    "ga": plan_ga,
    "sa": plan_sa,
    "pso": plan_pso,
    "exact_pareto_dp": plan_exact_pareto_dp,
}


@dataclass(frozen=True)
class PlannerSpec:
    """正式协议中的算法角色、随机性与停止预算。"""

    name: str
    deterministic: bool
    profile: str
    max_evaluations: Optional[int]
    time_limit_s: Optional[float]

    def budget(self) -> PlannerBudget:
        return PlannerBudget(
            max_evaluations=self.max_evaluations,
            time_limit_s=self.time_limit_s,
        )


PLANNER_SPECS: Dict[str, PlannerSpec] = {
    "nearest_feasible": PlannerSpec(
        "nearest_feasible", True, "main", None, None
    ),
    "priority_resource_greedy": PlannerSpec(
        "priority_resource_greedy", True, "main", None, None
    ),
    "aco": PlannerSpec("aco", False, "main", 50_000, None),
    "ga": PlannerSpec("ga", False, "main", 50_000, None),
    "sa": PlannerSpec("sa", False, "main", 50_000, None),
    "milp_orienteering": PlannerSpec(
        "milp_orienteering", True, "main", None, 60.0
    ),
    "a_star": PlannerSpec("a_star", True, "supplementary", None, 60.0),
    "pso": PlannerSpec("pso", False, "supplementary", 50_000, None),
    "exact_pareto_dp": PlannerSpec(
        "exact_pareto_dp", True, "supplementary", None, 60.0
    ),
}

if set(PLANNER_SPECS) != set(PLANNERS):  # pragma: no cover - 导入期防漂移
    raise RuntimeError("PLANNER_SPECS与PLANNERS注册表不一致。")

DETERMINISTIC_PLANNERS = {
    name for name, spec in PLANNER_SPECS.items() if spec.deterministic
}


def planner_names(profile: str = "all") -> Tuple[str, ...]:
    key = str(profile).strip().lower()
    if key == "all":
        return tuple(PLANNER_SPECS)
    if key not in {"main", "supplementary"}:
        raise ValueError("profile必须是main、supplementary或all。")
    return tuple(
        name for name, spec in PLANNER_SPECS.items() if spec.profile == key
    )


def planner_budget(name: str) -> PlannerBudget:
    key = str(name).strip().lower()
    if key not in PLANNER_SPECS:
        raise ValueError(f"未知传统算法 {name!r}。")
    return PLANNER_SPECS[key].budget()


def run_planner(
    name: str,
    problem: ProblemInstance,
    *,
    seed: int = 42,
    budget: Optional[Any] = None,
    params: Optional[Any] = None,
) -> PlanningResult:
    """运行一个基线；budget既可传PlannerBudget，也可传JSON映射。"""

    key = str(name).strip().lower()
    if key not in PLANNERS:
        raise ValueError(f"未知传统算法 {name!r}；可选：{sorted(PLANNERS)}")
    resolved_budget = planner_budget(key) if budget is None else _coerce_budget(budget)
    return PLANNERS[key](
        problem,
        seed=int(seed),
        budget=resolved_budget,
        params=params,
    )


def run_baselines(
    names: Sequence[str],
    problem: ProblemInstance,
    *,
    seeds: Iterable[int] = (42,),
    budget: Optional[Any] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> List[PlanningResult]:
    """按给定算法和种子顺序运行，返回适合论文长表展开的结果列表。"""

    results: List[PlanningResult] = []
    parameter_map = dict(params or {})
    seed_values = tuple(int(seed) for seed in seeds)
    for name in names:
        key = str(name).strip().lower()
        # 确定性规划器只运行一次，避免把相同路线伪装成独立随机重复。
        planner_seeds = seed_values[:1] if key in DETERMINISTIC_PLANNERS else seed_values
        for seed in planner_seeds:
            results.append(
                run_planner(
                    name,
                    problem,
                    seed=int(seed),
                    budget=_budget_for(name, budget),
                    params=parameter_map.get(str(name)),
                )
            )
    return results


def _coerce_budget(value: Optional[Any]) -> Optional[PlannerBudget]:
    if value is None or isinstance(value, PlannerBudget):
        return value
    if isinstance(value, Mapping):
        return PlannerBudget(**dict(value))
    raise TypeError("budget 必须是 PlannerBudget、映射或 None。")


def _budget_for(name: str, value: Optional[Any]) -> Optional[Any]:
    if not isinstance(value, Mapping):
        return value
    if "max_evaluations" in value or "time_limit_s" in value:
        return value
    return value.get(str(name))


__all__ = [
    "AStarPathPlanning",
    "AntColonyOptimization",
    "GeneticAlgorithm",
    "MILPConfig",
    "PLANNERS",
    "PLANNER_SPECS",
    "PlannerSpec",
    "DETERMINISTIC_PLANNERS",
    "ParticleSwarmOptimization",
    "SimulatedAnnealing",
    "plan_a_star",
    "plan_aco",
    "plan_exact_pareto_dp",
    "plan_ga",
    "plan_nearest_feasible",
    "plan_milp_orienteering",
    "solve_resource_threshold_milp",
    "plan_priority_resource_greedy",
    "plan_pso",
    "plan_sa",
    "planner_budget",
    "planner_names",
    "run_baselines",
    "run_planner",
]

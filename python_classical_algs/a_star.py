#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""资源约束子集A*与N=16可用的精确Pareto动态规划参考。"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .common import (
    AStarConfig,
    MissionEvaluator,
    ParetoDPConfig,
    PlannerBudget,
    PlanningResult,
    ProblemInstance,
    RouteEvaluation,
    RouteLabel,
    SearchController,
    dominates,
)


def _config(value: Optional[Any], cls: Any) -> Any:
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls(**dict(value))
    raise TypeError(f"params 必须是 {cls.__name__} 或映射。")


def _task_upper_bound(evaluator: MissionEvaluator, mask: int) -> float:
    """忽略资源代价并假定剩余点全可访问，因而是可采纳的收益上界。"""

    total = 0.0
    for node in range(evaluator.n):
        if not (mask & (1 << node)):
            total += evaluator.marginal_task_gain(node)
    visited = tuple(node for node in range(evaluator.n) if mask & (1 << node))
    priority = sum(evaluator._priority_nonnegative[node] for node in visited)
    weighted = priority / evaluator._total_priority if evaluator._total_priority > 0 else len(visited) / evaluator.n
    weights = evaluator.template["cfg"]["reward_weights"]
    current_task = float(weights["priority"]) * weighted + float(weights["coverage"]) * len(visited) / evaluator.n
    return float(current_task + total)


def _insert_pareto(
    frontier: List[RouteLabel], label: RouteLabel, tolerance: float
) -> bool:
    """相同(mask,last)只保留三资源非支配标签。"""

    for existing in frontier:
        equal_resources = (
            abs(existing.energy_wh - label.energy_wh) <= tolerance
            and abs(existing.distance_m - label.distance_m) <= tolerance
            and abs(existing.time_s - label.time_s) <= tolerance
        )
        if dominates(existing, label, tolerance) or (
            equal_resources and existing.order <= label.order
        ):
            return False
    frontier[:] = [
        existing
        for existing in frontier
        if not dominates(label, existing, tolerance)
    ]
    frontier.append(label)
    return True


def plan_a_star(
    problem: ProblemInstance,
    *,
    seed: int = 42,
    budget: Optional[PlannerBudget] = None,
    params: Optional[Any] = None,
) -> PlanningResult:
    """按任务收益可采纳上界展开的资源约束子集标签A*。"""

    config = _config(params, AStarConfig)
    controller = SearchController(budget)
    evaluator = MissionEvaluator(problem)
    initial = evaluator.start_label()
    best = evaluator.finish_label(initial)
    controller.consume()
    counter = itertools.count()
    heap: List[Tuple[float, int, RouteLabel]] = [
        (-_task_upper_bound(evaluator, initial.mask), next(counter), initial)
    ]
    frontiers: Dict[Tuple[int, int], List[RouteLabel]] = {(0, evaluator.depot): [initial]}
    interrupted = False

    while heap:
        if controller.exhausted:
            interrupted = True
            break
        _, _, label = heapq.heappop(heap)
        if label not in frontiers.get((label.mask, label.last), ()):
            continue
        # 节点编号只作为确定性破同规则，不构成额外人工路线修复。
        remaining = [node for node in range(evaluator.n) if not label.mask & (1 << node)]
        remaining.sort(key=lambda node: (-evaluator.marginal_task_gain(node), node))
        for node in remaining:
            if not controller.consume():
                interrupted = True
                break
            candidate = evaluator.try_append(label, node)
            if candidate is None:
                continue
            key = (candidate.mask, candidate.last)
            frontier = frontiers.setdefault(key, [])
            if not _insert_pareto(frontier, candidate, config.dominance_tolerance):
                continue
            evaluated = evaluator.finish_label(candidate)
            if _better(evaluated, best):
                best = evaluated
            heapq.heappush(
                heap,
                (-_task_upper_bound(evaluator, candidate.mask), next(counter), candidate),
            )
        if interrupted:
            break

    optimality_certified = bool(not interrupted and not heap)
    status = "ok" if optimality_certified else "budget_exhausted"
    return evaluator.build_result(
        "a_star",
        best,
        controller,
        seed,
        status=status,
        metadata={
            "optimality_certified": optimality_certified,
            "optimality_gap": 0.0 if optimality_certified else None,
            "search_interrupted": bool(interrupted),
            "config": asdict(config),
            "open_labels_remaining": len(heap),
        },
    )


def plan_exact_pareto_dp(
    problem: ProblemInstance,
    *,
    seed: int = 42,
    budget: Optional[PlannerBudget] = None,
    params: Optional[Any] = None,
) -> PlanningResult:
    """Held–Karp式三资源Pareto-DP；完整结束且不截断时给出精确参考。"""

    config = _config(params, ParetoDPConfig)
    controller = SearchController(
        budget or PlannerBudget(max_evaluations=None, time_limit_s=60.0)
    )
    evaluator = MissionEvaluator(problem)
    initial = evaluator.start_label()
    best = evaluator.finish_label(initial)
    controller.consume()
    frontiers: Dict[Tuple[int, int], List[RouteLabel]] = {(0, evaluator.depot): [initial]}
    current_layer = [initial]
    truncated = False

    for _depth in range(evaluator.n):
        next_keys = set()
        for label in current_layer:
            if controller.exhausted:
                break
            for node in range(evaluator.n):
                if label.mask & (1 << node):
                    continue
                if not controller.consume():
                    break
                candidate = evaluator.try_append(label, node)
                if candidate is None:
                    continue
                key = (candidate.mask, candidate.last)
                frontier = frontiers.setdefault(key, [])
                if not _insert_pareto(frontier, candidate, config.dominance_tolerance):
                    continue
                if (
                    config.max_labels_per_state is not None
                    and len(frontier) > int(config.max_labels_per_state)
                ):
                    # 只有用户显式设置上限才截断；截断结果不会标记为精确最优。
                    frontier.sort(
                        key=lambda item: (
                            item.energy_wh / evaluator.energy_budget_wh
                            + item.distance_m / evaluator.distance_budget_m
                            + item.time_s / evaluator.time_budget_s,
                            item.order,
                        )
                    )
                    del frontier[int(config.max_labels_per_state) :]
                    truncated = True
                    if candidate not in frontier:
                        continue
                next_keys.add(key)
                evaluated = evaluator.finish_label(candidate)
                if _better(evaluated, best):
                    best = evaluated
        if controller.exhausted:
            break
        current_layer = [label for key in sorted(next_keys) for label in frontiers[key]]
        if not current_layer:
            break

    completed = not controller.exhausted
    status = "ok" if completed else "budget_exhausted"
    return evaluator.build_result(
        "exact_pareto_dp",
        best,
        controller,
        seed,
        status=status,
        metadata={
            "optimality_certified": bool(completed and not truncated),
            "optimality_gap": 0.0 if completed and not truncated else None,
            "reference_type": "pareto_dynamic_programming",
            "pareto_frontier_truncated": bool(truncated),
            "config": asdict(config),
            "pareto_states": len(frontiers),
        },
    )


def _better(candidate: RouteEvaluation, incumbent: RouteEvaluation) -> bool:
    if candidate.objective > incumbent.objective + 1e-12:
        return True
    if abs(candidate.objective - incumbent.objective) <= 1e-12:
        return (len(candidate.order), tuple(-i for i in candidate.order)) > (
            len(incumbent.order),
            tuple(-i for i in incumbent.order),
        )
    return False


# 保留旧调度器使用的展示名称。
AStarPathPlanning = plan_a_star

__all__ = ["AStarPathPlanning", "plan_a_star", "plan_exact_pareto_dp"]

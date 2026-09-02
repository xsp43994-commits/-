#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两个透明、无需训练的确定性贪心基线。"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .common import (
    MissionEvaluator,
    PlannerBudget,
    PlanningResult,
    ProblemInstance,
    SearchController,
)


def plan_nearest_feasible(
    problem: ProblemInstance,
    *,
    seed: int = 42,
    budget: Optional[PlannerBudget] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> PlanningResult:
    """反复访问飞行距离最近且仍能返航的点。"""

    del params
    controller = SearchController(budget)
    evaluator = MissionEvaluator(problem)
    label = evaluator.start_label()
    controller.consume()
    while not controller.exhausted:
        candidates = []
        for node in range(evaluator.n):
            appended = evaluator.try_append(label, node)
            if appended is None:
                continue
            if not controller.consume():
                break
            edge = evaluator._segments[(label.last, node)]
            # 差距相同时用节点编号稳定破同，保证跨进程复现。
            candidates.append((float(edge.distance_m), int(node), appended))
        if not candidates:
            break
        _, _, label = min(candidates, key=lambda item: (item[0], item[1]))
    evaluation = evaluator.finish_label(label)
    status = "budget_exhausted" if controller.exhausted else "ok"
    return evaluator.build_result(
        "nearest_feasible", evaluation, controller, seed, status=status
    )


def plan_priority_resource_greedy(
    problem: ProblemInstance,
    *,
    seed: int = 42,
    budget: Optional[PlannerBudget] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> PlanningResult:
    """按优先级任务收益/三资源增量选择，收益不再上升时主动返航。"""

    del params
    controller = SearchController(budget)
    evaluator = MissionEvaluator(problem)
    label = evaluator.start_label()
    current = evaluator.finish_label(label)
    controller.consume()
    while not controller.exhausted:
        candidates = []
        for node in range(evaluator.n):
            appended = evaluator.try_append(label, node)
            if appended is None:
                continue
            if not controller.consume():
                break
            result = evaluator.finish_label(appended)
            resource_delta = (
                max(0.0, result.energy_wh - current.energy_wh) / evaluator.energy_budget_wh
                + max(0.0, result.distance_m - current.distance_m) / evaluator.distance_budget_m
                + max(0.0, result.time_s - current.time_s) / evaluator.time_budget_s
            )
            density = evaluator.marginal_task_gain(node) / max(resource_delta, 1e-12)
            candidates.append((float(density), float(result.objective), -int(node), appended, result))
        if not candidates:
            break
        _, _, _, candidate_label, candidate_result = max(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        if candidate_result.objective <= current.objective + 1e-12:
            break
        label, current = candidate_label, candidate_result
    status = "budget_exhausted" if controller.exhausted else "ok"
    return evaluator.build_result(
        "priority_resource_greedy", current, controller, seed, status=status
    )


__all__ = ["plan_nearest_feasible", "plan_priority_resource_greedy"]

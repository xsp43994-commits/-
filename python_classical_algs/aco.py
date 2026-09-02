#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""面向资源约束定向越野问题的蚁群搜索基线。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Optional

import numpy as np

from .common import (
    ACOConfig,
    MissionEvaluator,
    PlannerBudget,
    PlanningResult,
    ProblemInstance,
    RouteEvaluation,
    SearchController,
)


def plan_aco(
    problem: ProblemInstance,
    *,
    seed: int = 42,
    budget: Optional[PlannerBudget] = None,
    params: Optional[Any] = None,
) -> PlanningResult:
    config = _config(params)
    _validate(config)
    rng = np.random.default_rng(int(seed))
    controller = SearchController(budget)
    evaluator = MissionEvaluator(problem)
    pheromone = np.full(
        (evaluator.n + 1, evaluator.n + 1),
        float(config.initial_pheromone),
        dtype=np.float64,
    )
    best = evaluator.finish_label(evaluator.start_label())

    for _iteration in range(int(config.iterations)):
        if controller.exhausted:
            break
        iteration_best: Optional[RouteEvaluation] = None
        for _ant in range(int(config.ants)):
            if controller.exhausted:
                break
            label = evaluator.start_label()
            while len(label.order) < evaluator.n:
                nodes = []
                weights = []
                for node in range(evaluator.n):
                    appended = evaluator.try_append(label, node)
                    if appended is None:
                        continue
                    edge = evaluator._segments[(label.last, node)]
                    normalized_cost = (
                        (edge.energy_wh + evaluator.service_energy_wh[node])
                        / evaluator.energy_budget_wh
                        + edge.distance_m / evaluator.distance_budget_m
                        + (edge.time_s + evaluator.service_times_s[node])
                        / evaluator.time_budget_s
                    )
                    heuristic = evaluator.marginal_task_gain(node) / max(normalized_cost, 1e-12)
                    desirability = (
                        pheromone[label.last, node] ** float(config.alpha)
                        * max(heuristic, 1e-12) ** float(config.beta)
                    )
                    nodes.append((node, appended))
                    weights.append(desirability)
                if not nodes:
                    break
                # 返航是显式候选动作；ACO可以主动结束，而不是事后删点修路。
                stop_desirability = (
                    float(config.stop_weight)
                    * pheromone[label.last, evaluator.depot] ** float(config.alpha)
                )
                probabilities = np.asarray(weights + [stop_desirability], dtype=np.float64)
                probabilities /= float(np.sum(probabilities))
                selected = int(rng.choice(len(probabilities), p=probabilities))
                if selected == len(nodes):
                    break
                label = nodes[selected][1]
            if not controller.consume():
                break
            candidate = evaluator.finish_label(label)
            if iteration_best is None or _better(candidate, iteration_best):
                iteration_best = candidate
            if _better(candidate, best):
                best = candidate

        pheromone *= 1.0 - float(config.evaporation)
        np.maximum(pheromone, 1e-12, out=pheromone)
        if iteration_best is not None:
            quality = float(config.deposit_scale) * max(iteration_best.objective + 0.35, 1e-6)
            previous = evaluator.depot
            for node in iteration_best.order:
                pheromone[previous, node] += quality
                previous = node
            pheromone[previous, evaluator.depot] += quality

    status = "budget_exhausted" if controller.exhausted else "ok"
    return evaluator.build_result(
        "aco",
        best,
        controller,
        seed,
        status=status,
        metadata={"config": asdict(config)},
    )


def _config(params: Optional[Any]) -> ACOConfig:
    if params is None:
        return ACOConfig()
    if isinstance(params, ACOConfig):
        return params
    if isinstance(params, Mapping):
        return ACOConfig(**dict(params))
    raise TypeError("params 必须是 ACOConfig 或映射。")


def _validate(config: ACOConfig) -> None:
    if config.ants <= 0 or config.iterations <= 0:
        raise ValueError("ACO蚂蚁数和迭代数必须为正。")
    if not 0.0 < config.evaporation < 1.0:
        raise ValueError("ACO蒸发率必须位于(0,1)。")
    if min(config.alpha, config.beta, config.deposit_scale, config.initial_pheromone) <= 0.0:
        raise ValueError("ACO强度、启发指数、沉积量和初始信息素必须为正。")
    if config.stop_weight < 0.0:
        raise ValueError("ACO返航候选权重不能为负。")


def _better(candidate: RouteEvaluation, incumbent: RouteEvaluation) -> bool:
    return (candidate.objective, len(candidate.order), tuple(-i for i in candidate.order)) > (
        incumbent.objective,
        len(incumbent.order),
        tuple(-i for i in incumbent.order),
    )


AntColonyOptimization = plan_aco

__all__ = ["AntColonyOptimization", "plan_aco"]

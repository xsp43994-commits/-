#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""排列＋前缀长度编码的模拟退火基线。"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .common import (
    MissionEvaluator,
    PlannerBudget,
    PlanningResult,
    ProblemInstance,
    RouteEvaluation,
    SAConfig,
    SearchController,
)


def plan_sa(
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
    order = np.lexsort((np.arange(evaluator.n), -problem.priorities)).astype(int)
    # 从显式空前缀开始，避免把不可行的优先级全序列当作已修复初始解。
    stop = 0
    if not controller.consume():
        empty = evaluator.finish_label(evaluator.start_label())
        return evaluator.build_result(
            "sa",
            empty,
            controller,
            seed,
            status="budget_exhausted",
            metadata={"config": asdict(config), "encoding": "permutation_plus_prefix_length"},
        )
    current = evaluator.evaluate_order(order.tolist(), prefix_length=stop)
    best = current

    for iteration in range(1, int(config.iterations)):
        if controller.exhausted:
            break
        # restart_interval 表示周期性重新升温；不替换当前解，保持状态与分数一致。
        candidate_order, candidate_stop = _neighbor(order, stop, rng)
        if not controller.consume():
            break
        candidate = evaluator.evaluate_order(
            candidate_order.tolist(), prefix_length=candidate_stop
        )
        phase = (iteration % int(config.restart_interval)) / max(
            int(config.restart_interval) - 1, 1
        )
        temperature = float(config.initial_temperature) * (
            float(config.final_temperature) / float(config.initial_temperature)
        ) ** phase
        delta = candidate.objective - current.objective
        if delta >= 0.0 or rng.random() < math.exp(max(delta / max(temperature, 1e-12), -700.0)):
            order, stop, current = candidate_order, candidate_stop, candidate
        if _better(candidate, best):
            best = candidate

    status = "budget_exhausted" if controller.exhausted else "ok"
    return evaluator.build_result(
        "sa",
        best,
        controller,
        seed,
        status=status,
        metadata={"config": asdict(config), "encoding": "permutation_plus_prefix_length"},
    )


def _neighbor(
    order: np.ndarray, stop: int, rng: np.random.Generator
) -> Tuple[np.ndarray, int]:
    candidate = order.copy()
    n = int(candidate.size)
    move = int(rng.integers(0, 3))
    if move == 0 and n >= 2:
        first, second = rng.choice(n, size=2, replace=False)
        candidate[int(first)], candidate[int(second)] = candidate[int(second)], candidate[int(first)]
    elif move == 1 and n >= 2:
        left, right = sorted(rng.choice(n, size=2, replace=False).tolist())
        candidate[left : right + 1] = candidate[left : right + 1][::-1]
    else:
        stop = int(np.clip(stop + rng.choice([-2, -1, 1, 2]), 0, n))
    return candidate, int(stop)


def _config(params: Optional[Any]) -> SAConfig:
    if params is None:
        return SAConfig()
    if isinstance(params, SAConfig):
        return params
    if isinstance(params, Mapping):
        return SAConfig(**dict(params))
    raise TypeError("params 必须是 SAConfig 或映射。")


def _validate(config: SAConfig) -> None:
    if config.iterations <= 0 or config.restart_interval <= 0:
        raise ValueError("SA迭代数和重启间隔必须为正。")
    if not 0.0 < config.final_temperature < config.initial_temperature:
        raise ValueError("SA温度必须满足 0 < final < initial。")


def _better(candidate: RouteEvaluation, incumbent: RouteEvaluation) -> bool:
    return (candidate.objective, len(candidate.order), tuple(-i for i in candidate.order)) > (
        incumbent.objective,
        len(incumbent.order),
        tuple(-i for i in incumbent.order),
    )


SimulatedAnnealing = plan_sa

__all__ = ["SimulatedAnnealing", "plan_sa"]

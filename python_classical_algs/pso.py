#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""random-key排列编码的粒子群优化基线。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .common import (
    MissionEvaluator,
    PlannerBudget,
    PlanningResult,
    ProblemInstance,
    PSOConfig,
    RouteEvaluation,
    SearchController,
)


def plan_pso(
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
    dimensions = evaluator.n + 1
    positions = rng.random((int(config.swarm_size), dimensions))
    velocities = rng.uniform(
        -float(config.max_velocity),
        float(config.max_velocity),
        size=positions.shape,
    )
    # 第一个粒子使用优先级排序作初始化种子，后续仍由标准PSO更新。
    priority_order = np.lexsort((np.arange(evaluator.n), -problem.priorities))
    positions[0, priority_order] = np.linspace(0.0, 0.95, evaluator.n)
    positions[0, -1] = np.nextafter(1.0, 0.0)
    personal_positions = positions.copy()
    personal_scores = np.full(int(config.swarm_size), -np.inf, dtype=np.float64)
    global_position = positions[0].copy()
    global_score = -np.inf
    best = evaluator.finish_label(evaluator.start_label())

    for _iteration in range(int(config.iterations)):
        if controller.exhausted:
            break
        for particle in range(int(config.swarm_size)):
            if not controller.consume():
                break
            order, stop = _decode(positions[particle], evaluator.n)
            candidate = evaluator.evaluate_order(order.tolist(), prefix_length=stop)
            score = float(candidate.objective)
            if score > personal_scores[particle]:
                personal_scores[particle] = score
                personal_positions[particle] = positions[particle].copy()
            if score > global_score:
                global_score = score
                global_position = positions[particle].copy()
            if _better(candidate, best):
                best = candidate
        if controller.exhausted:
            break
        r_personal = rng.random(positions.shape)
        r_global = rng.random(positions.shape)
        velocities = (
            float(config.inertia) * velocities
            + float(config.cognitive) * r_personal * (personal_positions - positions)
            + float(config.social) * r_global * (global_position - positions)
        )
        np.clip(
            velocities,
            -float(config.max_velocity),
            float(config.max_velocity),
            out=velocities,
        )
        positions += velocities
        np.clip(positions, 0.0, np.nextafter(1.0, 0.0), out=positions)

    status = "budget_exhausted" if controller.exhausted else "ok"
    return evaluator.build_result(
        "pso",
        best,
        controller,
        seed,
        status=status,
        metadata={"config": asdict(config), "encoding": "random_keys_plus_prefix_key"},
    )


def _decode(position: np.ndarray, point_count: int) -> Tuple[np.ndarray, int]:
    order = np.argsort(np.asarray(position[:point_count]), kind="stable").astype(int)
    raw_stop = int(float(position[point_count]) * (point_count + 1))
    return order, int(np.clip(raw_stop, 0, point_count))


def _config(params: Optional[Any]) -> PSOConfig:
    if params is None:
        return PSOConfig()
    if isinstance(params, PSOConfig):
        return params
    if isinstance(params, Mapping):
        return PSOConfig(**dict(params))
    raise TypeError("params 必须是 PSOConfig 或映射。")


def _validate(config: PSOConfig) -> None:
    if config.swarm_size <= 0 or config.iterations <= 0:
        raise ValueError("PSO粒子数和迭代数必须为正。")
    if min(config.inertia, config.cognitive, config.social, config.max_velocity) <= 0.0:
        raise ValueError("PSO惯性、学习因子和最大速度必须为正。")


def _better(candidate: RouteEvaluation, incumbent: RouteEvaluation) -> bool:
    return (candidate.objective, len(candidate.order), tuple(-i for i in candidate.order)) > (
        incumbent.objective,
        len(incumbent.order),
        tuple(-i for i in incumbent.order),
    )


ParticleSwarmOptimization = plan_pso

__all__ = ["ParticleSwarmOptimization", "plan_pso"]

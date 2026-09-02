#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""排列＋显式前缀长度编码的遗传算法基线。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np

from .common import (
    GAConfig,
    MissionEvaluator,
    PlannerBudget,
    PlanningResult,
    ProblemInstance,
    RouteEvaluation,
    SearchController,
)


Chromosome = Tuple[np.ndarray, int]


def plan_ga(
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
    population = _initial_population(evaluator, config, rng)
    best = evaluator.finish_label(evaluator.start_label())

    for _generation in range(int(config.generations)):
        scored: List[Tuple[RouteEvaluation, Chromosome]] = []
        for order, stop in population:
            if not controller.consume():
                break
            evaluation = evaluator.evaluate_order(order.tolist(), prefix_length=stop)
            scored.append((evaluation, (order, stop)))
            if _better(evaluation, best):
                best = evaluation
        if controller.exhausted or not scored:
            break
        scored.sort(
            key=lambda item: (
                item[0].objective,
                len(item[0].order),
                tuple(-i for i in item[0].order),
            ),
            reverse=True,
        )
        elites = [
            (item[1][0].copy(), int(item[1][1]))
            for item in scored[: int(config.elite_count)]
        ]
        population = elites
        while len(population) < int(config.population_size):
            parent_a = _tournament(scored, int(config.tournament_size), rng)
            parent_b = _tournament(scored, int(config.tournament_size), rng)
            if rng.random() < float(config.crossover_rate):
                child_order = _order_crossover(parent_a[0], parent_b[0], rng)
                child_stop = int(parent_a[1] if rng.random() < 0.5 else parent_b[1])
            else:
                child_order = parent_a[0].copy()
                child_stop = int(parent_a[1])
            if rng.random() < float(config.mutation_rate):
                child_order, child_stop = _mutate(child_order, child_stop, rng)
            population.append((child_order, child_stop))

    status = "budget_exhausted" if controller.exhausted else "ok"
    return evaluator.build_result(
        "ga",
        best,
        controller,
        seed,
        status=status,
        metadata={"config": asdict(config), "encoding": "permutation_plus_prefix_length"},
    )


def _initial_population(
    evaluator: MissionEvaluator, config: GAConfig, rng: np.random.Generator
) -> List[Chromosome]:
    n = evaluator.n
    population: List[Chromosome] = []
    # 两个确定性启发式种子只参与初始化，不会在输出后补点或修路。
    priority_order = np.lexsort((np.arange(n), -evaluator.problem.priorities)).astype(int)
    population.append((priority_order, n))
    if config.population_size > 1:
        population.append((np.arange(n, dtype=int), n))
    while len(population) < int(config.population_size):
        population.append((rng.permutation(n).astype(int), int(rng.integers(0, n + 1))))
    return population


def _tournament(
    scored: List[Tuple[RouteEvaluation, Chromosome]],
    size: int,
    rng: np.random.Generator,
) -> Chromosome:
    indices = rng.integers(0, len(scored), size=size)
    selected = max(
        (scored[int(index)] for index in indices),
        key=lambda item: (item[0].objective, len(item[0].order)),
    )[1]
    return selected[0].copy(), int(selected[1])


def _order_crossover(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    n = int(parent_a.size)
    if n < 2:
        return parent_a.copy()
    left, right = sorted(rng.choice(n, size=2, replace=False).tolist())
    right += 1
    child = np.full(n, -1, dtype=int)
    child[left:right] = parent_a[left:right]
    used = set(int(value) for value in child[left:right])
    fill = [int(value) for value in parent_b if int(value) not in used]
    slots = [index for index in range(n) if child[index] < 0]
    child[slots] = fill
    return child


def _mutate(
    order: np.ndarray, stop: int, rng: np.random.Generator
) -> Chromosome:
    result = order.copy()
    n = int(result.size)
    if n >= 2 and rng.random() < 0.75:
        first, second = rng.choice(n, size=2, replace=False)
        result[int(first)], result[int(second)] = result[int(second)], result[int(first)]
    else:
        stop = int(np.clip(stop + rng.choice([-2, -1, 1, 2]), 0, n))
    return result, int(stop)


def _config(params: Optional[Any]) -> GAConfig:
    if params is None:
        return GAConfig()
    if isinstance(params, GAConfig):
        return params
    if isinstance(params, Mapping):
        return GAConfig(**dict(params))
    raise TypeError("params 必须是 GAConfig 或映射。")


def _validate(config: GAConfig) -> None:
    if config.population_size < 2 or config.generations <= 0:
        raise ValueError("GA种群至少为2且代数必须为正。")
    if not 1 <= config.elite_count < config.population_size:
        raise ValueError("GA elite_count 必须位于[1,population_size)。")
    if not 1 <= config.tournament_size <= config.population_size:
        raise ValueError("GA tournament_size 超出种群范围。")
    if not 0.0 <= config.crossover_rate <= 1.0 or not 0.0 <= config.mutation_rate <= 1.0:
        raise ValueError("GA交叉率和变异率必须位于[0,1]。")


def _better(candidate: RouteEvaluation, incumbent: RouteEvaluation) -> bool:
    return (candidate.objective, len(candidate.order), tuple(-i for i in candidate.order)) > (
        incumbent.objective,
        len(incumbent.order),
        tuple(-i for i in incumbent.order),
    )


GeneticAlgorithm = plan_ga

__all__ = ["GeneticAlgorithm", "plan_ga"]

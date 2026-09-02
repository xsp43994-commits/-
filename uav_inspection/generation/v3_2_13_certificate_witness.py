#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic lower-bound witnesses for pre-test task certification.

This module is deliberately separate from every planner evaluated in the
paper.  It does not estimate an optimum and is never entered into result
tables.  Its only job is to construct one replay-valid route, which is a
mathematically sufficient lower bound when an independent MILP certificate
already supplies the upper bound.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp as scipy_milp
from scipy.sparse import csr_matrix

from uav_inspection.core import final_python_ppo_pointer as ppo
from python_classical_algs.common import (
    MissionEvaluator,
    ProblemInstance,
    RouteEvaluation,
    RouteLabel,
    make_problem,
)
from python_classical_algs.milp import _build_model, _extract_order


# 论文前测试集认证专用参数。增大它们只影响找“可行见证路线”的速度与成功率，
# 不改变任务难度上界、模型、奖励、正式评价预算或任何论文算法的参数。
DEFAULT_BEAM_WIDTH = 32_768
DEFAULT_MAX_LABEL_EXPANSIONS = 5_000_000
DEFAULT_TIME_LIMIT_S = 45.0


@dataclass(frozen=True)
class WitnessSearchResult:
    evaluation: Optional[RouteEvaluation]
    best_partial_evaluation: Optional[RouteEvaluation]
    best_partial_priority_weight: float
    expansions: int
    runtime_s: float
    exhausted_reason: str

    def as_dict(self) -> Dict[str, Any]:
        evaluation = self.evaluation
        best_partial = self.best_partial_evaluation
        return {
            "found": evaluation is not None,
            "visit_order": (
                list(evaluation.order) if evaluation is not None else None
            ),
            "weighted_coverage": (
                float(evaluation.weighted_coverage)
                if evaluation is not None
                else None
            ),
            "energy_wh": (
                float(evaluation.energy_wh) if evaluation is not None else None
            ),
            "distance_m": (
                float(evaluation.distance_m)
                if evaluation is not None
                else None
            ),
            "time_s": (
                float(evaluation.time_s) if evaluation is not None else None
            ),
            "returned": (
                bool(evaluation.returned) if evaluation is not None else False
            ),
            "best_partial_visit_order": (
                list(best_partial.order) if best_partial is not None else None
            ),
            "best_partial_weighted_coverage": (
                float(best_partial.weighted_coverage)
                if best_partial is not None
                else None
            ),
            "best_partial_priority_weight": float(
                self.best_partial_priority_weight
            ),
            "expansions": int(self.expansions),
            "runtime_s": float(self.runtime_s),
            "exhausted_reason": str(self.exhausted_reason),
        }


def build_frozen_problem(
    record: Mapping[str, Any],
    provider: Any,
) -> ProblemInstance:
    """Build exactly the frozen scenario used by the task certifier."""

    context = provider(record)
    base_cfg = ppo.resolve_config(
        {
            "reward_schema": "multimap_v3_1",
            "coordinate_scale_m_per_unit": context["cfg_overrides"][
                "coordinate_scale_m_per_unit"
            ],
            "point_z_mode": "terrain",
            "terrain_clearance_m": 18.0,
            "service_times_s": record["service_times_s"],
        }
    )
    scenario_cfg, scenario_wind = ppo.apply_frozen_domain_instance(
        base_cfg, context["wind_data"], record
    )
    return make_problem(
        context["start_pos"],
        np.asarray(record["inspection_points_xyz"], dtype=np.float32),
        np.asarray(record["priorities"], dtype=np.float32),
        context["terrain"],
        scenario_cfg,
        scenario_wind,
        name=str(record["id"]),
    )


def _label_score(
    evaluator: MissionEvaluator,
    label: RouteLabel,
) -> Tuple[float, float, float, Tuple[int, ...]]:
    """Rank only search labels; acceptance never depends on this score."""

    evaluation = evaluator.finish_label(label)
    max_utilization = max(
        evaluation.energy_wh / evaluator.energy_budget_wh,
        evaluation.distance_m / evaluator.distance_budget_m,
        evaluation.time_s / evaluator.time_budget_s,
    )
    utilization_sum = (
        evaluation.energy_wh / evaluator.energy_budget_wh
        + evaluation.distance_m / evaluator.distance_budget_m
        + evaluation.time_s / evaluator.time_budget_s
    )
    # 每一层访问点数相同：优先保留权重高、资源余量大的标签，并以节点序列稳定破同。
    priority_weight = float(
        sum(evaluator._priority_nonnegative[node] for node in label.order)
    )
    return (
        -priority_weight,
        float(max_utilization),
        float(utilization_sum),
        tuple(label.order),
    )


def _greedy_witness(
    evaluator: MissionEvaluator,
    required: float,
) -> Tuple[Optional[RouteEvaluation], RouteEvaluation, int]:
    """Try a fixed family of cheap constructive orders before beam search."""

    # 这些权重只决定证书见证路线的搜索次序，不进入任务接受条件。
    resource_weights = (
        (1.0, 1.0, 1.0),
        (3.0, 1.0, 1.0),
        (1.0, 3.0, 1.0),
        (1.0, 1.0, 3.0),
        (2.0, 2.0, 1.0),
        (2.0, 1.0, 2.0),
        (1.0, 2.0, 2.0),
    )
    priority_powers = (0.5, 1.0, 1.5, 2.0, 3.0)
    best = evaluator.finish_label(evaluator.start_label())
    evaluations = 0
    first_nodes = (None,) + tuple(range(evaluator.n))
    for first in first_nodes:
        for weights in resource_weights:
            for power in priority_powers:
                label = evaluator.start_label()
                if first is not None:
                    appended = evaluator.try_append(label, first)
                    evaluations += 1
                    if appended is None:
                        continue
                    label = appended
                while True:
                    current = evaluator.finish_label(label)
                    candidates = []
                    for node in range(evaluator.n):
                        if label.mask & (1 << node):
                            continue
                        appended = evaluator.try_append(label, node)
                        evaluations += 1
                        if appended is None:
                            continue
                        finished = evaluator.finish_label(appended)
                        delta = (
                            weights[0]
                            * max(0.0, finished.energy_wh - current.energy_wh)
                            / evaluator.energy_budget_wh
                            + weights[1]
                            * max(
                                0.0,
                                finished.distance_m - current.distance_m,
                            )
                            / evaluator.distance_budget_m
                            + weights[2]
                            * max(0.0, finished.time_s - current.time_s)
                            / evaluator.time_budget_s
                        )
                        priority = float(
                            evaluator._priority_nonnegative[node]
                        )
                        density = priority**power / max(delta, 1e-12)
                        candidates.append(
                            (
                                -density,
                                -priority,
                                float(delta),
                                int(node),
                                appended,
                                finished,
                            )
                        )
                    if not candidates:
                        break
                    (
                        _density,
                        _priority,
                        _delta,
                        _node,
                        label,
                        current,
                    ) = min(candidates, key=lambda item: item[:4])
                    current_priority = float(
                        sum(
                            evaluator._priority_nonnegative[index]
                            for index in label.order
                        )
                    )
                    best_priority = float(
                        sum(
                            evaluator._priority_nonnegative[index]
                            for index in best.order
                        )
                    )
                    if current_priority > best_priority + 1e-9:
                        best = current
                    if current_priority + 1e-9 >= required:
                        return current, best, evaluations
    return None, best, evaluations


def construct_threshold_witness(
    problem: ProblemInstance,
    *,
    minimum_priority_weight: float,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    max_label_expansions: int = DEFAULT_MAX_LABEL_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> WitnessSearchResult:
    """Find one safe route meeting a discrete priority threshold.

    Every retained label has already passed the shared environment's
    return-reserve checks through ``MissionEvaluator.try_append``.  The final
    route is replayed once more by ``finish_label`` before it is returned.
    """

    required = float(minimum_priority_weight)
    width = int(beam_width)
    limit = int(max_label_expansions)
    wall_limit = float(time_limit_s)
    if not math.isfinite(required) or required <= 0.0:
        raise ValueError("minimum_priority_weight must be finite and positive")
    if width <= 0 or limit <= 0 or wall_limit <= 0.0:
        raise ValueError("beam search budgets must be positive")

    evaluator = MissionEvaluator(problem)
    started = time.perf_counter()
    greedy, best_partial, expansions = _greedy_witness(
        evaluator, required
    )
    if greedy is not None:
        return WitnessSearchResult(
            evaluation=greedy,
            best_partial_evaluation=best_partial,
            best_partial_priority_weight=float(
                sum(
                    evaluator._priority_nonnegative[index]
                    for index in best_partial.order
                )
            ),
            expansions=expansions,
            runtime_s=float(time.perf_counter() - started),
            exhausted_reason="greedy_threshold_reached",
        )
    beam = [evaluator.start_label()]
    best: Optional[RouteEvaluation] = None
    best_partial_weight = float(
        sum(
            evaluator._priority_nonnegative[index]
            for index in best_partial.order
        )
    )
    reason = "state_space_exhausted"

    for _depth in range(1, evaluator.n + 1):
        next_labels: Dict[Tuple[int, int], RouteLabel] = {}
        for label in beam:
            for node in range(evaluator.n):
                if expansions >= limit:
                    reason = "max_label_expansions"
                    break
                if time.perf_counter() - started >= wall_limit:
                    reason = "time_limit"
                    break
                if label.mask & (1 << node):
                    continue
                expansions += 1
                appended = evaluator.try_append(label, node)
                if appended is None:
                    continue
                priority_weight = float(
                    sum(
                        evaluator._priority_nonnegative[index]
                        for index in appended.order
                    )
                )
                if priority_weight > best_partial_weight + 1e-9:
                    best_partial = evaluator.finish_label(appended)
                    best_partial_weight = priority_weight
                if priority_weight + 1e-9 >= required:
                    evaluation = evaluator.finish_label(appended)
                    if evaluation.returned and (
                        best is None
                        or _label_score(evaluator, appended)
                        < (
                            -sum(
                                evaluator._priority_nonnegative[index]
                                for index in best.order
                            ),
                            max(
                                best.energy_wh / evaluator.energy_budget_wh,
                                best.distance_m
                                / evaluator.distance_budget_m,
                                best.time_s / evaluator.time_budget_s,
                            ),
                            best.energy_wh / evaluator.energy_budget_wh
                            + best.distance_m / evaluator.distance_budget_m
                            + best.time_s / evaluator.time_budget_s,
                            tuple(best.order),
                        )
                    ):
                        best = evaluation
                    # 第一个达到阈值的深度已经最少访问节点；继续本层可提高余量。
                    continue
                key = (int(appended.mask), int(appended.last))
                previous = next_labels.get(key)
                if previous is None or _label_score(
                    evaluator, appended
                ) < _label_score(evaluator, previous):
                    next_labels[key] = appended
            if reason in {"max_label_expansions", "time_limit"}:
                break
        if best is not None:
            reason = "threshold_reached"
            break
        if reason in {"max_label_expansions", "time_limit"}:
            break
        if not next_labels:
            reason = "state_space_exhausted"
            break
        beam = sorted(
            next_labels.values(),
            key=lambda item: _label_score(evaluator, item),
        )[:width]

    return WitnessSearchResult(
        evaluation=best,
        best_partial_evaluation=best_partial,
        best_partial_priority_weight=float(best_partial_weight),
        expansions=expansions,
        runtime_s=float(time.perf_counter() - started),
        exhausted_reason=reason,
    )


def prove_priority_threshold_infeasible(
    problem: ProblemInstance,
    *,
    minimum_priority_weight: float,
    time_limit_s: float,
) -> Dict[str, Any]:
    """Solve the direct threshold feasibility problem with all budgets active."""

    evaluator = MissionEvaluator(problem)
    required = float(minimum_priority_weight)
    total_priority = float(np.sum(evaluator._priority_nonnegative))
    if not 0.0 < required <= total_priority + 1e-9:
        raise ValueError("priority threshold is outside the valid range")
    (
        _objective,
        integrality,
        bounds,
        constraints,
        indices,
        _empty_objective,
    ) = _build_model(evaluator, objective_mode="weighted_coverage")
    threshold_row = np.zeros(indices.size, dtype=np.float64)
    for node, column in enumerate(indices.visits):
        threshold_row[column] = float(
            evaluator._priority_nonnegative[node]
        )
    threshold_constraint = LinearConstraint(
        threshold_row, required, math.inf
    )
    # 零目标只回答“是否存在”，避免把计算花在与证书无关的二次优化上。
    feasibility_objective = np.zeros(indices.size, dtype=np.float64)
    started = time.perf_counter()
    result = scipy_milp(
        feasibility_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=[constraints, threshold_constraint],
        options={
            "time_limit": float(time_limit_s),
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    runtime_s = float(time.perf_counter() - started)
    status = int(result.status) if result.status is not None else None
    order = (
        _extract_order(result.x, indices, evaluator.n)
        if getattr(result, "x", None) is not None
        else None
    )
    evaluation = (
        evaluator.evaluate_order(order) if order is not None else None
    )
    return {
        "minimum_priority_weight": required,
        "required_weighted_coverage": required / total_priority,
        "total_priority": total_priority,
        "threshold_infeasible": bool(status == 2),
        "solver_status": status,
        "solver_success": bool(getattr(result, "success", False)),
        "solver_message": str(getattr(result, "message", "")),
        "runtime_s": runtime_s,
        "visit_order": list(order) if order is not None else None,
        "feasible_route_evaluation": (
            {
                "returned": bool(evaluation.returned),
                "weighted_coverage": float(
                    evaluation.weighted_coverage
                ),
                "energy_wh": float(evaluation.energy_wh),
                "distance_m": float(evaluation.distance_m),
                "time_s": float(evaluation.time_s),
                "visit_order": list(evaluation.order),
            }
            if evaluation is not None
            else None
        ),
    }


def prove_threshold_assignment_relaxation_infeasible(
    problem: ProblemInstance,
    *,
    minimum_priority_weight: float,
    time_limit_s: float,
) -> Dict[str, Any]:
    """Prove infeasibility in a superset that omits all subtour constraints."""

    evaluator = MissionEvaluator(problem)
    required = float(minimum_priority_weight)
    total_priority = float(np.sum(evaluator._priority_nonnegative))
    depot = evaluator.depot
    vertices = range(evaluator.n + 1)
    arcs = [
        (source, target)
        for source in vertices
        for target in vertices
        if source != target
        and (source, target) in evaluator._segments
        and evaluator._segments[(source, target)].feasible
    ]
    arc_index = {arc: index for index, arc in enumerate(arcs)}
    visit_offset = len(arcs)
    size = visit_offset + evaluator.n
    rows = []
    lower = []
    upper = []

    def add_row(coefficients: Dict[int, float], low: float, high: float) -> None:
        rows.append(coefficients)
        lower.append(float(low))
        upper.append(float(high))

    for node in range(evaluator.n):
        incoming = {
            arc_index[arc]: 1.0
            for arc in arcs
            if arc[1] == node
        }
        incoming[visit_offset + node] = -1.0
        add_row(incoming, 0.0, 0.0)
        outgoing = {
            arc_index[arc]: 1.0
            for arc in arcs
            if arc[0] == node
        }
        outgoing[visit_offset + node] = -1.0
        add_row(outgoing, 0.0, 0.0)
    add_row(
        {
            arc_index[arc]: 1.0
            for arc in arcs
            if arc[0] == depot
        },
        1.0,
        1.0,
    )
    add_row(
        {
            arc_index[arc]: 1.0
            for arc in arcs
            if arc[1] == depot
        },
        1.0,
        1.0,
    )
    add_row(
        {
            visit_offset + node: float(
                evaluator._priority_nonnegative[node]
            )
            for node in range(evaluator.n)
        },
        required,
        math.inf,
    )
    resource_specs = (
        (
            "energy",
            "energy_wh",
            evaluator.service_energy_wh,
            evaluator.energy_budget_wh,
        ),
        (
            "distance",
            "distance_m",
            np.zeros(evaluator.n, dtype=np.float64),
            evaluator.distance_budget_m,
        ),
        (
            "time",
            "time_s",
            evaluator.service_times_s,
            evaluator.time_budget_s,
        ),
    )
    for _name, attribute, service, budget in resource_specs:
        coefficients = {
            arc_index[arc]: float(
                getattr(evaluator._segments[arc], attribute)
            )
            for arc in arcs
        }
        for node in range(evaluator.n):
            coefficients[visit_offset + node] = float(service[node])
        add_row(coefficients, -math.inf, float(budget))

    matrix = np.zeros((len(rows), size), dtype=np.float64)
    for row_index, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
    started = time.perf_counter()
    result = scipy_milp(
        np.zeros(size, dtype=np.float64),
        integrality=np.ones(size, dtype=np.int8),
        bounds=Bounds(np.zeros(size), np.ones(size)),
        constraints=LinearConstraint(
            csr_matrix(matrix),
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        ),
        options={
            "time_limit": float(time_limit_s),
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    runtime_s = float(time.perf_counter() - started)
    status = int(result.status) if result.status is not None else None
    selected = None
    if getattr(result, "x", None) is not None:
        selected = [
            node
            for node in range(evaluator.n)
            if float(result.x[visit_offset + node]) >= 0.5
        ]
    return {
        "relaxation": (
            "binary directed assignment with all three resource budgets; "
            "subtour and connectivity constraints omitted"
        ),
        "minimum_priority_weight": required,
        "required_weighted_coverage": required / total_priority,
        "total_priority": total_priority,
        "relaxation_infeasible": bool(status == 2),
        "original_problem_infeasible_if_true": bool(status == 2),
        "solver_status": status,
        "solver_success": bool(getattr(result, "success", False)),
        "solver_message": str(getattr(result, "message", "")),
        "runtime_s": runtime_s,
        "selected_nodes_if_feasible": selected,
    }


def prove_threshold_flow_infeasible(
    problem: ProblemInstance,
    *,
    minimum_priority_weight: float,
    time_limit_s: float,
    objective_resource: Optional[str] = None,
) -> Dict[str, Any]:
    """Exact additive-resource tour feasibility with one commodity flow."""

    evaluator = MissionEvaluator(problem)
    required = float(minimum_priority_weight)
    total_priority = float(np.sum(evaluator._priority_nonnegative))
    depot = evaluator.depot
    vertices = range(evaluator.n + 1)
    arcs = [
        (source, target)
        for source in vertices
        for target in vertices
        if source != target
        and (source, target) in evaluator._segments
        and evaluator._segments[(source, target)].feasible
    ]
    arc_index = {arc: index for index, arc in enumerate(arcs)}
    visit_offset = len(arcs)
    flow_offset = visit_offset + evaluator.n
    size = flow_offset + len(arcs)
    rows = []
    lower = []
    upper = []

    def add_row(coefficients: Dict[int, float], low: float, high: float) -> None:
        rows.append(coefficients)
        lower.append(float(low))
        upper.append(float(high))

    for node in range(evaluator.n):
        incoming = {
            arc_index[arc]: 1.0 for arc in arcs if arc[1] == node
        }
        incoming[visit_offset + node] = -1.0
        add_row(incoming, 0.0, 0.0)
        outgoing = {
            arc_index[arc]: 1.0 for arc in arcs if arc[0] == node
        }
        outgoing[visit_offset + node] = -1.0
        add_row(outgoing, 0.0, 0.0)
    add_row(
        {
            arc_index[arc]: 1.0
            for arc in arcs
            if arc[0] == depot
        },
        1.0,
        1.0,
    )
    add_row(
        {
            arc_index[arc]: 1.0
            for arc in arcs
            if arc[1] == depot
        },
        1.0,
        1.0,
    )
    add_row(
        {
            visit_offset + node: float(
                evaluator._priority_nonnegative[node]
            )
            for node in range(evaluator.n)
        },
        required,
        math.inf,
    )
    resource_specs = (
        (
            "energy",
            "energy_wh",
            evaluator.service_energy_wh,
            evaluator.energy_budget_wh,
        ),
        (
            "distance",
            "distance_m",
            np.zeros(evaluator.n, dtype=np.float64),
            evaluator.distance_budget_m,
        ),
        (
            "time",
            "time_s",
            evaluator.service_times_s,
            evaluator.time_budget_s,
        ),
    )
    if objective_resource not in {None, "energy", "distance", "time"}:
        raise ValueError("objective_resource is invalid")
    objective = np.zeros(size, dtype=np.float64)
    actual_objective_budget = None
    for resource_name, attribute, service, budget in resource_specs:
        coefficients = {
            arc_index[arc]: float(
                getattr(evaluator._segments[arc], attribute)
            )
            for arc in arcs
        }
        for node in range(evaluator.n):
            coefficients[visit_offset + node] = float(service[node])
        if resource_name == objective_resource:
            actual_objective_budget = float(budget)
            for column, value in coefficients.items():
                objective[column] = value
            continue
        add_row(coefficients, -math.inf, float(budget))

    # 流从返航点出发，每访问一个节点消耗一单位；断开的子环无法获得流。
    for node in range(evaluator.n):
        flow_balance = {
            flow_offset + arc_index[arc]: 1.0
            for arc in arcs
            if arc[1] == node
        }
        for arc in arcs:
            if arc[0] == node:
                column = flow_offset + arc_index[arc]
                flow_balance[column] = flow_balance.get(column, 0.0) - 1.0
        flow_balance[visit_offset + node] = -1.0
        add_row(flow_balance, 0.0, 0.0)
    depot_flow = {}
    for arc in arcs:
        column = flow_offset + arc_index[arc]
        if arc[0] == depot:
            depot_flow[column] = depot_flow.get(column, 0.0) + 1.0
        if arc[1] == depot:
            depot_flow[column] = depot_flow.get(column, 0.0) - 1.0
    for node in range(evaluator.n):
        depot_flow[visit_offset + node] = -1.0
    add_row(depot_flow, 0.0, 0.0)
    for arc, column in arc_index.items():
        add_row(
            {
                flow_offset + column: 1.0,
                column: -float(evaluator.n),
            },
            -math.inf,
            0.0,
        )

    matrix = np.zeros((len(rows), size), dtype=np.float64)
    for row_index, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
    variable_lower = np.zeros(size, dtype=np.float64)
    variable_upper = np.ones(size, dtype=np.float64)
    variable_upper[flow_offset:] = float(evaluator.n)
    integrality = np.zeros(size, dtype=np.int8)
    integrality[:flow_offset] = 1
    started = time.perf_counter()
    result = scipy_milp(
        objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(
            csr_matrix(matrix),
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        ),
        options={
            "time_limit": float(time_limit_s),
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    runtime_s = float(time.perf_counter() - started)
    status = int(result.status) if result.status is not None else None
    primal = (
        float(result.fun)
        if getattr(result, "fun", None) is not None
        and math.isfinite(float(result.fun))
        else None
    )
    raw_dual = getattr(result, "mip_dual_bound", None)
    dual = (
        float(raw_dual)
        if raw_dual is not None and math.isfinite(float(raw_dual))
        else None
    )
    bound_excludes_budget = bool(
        objective_resource is not None
        and actual_objective_budget is not None
        and dual is not None
        and dual > actual_objective_budget + 1e-7
    )
    return {
        "formulation": (
            "exact directed visit/degree model with additive energy, "
            "distance, time, priority threshold, and single commodity flow"
        ),
        "minimum_priority_weight": required,
        "required_weighted_coverage": required / total_priority,
        "total_priority": total_priority,
        "objective_resource": objective_resource,
        "actual_objective_resource_budget": actual_objective_budget,
        "resource_primal_value": primal,
        "resource_dual_bound": dual,
        "bound_excludes_actual_budget": bound_excludes_budget,
        "threshold_infeasible": bool(status == 2 or bound_excludes_budget),
        "solver_status": status,
        "solver_success": bool(getattr(result, "success", False)),
        "solver_message": str(getattr(result, "message", "")),
        "runtime_s": runtime_s,
        "has_feasible_vector": bool(getattr(result, "x", None) is not None),
    }


def prove_threshold_by_subtour_cuts(
    problem: ProblemInstance,
    *,
    minimum_priority_weight: float,
    time_limit_s: float,
    maximum_iterations: int = 1000,
    initial_cut_node_sets: Tuple[Tuple[int, ...], ...] = (),
) -> Dict[str, Any]:
    """Exact branch-and-cut loop over the optional-node assignment model."""

    evaluator = MissionEvaluator(problem)
    required = float(minimum_priority_weight)
    total_priority = float(np.sum(evaluator._priority_nonnegative))
    depot = evaluator.depot
    vertices = range(evaluator.n + 1)
    arcs = [
        (source, target)
        for source in vertices
        for target in vertices
        if source != target
        and (source, target) in evaluator._segments
        and evaluator._segments[(source, target)].feasible
    ]
    arc_index = {arc: index for index, arc in enumerate(arcs)}
    visit_offset = len(arcs)
    size = visit_offset + evaluator.n
    base_rows = []
    base_lower = []
    base_upper = []

    def add_base(
        coefficients: Dict[int, float], low: float, high: float
    ) -> None:
        base_rows.append(coefficients)
        base_lower.append(float(low))
        base_upper.append(float(high))

    for node in range(evaluator.n):
        incoming = {
            arc_index[arc]: 1.0 for arc in arcs if arc[1] == node
        }
        incoming[visit_offset + node] = -1.0
        add_base(incoming, 0.0, 0.0)
        outgoing = {
            arc_index[arc]: 1.0 for arc in arcs if arc[0] == node
        }
        outgoing[visit_offset + node] = -1.0
        add_base(outgoing, 0.0, 0.0)
    add_base(
        {
            arc_index[arc]: 1.0
            for arc in arcs
            if arc[0] == depot
        },
        1.0,
        1.0,
    )
    add_base(
        {
            arc_index[arc]: 1.0
            for arc in arcs
            if arc[1] == depot
        },
        1.0,
        1.0,
    )
    add_base(
        {
            visit_offset + node: float(
                evaluator._priority_nonnegative[node]
            )
            for node in range(evaluator.n)
        },
        required,
        math.inf,
    )
    for attribute, service, budget in (
        (
            "energy_wh",
            evaluator.service_energy_wh,
            evaluator.energy_budget_wh,
        ),
        (
            "distance_m",
            np.zeros(evaluator.n, dtype=np.float64),
            evaluator.distance_budget_m,
        ),
        (
            "time_s",
            evaluator.service_times_s,
            evaluator.time_budget_s,
        ),
    ):
        coefficients = {
            arc_index[arc]: float(
                getattr(evaluator._segments[arc], attribute)
            )
            for arc in arcs
        }
        for node in range(evaluator.n):
            coefficients[visit_offset + node] = float(service[node])
        add_base(coefficients, -math.inf, float(budget))

    cut_sets = {
        tuple(sorted(int(node) for node in raw))
        for raw in initial_cut_node_sets
        if raw and depot not in raw
    }
    cuts = [
        (
            {
                arc_index[arc]: 1.0
                for arc in arcs
                if arc[0] in key and arc[1] in key
            },
            float(len(key) - 1),
        )
        for key in sorted(cut_sets)
    ]
    started = time.perf_counter()
    iterations = 0
    last_status = None
    last_message = ""
    connected_order = None
    while iterations < int(maximum_iterations):
        remaining = float(time_limit_s) - (
            time.perf_counter() - started
        )
        if remaining <= 0.0:
            break
        all_rows = list(base_rows) + [item[0] for item in cuts]
        matrix = np.zeros((len(all_rows), size), dtype=np.float64)
        for row_index, coefficients in enumerate(all_rows):
            for column, value in coefficients.items():
                matrix[row_index, column] = value
        lower = np.asarray(
            base_lower + [-math.inf] * len(cuts), dtype=np.float64
        )
        upper = np.asarray(
            base_upper
            + [float(item[1]) for item in cuts],
            dtype=np.float64,
        )
        result = scipy_milp(
            np.zeros(size, dtype=np.float64),
            integrality=np.ones(size, dtype=np.int8),
            bounds=Bounds(np.zeros(size), np.ones(size)),
            constraints=LinearConstraint(
                csr_matrix(matrix), lower, upper
            ),
            options={
                "time_limit": remaining,
                "mip_rel_gap": 0.0,
                "presolve": True,
            },
        )
        iterations += 1
        last_status = (
            int(result.status) if result.status is not None else None
        )
        last_message = str(getattr(result, "message", ""))
        if last_status == 2:
            break
        if getattr(result, "x", None) is None:
            break
        successor = {
            source: target
            for (source, target), column in arc_index.items()
            if float(result.x[column]) >= 0.5
        }
        cycles = []
        unseen = set(successor)
        while unseen:
            start = min(unseen)
            cycle = []
            current = start
            while current not in cycle and current in successor:
                cycle.append(current)
                unseen.discard(current)
                current = successor[current]
            if current in cycle:
                cycle = cycle[cycle.index(current) :]
            cycles.append(tuple(cycle))
        depot_cycle = next(
            (cycle for cycle in cycles if depot in cycle), ()
        )
        selected_count = int(
            sum(
                float(result.x[visit_offset + node]) >= 0.5
                for node in range(evaluator.n)
            )
        )
        if depot_cycle and len(depot_cycle) == selected_count + 1:
            connected_order = []
            current = successor[depot]
            while current != depot:
                connected_order.append(int(current))
                current = successor[current]
            break
        added = 0
        for cycle in cycles:
            if depot in cycle or len(cycle) <= 1:
                continue
            key = tuple(sorted(int(node) for node in cycle))
            if key in cut_sets:
                continue
            cut_sets.add(key)
            cuts.append(
                (
                    {
                        arc_index[arc]: 1.0
                        for arc in arcs
                        if arc[0] in key and arc[1] in key
                    },
                    float(len(key) - 1),
                )
            )
            added += 1
        if added == 0:
            break

    runtime_s = float(time.perf_counter() - started)
    connected_evaluation = (
        evaluator.evaluate_order(connected_order)
        if connected_order is not None
        else None
    )
    return {
        "formulation": (
            "exact optional-node assignment with iterative directed subtour "
            "elimination cuts and all three resource budgets"
        ),
        "minimum_priority_weight": required,
        "required_weighted_coverage": required / total_priority,
        "total_priority": total_priority,
        "threshold_infeasible": bool(last_status == 2),
        "solver_status": last_status,
        "solver_message": last_message,
        "runtime_s": runtime_s,
        "iterations": iterations,
        "subtour_cut_count": len(cuts),
        "subtour_cut_node_sets": [
            list(key) for key in sorted(cut_sets)
        ],
        "connected_route": connected_order,
        "connected_route_returned": bool(
            connected_evaluation is not None
            and connected_evaluation.returned
        ),
    }


__all__ = [
    "DEFAULT_BEAM_WIDTH",
    "DEFAULT_MAX_LABEL_EXPANSIONS",
    "DEFAULT_TIME_LIMIT_S",
    "WitnessSearchResult",
    "build_frozen_problem",
    "construct_threshold_witness",
    "prove_priority_threshold_infeasible",
    "prove_threshold_assignment_relaxation_infeasible",
    "prove_threshold_flow_infeasible",
    "prove_threshold_by_subtour_cuts",
]

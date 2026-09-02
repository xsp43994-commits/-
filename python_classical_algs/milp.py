#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于 SciPy/HiGHS 的有向资源约束 Orienteering MILP 参考。"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp as scipy_milp
from scipy.sparse import coo_matrix

from .common import (
    EPS,
    MissionEvaluator,
    PlannerBudget,
    PlanningResult,
    ProblemInstance,
    RouteEvaluation,
    SearchController,
)


@dataclass(frozen=True)
class MILPConfig:
    """HiGHS 求解与严格最优认证参数。"""

    mip_rel_gap: float = 0.0
    presolve: bool = True
    objective_mode: str = "mission_objective"

    def __post_init__(self) -> None:
        if not math.isfinite(self.mip_rel_gap) or self.mip_rel_gap < 0.0:
            raise ValueError("mip_rel_gap 必须是有限非负数。")
        if self.objective_mode not in {
            "mission_objective",
            "weighted_coverage",
        }:
            raise ValueError(
                "objective_mode只能是'mission_objective'或'weighted_coverage'。"
            )


@dataclass(frozen=True)
class _VariableIndex:
    arcs: Mapping[Tuple[int, int], int]
    visits: Tuple[int, ...]
    route_used: int
    orders: Tuple[int, ...]
    cumulative: Mapping[str, Tuple[int, ...]]
    flows: Mapping[Tuple[int, int], int]
    size: int


class _ConstraintBuilder:
    def __init__(self, variable_count: int) -> None:
        self.variable_count = int(variable_count)
        self.row_indices: List[int] = []
        self.column_indices: List[int] = []
        self.values: List[float] = []
        self.lower: List[float] = []
        self.upper: List[float] = []

    def add(
        self,
        coefficients: Mapping[int, float],
        *,
        lower: float = -math.inf,
        upper: float = math.inf,
    ) -> None:
        row = len(self.lower)
        for column, value in coefficients.items():
            if abs(float(value)) <= EPS:
                continue
            self.row_indices.append(row)
            self.column_indices.append(int(column))
            self.values.append(float(value))
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def build(self) -> LinearConstraint:
        matrix = coo_matrix(
            (self.values, (self.row_indices, self.column_indices)),
            shape=(len(self.lower), self.variable_count),
            dtype=np.float64,
        ).tocsr()
        return LinearConstraint(
            matrix,
            np.asarray(self.lower, dtype=np.float64),
            np.asarray(self.upper, dtype=np.float64),
        )


def _config(value: Optional[Any]) -> MILPConfig:
    if value is None:
        return MILPConfig()
    if isinstance(value, MILPConfig):
        return value
    if isinstance(value, Mapping):
        return MILPConfig(**dict(value))
    raise TypeError("params 必须是 MILPConfig、映射或 None。")


def _make_indices(node_count: int) -> _VariableIndex:
    depot = int(node_count)
    cursor = 0
    arcs: Dict[Tuple[int, int], int] = {}
    for source in range(node_count + 1):
        for target in range(node_count + 1):
            if source == target:
                continue
            arcs[(source, target)] = cursor
            cursor += 1
    visits = tuple(range(cursor, cursor + node_count))
    cursor += node_count
    route_used = cursor
    cursor += 1
    orders = tuple(range(cursor, cursor + node_count))
    cursor += node_count
    cumulative: Dict[str, Tuple[int, ...]] = {}
    for resource_name in ("energy", "distance", "time"):
        cumulative[resource_name] = tuple(range(cursor, cursor + node_count))
        cursor += node_count
    flows = {
        arc: cursor + offset for offset, arc in enumerate(arcs)
    }
    cursor += len(flows)
    assert depot == node_count
    return _VariableIndex(
        arcs,
        visits,
        route_used,
        orders,
        cumulative,
        flows,
        cursor,
    )


def _resource_specs(
    evaluator: MissionEvaluator,
    budget_overrides: Optional[Mapping[str, float]] = None,
) -> Tuple[Tuple[str, float, np.ndarray, str], ...]:
    overrides = dict(budget_overrides or {})
    return (
        (
            "energy",
            float(overrides.get("energy", evaluator.energy_budget_wh)),
            np.asarray(evaluator.service_energy_wh, dtype=np.float64),
            "energy_wh",
        ),
        (
            "distance",
            float(overrides.get("distance", evaluator.distance_budget_m)),
            np.zeros(evaluator.n, dtype=np.float64),
            "distance_m",
        ),
        (
            "time",
            float(overrides.get("time", evaluator.time_budget_s)),
            np.asarray(evaluator.service_times_s, dtype=np.float64),
            "time_s",
        ),
    )


def _build_model(
    evaluator: MissionEvaluator,
    *,
    objective_mode: str = "mission_objective",
    resource_budget_overrides: Optional[Mapping[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Bounds, LinearConstraint, _VariableIndex, float]:
    n = evaluator.n
    depot = evaluator.depot
    indices = _make_indices(n)
    lower_bounds = np.zeros(indices.size, dtype=np.float64)
    upper_bounds = np.full(indices.size, np.inf, dtype=np.float64)
    integrality = np.zeros(indices.size, dtype=np.int8)
    objective = np.zeros(indices.size, dtype=np.float64)

    for arc, column in indices.arcs.items():
        integrality[column] = 1
        upper_bounds[column] = 1.0 if evaluator._segments[arc].feasible else 0.0
    for column in indices.visits:
        integrality[column] = 1
        upper_bounds[column] = 1.0
    integrality[indices.route_used] = 1
    upper_bounds[indices.route_used] = 1.0
    for column in indices.orders:
        upper_bounds[column] = float(n)
    for column in indices.flows.values():
        upper_bounds[column] = float(n)

    resource_specs = _resource_specs(evaluator, resource_budget_overrides)
    for resource_name, budget, _service, _attribute in resource_specs:
        for column in indices.cumulative[resource_name]:
            upper_bounds[column] = budget

    empty_evaluation = evaluator.finish_label(evaluator.start_label())
    if objective_mode == "weighted_coverage":
        # 困难场景认证只最大化优先级覆盖，能量/航程/时间仍保持硬约束。
        priority_values = np.asarray(
            evaluator.problem.priorities, dtype=np.float64
        )
        total_priority = float(np.sum(np.clip(priority_values, 0.0, None)))
        if not math.isfinite(total_priority) or total_priority <= 0.0:
            raise ValueError("weighted_coverage认证要求优先级权重和为有限正数。")
        for node, column in enumerate(indices.visits):
            objective[column] = -max(0.0, float(priority_values[node])) / total_priority
        empty_objective = 0.0
    else:
        weights = evaluator.template["cfg"]["reward_weights"]
        normalized_weights = {
            "energy": float(weights["energy"]) / max(
                evaluator.energy_budget_wh, EPS
            ),
            "distance": float(weights["distance"]) / max(
                evaluator.distance_budget_m, EPS
            ),
            "time": float(weights["time"]) / max(evaluator.time_budget_s, EPS),
        }
        empty_objective = float(empty_evaluation.objective)

        # SciPy 求最小值；以下系数取负后与 MissionEvaluator 的最大化目标逐项一致。
        for node, column in enumerate(indices.visits):
            gain = evaluator.marginal_task_gain(node)
            gain -= normalized_weights["energy"] * float(
                evaluator.service_energy_wh[node]
            )
            gain -= normalized_weights["time"] * float(
                evaluator.service_times_s[node]
            )
            objective[column] = -gain
        for arc, column in indices.arcs.items():
            segment = evaluator._segments[arc]
            if not segment.feasible:
                continue
            gain = -(
                normalized_weights["energy"] * float(segment.energy_wh)
                + normalized_weights["distance"] * float(segment.distance_m)
                + normalized_weights["time"] * float(segment.time_s)
            )
            objective[column] = -gain
        # 空路线也在模型中；常数项省略，只需消除启用非空路线时的空路线基准值。
        objective[indices.route_used] = empty_objective

    constraints = _ConstraintBuilder(indices.size)
    for node in range(n):
        outgoing = {
            indices.arcs[(node, target)]: 1.0
            for target in range(n + 1)
            if target != node
        }
        outgoing[indices.visits[node]] = -1.0
        constraints.add(outgoing, lower=0.0, upper=0.0)
        incoming = {
            indices.arcs[(source, node)]: 1.0
            for source in range(n + 1)
            if source != node
        }
        incoming[indices.visits[node]] = -1.0
        constraints.add(incoming, lower=0.0, upper=0.0)
        constraints.add(
            {indices.visits[node]: 1.0, indices.route_used: -1.0}, upper=0.0
        )

    depot_out = {
        indices.arcs[(depot, node)]: 1.0 for node in range(n)
    }
    depot_out[indices.route_used] = -1.0
    constraints.add(depot_out, lower=0.0, upper=0.0)
    depot_in = {
        indices.arcs[(node, depot)]: 1.0 for node in range(n)
    }
    depot_in[indices.route_used] = -1.0
    constraints.add(depot_in, lower=0.0, upper=0.0)

    # 单商品流要求所有已选节点与机场连通，并显著强化MTZ的线性松弛。
    for arc, flow_column in indices.flows.items():
        constraints.add(
            {
                flow_column: 1.0,
                indices.arcs[arc]: -float(n),
            },
            upper=0.0,
        )
    depot_flow = {
        indices.flows[(depot, node)]: 1.0 for node in range(n)
    }
    depot_flow.update(
        {
            indices.flows[(node, depot)]: -1.0
            for node in range(n)
        }
    )
    for column in indices.visits:
        depot_flow[column] = -1.0
    constraints.add(depot_flow, lower=0.0, upper=0.0)
    for node in range(n):
        node_flow = {
            indices.flows[(source, node)]: 1.0
            for source in range(n + 1)
            if source != node
        }
        node_flow.update(
            {
                indices.flows[(node, target)]: -1.0
                for target in range(n + 1)
                if target != node
            }
        )
        node_flow[indices.visits[node]] = -1.0
        constraints.add(node_flow, lower=0.0, upper=0.0)

    for node in range(n):
        constraints.add(
            {indices.orders[node]: 1.0, indices.visits[node]: -float(n)},
            upper=0.0,
        )
        constraints.add(
            {indices.orders[node]: -1.0, indices.visits[node]: 1.0},
            upper=0.0,
        )
    for source in range(n):
        for target in range(n):
            if source == target or upper_bounds[indices.arcs[(source, target)]] == 0.0:
                continue
            # MTZ 顺序变量切断所有不经过机场的巡检点子环。
            constraints.add(
                {
                    indices.orders[source]: 1.0,
                    indices.orders[target]: -1.0,
                    indices.arcs[(source, target)]: float(n),
                },
                upper=float(n - 1),
            )

    for resource_name, budget, service_costs, segment_attribute in resource_specs:
        cumulative = indices.cumulative[resource_name]
        # 完整闭环路线的资源消耗可直接线性求和；该约束显著收紧覆盖率认证的上界。
        total_resource = {
            column: float(
                getattr(evaluator._segments[arc], segment_attribute)
            )
            for arc, column in indices.arcs.items()
            if evaluator._segments[arc].feasible
            and upper_bounds[column] > 0.0
        }
        for node, column in enumerate(indices.visits):
            total_resource[column] = (
                total_resource.get(column, 0.0) + float(service_costs[node])
            )
        constraints.add(total_resource, upper=budget)

        for node in range(n):
            return_segment = evaluator._segments[(node, depot)]
            return_cost = (
                float(getattr(return_segment, segment_attribute))
                if return_segment.feasible
                else math.inf
            )
            prefix_capacity = budget - return_cost
            if not math.isfinite(prefix_capacity) or prefix_capacity < -1e-9:
                upper_bounds[indices.visits[node]] = 0.0
                prefix_capacity = 0.0
            constraints.add(
                {
                    cumulative[node]: 1.0,
                    indices.visits[node]: -max(0.0, prefix_capacity),
                },
                upper=0.0,
            )

            start_segment = evaluator._segments[(depot, node)]
            if start_segment.feasible:
                start_cost = float(getattr(start_segment, segment_attribute)) + float(
                    service_costs[node]
                )
                constraints.add(
                    {
                        cumulative[node]: 1.0,
                        indices.arcs[(depot, node)]: -start_cost,
                    },
                    lower=0.0,
                )

        for source in range(n):
            for target in range(n):
                if source == target:
                    continue
                arc_column = indices.arcs[(source, target)]
                segment = evaluator._segments[(source, target)]
                if not segment.feasible or upper_bounds[arc_column] == 0.0:
                    continue
                transition_cost = float(getattr(segment, segment_attribute)) + float(
                    service_costs[target]
                )
                big_m = budget + transition_cost
                # 到达累计资源受后继传播并留出该前缀直接返航的资源余量。
                constraints.add(
                    {
                        cumulative[target]: 1.0,
                        cumulative[source]: -1.0,
                        arc_column: -big_m,
                    },
                    lower=-budget,
                )

    return (
        objective,
        integrality,
        Bounds(lower_bounds, upper_bounds),
        constraints.build(),
        indices,
        empty_objective,
    )


def _extract_order(
    solution: Sequence[float], indices: _VariableIndex, node_count: int
) -> Optional[Tuple[int, ...]]:
    values = np.asarray(solution, dtype=np.float64).reshape(-1)
    if values.size != indices.size or not np.all(np.isfinite(values)):
        return None
    if values[indices.route_used] <= 0.5:
        return ()

    selected = {
        node for node, column in enumerate(indices.visits) if values[column] > 0.5
    }
    depot = int(node_count)
    current = depot
    order: List[int] = []
    for _ in range(node_count + 1):
        next_nodes = [
            target
            for (source, target), column in indices.arcs.items()
            if source == current and values[column] > 0.5
        ]
        if len(next_nodes) != 1:
            return None
        target = int(next_nodes[0])
        if target == depot:
            break
        if target in order or target not in selected:
            return None
        order.append(target)
        current = target
    else:
        return None
    if set(order) != selected:
        return None
    return tuple(order)


def _optional_float(result: Any, field_name: str) -> Optional[float]:
    value = getattr(result, field_name, None)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(result: Any, field_name: str) -> Optional[int]:
    value = getattr(result, field_name, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def solve_resource_threshold_milp(
    problem: ProblemInstance,
    *,
    resource_name: str,
    minimum_priority_weight: float,
    time_limit_s: float,
) -> Dict[str, Any]:
    """求达到给定优先级覆盖阈值所需的最小单项资源，并返回严格dual下界。"""

    resource_name = str(resource_name)
    if resource_name not in {"energy", "distance", "time"}:
        raise ValueError("resource_name必须是energy、distance或time。")
    required_weight = float(minimum_priority_weight)
    limit = float(time_limit_s)
    if not math.isfinite(required_weight) or required_weight <= 0.0:
        raise ValueError("minimum_priority_weight必须是有限正数。")
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("time_limit_s必须是有限正数。")

    evaluator = MissionEvaluator(problem)
    priorities = np.clip(
        np.asarray(problem.priorities, dtype=np.float64), 0.0, None
    )
    total_priority = float(np.sum(priorities))
    if required_weight > total_priority + 1e-9:
        raise ValueError("minimum_priority_weight不得超过总优先级权重。")
    resource_fields = {
        "energy": (
            "energy_wh",
            np.asarray(evaluator.service_energy_wh, dtype=np.float64),
            float(evaluator.energy_budget_wh),
        ),
        "distance": (
            "distance_m",
            np.zeros(evaluator.n, dtype=np.float64),
            float(evaluator.distance_budget_m),
        ),
        "time": (
            "time_s",
            np.asarray(evaluator.service_times_s, dtype=np.float64),
            float(evaluator.time_budget_s),
        ),
    }
    segment_attribute, service_costs, actual_budget = resource_fields[
        resource_name
    ]
    feasible_segment_costs = [
        float(getattr(segment, segment_attribute))
        for segment in evaluator._segments.values()
        if segment.feasible
        and math.isfinite(float(getattr(segment, segment_attribute)))
    ]
    if not feasible_segment_costs:
        raise RuntimeError("阈值MILP没有可用航段。")
    # 任意简单闭环至多含n+1条航段；该上界只用于移除目标资源约束。
    relaxed_budget = max(
        actual_budget * 2.0,
        (evaluator.n + 1) * max(feasible_segment_costs)
        + float(np.sum(service_costs))
        + 1.0,
    )
    (
        _coverage_objective,
        integrality,
        bounds,
        constraints,
        indices,
        _empty_objective,
    ) = _build_model(
        evaluator,
        objective_mode="weighted_coverage",
        resource_budget_overrides={resource_name: relaxed_budget},
    )
    resource_objective = np.zeros(indices.size, dtype=np.float64)
    for arc, column in indices.arcs.items():
        segment = evaluator._segments[arc]
        if segment.feasible:
            resource_objective[column] = float(
                getattr(segment, segment_attribute)
            )
    for node, column in enumerate(indices.visits):
        resource_objective[column] += float(service_costs[node])
    threshold_row = np.zeros(indices.size, dtype=np.float64)
    for node, column in enumerate(indices.visits):
        threshold_row[column] = float(priorities[node])
    threshold_constraint = LinearConstraint(
        threshold_row, required_weight, math.inf
    )

    started = time.perf_counter()
    result = scipy_milp(
        resource_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=[constraints, threshold_constraint],
        options={
            "time_limit": limit,
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    runtime_s = float(time.perf_counter() - started)
    primal_value = _optional_float(result, "fun")
    dual_bound = _optional_float(result, "mip_dual_bound")
    status = _optional_int(result, "status")
    order = (
        _extract_order(result.x, indices, evaluator.n)
        if getattr(result, "x", None) is not None
        else None
    )
    evaluation = (
        evaluator.evaluate_order(order) if order is not None else None
    )
    evaluation_payload = None
    if evaluation is not None:
        evaluation_objective = float(evaluation.objective)
        evaluation_payload = {
            "order": list(evaluation.order),
            "objective": (
                evaluation_objective
                if math.isfinite(evaluation_objective)
                else None
            ),
            "energy_wh": float(evaluation.energy_wh),
            "distance_m": float(evaluation.distance_m),
            "time_s": float(evaluation.time_s),
            "coverage": float(evaluation.coverage),
            "weighted_coverage": float(evaluation.weighted_coverage),
            "returned": bool(evaluation.returned),
            "termination_reason": str(evaluation.termination_reason),
        }
    return {
        "resource_name": resource_name,
        "minimum_priority_weight": required_weight,
        "required_weighted_coverage": required_weight / total_priority,
        "total_priority": total_priority,
        "actual_resource_budget": actual_budget,
        "relaxed_resource_budget": relaxed_budget,
        "resource_primal_value": primal_value,
        "resource_dual_bound": dual_bound,
        "threshold_impossible_under_actual_budget": bool(
            status == 2
            or (
                dual_bound is not None
                and dual_bound > actual_budget + 1e-7
            )
        ),
        "solver_status": status,
        "solver_success": bool(getattr(result, "success", False)),
        "solver_message": str(getattr(result, "message", "")),
        "runtime_s": runtime_s,
        "visit_order": list(order) if order is not None else None,
        "actual_budget_evaluation": evaluation_payload,
    }


def plan_milp_orienteering(
    problem: ProblemInstance,
    *,
    seed: int = 42,
    budget: Optional[PlannerBudget] = None,
    params: Optional[Any] = None,
) -> PlanningResult:
    """求解有向三资源 Orienteering MILP，并用统一环境重放最终路线。"""

    config = _config(params)
    effective_budget = budget or PlannerBudget(
        max_evaluations=None, time_limit_s=60.0
    )
    controller = SearchController(effective_budget)
    evaluator = MissionEvaluator(problem)
    empty = evaluator.finish_label(evaluator.start_label())
    best: RouteEvaluation = empty
    controller.consume()

    objective, integrality, bounds, constraints, indices, empty_objective = (
        _build_model(evaluator, objective_mode=config.objective_mode)
    )
    options: Dict[str, Any] = {
        "disp": False,
        "presolve": bool(config.presolve),
        "mip_rel_gap": float(config.mip_rel_gap),
    }
    if effective_budget.time_limit_s is not None:
        remaining = float(effective_budget.time_limit_s) - controller.elapsed_s
        options["time_limit"] = max(1e-6, remaining)

    solver_result: Any = None
    solver_exception: Optional[str] = None
    try:
        solver_result = scipy_milp(
            objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )
    except Exception as exc:  # pragma: no cover - HiGHS 运行时故障的防御路径
        solver_exception = f"{type(exc).__name__}: {exc}"

    incumbent_order: Optional[Tuple[int, ...]] = None
    incumbent_replay_valid = False
    if solver_result is not None and getattr(solver_result, "x", None) is not None:
        incumbent_order = _extract_order(solver_result.x, indices, evaluator.n)
        if incumbent_order is not None:
            incumbent = evaluator.evaluate_order(incumbent_order)
            incumbent_replay_valid = bool(
                incumbent.returned and math.isfinite(incumbent.objective)
            )
            if incumbent_replay_valid and incumbent.objective > best.objective + 1e-12:
                best = incumbent

    solver_status = _optional_int(solver_result, "status")
    mip_gap = _optional_float(solver_result, "mip_gap")
    mip_dual_bound = _optional_float(solver_result, "mip_dual_bound")
    solver_fun = _optional_float(solver_result, "fun")
    incumbent_evaluation = (
        evaluator.evaluate_order(incumbent_order or ())
        if incumbent_replay_valid
        else None
    )
    incumbent_target_value = (
        float(incumbent_evaluation.weighted_coverage)
        if incumbent_evaluation is not None
        and config.objective_mode == "weighted_coverage"
        else (
            float(incumbent_evaluation.objective)
            if incumbent_evaluation is not None
            else None
        )
    )
    solver_target_value = (
        empty_objective - solver_fun if solver_fun is not None else None
    )
    incumbent_matches_solver = bool(
        incumbent_target_value is not None
        and solver_target_value is not None
        and math.isclose(
            incumbent_target_value,
            solver_target_value,
            rel_tol=1e-7,
            abs_tol=2e-6,
        )
    )
    if (
        incumbent_evaluation is not None
        and config.objective_mode == "weighted_coverage"
    ):
        best = incumbent_evaluation
    optimality_certified = bool(
        solver_status == 0
        and bool(getattr(solver_result, "success", False))
        and mip_gap is not None
        # 只接受 HiGHS 报告的零 MIP gap；调宽求解终止 gap 不会变成“精确最优”。
        and mip_gap == 0.0
        and incumbent_replay_valid
        and incumbent_matches_solver
    )

    if optimality_certified:
        status = "ok"
    elif solver_status == 1:
        status = "budget_exhausted"
    elif incumbent_replay_valid:
        status = "solver_nonoptimal"
    else:
        status = "solver_no_incumbent"

    objective_dual_bound = (
        empty_objective - mip_dual_bound if mip_dual_bound is not None else None
    )
    metadata = {
        "reference_type": "milp_orienteering",
        "optimization_target": config.objective_mode,
        "solver": "scipy.optimize.milp/HiGHS",
        "solver_status": solver_status,
        "solver_success": bool(getattr(solver_result, "success", False)),
        "solver_message": (
            str(getattr(solver_result, "message", ""))
            if solver_result is not None
            else solver_exception or "solver did not run"
        ),
        "mip_gap": mip_gap,
        "mip_dual_bound": mip_dual_bound,
        "objective_dual_bound": objective_dual_bound,
        "weighted_coverage_incumbent": (
            float(best.weighted_coverage)
            if config.objective_mode == "weighted_coverage"
            else None
        ),
        "weighted_coverage_upper_bound": (
            min(1.0, max(0.0, float(objective_dual_bound)))
            if config.objective_mode == "weighted_coverage"
            and objective_dual_bound is not None
            else None
        ),
        "mip_node_count": _optional_int(solver_result, "mip_node_count"),
        "optimality_certified": optimality_certified,
        "optimality_gap": 0.0 if optimality_certified else mip_gap,
        "incumbent_available": bool(incumbent_order is not None),
        "incumbent_replay_valid": incumbent_replay_valid,
        "incumbent_matches_solver_objective": incumbent_matches_solver,
        "config": asdict(config),
    }
    return evaluator.build_result(
        "milp_orienteering",
        best,
        controller,
        seed,
        status=status,
        metadata=metadata,
    )


__all__ = [
    "MILPConfig",
    "plan_milp_orienteering",
    "solve_resource_threshold_milp",
]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic, cached MILP budget search for v3.2.12 test inputs.

The inspection-point geometry is fixed before this module is called.  The
module only changes the three registered resource budgets and only reads MILP
certificates.  No learned-policy or comparison-planner result is available to
the search.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

import paper_multimap_experiments as multimap


RESOURCE_PARAMETERS = (
    "initial_soc",
    "distance_budget_scale",
    "time_budget_scale",
)
RESOURCE_NAMES = {
    "initial_soc": "energy",
    "distance_budget_scale": "distance",
    "time_budget_scale": "time",
}
NAME_TO_PARAMETER = {value: key for key, value in RESOURCE_NAMES.items()}


def _config(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    return protocol["pretest_parametric_certificate_search"]


def _bounds(protocol: Mapping[str, Any], parameter: str) -> Tuple[float, float]:
    values = protocol["task_generation"]["single_constraint_budget_calibration"][
        "parameter_bounds"
    ][parameter]
    lower, upper = float(values[0]), float(values[1])
    if parameter == "initial_soc":
        lower = max(
            lower,
            float(
                protocol["task_generation"]["evaluator_safety_bounds"][
                    "minimum_initial_soc"
                ]
            ),
        )
    return lower, upper


def _candidate_key(candidate: Mapping[str, Any], time_limit_s: float) -> str:
    excluded = {
        "certificate",
        "certificate_search_trace",
        "task_hash",
    }
    payload = {
        key: value
        for key, value in candidate.items()
        if key not in excluded
    }
    payload["__time_limit_s"] = float(time_limit_s)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _with_values(
    candidate: Mapping[str, Any],
    values: Mapping[str, float],
    protocol: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> Dict[str, Any]:
    result = copy.deepcopy(dict(candidate))
    for parameter, raw in values.items():
        lower, upper = _bounds(protocol, parameter)
        result[parameter] = float(np.clip(float(raw), lower, upper))
    history = list(result.get("certificate_search_trace", ()))
    history.append(dict(trace))
    result["certificate_search_trace"] = history
    return result


def _global_scaled(
    anchor: Mapping[str, Any],
    factor: float,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    values = {
        parameter: float(anchor[parameter]) * float(factor)
        for parameter in RESOURCE_PARAMETERS
    }
    return _with_values(
        anchor,
        values,
        protocol,
        {"stage": "global_scale", "factor_from_geometry_anchor": float(factor)},
    )


def _resource_scaled(
    anchor: Mapping[str, Any],
    parameter: str,
    factor: float,
    protocol: Mapping[str, Any],
    *,
    stage: str,
) -> Dict[str, Any]:
    return _with_values(
        anchor,
        {parameter: float(anchor[parameter]) * float(factor)},
        protocol,
        {
            "stage": str(stage),
            "parameter": str(parameter),
            "factor_from_stage_anchor": float(factor),
        },
    )


def _finite_interval(certificate: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        lower = float(certificate["weighted_coverage_lower_bound"])
        upper = float(certificate["weighted_coverage_upper_bound"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(lower) or not math.isfinite(upper):
        return None
    return min(lower, upper), max(lower, upper)


def _relation(
    certificate: Mapping[str, Any],
    difficulty: str,
    parent_protocol: Mapping[str, Any],
) -> str:
    interval = _finite_interval(certificate)
    if interval is None:
        return "unknown"
    lower, upper = interval
    band_low, band_high = (
        float(value)
        for value in parent_protocol["difficulty_bands"][difficulty]
    )
    tolerance = float(parent_protocol["certification"]["band_tolerance"])
    if upper < band_low - tolerance:
        return "below"
    if lower > band_high + tolerance:
        return "above"
    return "intersects"


@dataclass
class ProbeCache:
    """Per-process exact-budget MILP cache; runtime-dependent limits stay separate."""

    values: Dict[str, Tuple[bool, Dict[str, Any], str]] = field(
        default_factory=dict
    )
    hits: int = 0
    misses: int = 0

    def run(
        self,
        candidate: Mapping[str, Any],
        provider: Any,
        parent_protocol: Mapping[str, Any],
        time_limit_s: float,
    ) -> Tuple[bool, Dict[str, Any], str]:
        key = _candidate_key(candidate, time_limit_s)
        if key in self.values:
            self.hits += 1
            accepted, certificate, reason = self.values[key]
            return accepted, copy.deepcopy(certificate), reason
        self.misses += 1
        accepted, certificate, reason = multimap._certify_multimap_task(
            candidate,
            provider,
            parent_protocol,
            time_limit_s=float(time_limit_s),
        )
        stored = (bool(accepted), copy.deepcopy(dict(certificate)), str(reason))
        self.values[key] = stored
        return stored[0], copy.deepcopy(stored[1]), stored[2]


def _probe(
    candidate: Mapping[str, Any],
    provider: Any,
    parent_protocol: Mapping[str, Any],
    protocol: Mapping[str, Any],
    cache: ProbeCache,
) -> Tuple[bool, Dict[str, Any], str]:
    config = _config(protocol)
    fast_limit = float(config["fast_probe_time_limit_s"])
    result = cache.run(candidate, provider, parent_protocol, fast_limit)
    relation = _relation(
        result[1], str(candidate["difficulty"]), parent_protocol
    )
    missing = _finite_interval(result[1]) is None
    if not missing and relation != "intersects":
        return result
    return cache.run(
        candidate,
        provider,
        parent_protocol,
        float(protocol["certification"]["candidate_screening_time_limit_s"]),
    )


def _strict_outcome(
    candidate: Mapping[str, Any],
    screen_certificate: Mapping[str, Any],
    screen_reason: str,
    provider: Any,
    parent_protocol: Mapping[str, Any],
    protocol: Mapping[str, Any],
    cache: ProbeCache,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], str]:
    accepted, certificate, reason = cache.run(
        candidate,
        provider,
        parent_protocol,
        float(protocol["certification"]["time_limit_s"]),
    )
    if not accepted:
        return None, certificate, reason
    result = copy.deepcopy(dict(candidate))
    final_certificate = dict(certificate)
    final_certificate["screening"] = {
        "time_limit_s": float(
            protocol["certification"]["candidate_screening_time_limit_s"]
        ),
        "reason": str(screen_reason),
        "weighted_coverage_lower_bound": screen_certificate.get(
            "weighted_coverage_lower_bound"
        ),
        "weighted_coverage_upper_bound": screen_certificate.get(
            "weighted_coverage_upper_bound"
        ),
        "mip_gap": screen_certificate.get("mip_gap"),
    }
    final_certificate["certification_source"] = (
        "v3_2_12_parametric_fixed_budget_final"
    )
    final_certificate["certification_time_limit_s_used"] = float(
        protocol["certification"]["time_limit_s"]
    )
    final_certificate["parametric_probe_cache"] = {
        "hits": int(cache.hits),
        "misses": int(cache.misses),
    }
    result["certificate"] = final_certificate
    result["task_hash"] = multimap._canonical_hash(
        result, excluded=("task_hash",)
    )
    return result, certificate, reason


def _resource_order(certificate: Mapping[str, Any]) -> Sequence[str]:
    pairs = []
    for index, name in enumerate(("energy", "distance", "time")):
        try:
            value = float(certificate[f"{name}_utilization"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            pairs.append((-value, index, name))
    return tuple(item[2] for item in sorted(pairs))


def _activation_parameter(
    candidate: Mapping[str, Any],
    certificate: Mapping[str, Any],
) -> Optional[str]:
    intended = str(candidate["constraint_type"])
    active = set(certificate.get("bottleneck_resources") or ())
    if intended != "mixed":
        return None if intended in active else NAME_TO_PARAMETER.get(intended)
    minimum = 2
    if len(active) >= minimum:
        return None
    for resource in _resource_order(certificate):
        if resource not in active:
            return NAME_TO_PARAMETER[resource]
    return None


def _try_activation(
    anchor: Mapping[str, Any],
    strict_certificate: Mapping[str, Any],
    provider: Any,
    parent_protocol: Mapping[str, Any],
    protocol: Mapping[str, Any],
    cache: ProbeCache,
) -> Optional[Dict[str, Any]]:
    parameter = _activation_parameter(anchor, strict_certificate)
    if parameter is None:
        return None
    factors = tuple(
        float(value)
        for value in _config(protocol)["activation_tighten_factors"]
    )
    for factor in factors:
        trial = _resource_scaled(
            anchor,
            parameter,
            factor,
            protocol,
            stage="activate_required_bottleneck",
        )
        probe_ok, probe_certificate, probe_reason = _probe(
            trial, provider, parent_protocol, protocol, cache
        )
        relation = _relation(
            probe_certificate, str(trial["difficulty"]), parent_protocol
        )
        if relation == "below":
            break
        if not probe_ok and relation != "intersects":
            continue
        accepted, strict_certificate, strict_reason = _strict_outcome(
            trial,
            probe_certificate,
            probe_reason,
            provider,
            parent_protocol,
            protocol,
            cache,
        )
        if accepted is not None:
            return accepted
        if strict_reason not in {
            "mixed_bottleneck_not_active",
            "intended_bottleneck_not_active",
        }:
            continue
    return None


def _single_constraint_fallback(
    anchor: Mapping[str, Any],
    provider: Any,
    parent_protocol: Mapping[str, Any],
    protocol: Mapping[str, Any],
    cache: ProbeCache,
) -> Optional[Dict[str, Any]]:
    intended = str(anchor["constraint_type"])
    parameter = NAME_TO_PARAMETER.get(intended)
    if parameter is None:
        return None
    values = {}
    for other in RESOURCE_PARAMETERS:
        if other != parameter:
            values[other] = _bounds(protocol, other)[1]
    relaxed = _with_values(
        anchor,
        values,
        protocol,
        {
            "stage": "single_constraint_release_nuisance_resources",
            "intended_parameter": parameter,
        },
    )
    lower_bound, upper_bound = _bounds(protocol, parameter)
    factors = tuple(
        float(value)
        for value in _config(protocol)["single_constraint_absolute_fractions"]
    )
    for fraction in factors:
        value = lower_bound + fraction * (upper_bound - lower_bound)
        trial = _with_values(
            relaxed,
            {parameter: value},
            protocol,
            {
                "stage": "single_constraint_absolute_search",
                "parameter": parameter,
                "registered_range_fraction": fraction,
            },
        )
        probe_ok, probe_certificate, probe_reason = _probe(
            trial, provider, parent_protocol, protocol, cache
        )
        relation = _relation(
            probe_certificate, str(trial["difficulty"]), parent_protocol
        )
        if not probe_ok and relation != "intersects":
            continue
        accepted, strict_certificate, strict_reason = _strict_outcome(
            trial,
            probe_certificate,
            probe_reason,
            provider,
            parent_protocol,
            protocol,
            cache,
        )
        if accepted is not None:
            return accepted
        if strict_reason == "intended_bottleneck_not_active":
            activated = _try_activation(
                trial,
                strict_certificate,
                provider,
                parent_protocol,
                protocol,
                cache,
            )
            if activated is not None:
                return activated
    return None


def _factor_schedule(
    first_relation: str, protocol: Mapping[str, Any]
) -> Tuple[float, ...]:
    config = _config(protocol)
    if first_relation == "above":
        return tuple(float(value) for value in config["global_tighten_factors"])
    if first_relation == "below":
        return tuple(float(value) for value in config["global_loosen_factors"])
    return tuple(
        float(value)
        for value in config["global_tighten_factors"]
    ) + tuple(
        float(value)
        for value in config["global_loosen_factors"]
        if float(value) != 1.0
    )


def certify_candidate_with_parametric_search(
    candidate: Mapping[str, Any],
    provider: Any,
    parent_protocol: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    cache: Optional[ProbeCache] = None,
) -> Optional[Dict[str, Any]]:
    """Return one unchanged 60-second certificate or ``None``.

    Search decisions use only finite MILP coverage intervals and registered
    resource-utilization bottleneck checks.  The exact candidate passed to the
    final solve is never recalibrated after the handoff.
    """

    local_cache = cache if cache is not None else ProbeCache()
    first = _global_scaled(candidate, 1.0, protocol)
    first_ok, first_certificate, first_reason = _probe(
        first, provider, parent_protocol, protocol, local_cache
    )
    first_relation = _relation(
        first_certificate, str(first["difficulty"]), parent_protocol
    )
    schedule = _factor_schedule(first_relation, protocol)
    probed: Dict[float, str] = {}

    def examine(
        trial: Mapping[str, Any],
        probe_ok: bool,
        probe_certificate: Mapping[str, Any],
        probe_reason: str,
    ) -> Optional[Dict[str, Any]]:
        relation = _relation(
            probe_certificate, str(trial["difficulty"]), parent_protocol
        )
        if not probe_ok and relation != "intersects":
            return None
        accepted, strict_certificate, strict_reason = _strict_outcome(
            trial,
            probe_certificate,
            probe_reason,
            provider,
            parent_protocol,
            protocol,
            local_cache,
        )
        if accepted is not None:
            return accepted
        if strict_reason in {
            "mixed_bottleneck_not_active",
            "intended_bottleneck_not_active",
        }:
            return _try_activation(
                trial,
                strict_certificate,
                provider,
                parent_protocol,
                protocol,
                local_cache,
            )
        return None

    accepted = examine(first, first_ok, first_certificate, first_reason)
    if accepted is not None:
        return accepted
    probed[1.0] = first_relation

    for factor in schedule:
        if factor in probed:
            continue
        trial = _global_scaled(candidate, factor, protocol)
        probe_ok, certificate, reason = _probe(
            trial, provider, parent_protocol, protocol, local_cache
        )
        relation = _relation(
            certificate, str(trial["difficulty"]), parent_protocol
        )
        probed[factor] = relation
        accepted = examine(trial, probe_ok, certificate, reason)
        if accepted is not None:
            return accepted

        # 只在已找到“过松/过紧”夹逼时细化，避免固定密网格浪费 MILP。
        above = [value for value, state in probed.items() if state == "above"]
        below = [value for value, state in probed.items() if state == "below"]
        if not above or not below:
            continue
        low_factor = max(value for value in below if value < max(above)) if any(
            value < max(above) for value in below
        ) else None
        high_factor = min(value for value in above if low_factor is not None and value > low_factor) if low_factor is not None and any(
            value > low_factor for value in above
        ) else None
        if low_factor is None or high_factor is None:
            continue
        left, right = float(low_factor), float(high_factor)
        for _ in range(int(_config(protocol)["bracket_refinement_steps"])):
            middle = math.sqrt(left * right)
            if any(abs(middle - known) <= 1e-9 for known in probed):
                break
            refined = _global_scaled(candidate, middle, protocol)
            refined_ok, refined_certificate, refined_reason = _probe(
                refined, provider, parent_protocol, protocol, local_cache
            )
            state = _relation(
                refined_certificate,
                str(refined["difficulty"]),
                parent_protocol,
            )
            probed[middle] = state
            accepted = examine(
                refined,
                refined_ok,
                refined_certificate,
                refined_reason,
            )
            if accepted is not None:
                return accepted
            if state == "above":
                right = middle
            elif state == "below":
                left = middle
            else:
                break
        break

    if str(candidate["constraint_type"]) != "mixed":
        return _single_constraint_fallback(
            candidate,
            provider,
            parent_protocol,
            protocol,
            local_cache,
        )
    return None

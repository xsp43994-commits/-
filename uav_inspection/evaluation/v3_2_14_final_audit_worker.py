#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wait for every frozen evaluation job, then audit all 21,648 rows."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Dict, Iterable, Mapping, Sequence

from uav_inspection.experiments import paper_v3_2_experiments as v32
from uav_inspection.evaluation import v3_2_14_evaluation_smoke as smoke


ROOT = WORKSPACE_ROOT
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14"
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/protocol.json"
)
FORMAL = OUTPUT / "formal_evaluation"
RESULTS = FORMAL / "results"
MATRIX = FORMAL / "evaluation_matrix.jsonl"
MATRIX_MANIFEST = FORMAL / "evaluation_matrix_manifest.json"
STATUS = RESULTS / "final_audit_status.json"
DESTINATION = RESULTS / "final_results.jsonl"
AUDIT = RESULTS / "final_audit.json"
EXPECTED_FAMILIES = {
    "synthetic_learning": 7560,
    "synthetic_main_baselines": 3888,
    "synthetic_supplementary": 504,
    "real_learning": 5040,
    "real_baselines": 1152,
    "known_domain_shift": 1008,
    "hidden_model_perception_mismatch": 2496,
}


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["family"]),
        str(row["task_id"]),
        str(row["model"]),
        (
            int(row["training_seed"])
            if row.get("training_seed") is not None
            else None
        ),
        (
            int(row["planner_seed"])
            if row.get("planner_seed") is not None
            else None
        ),
        str(row["condition"]),
    )


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _status_completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return (
            json.loads(path.read_text(encoding="utf-8")).get("state")
            == "completed"
        )
    except (OSError, json.JSONDecodeError):
        return False


def _ready() -> bool:
    return all(
        _status_completed(path)
        for path in (
            RESULTS / "learning_chain_status.json",
            RESULTS / "baseline_chain_status.json",
            RESULTS / "robustness_learning_chain_status.json",
        )
    ) and all(
        _status_completed(
            RESULTS
            / family
            / "jobs"
            / "priority_resource_greedy__plan42"
            / "status.json"
        )
        for family in (
            "known_domain_shift",
            "hidden_model_perception_mismatch",
        )
    )


def _source_groups() -> Iterable[
    tuple[Path, Sequence[Path]]
]:
    for family in ("synthetic_learning", "real_learning"):
        family_dir = RESULTS / family
        yield (
            family_dir / "results.jsonl",
            tuple(
                sorted(
                    path / "routes"
                    for path in (family_dir / "shards").iterdir()
                    if path.is_dir()
                )
            ),
        )
    for family in (
        "synthetic_main_baselines",
        "synthetic_supplementary",
        "real_baselines",
        "known_domain_shift",
        "hidden_model_perception_mismatch",
    ):
        jobs = RESULTS / family / "jobs"
        for run_dir in sorted(path for path in jobs.iterdir() if path.is_dir()):
            yield run_dir / "results.jsonl", (run_dir / "routes",)


def _route_hashes(route_dirs: Sequence[Path]) -> set[str]:
    hashes: set[str] = set()
    for directory in route_dirs:
        if not directory.is_dir():
            raise RuntimeError(f"missing route directory: {directory}")
        for path in directory.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not _finite(payload):
                raise RuntimeError(f"non-finite route payload: {path}")
            route_hash = smoke._canonical_hash(payload)
            if route_hash in hashes:
                raise RuntimeError(f"duplicate route payload hash: {path}")
            hashes.add(route_hash)
    return hashes


def _write_waiting_status() -> None:
    smoke._atomic_json(
        STATUS,
        {
            "schema_version": 1,
            "state": "waiting_for_all_evaluation_jobs",
            "matrix_sha256": v32._sha256_file(MATRIX),
        },
    )


def audit() -> Dict[str, Any]:
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    manifest = json.loads(MATRIX_MANIFEST.read_text(encoding="utf-8"))
    if (
        int(manifest["row_count"]) != 21648
        or manifest["protocol_hash"] != protocol["protocol_hash"]
        or manifest["matrix_sha256"] != v32._sha256_file(MATRIX)
    ):
        raise RuntimeError("final audit matrix identity mismatch")
    matrix_rows = v32._read_jsonl(MATRIX)
    expected_by_key = {_key(row): row for row in matrix_rows}
    if len(expected_by_key) != 21648:
        raise RuntimeError("final matrix keys are not unique")

    actual: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    source_hashes: Dict[str, str] = {}
    route_count = 0
    for results_path, route_dirs in _source_groups():
        if not results_path.is_file():
            raise RuntimeError(f"missing result source: {results_path}")
        rows = v32._read_jsonl(results_path)
        route_hashes = _route_hashes(route_dirs)
        if len(route_hashes) != len(rows):
            raise RuntimeError(
                f"route/result count mismatch: {results_path}"
            )
        route_count += len(route_hashes)
        source_hashes[str(results_path.relative_to(RESULTS))] = (
            v32._sha256_file(results_path)
        )
        for row in rows:
            key = _key(row)
            if key in actual:
                raise RuntimeError(f"duplicate final result key: {key}")
            matrix_row = expected_by_key.get(key)
            if matrix_row is None:
                raise RuntimeError(f"unexpected final result key: {key}")
            if row.get("matrix_row_hash") != smoke._canonical_hash(
                matrix_row
            ):
                raise RuntimeError(f"matrix row hash mismatch: {key}")
            if row.get("task_hash") != matrix_row.get("task_hash"):
                raise RuntimeError(f"task hash mismatch: {key}")
            if row.get("protocol_hash") != protocol["protocol_hash"]:
                raise RuntimeError(f"protocol hash mismatch: {key}")
            if row.get("route_hash") not in route_hashes:
                raise RuntimeError(f"route hash missing: {key}")
            result_payload = {
                field: value
                for field, value in row.items()
                if field != "result_hash"
            }
            if row.get("result_hash") != smoke._canonical_hash(
                result_payload
            ):
                raise RuntimeError(f"result hash mismatch: {key}")
            if not row.get("paper_eligible") or not _finite(row):
                raise RuntimeError(f"invalid paper result: {key}")
            if row["family"] == "known_domain_shift":
                if (
                    row.get("observed_input_hash")
                    != row.get("execution_truth_hash")
                ):
                    raise RuntimeError(
                        f"known-shift input hash mismatch: {key}"
                    )
            elif row["family"] == "hidden_model_perception_mismatch":
                if (
                    row.get("observed_input_hash")
                    == row.get("execution_truth_hash")
                ):
                    raise RuntimeError(
                        f"hidden-mismatch input hashes equal: {key}"
                    )
            elif not (
                row.get("nominal_input_hash")
                == row.get("observed_input_hash")
                == row.get("execution_truth_hash")
            ):
                raise RuntimeError(f"nominal input hash mismatch: {key}")
            actual[key] = row

    if set(actual) != set(expected_by_key):
        missing = len(set(expected_by_key) - set(actual))
        extra = len(set(actual) - set(expected_by_key))
        raise RuntimeError(
            f"final result matrix incomplete: missing={missing}, extra={extra}"
        )
    ordered = [actual[_key(row)] for row in matrix_rows]
    counts = Counter(str(row["family"]) for row in ordered)
    if dict(counts) != EXPECTED_FAMILIES:
        raise RuntimeError(f"final family counts mismatch: {dict(counts)}")
    if any(str(row["model"]) == "ppo_mlp" for row in ordered):
        raise RuntimeError("archived ppo_mlp entered final results")
    if not any(str(row["model"]) == "traditional_ppo" for row in ordered):
        raise RuntimeError("traditional_ppo missing from final results")

    smoke._atomic_text(DESTINATION, smoke._jsonl(ordered))
    payload = {
        "schema_version": 1,
        "state": "completed",
        "passed": True,
        "row_count": len(ordered),
        "unique_key_count": len(actual),
        "route_count": route_count,
        "family_counts": dict(counts),
        "matrix_sha256": manifest["matrix_sha256"],
        "results_sha256": v32._sha256_file(DESTINATION),
        "source_results_sha256": source_hashes,
        "ppo_mlp_absent": True,
        "traditional_ppo_present": True,
    }
    payload["manifest_hash"] = smoke._canonical_hash(payload)
    smoke._atomic_json(AUDIT, payload)
    smoke._atomic_json(STATUS, payload)
    return payload


def run(*, poll_seconds: float) -> Dict[str, Any]:
    while not _ready():
        _write_waiting_status()
        time.sleep(poll_seconds)
    smoke._atomic_json(
        STATUS,
        {
            "schema_version": 1,
            "state": "auditing",
            "matrix_sha256": v32._sha256_file(MATRIX),
        },
    )
    return audit()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    payload = run(poll_seconds=float(args.poll_seconds))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

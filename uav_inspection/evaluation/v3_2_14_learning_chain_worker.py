#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supervise, merge, and audit the two nominal learning families."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Dict, Mapping, Sequence

from uav_inspection.experiments import paper_v3_2_experiments as v32
from uav_inspection.evaluation import v3_2_14_evaluation_smoke as smoke


ROOT = WORKSPACE_ROOT
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14"
MATRIX = OUTPUT / "formal_evaluation/evaluation_matrix.jsonl"
MATRIX_MANIFEST = (
    OUTPUT / "formal_evaluation/evaluation_matrix_manifest.json"
)
WORKER = ROOT / "v3_2_14_nominal_learning_worker.py"
DIAGNOSTICS = OUTPUT / "diagnostics"
CHAIN_STATUS = (
    OUTPUT / "formal_evaluation/results/learning_chain_status.json"
)
EXPECTED = {"synthetic_learning": 7560, "real_learning": 5040}


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["family"]),
        str(row["task_id"]),
        str(row["model"]),
        int(row["training_seed"]),
        str(row["condition"]),
    )


def _write_status(payload: Mapping[str, Any]) -> None:
    smoke._atomic_json(CHAIN_STATUS, payload)


def _shard_dirs(family: str, shard_count: int) -> list[Path]:
    root = (
        OUTPUT
        / "formal_evaluation"
        / "results"
        / family
        / "shards"
    )
    return [
        root / f"shard_{index:02d}_of_{shard_count:02d}"
        for index in range(shard_count)
    ]


def _family_progress(
    family: str, shard_count: int
) -> Dict[str, Any]:
    shards = []
    for directory in _shard_dirs(family, shard_count):
        status_path = directory / "status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file()
            else {"state": "missing", "completed": 0, "total": 0}
        )
        shards.append(
            {
                "name": directory.name,
                "state": status.get("state", "unknown"),
                "completed": int(status.get("completed", 0)),
                "total": int(status.get("total", 0)),
            }
        )
    return {
        "family": family,
        "completed": sum(item["completed"] for item in shards),
        "expected": EXPECTED[family],
        "all_shards_completed": all(
            item["state"] == "completed" for item in shards
        ),
        "shards": shards,
    }


def _audit_and_merge(family: str, shard_count: int) -> Dict[str, Any]:
    manifest = json.loads(MATRIX_MANIFEST.read_text(encoding="utf-8"))
    if (
        int(manifest["row_count"]) != 21648
        or manifest["matrix_sha256"] != v32._sha256_file(MATRIX)
    ):
        raise RuntimeError("matrix drift before learning-family merge")
    expected_rows = [
        row
        for row in v32._read_jsonl(MATRIX)
        if str(row["family"]) == family
    ]
    expected_by_key = {_key(row): row for row in expected_rows}
    if len(expected_by_key) != EXPECTED[family]:
        raise RuntimeError("matrix family key count mismatch")

    actual: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    shard_hashes = []
    route_count = 0
    for directory in _shard_dirs(family, shard_count):
        status = json.loads(
            (directory / "status.json").read_text(encoding="utf-8")
        )
        if status.get("state") != "completed":
            raise RuntimeError(f"incomplete learning shard {directory}")
        results_path = directory / "results.jsonl"
        shard_hashes.append(v32._sha256_file(results_path))
        rows = v32._read_jsonl(results_path)
        routes = list((directory / "routes").glob("*.json"))
        if len(routes) != len(rows):
            raise RuntimeError(f"route/result count mismatch in {directory}")
        route_count += len(routes)
        route_hashes = {
            smoke._canonical_hash(
                json.loads(path.read_text(encoding="utf-8"))
            )
            for path in routes
        }
        for row in rows:
            key = _key(row)
            if key in actual:
                raise RuntimeError(f"duplicate learning result key {key}")
            matrix_row = expected_by_key.get(key)
            if matrix_row is None:
                raise RuntimeError(f"unexpected learning result key {key}")
            if row.get("matrix_row_hash") != smoke._canonical_hash(
                matrix_row
            ):
                raise RuntimeError(f"matrix-row hash mismatch {key}")
            if row.get("route_hash") not in route_hashes:
                raise RuntimeError(f"route hash missing {key}")
            result_payload = {
                field: value
                for field, value in row.items()
                if field != "result_hash"
            }
            if row.get("result_hash") != smoke._canonical_hash(
                result_payload
            ):
                raise RuntimeError(f"result hash mismatch {key}")
            for value in row.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise RuntimeError(f"non-finite learning result {key}")
            actual[key] = row
    if set(actual) != set(expected_by_key):
        raise RuntimeError("merged learning keys are not matrix-complete")
    ordered = [actual[_key(row)] for row in expected_rows]
    family_dir = (
        OUTPUT / "formal_evaluation" / "results" / family
    )
    destination = family_dir / "results.jsonl"
    smoke._atomic_text(destination, smoke._jsonl(ordered))
    merged = {
        "schema_version": 1,
        "family": family,
        "passed": True,
        "row_count": len(ordered),
        "unique_key_count": len(actual),
        "route_count": route_count,
        "matrix_sha256": manifest["matrix_sha256"],
        "results_sha256": v32._sha256_file(destination),
        "shard_results_sha256": shard_hashes,
    }
    merged["manifest_hash"] = smoke._canonical_hash(merged)
    smoke._atomic_json(family_dir / "merged_audit.json", merged)
    return merged


def _launch_family(family: str, shard_count: int) -> list[int]:
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    pids = []
    for index, directory in enumerate(
        _shard_dirs(family, shard_count)
    ):
        status_path = directory / "status.json"
        resume = status_path.is_file()
        if resume:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("state") == "completed":
                continue
        command = [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            str(WORKER),
            "--family",
            family,
            "--device",
            "cuda",
            "--shard-index",
            str(index),
            "--shard-count",
            str(shard_count),
        ]
        if resume:
            command.append("--resume")
        stdout_path = (
            DIAGNOSTICS
            / f"formal_{family}_shard{index:02d}.stdout.log"
        )
        stderr_path = (
            DIAGNOSTICS
            / f"formal_{family}_shard{index:02d}.stderr.log"
        )
        with stdout_path.open("a", encoding="utf-8") as stdout, (
            stderr_path.open("a", encoding="utf-8")
        ) as stderr:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if os.name == "nt"
                    else 0
                ),
            )
        pids.append(int(process.pid))
    return pids


def run(*, poll_seconds: float, shard_count: int) -> Dict[str, Any]:
    stages: Dict[str, Any] = {}
    for family in ("synthetic_learning", "real_learning"):
        progress = _family_progress(family, shard_count)
        if (
            not progress["all_shards_completed"]
            and progress["completed"] == 0
        ):
            pids = _launch_family(family, shard_count)
        else:
            pids = []
        while True:
            progress = _family_progress(family, shard_count)
            _write_status(
                {
                    "schema_version": 1,
                    "state": "running",
                    "current_family": family,
                    "progress": progress,
                    "launched_pids": pids,
                    "completed_families": list(stages),
                }
            )
            if progress["all_shards_completed"]:
                break
            time.sleep(poll_seconds)
        stages[family] = _audit_and_merge(family, shard_count)
    report = {
        "schema_version": 1,
        "state": "completed",
        "families": stages,
        "total_rows": sum(
            int(stage["row_count"]) for stage in stages.values()
        ),
    }
    _write_status(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--shard-count", type=int, default=4)
    args = parser.parse_args(argv)
    report = run(
        poll_seconds=float(args.poll_seconds),
        shard_count=int(args.shard_count),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

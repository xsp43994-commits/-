#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wait for nominal learning, then schedule frozen robustness learning jobs."""

from __future__ import annotations

import argparse
import json
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
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/protocol.json"
)
MATRIX = OUTPUT / "formal_evaluation/evaluation_matrix.jsonl"
MATRIX_MANIFEST = (
    OUTPUT / "formal_evaluation/evaluation_matrix_manifest.json"
)
ROBUSTNESS_MANIFEST = (
    OUTPUT / "formal_evaluation/robustness_implementation_manifest.json"
)
RESULTS = OUTPUT / "formal_evaluation/results"
DIAGNOSTICS = OUTPUT / "diagnostics"
WORKER = ROOT / "v3_2_14_robustness_worker.py"
STATUS = RESULTS / "robustness_learning_chain_status.json"
FAMILIES = (
    "known_domain_shift",
    "hidden_model_perception_mismatch",
)


def _identity(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["family"]),
        str(row["model"]),
        int(row["training_seed"]),
    )


def _job_dir(identity: tuple[str, str, int]) -> Path:
    family, model, seed = identity
    return RESULTS / family / "jobs" / f"{model}__train{seed}"


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _completed(
    identity: tuple[str, str, int], expected: int
) -> bool:
    run_dir = _job_dir(identity)
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        status.get("state") == "completed"
        and int(status.get("completed", -1)) == expected
        and int(status.get("total", -1)) == expected
        and _line_count(run_dir / "results.jsonl") == expected
    )


def _freeze_audit() -> tuple[
    Dict[tuple[str, str, int], int], Dict[str, Any]
]:
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    matrix_manifest = json.loads(
        MATRIX_MANIFEST.read_text(encoding="utf-8")
    )
    robustness = json.loads(
        ROBUSTNESS_MANIFEST.read_text(encoding="utf-8")
    )
    if (
        matrix_manifest["protocol_hash"] != protocol["protocol_hash"]
        or matrix_manifest["matrix_sha256"] != v32._sha256_file(MATRIX)
        or robustness["parent_protocol_hash"] != protocol["protocol_hash"]
        or robustness["matrix_sha256"]
        != matrix_manifest["matrix_sha256"]
        or robustness["implementation_sha256"]
        != v32._sha256_file(WORKER)
    ):
        raise RuntimeError("robustness chain freeze identity mismatch")
    expected: Dict[tuple[str, str, int], int] = {}
    for row in v32._read_jsonl(MATRIX):
        if (
            str(row["family"]) in FAMILIES
            and row.get("training_seed") is not None
        ):
            key = _identity(row)
            expected[key] = expected.get(key, 0) + 1
    if len(expected) != 45 or sum(expected.values()) != 3360:
        raise RuntimeError("robustness learning job matrix mismatch")
    return expected, robustness


def _nominal_real_passed() -> bool:
    path = RESULTS / "real_learning/merged_audit.json"
    if not path.is_file():
        return False
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        audit.get("passed")
        and int(audit.get("row_count", -1)) == 5040
        and int(audit.get("route_count", -1)) == 5040
    )


def _write_status(
    *,
    state: str,
    expected: Mapping[tuple[str, str, int], int],
    running: Mapping[tuple[str, str, int], subprocess.Popen[Any]],
    robustness: Mapping[str, Any],
) -> None:
    completed = {
        identity
        for identity, total in expected.items()
        if _completed(identity, total)
    }
    completed_rows = sum(expected[item] for item in completed)
    payload = {
        "schema_version": 1,
        "state": state,
        "robustness_manifest_hash": robustness["manifest_hash"],
        "expected_jobs": len(expected),
        "expected_rows": sum(expected.values()),
        "completed_jobs": len(completed),
        "completed_rows": completed_rows,
        "running": [
            {
                "family": identity[0],
                "model": identity[1],
                "training_seed": identity[2],
                "pid": process.pid,
            }
            for identity, process in sorted(running.items())
        ],
    }
    smoke._atomic_json(STATUS, payload)


def run(*, max_concurrency: int, poll_seconds: float) -> None:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    expected, robustness = _freeze_audit()
    ordered = sorted(
        expected,
        key=lambda item: (
            FAMILIES.index(item[0]),
            item[1],
            item[2],
        ),
    )
    running: Dict[
        tuple[str, str, int], subprocess.Popen[Any]
    ] = {}
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    while not _nominal_real_passed():
        _write_status(
            state="waiting_for_real_learning_audit",
            expected=expected,
            running=running,
            robustness=robustness,
        )
        time.sleep(poll_seconds)

    creationflags = (
        int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if sys.platform == "win32"
        else 0
    )
    while True:
        for identity, process in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del running[identity]
            if return_code != 0:
                raise RuntimeError(
                    "robustness job failed: "
                    f"{identity}, return_code={return_code}"
                )

        completed = {
            identity
            for identity, total in expected.items()
            if _completed(identity, total)
        }
        if len(completed) == len(expected):
            _write_status(
                state="completed",
                expected=expected,
                running=running,
                robustness=robustness,
            )
            return

        for identity in ordered:
            if len(running) >= max_concurrency:
                break
            if identity in completed or identity in running:
                continue
            family, model, seed = identity
            run_dir = _job_dir(identity)
            command = [
                sys.executable,
                "-X",
                "utf8",
                "-B",
                str(WORKER),
                "--family",
                family,
                "--model",
                model,
                "--seed",
                str(seed),
                "--device",
                "cuda",
            ]
            if (run_dir / "results.jsonl").exists():
                command.append("--resume")
            label = f"{family}__{model}__train{seed}"
            stdout_path = DIAGNOSTICS / f"formal_{label}.stdout.log"
            stderr_path = DIAGNOSTICS / f"formal_{label}.stderr.log"
            with stdout_path.open("a", encoding="utf-8") as stdout, (
                stderr_path.open("a", encoding="utf-8")
            ) as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creationflags,
                )
            running[identity] = process

        _write_status(
            state="running",
            expected=expected,
            running=running,
            robustness=robustness,
        )
        time.sleep(poll_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)
    run(
        max_concurrency=int(args.max_concurrency),
        poll_seconds=float(args.poll_seconds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

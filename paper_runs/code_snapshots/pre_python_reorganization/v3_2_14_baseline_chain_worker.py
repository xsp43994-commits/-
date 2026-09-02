#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schedule all remaining frozen nominal baseline jobs with bounded concurrency."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import paper_v3_2_experiments as v32
import v3_2_14_evaluation_smoke as smoke


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14"
PROTOCOL = (
    ROOT
    / "paper_runs/protocols/multimap_generalization_v3_2_14/protocol.json"
)
MATRIX = OUTPUT / "formal_evaluation/evaluation_matrix.jsonl"
MATRIX_MANIFEST = (
    OUTPUT / "formal_evaluation/evaluation_matrix_manifest.json"
)
RESULTS = OUTPUT / "formal_evaluation/results"
DIAGNOSTICS = OUTPUT / "diagnostics"
WORKER = ROOT / "v3_2_14_nominal_baseline_worker.py"
STATUS = RESULTS / "baseline_chain_status.json"
FAMILIES = (
    "synthetic_main_baselines",
    "synthetic_supplementary",
    "real_baselines",
)
INITIAL_EXTERNAL_JOBS = (
    ("synthetic_main_baselines", "aco", 42),
    ("synthetic_main_baselines", "aco", 43),
    ("synthetic_main_baselines", "aco", 44),
    ("synthetic_main_baselines", "aco", 45),
)


def _identity(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["family"]),
        str(row["model"]),
        int(row["planner_seed"]),
    )


def _job_dir(identity: tuple[str, str, int]) -> Path:
    family, model, seed = identity
    return RESULTS / family / "jobs" / f"{model}__seed{seed}"


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


def _freeze_audit() -> Dict[tuple[str, str, int], int]:
    protocol = v32.load_v3_2_protocol(PROTOCOL)
    manifest = json.loads(MATRIX_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest["protocol_hash"] != protocol["protocol_hash"]
        or manifest["matrix_sha256"] != v32._sha256_file(MATRIX)
        or int(manifest["row_count"]) != 21648
    ):
        raise RuntimeError("baseline chain freeze identity mismatch")
    expected: Dict[tuple[str, str, int], int] = {}
    for row in v32._read_jsonl(MATRIX):
        if str(row["family"]) in FAMILIES:
            key = _identity(row)
            expected[key] = expected.get(key, 0) + 1
    if len(expected) != 33 or sum(expected.values()) != 5544:
        raise RuntimeError("baseline job matrix mismatch")
    return expected


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
) -> None:
    completed = {
        identity
        for identity, total in expected.items()
        if _completed(identity, total)
    }
    smoke._atomic_json(
        STATUS,
        {
            "schema_version": 1,
            "state": state,
            "matrix_sha256": v32._sha256_file(MATRIX),
            "expected_jobs": len(expected),
            "expected_rows": sum(expected.values()),
            "completed_jobs": len(completed),
            "completed_rows": sum(expected[item] for item in completed),
            "running": [
                {
                    "family": identity[0],
                    "model": identity[1],
                    "planner_seed": identity[2],
                    "pid": process.pid,
                }
                for identity, process in sorted(running.items())
            ],
        },
    )


def run(*, max_concurrency: int, poll_seconds: float) -> None:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    expected = _freeze_audit()
    running: Dict[
        tuple[str, str, int], subprocess.Popen[Any]
    ] = {}
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)

    # 真实学习释放资源后才扩展传统规划器并发，避免 16 GB 内存换页。
    while not _nominal_real_passed():
        _write_status(
            state="waiting_for_real_learning_audit",
            expected=expected,
            running=running,
        )
        time.sleep(poll_seconds)

    ordered = sorted(
        expected,
        key=lambda item: (
            FAMILIES.index(item[0]),
            item[1],
            item[2],
        ),
    )
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
                    "baseline job failed: "
                    f"{identity}, return_code={return_code}"
                )
        completed = {
            identity
            for identity, total in expected.items()
            if _completed(identity, total)
        }
        if len(completed) == len(expected):
            _write_status(
                state="completed", expected=expected, running=running
            )
            return
        external_active = sum(
            1
            for identity in INITIAL_EXTERNAL_JOBS
            if identity not in completed
        )
        for identity in ordered:
            if len(running) + external_active >= max_concurrency:
                break
            if (
                identity in completed
                or identity in running
                or identity in INITIAL_EXTERNAL_JOBS
            ):
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
                "--planner-seed",
                str(seed),
            ]
            if (run_dir / "results.jsonl").exists():
                command.append("--resume")
            label = f"{family}__{model}__seed{seed}"
            with (
                (DIAGNOSTICS / f"formal_{label}.stdout.log").open(
                    "a", encoding="utf-8"
                ) as stdout,
                (DIAGNOSTICS / f"formal_{label}.stderr.log").open(
                    "a", encoding="utf-8"
                ) as stderr,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creationflags,
                )
            running[identity] = process
        _write_status(
            state="running", expected=expected, running=running
        )
        time.sleep(poll_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)
    run(
        max_concurrency=int(args.max_concurrency),
        poll_seconds=float(args.poll_seconds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文工作区的统一命令行入口。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from uav_inspection.paths import WORKSPACE_ROOT


def _audit_workspace() -> int:
    from tools.maintenance.paper_workspace_audit import audit

    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def _audit_checkpoints(extra: Sequence[str]) -> int:
    from uav_inspection.experiments.paper_v3_2_experiments import main

    return int(main([*extra, "audit-checkpoints"]))


def _statistics(extra: Sequence[str]) -> int:
    from uav_inspection.analysis.v3_2_14_statistics import main

    return int(main(extra))


def _figures() -> int:
    from uav_inspection.figures.v3_2_14_split_publication_figures import main

    main()
    return 0


def _show_paths() -> int:
    paths = {
        "workspace": WORKSPACE_ROOT,
        "protocols": WORKSPACE_ROOT / "paper_runs/protocols",
        "formal_results": WORKSPACE_ROOT
        / "paper_runs/multimap_v3_2_14/formal_evaluation/results",
        "analysis": WORKSPACE_ROOT / "paper_runs/multimap_v3_2_14/analysis",
        "figures": WORKSPACE_ROOT
        / "paper_runs/multimap_v3_2_14/figures/paper_final",
        "source_snapshot": WORKSPACE_ROOT
        / "paper_runs/code_snapshots/pre_python_reorganization",
    }
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="论文训练、评价、统计和制图统一入口")
    parser.add_argument(
        "command",
        choices=(
            "audit-workspace",
            "audit-checkpoints",
            "statistics",
            "figures",
            "show-paths",
        ),
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="透传给对应正式入口的参数")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit-workspace":
        return _audit_workspace()
    if args.command == "audit-checkpoints":
        return _audit_checkpoints(args.args)
    if args.command == "statistics":
        return _statistics(args.args)
    if args.command == "figures":
        if args.args:
            raise SystemExit("figures命令不接受额外参数")
        return _figures()
    return _show_paths()


if __name__ == "__main__":
    raise SystemExit(main())

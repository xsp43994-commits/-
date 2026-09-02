#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文专用工作区的可审计清理工具。

本工具只把明确列出的旧资产移动到隔离区，不直接永久删除。正式结果、
冻结协议、best_safe checkpoint、地图资产和最终论文图均不属于候选范围。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Iterable


ROOT = WORKSPACE_ROOT
AUDIT_DIR = ROOT / "paper_runs" / "cleanup_audit_20260802"
QUARANTINE_DIR = ROOT / "_cleanup_quarantine_20260802"


LEGACY_TOP_LEVEL_DIRS = (
    "training_runs",
    "paper_runs/training",
    "paper_runs/baselines",
    "paper_runs/evaluation",
    "paper_runs/publication",
    "paper_runs/invalidated",
    "paper_runs/difficulty_v2_1",
    "paper_runs/difficulty_v2_1_probe_round1",
    "paper_runs/difficulty_v2_1_probe_round2",
    "paper_runs/analysis",
    "paper_runs/validation",
    "paper_runs/manifests",
    "paper_runs/logs",
    "paper_runs/audits",
    "paper_runs/dispatcher_smoke",
)

OLD_PROTOCOL_OUTPUTS = tuple(
    f"paper_runs/multimap_v3_2_{version}" for version in range(1, 14) if version != 4
)

V31_TRANSIENT_DIRS = tuple(
    f"paper_runs/multimap_v3_1/{name}"
    for name in (
        "diagnostics",
        "logs",
        "monitoring",
        "parallel_smoke",
        "pilot",
        "smoke",
        "supervisor",
        "audits/formal_full_seed42_3000ep_runtime_io_failure_v3_1_16",
        "audits/pilot_full_seed42_600ep_preepisode_failure_v3_1_15",
    )
)

V32_TRANSIENT_PATHS = (
    "paper_runs/multimap_v3_2/formal_evaluation",
    "paper_runs/multimap_v3_2/pilot",
    "paper_runs/multimap_v3_2/environment_freeze.pre_formal_assessment_path_fix.json",
    "paper_runs/multimap_v3_2/environment_freeze.pre_formal_callback_schema_fix.json",
    "paper_runs/multimap_v3_2/environment_freeze.superseded_callback_fix.json",
    "paper_runs/multimap_v3_2/environment_freeze.superseded_pre_evaluator.json",
    "paper_runs/multimap_v3_2/formal_training/formal_traditional_ppo_seed42_3000ep_failed_callback_schema",
)

V3214_TRANSIENT_PATHS = (
    "paper_runs/multimap_v3_2_14/diagnostics",
    "paper_runs/multimap_v3_2_14/smoke",
    "paper_runs/multimap_v3_2_14/formal_evaluation/robustness_v1_failed_localization_archive",
    "paper_runs/multimap_v3_2_14/formal_evaluation/robustness_v2_failed_dem_start_archive",
    "paper_runs/multimap_v3_2_14/formal_evaluation/results/synthetic_learning_serial_probe_archive_20260730",
)

LEGACY_ROOT_FILES = (
    "export_v2_modeling_image.py",
    "PAPER_TRAINING_GUIDE_CN.md",
    "paper_publication_analysis.py",
    "paper_publication_figures.py",
    "PPO_Pointer综合评价统计判优与顶刊级制图计划_v3.md",
    "render_animations_to_videos.py",
    "run_paper_training.ps1",
    "run_algorithms_3d.py",
    "v2_3d_visualization.py",
    "v3_export_static_images.py",
    "v3_multi_window_compare.py",
    "v4_metrics_evaluation.py",
    "v5_fpv_animation.py",
    "v6_fpv_animation.py",
    "v3_2_12_mixed_threshold_probe.py",
    "v3_2_12_post_generation_worker.py",
    "v3_2_12_real_attempt_shard.py",
    "v3_2_12_synthetic_attempt_shard.py",
    "v3_2_12_threshold_probe.py",
    "v3_2_13_assignment_relaxation_diagnostic.py",
    "v3_2_13_compact_resource_bound_diagnostic.py",
    "v3_2_13_flow_feasibility_diagnostic.py",
    "v3_2_13_high_threshold_diagnostic.py",
    "v3_2_13_low_route_calibration_diagnostic.py",
    "v3_2_13_low_threshold_upper_diagnostic.py",
    "v3_2_13_migrate_real_shards.py",
    "v3_2_13_subtour_cut_diagnostic.py",
    "v3_2_13_synthetic_tasks.py",
    "v3_2_13_threshold_feasibility_diagnostic.py",
    "v3_2_13_witness_diagnostic.py",
    "v3_2_4_certificate_search.py",
)

ORPHAN_TESTS = (
    "tests/test_paper_publication_analysis.py",
    "tests/test_run_algorithms_3d.py",
    "tests/test_v3_2_2_coordinate_boundary.py",
    "tests/test_v3_2_4_certificate_search.py",
)

ENVIRONMENT_CLUTTER = (
    "__pycache__",
    "logs",
    "static_images",
    "trajectories_3d",
    "rendered_videos",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _tree_size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _record(candidates: list[dict[str, Any]], rel: str, category: str, reason: str) -> None:
    path = ROOT / rel
    if not path.exists() and not path.is_symlink():
        return
    candidates.append(
        {
            "path": rel.replace("\\", "/"),
            "category": category,
            "reason": reason,
            "kind": "directory" if path.is_dir() and not path.is_symlink() else "file",
            "bytes": _tree_size(path),
        }
    )


def collect_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for rel in LEGACY_TOP_LEVEL_DIRS:
        _record(candidates, rel, "legacy_experiment", "已被v3.2.14第二次正式实验替代")
    for rel in OLD_PROTOCOL_OUTPUTS:
        _record(candidates, rel, "superseded_protocol_output", "旧协议生成失败或中间产物")

    # v3.2.4仅保留最终协议仍引用的real_corridor_assets。
    v324 = ROOT / "paper_runs" / "multimap_v3_2_4"
    if v324.exists():
        for child in v324.iterdir():
            if child.name != "real_corridor_assets":
                _record(candidates, _relative(child), "superseded_protocol_output", "v3.2.4非正式道路资产")

    for rel in V31_TRANSIENT_DIRS:
        _record(candidates, rel, "training_transient", "训练试跑、监控或诊断产物")
    for rel in V32_TRANSIENT_PATHS:
        _record(candidates, rel, "training_transient", "传统PPO试跑、失败运行或已取代冻结文件")
    for rel in V3214_TRANSIENT_PATHS:
        _record(candidates, rel, "evaluation_transient", "正式评价前的失败或探测产物")

    v31_training = ROOT / "paper_runs" / "multimap_v3_1" / "formal_training"
    for seed in range(42, 47):
        _record(
            candidates,
            _relative(v31_training / f"formal_ppo_mlp_seed{seed}_3000ep"),
            "excluded_model",
            "ppo_mlp已被traditional_ppo替代并排除出论文",
        )
    for training_root in (
        v31_training,
        ROOT / "paper_runs" / "multimap_v3_2" / "formal_training",
    ):
        if not training_root.exists():
            continue
        for run_dir in training_root.iterdir():
            if not run_dir.is_dir() or run_dir.name == "group_health" or "ppo_mlp" in run_dir.name or "failed" in run_dir.name:
                continue
            for checkpoint in ("latest.pt", "best_candidate.pt"):
                path = run_dir / checkpoint
                if path.exists():
                    _record(candidates, _relative(path), "resume_checkpoint", "论文只使用best_safe.pt，不再恢复训练")

    figures_root = ROOT / "paper_runs" / "multimap_v3_2_14" / "figures"
    if figures_root.exists():
        for child in figures_root.iterdir():
            if child.name != "paper_final":
                _record(candidates, _relative(child), "superseded_figure", "已由唯一正式paper_final图包替代")

    for rel in LEGACY_ROOT_FILES:
        _record(candidates, rel, "legacy_code", "第一轮展示、旧分析或不可达诊断代码")
    for rel in ORPHAN_TESTS:
        _record(candidates, rel, "orphan_test", "仅测试已删除的旧模块")
    for rel in ENVIRONMENT_CLUTTER:
        _record(candidates, rel, "environment_clutter", "缓存、IDE配置或空展示目录")

    # 父目录已入选时不再重复记录其中的子项。
    ordered = sorted(candidates, key=lambda item: (len(Path(item["path"]).parts), item["path"]))
    kept: list[dict[str, Any]] = []
    selected_paths: list[Path] = []
    for item in ordered:
        path = Path(item["path"])
        if any(parent == path or parent in path.parents for parent in selected_paths):
            continue
        selected_paths.append(path)
        kept.append(item)
    return kept


def write_inventory() -> None:
    rows: list[dict[str, Any]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if QUARANTINE_DIR in path.parents or AUDIT_DIR in path.parents:
            continue
        stat = path.stat()
        rows.append(
            {
                "path": _relative(path),
                "bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    _write_rows(AUDIT_DIR / "inventory_before.csv", rows, ["path", "bytes", "modified_utc"])


def critical_hashes() -> list[dict[str, Any]]:
    files: set[Path] = {
        ROOT / "paper_runs/multimap_v3_2_14/formal_evaluation/evaluation_matrix.jsonl",
        ROOT / "paper_runs/multimap_v3_2_14/formal_evaluation/results/final_results.jsonl",
        ROOT / "paper_runs/multimap_v3_2_14/formal_evaluation/results/final_audit_status.json",
        ROOT / "paper_runs/multimap_v3_2_14/figures/paper_final/figure_manifest.json",
        ROOT / "paper_runs/multimap_v3_2_14/figures/paper_final/qa_report.json",
    }
    for root in (
        ROOT / "paper_runs/multimap_v3_1/formal_training",
        ROOT / "paper_runs/multimap_v3_2/formal_training",
    ):
        files.update(root.glob("formal_*/best_safe.pt"))
    files.update((ROOT / "paper_runs/protocols").rglob("*"))
    files.update((ROOT / "paper_runs/multimap_v3_2_14/analysis").rglob("*manifest*.json"))
    rows = []
    for path in sorted((p for p in files if p.is_file()), key=lambda p: _relative(p)):
        rows.append({"path": _relative(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return rows


def prepare() -> list[dict[str, Any]]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    write_inventory()
    hashes = critical_hashes()
    _write_json(AUDIT_DIR / "critical_hashes_before.json", {"created_utc": _utc_now(), "files": hashes})
    candidates = collect_candidates()
    _write_json(
        AUDIT_DIR / "cleanup_candidates.json",
        {
            "created_utc": _utc_now(),
            "candidate_count": len(candidates),
            "candidate_bytes": sum(item["bytes"] for item in candidates),
            "items": candidates,
        },
    )
    _write_rows(
        AUDIT_DIR / "cleanup_candidates.csv",
        candidates,
        ["path", "category", "reason", "kind", "bytes"],
    )
    return candidates


def quarantine(candidates: list[dict[str, Any]]) -> None:
    if QUARANTINE_DIR.exists():
        raise RuntimeError(f"隔离区已存在，拒绝覆盖：{QUARANTINE_DIR}")
    moved: list[dict[str, Any]] = []
    QUARANTINE_DIR.mkdir(parents=True)
    for item in candidates:
        source = ROOT / item["path"]
        if not source.exists() and not source.is_symlink():
            raise FileNotFoundError(source)
        destination = QUARANTINE_DIR / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved.append({**item, "quarantine_path": _relative(destination)})
    _write_json(
        AUDIT_DIR / "cleanup_manifest.json",
        {
            "schema": "paper-workspace-cleanup-v1",
            "quarantined_utc": _utc_now(),
            "quarantine_dir": _relative(QUARANTINE_DIR),
            "permanently_deleted": False,
            "audit_passed": False,
            "item_count": len(moved),
            "bytes": sum(item["bytes"] for item in moved),
            "items": moved,
        },
    )
    _write_rows(
        AUDIT_DIR / "cleanup_manifest.csv",
        moved,
        ["path", "quarantine_path", "category", "reason", "kind", "bytes"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成论文工作区清理清单并安全隔离旧资产。")
    parser.add_argument("action", choices=("prepare", "quarantine"))
    args = parser.parse_args()
    candidates = prepare()
    if args.action == "quarantine":
        quarantine(candidates)
    print(
        json.dumps(
            {
                "action": args.action,
                "candidate_count": len(candidates),
                "candidate_gib": round(sum(item["bytes"] for item in candidates) / 1024**3, 3),
                "audit_dir": str(AUDIT_DIR),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

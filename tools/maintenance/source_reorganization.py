#!/usr/bin/env python3
"""将整理后的活动源码改为包导入，并维护旧路径到新路径的映射。"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "paper_runs/code_snapshots/pre_python_reorganization"

MODULE_PATHS = {
    "final_python_ppo_pointer": "uav_inspection.core.final_python_ppo_pointer",
    "ppo_training_scenario": "uav_inspection.core.ppo_training_scenario",
    "paper_protocol": "uav_inspection.core.paper_protocol",
    "paper_evaluation": "uav_inspection.core.paper_evaluation",
    "paper_experiments": "uav_inspection.experiments.paper_experiments",
    "paper_multimap_experiments": "uav_inspection.experiments.paper_multimap_experiments",
    "paper_difficulty_experiments": "uav_inspection.experiments.paper_difficulty_experiments",
    "paper_v3_2_experiments": "uav_inspection.experiments.paper_v3_2_experiments",
    "prepare_v3_2_1_real_contexts": "uav_inspection.generation.prepare_v3_2_1_real_contexts",
    "v3_2_1_real_task_shards": "uav_inspection.generation.v3_2_1_real_task_shards",
    "v3_2_12_parametric_certificate_search": "uav_inspection.generation.v3_2_12_parametric_certificate_search",
    "v3_2_12_synthetic_tasks": "uav_inspection.generation.v3_2_12_synthetic_tasks",
    "v3_2_13_calibrated_mixed_diagnostic": "uav_inspection.generation.v3_2_13_calibrated_mixed_diagnostic",
    "v3_2_13_certificate_composition": "uav_inspection.generation.v3_2_13_certificate_composition",
    "v3_2_13_certificate_witness": "uav_inspection.generation.v3_2_13_certificate_witness",
    "v3_2_14_direct_threshold_probe": "uav_inspection.generation.v3_2_14_direct_threshold_probe",
    "v3_2_14_inject_direct_task": "uav_inspection.generation.v3_2_14_inject_direct_task",
    "v3_2_14_migrate_real_shards": "uav_inspection.generation.v3_2_14_migrate_real_shards",
    "v3_2_14_migrate_synthetic": "uav_inspection.generation.v3_2_14_migrate_synthetic",
    "v3_2_14_post_generation_worker": "uav_inspection.generation.v3_2_14_post_generation_worker",
    "v3_2_14_evaluation_smoke": "uav_inspection.evaluation.v3_2_14_evaluation_smoke",
    "v3_2_14_nominal_baseline_worker": "uav_inspection.evaluation.v3_2_14_nominal_baseline_worker",
    "v3_2_14_nominal_learning_worker": "uav_inspection.evaluation.v3_2_14_nominal_learning_worker",
    "v3_2_14_learning_chain_worker": "uav_inspection.evaluation.v3_2_14_learning_chain_worker",
    "v3_2_14_baseline_chain_worker": "uav_inspection.evaluation.v3_2_14_baseline_chain_worker",
    "v3_2_14_robustness_worker": "uav_inspection.evaluation.v3_2_14_robustness_worker",
    "v3_2_14_robustness_chain_worker": "uav_inspection.evaluation.v3_2_14_robustness_chain_worker",
    "v3_2_14_final_audit_worker": "uav_inspection.evaluation.v3_2_14_final_audit_worker",
    "v3_2_14_statistics": "uav_inspection.analysis.v3_2_14_statistics",
    "manuscript_multiobjective_v1": "uav_inspection.analysis.manuscript_multiobjective_v1",
    "manuscript_training_aware_v2": "uav_inspection.analysis.manuscript_training_aware_v2",
    "manuscript_training_priority_v3": "uav_inspection.analysis.manuscript_training_priority_v3",
    "manuscript_operational_band_v4": "uav_inspection.analysis.manuscript_operational_band_v4",
    "manuscript_preplot_closure_v5": "uav_inspection.analysis.manuscript_preplot_closure_v5",
    "audit_manuscript_multiobjective_v1": "uav_inspection.analysis.audit_manuscript_multiobjective_v1",
    "audit_manuscript_training_aware_v2": "uav_inspection.analysis.audit_manuscript_training_aware_v2",
    "audit_manuscript_preplot_closure_v5": "uav_inspection.analysis.audit_manuscript_preplot_closure_v5",
    "v3_2_14_analysis_chain_worker": "uav_inspection.analysis.v3_2_14_analysis_chain_worker",
    "v3_2_14_publication_figures": "uav_inspection.figures.v3_2_14_publication_figures",
    "v3_2_14_split_publication_figures": "uav_inspection.figures.v3_2_14_split_publication_figures",
    "paper_workspace_audit": "tools.maintenance.paper_workspace_audit",
    "paper_workspace_cleanup": "tools.maintenance.paper_workspace_cleanup",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _active_python_files() -> list[Path]:
    roots = (
        ROOT / "uav_inspection",
        ROOT / "scripts",
        ROOT / "tools",
        ROOT / "tests",
        ROOT / "python_classical_algs",
    )
    return sorted(path for base in roots for path in base.rglob("*.py"))


def rewrite_imports() -> list[str]:
    """执行可重复的机械导入改写，不接触冻结证据目录。"""

    changed: list[str] = []
    for path in _active_python_files():
        if path == Path(__file__).resolve():
            continue
        original = path.read_text(encoding="utf-8-sig")
        text = original
        for old, new in MODULE_PATHS.items():
            parent, name = new.rsplit(".", 1)
            text = re.sub(
                rf"(?m)^(\s*)import\s+{re.escape(old)}\s+as\s+(\w+)(\s*(?:#.*)?)$",
                rf"\1from {parent} import {name} as \2\3",
                text,
            )
            text = re.sub(
                rf"(?m)^(\s*)import\s+{re.escape(old)}(\s*(?:#.*)?)$",
                rf"\1from {parent} import {name}\2",
                text,
            )
            text = re.sub(
                rf"(?m)^(\s*)from\s+{re.escape(old)}\s+import\s+",
                rf"\1from {new} import ",
                text,
            )
            text = text.replace(
                f'importlib.import_module("{old}")',
                f'importlib.import_module("{new}")',
            )
            text = text.replace(
                f"importlib.import_module('{old}')",
                f"importlib.import_module('{new}')",
            )
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed.append(path.relative_to(ROOT).as_posix())
    return changed


def rewrite_workspace_roots() -> list[str]:
    """把依赖脚本目录的ROOT定义统一替换为工作区根目录。"""

    changed: list[str] = []
    patterns = (
        "ROOT = Path(__file__).resolve().parent",
        'DEFAULT_DEM_PATH = Path(__file__).resolve().parent / "map_data/AP_15010_FBS_F2760_RT1.dem.tif"',
    )
    for path in _active_python_files():
        if path == Path(__file__).resolve() or path.name == "paths.py":
            continue
        original = path.read_text(encoding="utf-8-sig")
        text = re.sub(
            rf"(?m)^{re.escape(patterns[0])}$",
            "ROOT = WORKSPACE_ROOT",
            original,
        )
        text = text.replace(
            patterns[1],
            'DEFAULT_DEM_PATH = WORKSPACE_ROOT / "map_data/AP_15010_FBS_F2760_RT1.dem.tif"',
        )
        text = text.replace(
            "Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent",
            "Path(repo_root) if repo_root is not None else WORKSPACE_ROOT",
        )
        if text != original and "WORKSPACE_ROOT" in text and "from uav_inspection.paths import WORKSPACE_ROOT" not in text:
            marker = "from pathlib import Path\n"
            if marker not in text:
                raise RuntimeError(f"无法为{path}插入WORKSPACE_ROOT导入")
            text = text.replace(
                marker,
                marker + "\nfrom uav_inspection.paths import WORKSPACE_ROOT\n",
                1,
            )
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed.append(path.relative_to(ROOT).as_posix())
    return changed


def relocation_rows() -> list[dict[str, object]]:
    manifest = json.loads((SNAPSHOT / "source_snapshot_manifest.json").read_text(encoding="utf-8-sig"))
    rows: list[dict[str, object]] = []
    protocol_names = {path.name for path in (ROOT / "scripts/protocol_builders").glob("*.py") if path.name != "__init__.py"}
    for record in manifest:
        old = str(record["original_relative_path"])
        if old.endswith(".ps1"):
            active = ""
            status = "snapshot_only_legacy_launcher"
        elif old in protocol_names:
            active = f"scripts/protocol_builders/{old}"
            status = "active_protocol_builder"
        else:
            module = MODULE_PATHS.get(Path(old).stem)
            active = "" if module is None else module.replace(".", "/") + ".py"
            status = "snapshot_only" if not active else "active_module"
        active_path = ROOT / active if active else None
        rows.append(
            {
                "original_relative_path": old,
                "snapshot_relative_path": str(record["snapshot_relative_path"]),
                "active_relative_path": active,
                "status": status,
                "original_size_bytes": int(record["size_bytes"]),
                "original_sha256": str(record["sha256"]),
                "active_sha256": _sha256(active_path) if active_path and active_path.is_file() else "",
            }
        )
    return rows


def write_relocation_map() -> None:
    rows = relocation_rows()
    fields = list(rows[0])
    with (SNAPSHOT / "source_relocation_map.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema": "paper-source-relocation-v1",
        "source_count": len(rows),
        "root_python_target": 1,
        "rows": rows,
    }
    (SNAPSHOT / "source_relocation_map.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_reorganization_summary() -> None:
    """记录整理前后数量、目录边界和正式资产身份。"""

    categories = {
        "core": ROOT / "uav_inspection/core",
        "experiments": ROOT / "uav_inspection/experiments",
        "generation": ROOT / "uav_inspection/generation",
        "evaluation": ROOT / "uav_inspection/evaluation",
        "analysis": ROOT / "uav_inspection/analysis",
        "figures": ROOT / "uav_inspection/figures",
        "protocol_builders": ROOT / "scripts/protocol_builders",
        "maintenance": ROOT / "tools/maintenance",
    }
    payload = {
        "schema": "paper-python-reorganization-v1",
        "workspace": str(ROOT),
        "before": {"root_python_files": 64, "root_powershell_files": 3},
        "after": {
            "root_python_files": len(list(ROOT.glob("*.py"))),
            "root_powershell_files": len(list(ROOT.glob("*.ps1"))),
            "root_python_names": sorted(path.name for path in ROOT.glob("*.py")),
        },
        "active_category_python_counts": {
            name: len(list(path.glob("*.py"))) for name, path in categories.items()
        },
        "snapshot_source_count": len(relocation_rows()),
        "legacy_launchers_snapshot_only": 3,
        "source_snapshot_manifest_sha256": _sha256(
            SNAPSHOT / "source_snapshot_manifest.json"
        ),
        "source_relocation_map_sha256": _sha256(
            SNAPSHOT / "source_relocation_map.csv"
        ),
        "formal_matrix_sha256": "48a31ee9b58d41a617fff61acb6eba6a2d9a930767d7af15856f70a964686224",
        "formal_results_sha256": "4b620c21566c2e33c875f6bea2017b741b02a7d30d70aa50add60a6d06214a2c",
        "post_reorganization_audit": (
            "paper_runs/cleanup_audit_20260802/post_cleanup_audit.json"
        ),
    }
    destination = ROOT / "paper_runs/cleanup_audit_20260802/python_reorganization_manifest.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    changed_imports = rewrite_imports()
    changed_roots = rewrite_workspace_roots()
    write_relocation_map()
    write_reorganization_summary()
    print(
        json.dumps(
            {
                "changed_import_files": len(changed_imports),
                "changed_root_files": len(changed_roots),
                "relocation_rows": len(relocation_rows()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

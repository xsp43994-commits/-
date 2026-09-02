from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


# 关键参数集中在此处：所有输入均指向冻结的 v3.2.14，输出仅写入交付目录。
WORKSPACE = Path(r"C:\Users\xsp\Desktop\DRL代码")
FROZEN_ROOT = WORKSPACE / "paper_runs" / "multimap_v3_2_14"
DELIVERY_ROOT = WORKSPACE / "paper_delivery" / "EAAI_2026-08-09"

KEY_FILES = {
    "handoff": WORKSPACE / "HANDOFF.md",
    "writing_specification": Path(r"C:\Users\xsp\Desktop\论文仿写规范_AI直接执行版.md"),
    "final_results": FROZEN_ROOT / "formal_evaluation" / "results" / "final_results.jsonl",
    "final_audit_status": FROZEN_ROOT / "formal_evaluation" / "results" / "final_audit_status.json",
    "evaluation_matrix_manifest": FROZEN_ROOT / "formal_evaluation" / "evaluation_matrix_manifest.json",
    "analysis_manifest": FROZEN_ROOT / "analysis" / "pre_plot_statistics" / "analysis_manifest.json",
    "descriptive_metrics": FROZEN_ROOT / "analysis" / "pre_plot_statistics" / "descriptive_metrics.csv",
    "confirmatory_pairwise": FROZEN_ROOT / "analysis" / "pre_plot_statistics" / "confirmatory_pairwise.csv",
    "confirmatory_omnibus": FROZEN_ROOT / "analysis" / "pre_plot_statistics" / "confirmatory_omnibus.csv",
    "training_dimension_scores": FROZEN_ROOT / "analysis" / "manuscript_training_aware_v2" / "training_dimension_scores.csv",
    "training_seed_metrics": FROZEN_ROOT / "analysis" / "manuscript_training_aware_v2" / "training_seed_metrics.csv",
    "operational_scores": FROZEN_ROOT / "analysis" / "manuscript_operational_band_v4" / "selected_operational_scores_100.csv",
    "bootstrap_summary": FROZEN_ROOT / "analysis" / "manuscript_preplot_closure_v5" / "hierarchical_bootstrap_summary.csv",
}

FIGURE_ROOT = FROZEN_ROOT / "figures" / "paper_redraw_multibackend_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def copy_key_sources() -> None:
    target = DELIVERY_ROOT / "reproducibility" / "frozen_source_extract"
    target.mkdir(parents=True, exist_ok=True)
    for name, path in KEY_FILES.items():
        if name in {"handoff", "writing_specification"}:
            continue
        shutil.copy2(path, target / path.name)

    # 图件 Source Data 全量复制，保持文件名和层级，便于逐图复核。
    source_data = FIGURE_ROOT / "source_data"
    copied = DELIVERY_ROOT / "source_data" / "frozen_figure_source_data"
    if copied.exists():
        shutil.rmtree(copied)
    shutil.copytree(source_data, copied)


def build_snapshot() -> dict:
    missing = [str(path) for path in KEY_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen evidence:\n" + "\n".join(missing))

    audit = json.loads(KEY_FILES["final_audit_status"].read_text(encoding="utf-8-sig"))
    results_lines = count_lines(KEY_FILES["final_results"])
    if results_lines != 21648:
        raise RuntimeError(f"Unexpected final_results.jsonl line count: {results_lines}")
    if not audit.get("passed") or not audit.get("ppo_mlp_absent"):
        raise RuntimeError("Frozen audit gate did not pass or ppo_mlp_absent is false")

    descriptive = read_csv(KEY_FILES["descriptive_metrics"])
    pairwise = read_csv(KEY_FILES["confirmatory_pairwise"])
    omnibus = read_csv(KEY_FILES["confirmatory_omnibus"])
    training = read_csv(KEY_FILES["training_dimension_scores"])
    operational = read_csv(KEY_FILES["operational_scores"])
    bootstrap = read_csv(KEY_FILES["bootstrap_summary"])

    file_manifest = []
    for label, path in KEY_FILES.items():
        file_manifest.append(
            {
                "label": label,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    figure_manifest = []
    for path in sorted(FIGURE_ROOT.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".png", ".pdf", ".svg", ".tiff", ".m", ".opju"}:
            figure_manifest.append(
                {
                    "relative_path": str(path.relative_to(FIGURE_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    return {
        "snapshot_created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_version": "v3.2.14",
        "protocol_identity": "multimap_v3_2_14",
        "assertions": {
            "final_results_line_count": results_lines,
            "audit_passed": bool(audit.get("passed")),
            "ppo_mlp_absent": bool(audit.get("ppo_mlp_absent")),
            "no_training_or_evaluation_rerun": True,
            "map_is_independent_statistical_unit": True,
        },
        "audit_status": audit,
        "tables": {
            "descriptive_metrics": descriptive,
            "confirmatory_pairwise": pairwise,
            "confirmatory_omnibus": omnibus,
            "training_dimension_scores": training,
            "selected_operational_scores_100": operational,
            "hierarchical_bootstrap_summary": bootstrap,
        },
        "key_file_manifest": file_manifest,
        "frozen_figure_manifest": figure_manifest,
    }


def write_manifest_csv(snapshot: dict) -> None:
    path = DELIVERY_ROOT / "evidence" / "frozen_file_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = snapshot["key_file_manifest"] + [
        {
            "label": "figure:" + row["relative_path"],
            "path": str(FIGURE_ROOT / row["relative_path"]),
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in snapshot["frozen_figure_manifest"]
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    DELIVERY_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot = build_snapshot()
    copy_key_sources()
    evidence_path = DELIVERY_ROOT / "evidence" / "evidence_snapshot.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest_csv(snapshot)
    print(json.dumps(snapshot["assertions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

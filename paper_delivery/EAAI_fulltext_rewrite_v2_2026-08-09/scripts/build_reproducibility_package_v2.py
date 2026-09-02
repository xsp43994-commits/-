from __future__ import annotations

"""构建匿名可复现包；只复制冻结证据，不包含原始 Copernicus 栅格。"""

import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproducibility" / "anonymized_package_v2"


FILES = {
    WORKSPACE / "paper_runs" / "protocols" / "multimap_generalization_v3_2_14" / "protocol.json": "protocol/protocol_v3_2_14.json",
    WORKSPACE / "paper_runs" / "protocols" / "multimap_generalization_v3_2_14" / "analysis_protocol.json": "protocol/analysis_protocol_v3_2_14.json",
    WORKSPACE / "paper_runs" / "multimap_v3_2_14" / "formal_evaluation" / "evaluation_matrix.jsonl": "evaluation/evaluation_matrix.jsonl",
    WORKSPACE / "paper_runs" / "multimap_v3_2_14" / "formal_evaluation" / "evaluation_matrix_manifest.json": "evaluation/evaluation_matrix_manifest.json",
    WORKSPACE / "paper_runs" / "multimap_v3_2_14" / "formal_evaluation" / "results" / "final_results.jsonl": "results/final_results.jsonl",
    WORKSPACE / "paper_runs" / "multimap_v3_2_14" / "formal_evaluation" / "results" / "final_audit_status.json": "results/final_audit_status.json",
    WORKSPACE / "uav_inspection" / "core" / "final_python_ppo_pointer.py": "code/final_python_ppo_pointer.py",
    WORKSPACE / "uav_inspection" / "experiments" / "paper_multimap_experiments.py": "code/paper_multimap_experiments.py",
    WORKSPACE / "uav_inspection" / "analysis" / "v3_2_14_statistics.py": "code/v3_2_14_statistics.py",
    WORKSPACE / "paper_cli.py": "code/paper_cli.py",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_readme() -> None:
    text = """# Anonymous reproducibility package v2

This package accompanies the EAAI full-text rewrite. It contains the frozen v3.2.14 protocol, task/evaluation matrix, 21,648 formal result rows, audit state, pre-plot statistics, figure Source Data, implementation snapshots and file hashes. It does not retrain models or recompute route evaluations.

## Evidence identity

- Confirmatory endpoint: map-level `safe_weighted_coverage`.
- Independent inferential unit: map.
- Formal result rows: 21,648.
- Paper-eligible conventional PPO: fixed-slot `FlatMLPActorCritic` (`traditional_ppo`).
- The excluded historical attention-containing prototype is not included.

## Suggested checks

1. Verify `manifest_sha256_v2.json`.
2. Confirm that `results/final_results.jsonl` has 21,648 lines.
3. Inspect `results/final_audit_status.json` before using any result.
4. Run analysis only with the protocol and map-level aggregation rules supplied here.

Example from the workspace checkout:

```powershell
python -X utf8 -B paper_cli.py audit-workspace
python -X utf8 -B paper_cli.py show-paths
```

## Copernicus data

Raw Copernicus DEM/DSM assets are intentionally not redistributed in this anonymous package. The simulation used Copernicus DEM GLO-30 terrain inputs. Reconstruct the geographic task set from the official product record (DOI: 10.5270/ESA-c5d3d65), the public region/task identifiers in the evaluation matrix, and the task-building code. Before public release, the authors must confirm that the region identifiers and attribution wording are suitable for their repository.

## Scope boundary

The geographic results are zero-shot DSM simulation transfer. They are not real-flight validation, deployment certification, or evidence of extrapolation beyond the trained node counts.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def redact_local_paths() -> list[str]:
    """仅匿名化本地账户/工作区路径；不改变数值、任务身份或实验字段。"""
    changed = []
    text_suffixes = {".json", ".jsonl", ".md", ".txt", ".csv", ".py"}
    replacements = [
        (r"C:\\Users\\xsp\\Desktop\\DRL代码", r"<WORKSPACE_ROOT>"),
        (r"C:\Users\xsp\Desktop\DRL代码", r"<WORKSPACE_ROOT>"),
        (r"C:\\Users\\xsp", r"<USER_HOME>"),
        (r"C:\Users\xsp", r"<USER_HOME>"),
    ]
    for path in OUT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        original = text
        for old, new in replacements:
            text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed.append(str(path.relative_to(OUT)).replace("\\", "/"))
    return changed


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    copied = []
    for src, rel in FILES.items():
        if not src.exists():
            raise FileNotFoundError(src)
        dst = OUT / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        copied.append({"source": str(src.relative_to(WORKSPACE)), "path": rel, "source_sha256": sha(src), "copy_sha256": sha(dst)})

    copy_tree(WORKSPACE / "paper_runs" / "multimap_v3_2_14" / "analysis" / "pre_plot_statistics", OUT / "statistics" / "pre_plot_statistics")
    copy_tree(ROOT / "figures" / "source_data", OUT / "source_data" / "figures")
    copy_tree(ROOT / "evidence", OUT / "evidence_architecture")
    write_readme()
    redacted_paths = redact_local_paths()
    for item in copied:
        copy_path = OUT / item["path"]
        item["copy_sha256"] = sha(copy_path)
        item["path_redacted"] = item["path"].replace("\\", "/") in redacted_paths

    env = {
        "platform": platform.platform(),
        "document_runtime_python": platform.python_version(),
        "project_python": "D:/Anaconda3/envs/Deeplearning/python.exe",
        "note": "Run model/statistics code with the project Deeplearning environment; paths are descriptive and not required to match.",
    }
    try:
        proc = subprocess.run([r"D:\Anaconda3\envs\Deeplearning\python.exe", "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60)
        env["pip_freeze"] = [line for line in proc.stdout.splitlines() if line.strip()]
    except Exception as exc:
        env["pip_freeze_error"] = type(exc).__name__
    (OUT / "environment_manifest_v2.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "manifest_sha256_v2.json":
            manifest.append({"path": str(p.relative_to(OUT)).replace("\\", "/"), "sha256": sha(p), "bytes": p.stat().st_size})
    (OUT / "manifest_sha256_v2.json").write_text(json.dumps({"files": manifest, "source_copies": copied}, ensure_ascii=False, indent=2), encoding="utf-8")

    identity_hits = []
    needles = (b"C:\\Users\\xsp", "DRL代码".encode("utf-8"))
    for p in OUT.rglob("*"):
        if p.is_file():
            data = p.read_bytes()
            if any(n in data for n in needles):
                identity_hits.append(str(p.relative_to(OUT)))
    report = {"files": len(manifest) + 1, "bytes": sum(x["bytes"] for x in manifest), "identity_hits": identity_hits,
              "path_redacted_files": redacted_paths,
              "raw_copernicus_assets_included": False}
    (ROOT / "qa" / "reproducibility_package_qa_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if identity_hits:
        raise RuntimeError(f"Identity scan failed: {identity_hits[:10]}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

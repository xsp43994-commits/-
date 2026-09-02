from __future__ import annotations

"""为第二轮完整交付目录生成最终文件清单与 SHA-256。"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "DELIVERY_MANIFEST_v2.json"
OUT_MD = ROOT / "DELIVERY_MANIFEST_v2.md"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    excluded = {OUT_JSON, OUT_MD}
    files = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and p not in excluded:
            files.append({"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size})
    core = [
        "deliverables/EAAI_manuscript_anonymized_v2.docx",
        "deliverables/EAAI_title_page_v2.docx",
        "deliverables/EAAI_supplementary_material_v2.docx",
        "deliverables/EAAI_highlights_v2.docx",
        "deliverables/EAAI_cover_letter_v2.docx",
        "qa/Final_Compliance_Report_v2.md",
        "literature/EAAI_12_Fulltext_Reading_Matrix.xlsx",
        "literature/Reference_Fulltext_Status_v2.xlsx",
        "literature/Claim_Citation_Map_v2.xlsx",
        "reproducibility/anonymized_reproducibility_package_v2.zip",
    ]
    payload = {
        "version": "EAAI_fulltext_rewrite_v2_2026-08-09",
        "status": "PASS_WITH_AUTHOR_INPUT",
        "file_count_excluding_manifest": len(files),
        "total_bytes_excluding_manifest": sum(f["bytes"] for f in files),
        "controlling_spec_sha256": "3344d4de6769ac9858f39398fab648b5a5d18db45fc85fe8b6d53ae931ff7ee3",
        "frozen_results_sha256": "4b620c21566c2e33c875f6bea2017b741b02a7d30d70aa50add60a6d06214a2c",
        "frozen_result_rows": 21648,
        "nature_skills_used": False,
        "experiments_rerun": False,
        "core_files_present": {p: any(f["path"] == p for f in files) for p in core},
        "files": files,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Delivery manifest v2", "", "- Status: **PASS_WITH_AUTHOR_INPUT**", f"- Files (excluding this manifest pair): **{len(files)}**", f"- Total bytes: **{payload['total_bytes_excluding_manifest']}**", "", "## Core files", ""]
    for path, present in payload["core_files_present"].items():
        lines.append(f"- {'PASS' if present else 'FAIL'} — `{path}`")
    lines += ["", "Complete per-file SHA-256 values are in `DELIVERY_MANIFEST_v2.json`."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(files), "bytes": payload["total_bytes_excluding_manifest"], "core": payload["core_files_present"]}, indent=2))


if __name__ == "__main__":
    main()

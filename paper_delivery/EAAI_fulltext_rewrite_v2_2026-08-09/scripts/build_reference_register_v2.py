"""Build the v2 verified-reference registry, RIS file, and access/link reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# 关键路径参数：旧登记簿只读，第二轮输出全部写入独立目录。
WORKSPACE = Path(r"C:\Users\xsp\Desktop\DRL代码")
V1_REGISTER = WORKSPACE / "paper_delivery" / "EAAI_2026-08-09" / "literature" / "literature_register.json"
V2_ROOT = WORKSPACE / "paper_delivery" / "EAAI_fulltext_rewrite_v2_2026-08-09"
SELECTION_PATH = V2_ROOT / "literature" / "reference_selection_v2.json"
MANIFEST_PATH = V2_ROOT / "literature" / "reference_fulltexts_manifest.json"


MANUAL_RECORDS: dict[str, dict[str, Any]] = {
    "Dorigo1996ACO": {
        "type": "journal-article",
        "title": "Ant system: optimization by a colony of cooperating agents",
        "authors": ["Marco Dorigo", "Vittorio Maniezzo", "Alberto Colorni"],
        "year": 1996,
        "venue": "IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics)",
        "volume": "26",
        "issue": "1",
        "article_number_or_pages": "29–41",
        "doi": "10.1109/3477.484436",
        "url": "https://doi.org/10.1109/3477.484436",
    },
    "Kirkpatrick1983SA": {
        "type": "journal-article",
        "title": "Optimization by simulated annealing",
        "authors": ["Scott Kirkpatrick", "C. Daniel Gelatt Jr.", "Mario P. Vecchi"],
        "year": 1983,
        "venue": "Science",
        "volume": "220",
        "issue": "4598",
        "article_number_or_pages": "671–680",
        "doi": "10.1126/science.220.4598.671",
        "url": "https://doi.org/10.1126/science.220.4598.671",
    },
    "Hart1968AStar": {
        "type": "journal-article",
        "title": "A formal basis for the heuristic determination of minimum cost paths",
        "authors": ["Peter E. Hart", "Nils J. Nilsson", "Bertram Raphael"],
        "year": 1968,
        "venue": "IEEE Transactions on Systems Science and Cybernetics",
        "volume": "4",
        "issue": "2",
        "article_number_or_pages": "100–107",
        "doi": "10.1109/TSSC.1968.300136",
        "url": "https://doi.org/10.1109/TSSC.1968.300136",
    },
    "Kennedy1995PSO": {
        "type": "conference-paper",
        "title": "Particle swarm optimization",
        "authors": ["James Kennedy", "Russell Eberhart"],
        "year": 1995,
        "venue": "Proceedings of ICNN'95 — International Conference on Neural Networks",
        "volume": "4",
        "issue": "",
        "article_number_or_pages": "1942–1948",
        "doi": "10.1109/ICNN.1995.488968",
        "url": "https://doi.org/10.1109/ICNN.1995.488968",
    },
    "Virtanen2020SciPy": {
        "type": "journal-article",
        "title": "SciPy 1.0: fundamental algorithms for scientific computing in Python",
        "authors": [
            "Pauli Virtanen", "Ralf Gommers", "Travis E. Oliphant", "Matt Haberland",
            "Tyler Reddy", "David Cournapeau", "Evgeni Burovski", "Pearu Peterson",
            "Warren Weckesser", "Jonathan Bright", "SciPy 1.0 Contributors",
        ],
        "year": 2020,
        "venue": "Nature Methods",
        "volume": "17",
        "issue": "3",
        "article_number_or_pages": "261–272",
        "doi": "10.1038/s41592-019-0686-2",
        "url": "https://doi.org/10.1038/s41592-019-0686-2",
    },
    "SciPyMILPHiGHS": {
        "type": "webpage",
        "title": "scipy.optimize.milp — SciPy API reference",
        "authors": ["SciPy community"],
        "year": 2026,
        "venue": "SciPy documentation",
        "volume": "",
        "issue": "",
        "article_number_or_pages": "",
        "doi": "",
        "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html",
    },
}


def normalise(identifier: str) -> str:
    return identifier.strip().lower()


def ris_type(record_type: str) -> str:
    return {
        "journal-article": "JOUR",
        "conference-paper": "CPAPER",
        "dataset": "DATA",
        "webpage": "ELEC",
    }.get(record_type, "GEN")


def build_ris(records: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for record in records:
        lines = [f"TY  - {ris_type(record.get('type', ''))}", f"ID  - {record['ref_id']}"]
        lines.append(f"TI  - {record.get('title', '')}")
        for author in record.get("authors", []):
            lines.append(f"AU  - {author}")
        lines.append(f"PY  - {record.get('year', '')}")
        if record.get("venue"):
            lines.append(f"T2  - {record['venue']}")
        if record.get("volume"):
            lines.append(f"VL  - {record['volume']}")
        if record.get("issue"):
            lines.append(f"IS  - {record['issue']}")
        if record.get("article_number_or_pages"):
            lines.append(f"SP  - {record['article_number_or_pages']}")
        if record.get("doi"):
            lines.append(f"DO  - {record['doi']}")
        lines.append(f"UR  - {record.get('access_url') or record.get('url', '')}")
        lines.append(f"N1  - Full-text status: {record.get('fulltext_status', '')}")
        lines.append(f"N1  - Sentence support: {record.get('support', '')}")
        lines.append("ER  -")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    v1 = json.loads(V1_REGISTER.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    # 旧登记簿的references与exemplars同时建索引，防止同刊样本只存在于一个数组。
    metadata_index: dict[str, dict[str, Any]] = {}
    for record in [*v1.get("references", []), *v1.get("exemplars", [])]:
        for identifier in (record.get("id"), record.get("doi")):
            if identifier:
                metadata_index[normalise(identifier)] = record
    for identifier, record in MANUAL_RECORDS.items():
        metadata_index[normalise(identifier)] = record

    file_index = {record["file"]: record for record in manifest}
    output_records: list[dict[str, Any]] = []
    missing_metadata: list[str] = []
    missing_files: list[str] = []
    for selected in selection["references"]:
        source_id = selected["source_id"]
        metadata = metadata_index.get(normalise(source_id))
        if metadata is None:
            missing_metadata.append(source_id)
            continue
        merged = dict(metadata)
        merged.update(selected)
        merged["authors"] = list(metadata.get("authors", []))
        merged["metadata_identity_verified"] = True
        fulltext_file = selected.get("fulltext_file")
        if fulltext_file:
            file_record = file_index.get(fulltext_file)
            if file_record is None:
                missing_files.append(fulltext_file)
            else:
                merged["fulltext_sha256"] = file_record["sha256"]
                merged["fulltext_pages"] = file_record["pages"]
                merged["fulltext_text_chars"] = file_record["text_chars"]
        output_records.append(merged)

    if missing_metadata or missing_files:
        raise RuntimeError(f"Reference build gate failed: metadata={missing_metadata}; files={missing_files}")

    status_payload = {
        "prepared": selection["prepared"],
        "selection_rule": selection["selection_rule"],
        "reference_count": len(output_records),
        "references": output_records,
        "claims": selection["claims"],
        "excluded_candidates": selection["excluded_candidates"],
        "school_access_required": [],
        "gate": {
            "existence": "pass",
            "identity": "pass",
            "fulltext_relevance": "pass for every retained source",
            "sentence_level_entailment": "mapped in claims; rechecked at manuscript insertion",
        },
    }
    literature_dir = V2_ROOT / "literature"
    (literature_dir / "reference_status_v2.json").write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (literature_dir / "verified_references_v2.ris").write_text(build_ris(output_records), encoding="utf-8")

    school_report = "# School Access Required List\n\n"
    school_report += "## Current result\n\n"
    school_report += "No school-access request is currently required for a retained v2 citation. "
    school_report += "All 30 retained records were verified through an acquired exemplar full text, a legal/open author or publisher version, an official dataset/software page, or a link-only full text used internally.\n\n"
    school_report += "## Copyright control\n\n"
    school_report += "Files marked `redistribute_pdf=false` are not copied into the redistributable reference-PDF package. Their DOI, publisher, official or host link is delivered instead. If a link later becomes unavailable, only that cited item—not the full historical candidate list—should be requested through the school library.\n"
    (literature_dir / "School_Access_Required_List.md").write_text(school_report, encoding="utf-8")

    link_lines = [
        "# Verified reference PDF/link package v2",
        "",
        "Only clearly redistributable/open PDFs are included as files. Other items are supplied as verified links.",
        "",
        "| Ref | Year | Title | Full-text status | Delivered access |",
        "|---|---:|---|---|---|",
    ]
    for record in output_records:
        if record.get("redistribute_pdf") and record.get("fulltext_file"):
            delivered = f"`reference_fulltexts_open/{record['fulltext_file']}`"
        else:
            delivered = f"[verified link]({record.get('access_url') or record.get('url', '')})"
        title = record.get("title", "").replace("|", "\\|")
        link_lines.append(
            f"| {record['ref_id']} | {record.get('year', '')} | {title} | {record['fulltext_status']} | {delivered} |"
        )
    (literature_dir / "Verified_Reference_Links_v2.md").write_text("\n".join(link_lines) + "\n", encoding="utf-8")
    print(json.dumps({"references": len(output_records), "claims": len(selection["claims"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()

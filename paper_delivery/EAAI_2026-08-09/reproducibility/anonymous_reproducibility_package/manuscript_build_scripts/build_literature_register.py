from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DELIVERY_ROOT = Path(r"C:\Users\xsp\Desktop\DRL代码\paper_delivery\EAAI_2026-08-09")
LITERATURE_ROOT = DELIVERY_ROOT / "literature"
CROSSREF_API = "https://api.crossref.org/works/"
USER_AGENT = "EAAI-manuscript-evidence-audit/1.0"


# 12 篇同刊 Original Research 样本。用途字段只描述可借鉴的科学功能，不复制文本。
EXEMPLARS = [
    ("10.1016/j.engappai.2026.115219", "Exact-neighbour UAV inspection allocation, endurance-aware pruning, and routing narrative"),
    ("10.1016/j.engappai.2026.113779", "Attention-assisted reinforcement-learning navigation and collision-free evaluation"),
    ("10.1016/j.engappai.2025.113518", "UAV swarm navigation under dynamic obstacles and situation awareness"),
    ("10.1016/j.engappai.2025.112090", "Quadcopter navigation under wind perturbations and controller benchmarking"),
    ("10.1016/j.engappai.2025.111219", "Aircraft trajectory planning, control integration, and engineering validation storyline"),
    ("10.1016/j.engappai.2025.110392", "Urban aerial motion planning with wind-field uncertainty"),
    ("10.1016/j.engappai.2024.109870", "Closed-loop RL path planning with field-test evidence boundaries"),
    ("10.1016/j.engappai.2024.108926", "PPO-based quadrotor planning and control integration"),
    ("10.1016/j.engappai.2024.108506", "Illegal-action handling for autonomous robot navigation"),
    ("10.1016/j.engappai.2024.109339", "Risk-aware traversability over elevation-defined uneven terrain"),
    ("10.1016/j.engappai.2023.106703", "Curriculum design and UAV manoeuvre decision-making"),
    ("10.1016/j.engappai.2023.105891", "Energy-aware multi-UAV trajectory optimisation"),
]


# DOI 文献用于支撑算法、约束安全、统计、无人机能耗和工程路径规划命题。
REFERENCE_DOIS = [
    "10.1016/j.engappai.2026.115219",
    "10.1016/j.engappai.2025.112090",
    "10.1016/j.engappai.2024.109870",
    "10.1016/j.engappai.2024.109339",
    "10.1016/j.engappai.2023.105891",
    "10.1016/j.engappai.2023.106703",
    "10.1016/j.engappai.2022.105321",
    "10.1016/j.engappai.2022.105182",
    "10.1109/TWC.2019.2902559",
    "10.1109/IROS.2017.8202133",
    "10.1007/s10994-021-05961-4",
    "10.2307/2279372",
    "10.2307/3001968",
    "10.1214/aoms/1177704172",
    "10.1214/aos/1176344552",
    "10.3102/10769986025002101",
    "10.1016/j.autcon.2024.105764",
]


MANUAL_REFERENCES = [
    {
        "id": "Schulman2017PPO",
        "type": "conference-paper",
        "title": "Proximal Policy Optimization Algorithms",
        "authors": ["John Schulman", "Filip Wolski", "Prafulla Dhariwal", "Alec Radford", "Oleg Klimov"],
        "year": 2017,
        "venue": "arXiv preprint arXiv:1707.06347",
        "url": "https://arxiv.org/abs/1707.06347",
        "pdf_url": "https://arxiv.org/pdf/1707.06347",
        "verification": "full text legally open",
        "claim": "Defines the clipped surrogate objective used by PPO.",
    },
    {
        "id": "Vinyals2015Pointer",
        "type": "conference-paper",
        "title": "Pointer Networks",
        "authors": ["Oriol Vinyals", "Meire Fortunato", "Navdeep Jaitly"],
        "year": 2015,
        "venue": "Advances in Neural Information Processing Systems 28",
        "url": "https://papers.nips.cc/paper/5866-pointer-networks",
        "pdf_url": "https://papers.nips.cc/paper_files/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Paper.pdf",
        "verification": "full text legally open",
        "claim": "Introduces pointer distributions over variable-length input positions for combinatorial outputs.",
    },
    {
        "id": "Vaswani2017Attention",
        "type": "conference-paper",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
        "year": 2017,
        "venue": "Advances in Neural Information Processing Systems 30",
        "url": "https://papers.nips.cc/paper/7181-attention-is-all-you-need",
        "pdf_url": "https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf",
        "verification": "full text legally open",
        "claim": "Provides the multi-head self-attention mechanism used in the node encoder.",
    },
    {
        "id": "Schulman2016GAE",
        "type": "conference-paper",
        "title": "High-Dimensional Continuous Control Using Generalized Advantage Estimation",
        "authors": ["John Schulman", "Philipp Moritz", "Sergey Levine", "Michael Jordan", "Pieter Abbeel"],
        "year": 2016,
        "venue": "International Conference on Learning Representations",
        "url": "https://arxiv.org/abs/1506.02438",
        "pdf_url": "https://arxiv.org/pdf/1506.02438",
        "verification": "full text legally open",
        "claim": "Defines generalized advantage estimation used in policy optimisation.",
    },
    {
        "id": "Mnih2016A3C",
        "type": "conference-paper",
        "title": "Asynchronous Methods for Deep Reinforcement Learning",
        "authors": ["Volodymyr Mnih", "Adria Puigdomenech Badia", "Mehdi Mirza", "Alex Graves", "Timothy Lillicrap", "Tim Harley", "David Silver", "Koray Kavukcuoglu"],
        "year": 2016,
        "venue": "Proceedings of the 33rd International Conference on Machine Learning",
        "url": "https://proceedings.mlr.press/v48/mniha16.html",
        "pdf_url": "https://proceedings.mlr.press/v48/mniha16.pdf",
        "verification": "full text legally open",
        "claim": "Provides the advantage actor-critic lineage used for the A2C comparator.",
    },
    {
        "id": "Kool2019AttentionRouting",
        "type": "conference-paper",
        "title": "Attention, Learn to Solve Routing Problems!",
        "authors": ["Wouter Kool", "Herke van Hoof", "Max Welling"],
        "year": 2019,
        "venue": "International Conference on Learning Representations",
        "url": "https://openreview.net/forum?id=ByxBFsRqYm",
        "pdf_url": "https://openreview.net/pdf?id=ByxBFsRqYm",
        "verification": "full text legally open",
        "claim": "Shows attention-based neural construction policies for routing problems.",
    },
    {
        "id": "Achiam2017CPO",
        "type": "conference-paper",
        "title": "Constrained Policy Optimization",
        "authors": ["Joshua Achiam", "David Held", "Aviv Tamar", "Pieter Abbeel"],
        "year": 2017,
        "venue": "Proceedings of the 34th International Conference on Machine Learning",
        "url": "https://proceedings.mlr.press/v70/achiam17a.html",
        "pdf_url": "https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf",
        "verification": "full text legally open",
        "claim": "Formalises policy optimisation under expected-cost constraints; used only as safe-RL context.",
    },
    {
        "id": "Garcia2015SafeRL",
        "type": "journal-article",
        "title": "A Comprehensive Survey on Safe Reinforcement Learning",
        "authors": ["Javier Garcia", "Fernando Fernandez"],
        "year": 2015,
        "venue": "Journal of Machine Learning Research 16",
        "url": "https://jmlr.org/papers/v16/garcia15a.html",
        "pdf_url": "https://jmlr.org/papers/volume16/garcia15a/garcia15a.pdf",
        "verification": "full text legally open",
        "claim": "Defines safe reinforcement-learning mechanisms and distinguishes constraint handling approaches.",
    },
    {
        "id": "CopernicusDEM2021",
        "type": "dataset-documentation",
        "title": "Copernicus DEM GLO-30 Public Dataset",
        "authors": ["European Space Agency", "Airbus"],
        "year": 2021,
        "venue": "Copernicus Data Space / AWS Registry of Open Data",
        "url": "https://registry.opendata.aws/copernicus-dem/",
        "pdf_url": "",
        "verification": "official dataset registry and access documentation",
        "claim": "Documents the public 30 m Copernicus digital surface model used for geographic simulation transfer.",
    },
    {
        "id": "Demsar2006",
        "type": "journal-article",
        "title": "Statistical Comparisons of Classifiers over Multiple Data Sets",
        "authors": ["Janez Demsar"],
        "year": 2006,
        "venue": "Journal of Machine Learning Research 7, 1-30",
        "url": "https://jmlr.org/papers/v7/demsar06a.html",
        "pdf_url": "https://jmlr.org/papers/volume7/demsar06a/demsar06a.pdf",
        "verification": "full text legally open",
        "claim": "Explains Friedman and paired Wilcoxon procedures for comparisons across independent data sets.",
    },
    {
        "id": "Holm1979",
        "type": "journal-article",
        "title": "A Simple Sequentially Rejective Multiple Test Procedure",
        "authors": ["Sture Holm"],
        "year": 1979,
        "venue": "Scandinavian Journal of Statistics 6, 65-70",
        "url": "https://doi.org/10.2307/4615733",
        "pdf_url": "",
        "verification": "DOI resolved; redistribution not attempted",
        "claim": "Defines the sequentially rejective familywise-error correction used for pairwise families.",
    },
]


CLAIM_MAP = [
    ("C01", "PPO uses a clipped surrogate objective for stable policy updates.", ["Schulman2017PPO"]),
    ("C02", "Pointer mechanisms select variable-length input positions and suit combinatorial sequence construction.", ["Vinyals2015Pointer", "Kool2019AttentionRouting"]),
    ("C03", "Multi-head attention represents interactions among candidate inspection points.", ["Vaswani2017Attention"]),
    ("C04", "A2C is an actor-critic comparator from the asynchronous advantage-learning lineage.", ["Mnih2016A3C"]),
    ("C05", "Safety in reinforcement learning can be imposed through action restriction or constrained optimisation, not only reward penalties.", ["Garcia2015SafeRL", "Achiam2017CPO"]),
    ("C06", "Domain randomisation is a recognised strategy for transfer across simulated condition variations.", ["10.1109/IROS.2017.8202133"]),
    ("C07", "Rotorcraft propulsion energy depends on speed and flight regime; simplified proxies require explicit limitations.", ["10.1109/TWC.2019.2902559"]),
    ("C08", "Map-level paired non-parametric tests and multiplicity correction avoid treating routes as independent replicates.", ["Demsar2006", "10.2307/2279372", "10.2307/3001968", "Holm1979"]),
    ("C09", "Copernicus DEM GLO-30 is a public 30 m digital surface model source.", ["CopernicusDEM2021"]),
    ("C10", "EAAI has published closely related UAV inspection, navigation, wind-robustness, and terrain-risk studies.", [x[0] for x in EXEMPLARS]),
]


def fetch_crossref(doi: str) -> dict:
    url = CROSSREF_API + urllib.parse.quote(doi, safe="")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["message"]


def normalise_crossref(message: dict) -> dict:
    authors = []
    for author in message.get("author", []):
        name = " ".join(x for x in [author.get("given", ""), author.get("family", "")] if x).strip()
        if name:
            authors.append(name)
    date_parts = (
        message.get("published-print", {}).get("date-parts")
        or message.get("published-online", {}).get("date-parts")
        or message.get("issued", {}).get("date-parts")
        or [[None]]
    )[0]
    links = message.get("link", [])
    pdf_links = [x.get("URL", "") for x in links if x.get("content-type") == "application/pdf"]
    licences = [x.get("URL", "") for x in message.get("license", [])]
    return {
        "id": message.get("DOI", "").lower(),
        "type": message.get("type", "journal-article"),
        "title": (message.get("title") or [""])[0],
        "authors": authors,
        "year": date_parts[0] if date_parts else None,
        "published_date_parts": date_parts,
        "venue": (message.get("container-title") or [""])[0],
        "volume": message.get("volume", ""),
        "issue": message.get("issue", ""),
        "article_number_or_pages": message.get("article-number") or message.get("page", ""),
        "doi": message.get("DOI", "").lower(),
        "url": message.get("URL", "") or ("https://doi.org/" + message.get("DOI", "")),
        "publisher": message.get("publisher", ""),
        "licences": licences,
        "publisher_pdf_links": pdf_links,
        "crossref_verified": True,
    }


def fetch_all() -> dict[str, dict]:
    all_dois = list(dict.fromkeys([x[0] for x in EXEMPLARS] + REFERENCE_DOIS))
    records: dict[str, dict] = {}
    failures = []
    for doi in all_dois:
        try:
            records[doi.lower()] = normalise_crossref(fetch_crossref(doi))
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            failures.append({"doi": doi, "error": repr(exc)})
        time.sleep(0.08)
    if failures:
        raise RuntimeError("Crossref verification failed: " + json.dumps(failures, ensure_ascii=False))
    return records


def format_ris(record: dict) -> str:
    kind = "JOUR" if record.get("type") == "journal-article" else "CONF"
    lines = [f"TY  - {kind}"]
    for author in record.get("authors", []):
        lines.append(f"AU  - {author}")
    lines.extend([f"TI  - {record.get('title', '')}", f"PY  - {record.get('year', '')}"])
    if record.get("venue"):
        lines.append(f"JO  - {record['venue']}")
    if record.get("volume"):
        lines.append(f"VL  - {record['volume']}")
    if record.get("issue"):
        lines.append(f"IS  - {record['issue']}")
    if record.get("article_number_or_pages"):
        lines.append(f"SP  - {record['article_number_or_pages']}")
    if record.get("doi"):
        lines.append(f"DO  - {record['doi']}")
    if record.get("url"):
        lines.append(f"UR  - {record['url']}")
    lines.append("ER  -")
    return "\n".join(lines)


def download_oa_manual(records: list[dict]) -> list[dict]:
    target = LITERATURE_ROOT / "open_access_pdfs"
    target.mkdir(parents=True, exist_ok=True)
    log = []
    for record in records:
        url = record.get("pdf_url", "")
        if not url:
            continue
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", record["id"]) + ".pdf"
        path = target / safe_name
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("Downloaded content is not a PDF")
            path.write_bytes(data)
            log.append({"id": record["id"], "url": url, "file": str(path), "bytes": len(data), "status": "downloaded"})
        except Exception as exc:
            log.append({"id": record["id"], "url": url, "file": "", "bytes": 0, "status": "link only", "error": repr(exc)})
    return log


def main() -> None:
    LITERATURE_ROOT.mkdir(parents=True, exist_ok=True)
    crossref = fetch_all()

    exemplars = []
    use_by_doi = dict(EXEMPLARS)
    for doi, _ in EXEMPLARS:
        record = dict(crossref[doi.lower()])
        record.update(
            {
                "scientific_function": use_by_doi[doi],
                "article_type_target": "Original Research / Research article",
                "verification_level": "DOI and Crossref metadata verified; publisher abstract/page to be used within access rights",
                "legal_access": "Publisher or DOI link; PDF only when openly licensed",
            }
        )
        exemplars.append(record)

    references = [dict(crossref[doi.lower()]) for doi in REFERENCE_DOIS]
    for record in references:
        record["verification"] = "DOI and Crossref metadata verified; claim use restricted to inspected title/abstract/full text availability"
    references.extend(MANUAL_REFERENCES)

    download_log = download_oa_manual(MANUAL_REFERENCES)
    payload = {
        "register_created": "2026-08-09",
        "target_journal": "Engineering Applications of Artificial Intelligence",
        "exemplars": exemplars,
        "references": references,
        "claim_citation_map": [
            {"claim_id": cid, "claim": claim, "reference_ids": ids} for cid, claim, ids in CLAIM_MAP
        ],
        "open_access_download_log": download_log,
        "verification_note": "A DOI-resolving metadata check does not by itself prove sentence-level support. The manuscript uses external claims only where the publisher abstract or legally accessible full text was inspected; access-limited records remain link-only.",
    }
    (LITERATURE_ROOT / "literature_register.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (LITERATURE_ROOT / "verified_references.ris").write_text(
        "\n\n".join(format_ris(x) for x in references) + "\n", encoding="utf-8"
    )

    link_lines = [
        "# EAAI exemplars and verified reference links",
        "",
        "The links below are the legal fallback whenever a publisher PDF cannot be redistributed.",
        "",
        "## Twelve EAAI Original Research exemplars",
        "",
    ]
    for idx, item in enumerate(exemplars, 1):
        link_lines.append(f"{idx}. [{item['title']}](https://doi.org/{item['doi']}) — {item['year']}; {item['scientific_function']}.")
    link_lines.extend(["", "## Reference links", ""])
    for item in references:
        identifier = item.get("doi") or item.get("id")
        link_lines.append(f"- [{item['title']}]({item['url']}) — {identifier}.")
    link_lines.extend(["", "## Open-access PDF download log", ""])
    for row in download_log:
        link_lines.append(f"- {row['id']}: {row['status']} — {row['url']}")
    (LITERATURE_ROOT / "README_links.md").write_text("\n".join(link_lines) + "\n", encoding="utf-8")
    print(json.dumps({"exemplars": len(exemplars), "references": len(references), "oa_downloaded": sum(x['status'] == 'downloaded' for x in download_log)}, indent=2))


if __name__ == "__main__":
    main()

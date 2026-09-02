"""从冻结 v3.2.14 证据生成第二轮论文证据架构。

脚本只读取冻结协议、统计表和图件登记，不修改实验资产。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


WORKSPACE = Path(r"C:\Users\xsp\Desktop\DRL代码")
ROOT = WORKSPACE / "paper_delivery" / "EAAI_fulltext_rewrite_v2_2026-08-09"
PRE = WORKSPACE / "paper_runs" / "multimap_v3_2_14" / "analysis" / "pre_plot_statistics"
TRAIN = WORKSPACE / "paper_runs" / "multimap_v3_2_14" / "analysis" / "manuscript_training_aware_v2"
PROTOCOL = WORKSPACE / "paper_runs" / "protocols" / "multimap_generalization_v3_2_14" / "protocol.json"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def one(items: list[dict[str, str]], **filters: str) -> dict[str, str]:
    found = [r for r in items if all(r.get(k) == v for k, v in filters.items())]
    if len(found) != 1:
        raise RuntimeError(f"Expected one row for {filters}, found {len(found)}")
    return found[0]


primary = rows(PRE / "algorithm_primary_summary.csv")
pairwise = rows(PRE / "confirmatory_pairwise.csv")
descriptive = rows(PRE / "descriptive_metrics.csv")
seven = rows(TRAIN / "seven_dimension_scores.csv")
protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))


def primary_fact(family: str, algorithm: str) -> dict[str, str]:
    return one(primary, statistical_family=family, algorithm=algorithm)


def pair(family: str, comparator: str) -> dict[str, str]:
    return one(pairwise, statistical_family=family, reference="full", comparator=comparator)


def metric(result_family: str, algorithm: str, name: str) -> dict[str, str]:
    return one(descriptive, result_family=result_family, algorithm=algorithm, metric=name)


def compact(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


facts: list[dict[str, str]] = []


def add_fact(fid: str, category: str, statement: str, value: str, unit: str,
             source: str, locator: str, strength: str, boundary: str) -> None:
    facts.append({
        "fact_id": fid,
        "category": category,
        "statement": statement,
        "value": value,
        "unit": unit,
        "source": source,
        "locator": locator,
        "strength": strength,
        "boundary": boundary,
    })


add_fact("F01", "frozen identity", "Formal result count", "21648", "routes/results rows",
         "formal_evaluation/results/final_results.jsonl", "exact line count and final_audit_status.json", "frozen", "Must remain unchanged")
add_fact("F02", "protocol", "Confirmatory endpoint", protocol["objective"]["confirmatory_metric"], "ratio",
         "protocol.json", "objective.confirmatory_metric", "frozen", "Post-hoc scores cannot replace it")
add_fact("F03", "training", "Paper-eligible learning models", str(protocol["formal_training"]["paper_eligible_model_count"]), "models",
         "protocol.json", "formal_training.paper_eligible_model_count", "frozen", "Excluded ppo_mlp is not paper evidence")
add_fact("F04", "training", "Paper-eligible training episodes", str(protocol["formal_training"]["paper_eligible_episode_count"]), "episodes",
         "protocol.json", "formal_training.paper_eligible_episode_count", "frozen", "35 models x 3000 episodes")

for fid, family, alg, label in [
    ("F05", "synthetic_main_algorithms", "full", "PPO-Pointer synthetic map mean"),
    ("F06", "synthetic_main_algorithms", "a2c_pointer", "A2C-Pointer synthetic map mean"),
    ("F07", "synthetic_main_algorithms", "traditional_ppo", "Flat-MLP PPO synthetic map mean"),
    ("F08", "synthetic_main_algorithms", "aco", "ACO synthetic map mean"),
    ("F09", "synthetic_main_algorithms", "sa", "SA synthetic map mean"),
    ("F10", "synthetic_main_algorithms", "milp", "MILP synthetic map mean"),
    ("F11", "real_main_algorithms", "full", "PPO-Pointer DSM map mean"),
    ("F12", "real_main_algorithms", "a2c_pointer", "A2C-Pointer DSM map mean"),
    ("F13", "real_main_algorithms", "traditional_ppo", "Flat-MLP PPO DSM map mean"),
    ("F14", "real_main_algorithms", "aco", "ACO DSM map mean"),
    ("F15", "real_main_algorithms", "milp", "MILP DSM map mean"),
]:
    r = primary_fact(family, alg)
    add_fact(fid, "primary outcome", label, compact(r["mean"]), "safe weighted coverage",
             "analysis/pre_plot_statistics/algorithm_primary_summary.csv",
             f"statistical_family={family}; algorithm={alg}; n_maps={r['map_count']}", "confirmatory descriptive",
             "Map is the independent unit")

for fid, family, comp, label in [
    ("F16", "synthetic_main_algorithms", "traditional_ppo", "PPO-Pointer minus Flat-MLP PPO on synthetic maps"),
    ("F17", "synthetic_main_algorithms", "a2c_pointer", "PPO-Pointer minus A2C-Pointer on synthetic maps"),
    ("F18", "real_main_algorithms", "traditional_ppo", "PPO-Pointer minus Flat-MLP PPO on DSM maps"),
    ("F19", "real_main_algorithms", "a2c_pointer", "PPO-Pointer minus A2C-Pointer on DSM maps"),
    ("F20", "synthetic_ablations", "no_return_reserve", "Return-reserve ablation difference on synthetic maps"),
    ("F21", "real_ablations", "no_return_reserve", "Return-reserve ablation difference on DSM maps"),
    ("F22", "hidden_model_perception_mismatch", "no_return_reserve", "Return-reserve ablation difference under hidden mismatch"),
]:
    r = pair(family, comp)
    value = f"mean diff {compact(r['mean_difference'])}; 95% bootstrap CI [{compact(r['bootstrap_ci_low'])}, {compact(r['bootstrap_ci_high'])}]; Holm p={float(r['p_holm']):.3g}"
    add_fact(fid, "paired inference", label, value, "absolute coverage difference",
             "analysis/pre_plot_statistics/confirmatory_pairwise.csv",
             f"family={family}; reference=full; comparator={comp}; n_maps={r['map_count']}", "confirmatory inferential",
             "Non-significance is not equivalence")

for fid, model, dim, label in [
    ("F23", "full", "D6", "PPO-Pointer training stability score"),
    ("F24", "a2c_pointer", "D6", "A2C-Pointer training stability score"),
    ("F25", "full", "D7", "PPO-Pointer sample-efficiency score"),
    ("F26", "a2c_pointer", "D7", "A2C-Pointer sample-efficiency score"),
]:
    r = one(seven, model=model)
    add_fact(fid, "training", label, compact(r[dim]), "normalised score",
             "analysis/manuscript_training_aware_v2/seven_dimension_scores.csv", f"model={model}; dimension={dim}",
             "post-hoc dimension", "Report with definition and do not claim causal mechanism")

for fid, fam, model, label in [
    ("F27", "hidden_model_perception_mismatch", "full", "PPO-Pointer hidden-mismatch mean"),
    ("F28", "hidden_model_perception_mismatch", "a2c_pointer", "A2C-Pointer hidden-mismatch mean"),
    ("F29", "hidden_model_perception_mismatch", "full", "PPO-Pointer hidden-mismatch safe rate"),
]:
    name = "safe_rate" if fid == "F29" else "safe_weighted_coverage"
    r = metric(fam, model, name)
    add_fact(fid, "robustness", label, compact(r["mean"]), name,
             "analysis/pre_plot_statistics/descriptive_metrics.csv", f"result_family={fam}; algorithm={model}; valid_runs={r['valid_run_count']}",
             "descriptive", "Hidden mismatch must remain separate from known shift")

for fid, fam, model, label in [
    ("F30", "synthetic_learning", "full", "PPO-Pointer synthetic planning time"),
    ("F31", "synthetic_main_baselines", "aco", "ACO synthetic planning time"),
    ("F32", "synthetic_main_baselines", "sa", "SA synthetic planning time"),
    ("F33", "synthetic_main_baselines", "milp", "MILP synthetic planning time"),
]:
    r = metric(fam, model, "planning_time_s")
    add_fact(fid, "online computation", label, compact(r["mean"], 3), "s",
             "analysis/pre_plot_statistics/descriptive_metrics.csv", f"result_family={fam}; algorithm={model}; valid_runs={r['valid_run_count']}",
             "descriptive", "Hardware/protocol-specific; MILP status and gap must also be disclosed")


claims = [
    {"claim_id": "K01", "section": "Methods", "claim": "The task selects a sequence of fixed, priority-labelled mountain-road inspection points under safe-return, energy, distance, time, terrain, wind and dynamics constraints.", "support": "F02; protocol objective and task generation", "max_strength": "definition", "forbidden": "continuous road-coverage claim"},
    {"claim_id": "K02", "section": "Methods", "claim": "The principal model combines PPO with a Pointer policy and a return-aware multi-resource feasibility mask.", "support": "protocol claim_boundaries; implementation", "max_strength": "implemented method", "forbidden": "independent contribution of each mask component"},
    {"claim_id": "K03", "section": "Methods", "claim": "Flat-MLP PPO removes Pointer, attention and node encoding while sharing the remaining frozen training and constraint protocol.", "support": "HANDOFF 3.2; implementation class FlatMLPActorCritic", "max_strength": "controlled architecture comparison", "forbidden": "use of excluded ppo_mlp"},
    {"claim_id": "K04", "section": "Results", "claim": "PPO-Pointer and A2C-Pointer achieved nearly identical final safe weighted coverage on unseen synthetic maps and DSM simulations.", "support": "F05; F06; F11; F12; F17; F19", "max_strength": "no detected difference", "forbidden": "equivalence or PPO coverage superiority"},
    {"claim_id": "K05", "section": "Results", "claim": "Both Pointer-based learners substantially exceeded Flat-MLP PPO in final safe weighted coverage.", "support": "F07; F13; F16; F18", "max_strength": "paired significant difference for full vs Flat-MLP", "forbidden": "attribution solely to Pointer without broader architecture caveat"},
    {"claim_id": "K06", "section": "Results", "claim": "ACO, SA and MILP attained higher safe weighted coverage than PPO-Pointer on the synthetic test; ACO and MILP also did so on DSM maps.", "support": "F08-F10; F14-F15; confirmatory_pairwise.csv", "max_strength": "paired observed advantage", "forbidden": "concealment of traditional-planner coverage advantage"},
    {"claim_id": "K07", "section": "Results", "claim": "The higher-coverage traditional planners incurred longer protocol-specific online planning times than PPO-Pointer.", "support": "F30-F33", "max_strength": "descriptive engineering trade-off", "forbidden": "hardware-independent real-time guarantee"},
    {"claim_id": "K08", "section": "Results", "claim": "The frozen post-hoc training dimensions favour PPO-Pointer over A2C-Pointer in stability and sample efficiency.", "support": "F23-F26", "max_strength": "post-hoc score comparison", "forbidden": "causal proof of PPO clipping"},
    {"claim_id": "K09", "section": "Results", "claim": "The return-reserve ablation produced the clearest and statistically supported deterioration across synthetic, DSM and hidden-mismatch families.", "support": "F20-F22", "max_strength": "composite mechanism contribution", "forbidden": "independent claims for energy/distance/time/dynamics submasks"},
    {"claim_id": "K10", "section": "Results", "claim": "Removing priority bias, domain randomisation or resource shaping yielded small and non-significant overall differences in the principal map-level comparisons.", "support": "confirmatory_pairwise.csv synthetic_ablations and real_ablations", "max_strength": "limited aggregate evidence", "forbidden": "equivalence or universal irrelevance"},
    {"claim_id": "K11", "section": "Results", "claim": "Under hidden model/perception mismatch, PPO-Pointer was directionally above A2C-Pointer, but the paired difference was not significant.", "support": "F27-F29; hidden pairwise row", "max_strength": "directional observation", "forbidden": "robustness superiority"},
    {"claim_id": "K12", "section": "Discussion", "claim": "PPO-Pointer offers an engineering balance rather than a universal single-metric optimum.", "support": "K04-K11; figures M01-M10", "max_strength": "evidence-bounded interpretation", "forbidden": "best overall algorithm without explicit criterion"},
    {"claim_id": "K13", "section": "Discussion", "claim": "DSM experiments represent zero-shot geographic simulation transfer across eight independent maps.", "support": "protocol claim_boundaries; F11-F15", "max_strength": "simulation transfer", "forbidden": "real flight, deployment or safety certification"},
    {"claim_id": "K14", "section": "Discussion", "claim": "Performance at 16, 20 and 24 nodes is within the trained size range.", "support": "protocol node_counts; HANDOFF 4.1", "max_strength": "within-range multi-scale performance", "forbidden": "out-of-range scale generalisation"},
    {"claim_id": "K15", "section": "Discussion", "claim": "The map is the statistical unit; routes, tasks and seeds are nested observations.", "support": "analysis protocol; F16-F22 locators", "max_strength": "statistical design", "forbidden": "inflated task-level n"},
    {"claim_id": "K16", "section": "Supplement", "claim": "The 100-point multiobjective score is a post-hoc sensitivity summary, not the confirmatory endpoint.", "support": "F02; manuscript_preplot_closure_v5", "max_strength": "diagnostic summary", "forbidden": "abstract champion conclusion"},
]


figures = [
    {"unit": "U1", "assets": "non-generative method/evaluation schematic; M01/M02 inputs", "purpose": "Task, method and frozen evaluation design", "claim_ids": "K01-K03; K15", "placement": "Methods", "boundary": "Diagram only verifiable implementation and protocol facts"},
    {"unit": "U2", "assets": "M01 + M03", "purpose": "Coverage distribution and priority strata", "claim_ids": "K04-K06", "placement": "Results 6.1", "boundary": "Show traditional-planner advantages"},
    {"unit": "U3", "assets": "M02 + M04", "purpose": "Safety/return and safe-route resource costs", "claim_ids": "K05; K09", "placement": "Results 6.2", "boundary": "Resource metrics conditional on safe routes"},
    {"unit": "U4", "assets": "M05 + S02", "purpose": "Planning-time ECDF and quality-time trade-off", "claim_ids": "K06-K07; K12", "placement": "Results 6.3", "boundary": "Protocol/hardware-specific timing"},
    {"unit": "U5", "assets": "M06 + M07", "purpose": "Training trajectories, stability and sample efficiency", "claim_ids": "K08", "placement": "Results 6.4", "boundary": "Training curves are not test curves"},
    {"unit": "U6", "assets": "M08 + V01/V02 as non-inferential", "purpose": "Unseen maps and DSM simulation transfer", "claim_ids": "K04-K06; K13-K14", "placement": "Results 6.5", "boundary": "No flight or unseen-size claim"},
    {"unit": "U7", "assets": "M09", "purpose": "Known shifts versus hidden mismatch", "claim_ids": "K09-K11", "placement": "Results 6.6", "boundary": "Never merge perturbation semantics"},
    {"unit": "U8", "assets": "M10", "purpose": "Four ablations", "claim_ids": "K09-K10", "placement": "Results 6.7", "boundary": "Composite return mechanism only"},
]


glossary = [
    ["PPO-Pointer", "PPO with Pointer policy", "Use for full; do not call coverage champion"],
    ["A2C-Pointer", "A2C comparator with Pointer policy", "Final coverage is close to PPO-Pointer"],
    ["Flat-MLP PPO", "traditional_ppo / FlatMLPActorCritic / flat_mlp_24", "Never use ppo_mlp"],
    ["SWC", "safe weighted coverage", "Confirmatory endpoint; unsafe/violating routes score zero"],
    ["DSM", "digital surface model", "Copernicus DEM GLO-30; simulation terrain input"],
    ["unseen synthetic maps", "24 held-out procedural maps", "Not unseen node-count generalisation"],
    ["zero-shot DSM simulation transfer", "Evaluation on 8 DSM maps without DSM-specific training", "Not real flight"],
    ["known domain shift", "policy observes the shifted condition used for execution", "Wind or power coefficient shift"],
    ["hidden mismatch", "planning observation/model differs from execution truth", "Keep separate from known shift"],
    ["return-aware multi-resource feasibility mask", "composite return reserve mechanism", "No independent submask claim"],
    ["map-level pairing", "task/seed aggregation within each map", "n=24 synthetic or n=8 DSM"],
    ["post-hoc 100-point score", "multiobjective sensitivity summary", "Supplement only; not abstract evidence"],
]


paragraphs = [
    ["3.1", "P01-P03", "Define road corridors, fixed points, priorities, depot and resource-constrained objective", "K01", "protocol; method schematic"],
    ["3.2", "P04-P06", "State, action, transition and confirmatory SWC objective", "K01; K15", "F02; protocol"],
    ["4.1", "P07-P10", "Pointer encoder/decoder and PPO optimisation", "K02", "implementation; R09-R13"],
    ["4.2", "P11-P13", "Return-aware feasibility evaluation and mask", "K02; K09", "protocol; implementation"],
    ["4.3", "P14-P16", "Flat-MLP and A2C comparators plus four ablations", "K03", "implementation; protocol"],
    ["5.1", "P17-P19", "Training/validation/test maps and task factorial design", "K13-K15", "protocol"],
    ["5.2", "P20-P22", "Traditional baselines and MILP status/gap reporting", "K06-K07", "protocol; R25-R30"],
    ["5.3", "P23-P25", "Known shifts and hidden mismatch", "K11", "protocol"],
    ["5.4", "P26-P28", "Map-level statistics, bootstrap and multiplicity", "K15", "F16-F22; R20-R23"],
    ["6.1", "P29-P32", "Synthetic and DSM SWC with honest algorithm ranking", "K04-K06", "F05-F19; U2"],
    ["6.2", "P33-P35", "Safety, return and resource costs", "K05; K09", "U3"],
    ["6.3", "P36-P38", "Planning time and quality-time trade-offs", "K06-K07", "F30-F33; U4"],
    ["6.4", "P39-P41", "Training curves, stability and sample efficiency", "K08", "F23-F26; U5"],
    ["6.5", "P42-P44", "Unseen-map and DSM simulation transfer", "K13-K14", "U6"],
    ["6.6", "P45-P47", "Known shifts and hidden mismatch", "K11", "F27-F29; U7"],
    ["6.7", "P48-P50", "Ablations, led by return reserve", "K09-K10", "F20-F22; U8"],
    ["7", "P51-P57", "Interpret engineering balance, compare planners, state uncertainty and limitations", "K12-K16", "all principal evidence"],
    ["1-2", "P58-P66", "Problem importance, mechanism-organised literature, gap and contributions", "C01-C19; K01-K03", "verified full-text references"],
    ["8/Abstract", "P67-P70", "Bounded conclusions and <=250-word abstract", "K04-K15", "principal evidence only"],
]


author_queue = [
    ["AQ01", "Author names and order", "Title page; CRediT", "[AUTHOR INPUT REQUIRED: names and order]", "open"],
    ["AQ02", "Affiliations and corresponding author", "Title page", "[AUTHOR INPUT REQUIRED: affiliations, email, postal address]", "open"],
    ["AQ03", "Funding information", "Funding statement", "[AUTHOR INPUT REQUIRED: funder and grant number, or none]", "open"],
    ["AQ04", "CRediT contributions", "Author contributions", "[AUTHOR INPUT REQUIRED: role assignment]", "open"],
    ["AQ05", "Competing interests", "Declaration", "[AUTHOR INPUT REQUIRED: conflicts or no conflict]", "open"],
    ["AQ06", "Acknowledgements", "End matter", "[AUTHOR INPUT REQUIRED: acknowledgements or none]", "open"],
    ["AQ07", "Repository permanent identifier", "Data/code availability", "[AUTHOR INPUT REQUIRED: DOI or permanent URL]", "open"],
    ["AQ08", "Copernicus region identifiers/asset reconstruction details", "Data availability; reproducibility", "[AUTHOR INPUT REQUIRED: confirm public region identifiers]", "open"],
    ["AQ09", "AI-use disclosure wording and human verification", "Declaration", "[AUTHOR INPUT REQUIRED: approve accurate disclosure before submission]", "open"],
]


architecture = {
    "schema_version": 2,
    "prepared": "2026-08-09",
    "source_version": "v3.2.14",
    "facts": facts,
    "claims": claims,
    "figure_storyline": figures,
    "glossary": [{"term": r[0], "definition": r[1], "control": r[2]} for r in glossary],
    "paragraph_plan": [{"section": r[0], "paragraphs": r[1], "function": r[2], "claims": r[3], "evidence": r[4]} for r in paragraphs],
    "author_verification_queue": [{"id": r[0], "item": r[1], "location": r[2], "placeholder": r[3], "status": r[4]} for r in author_queue],
}

# 冻结事实断言：若源表漂移，立即停止，不生成误导性架构。
assert primary_fact("synthetic_main_algorithms", "full")["map_count"] == "24"
assert primary_fact("real_main_algorithms", "full")["map_count"] == "8"
assert protocol["formal_training"]["archived_excluded_variants"] == ["ppo_mlp"]
assert protocol["claim_boundaries"]["real_claim"] == "zero_shot_geographic_DSM_simulation_transfer"

(ROOT / "evidence" / "evidence_architecture_source_v2.json").write_text(
    json.dumps(architecture, ensure_ascii=False, indent=2), encoding="utf-8"
)


def md_table(headers: list[str], body: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in body:
        out.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(out)


(ROOT / "evidence" / "Fact_Map_v2.md").write_text(
    "# Fact Map v2\n\nAll numeric facts below are read from frozen v3.2.14 evidence.\n\n" +
    md_table(["ID", "Category", "Statement", "Value", "Source/locator", "Boundary"], [
        [f["fact_id"], f["category"], f["statement"], f"{f['value']} {f['unit']}", f"{f['source']} — {f['locator']}", f["boundary"]] for f in facts
    ]) + "\n", encoding="utf-8"
)
(ROOT / "evidence" / "Claim_Map_v2.md").write_text(
    "# Claim Map v2\n\n" + md_table(["ID", "Section", "Permitted claim", "Support", "Maximum strength", "Forbidden inference"], [
        [c["claim_id"], c["section"], c["claim"], c["support"], c["max_strength"], c["forbidden"]] for c in claims
    ]) + "\n", encoding="utf-8"
)
(ROOT / "evidence" / "Figure_Storyline_v2.md").write_text(
    "# Figure Storyline v2\n\n" + md_table(["Unit", "Assets", "Purpose", "Claims", "Placement", "Boundary"], [
        [f["unit"], f["assets"], f["purpose"], f["claim_ids"], f["placement"], f["boundary"]] for f in figures
    ]) + "\n", encoding="utf-8"
)
(ROOT / "evidence" / "Terminology_and_Abbreviations_v2.md").write_text(
    "# Terminology and abbreviations v2\n\n" + md_table(["Term", "Definition", "Usage control"], glossary) + "\n", encoding="utf-8"
)
(ROOT / "evidence" / "Paragraph_Plan_v2.md").write_text(
    "# Paragraph Plan v2\n\n" + md_table(["Section", "Paragraphs", "Function", "Claims", "Evidence"], paragraphs) + "\n", encoding="utf-8"
)
(ROOT / "evidence" / "Author_Verification_Queue_v2.md").write_text(
    "# Author Verification Queue v2\n\nUnknown author-side metadata is never guessed.\n\n" +
    md_table(["ID", "Item", "Location", "Required placeholder", "Status"], author_queue) + "\n", encoding="utf-8"
)

print(json.dumps({"facts": len(facts), "claims": len(claims), "figure_units": len(figures), "paragraph_groups": len(paragraphs), "author_items": len(author_queue)}, ensure_ascii=False))

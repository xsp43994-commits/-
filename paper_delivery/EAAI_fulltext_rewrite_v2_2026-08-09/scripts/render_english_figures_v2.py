from __future__ import annotations

"""将冻结第三轮 Source Data 重新排版为英文投稿图；仅改变文字与布局。"""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts" / "prepare_submission_figures_v2.py"
spec = importlib.util.spec_from_file_location("figure_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

SRC = ROOT / "figures" / "source_data"
MAIN = ROOT / "figures" / "submission" / "main"
SUPP = ROOT / "figures" / "submission" / "supplementary"
SHOW = ROOT / "figures" / "submission" / "showcase"

MODEL_ORDER = ["full", "a2c_pointer", "traditional_ppo", "priority_resource_greedy", "aco", "milp"]
EXTRA_LABELS = {
    "no_priority_bias": "No priority bias",
    "no_domain_randomization": "No domain randomization",
    "no_resource_shaping": "No resource shaping",
    "no_return_reserve": "No return reserve",
    "a_star": "A*",
    "exact_pareto_dp": "Exact Pareto DP",
    "ga": "GA", "pso": "PSO", "sa": "SA",
    "nearest_feasible": "Nearest feasible",
}
DOMAIN = {"synthetic": "Unseen synthetic", "real": "DSM transfer", "未见合成": "Unseen synthetic", "真实DSM": "DSM transfer"}


def label(model: str) -> str:
    return base.LABELS.get(model, EXTRA_LABELS.get(model, model.replace("_", " ").title()))


def color(model: str) -> str:
    return base.COLORS.get(model, {"no_priority_bias": "#7F8DAA", "no_domain_randomization": "#8064A2",
                                   "no_resource_shaping": "#B07C48", "no_return_reserve": "#A94F5C"}.get(model, "#5F6B73"))


def sized(height: float = 3.8):
    return plt.subplots(figsize=(base.FULL_WIDTH_MM / 25.4, height))


def panel(ax, letter: str):
    ax.text(-0.075, 1.045, letter, transform=ax.transAxes, fontsize=10, weight="bold", va="bottom")


def m01():
    d = pd.read_csv(SRC / "M01_source_data.csv")
    fig, axes = plt.subplots(2, 1, figsize=(base.FULL_WIDTH_MM / 25.4, 5.15), sharex=True)
    for ax, dom, letter in zip(axes, ["synthetic", "real"], ["a", "b"]):
        dd = d[d.domain == dom]
        models = [m for m in MODEL_ORDER if m in set(dd.model)]
        for i, m in enumerate(models):
            values = dd[dd.model == m].weighted_coverage.to_numpy()
            x = np.linspace(-0.14, 0.14, len(values)) + i
            ax.scatter(values, x, s=18, facecolor="white", edgecolor=color(m), lw=0.8, zorder=3)
            med = np.median(values)
            ax.plot([med, med], [i - 0.22, i + 0.22], color=color(m), lw=2.2)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([label(m) for m in models])
        ax.set_title(f"{DOMAIN[dom]} maps (n = {dd.map_id.nunique()} maps)", loc="left")
        ax.grid(axis="x", color="#DDE2E7", lw=0.55)
        panel(ax, letter)
    axes[-1].set_xlabel("Map-level priority-weighted coverage")
    fig.tight_layout(h_pad=1.15)
    return base.save_all(fig, MAIN / "M01_priority_weighted_coverage_english")


def m02():
    d = pd.read_csv(SRC / "M02_source_data.csv")
    fig, axes = plt.subplots(1, 2, figsize=(base.FULL_WIDTH_MM / 25.4, 4.0), sharey=False)
    for ax, layer, title, letter in zip(axes, ["known_domain_shift", "hidden_model_perception_mismatch"],
                                        ["Known domain shift", "Hidden model/perception mismatch"], ["a", "b"]):
        dd = d[d.layer == layer].copy()
        dd["row"] = dd.metric.map({"safe": "Safe route", "returned": "Return to depot"}) + " | " + dd.comparator.map(label)
        y = np.arange(len(dd))[::-1]
        for yi, (_, r) in zip(y, dd.iterrows()):
            ax.errorbar(r.estimate_pp, yi, xerr=[[r.estimate_pp-r.ci_low_pp], [r.ci_high_pp-r.estimate_pp]],
                        fmt="o", ms=5, color=color(r.comparator), capsize=2.5, lw=1.15)
        ax.axvline(0, color="#444444", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(dd.row)
        ax.set_title(title); ax.set_xlabel("Difference (percentage points)")
        ax.grid(axis="x", color="#E1E5E9", lw=0.5); panel(ax, letter)
    fig.tight_layout(w_pad=1.4)
    return base.save_all(fig, MAIN / "M02_safety_and_return_effects_english")


def m03():
    d = pd.read_csv(SRC / "M03_source_data.csv")
    strata = {"高优先级": "High priority", "中优先级": "Medium priority", "低优先级": "Low priority", "远端高优先级冲突": "Remote high-priority conflict"}
    fig, axes = plt.subplots(1, 2, figsize=(base.FULL_WIDTH_MM / 25.4, 3.55), sharey=True)
    for ax, dom, letter in zip(axes, ["未见合成", "真实DSM"], ["a", "b"]):
        dd = d[d.domain == dom]
        y = np.arange(len(dd))
        ax.barh(y, dd.effect, color="#1764AB", alpha=0.82)
        ax.axvline(0, color="#333333", lw=0.75)
        ax.set_yticks(y); ax.set_yticklabels([strata.get(v, v) for v in dd.stratum])
        ax.set_title(DOMAIN[dom]); ax.set_xlabel("Full model minus ablation")
        ax.grid(axis="x", color="#E1E5E9", lw=0.5); panel(ax, letter)
    fig.tight_layout(w_pad=1.4)
    return base.save_all(fig, MAIN / "M03_priority_stratum_effects_english")


def m04():
    d = pd.read_csv(SRC / "M04_source_data.csv")
    metric_map = {"能耗": "Energy", "航程": "Range", "总任务时间": "Mission time"}
    fig, axes = plt.subplots(1, 3, figsize=(base.FULL_WIDTH_MM / 25.4, 3.55), sharey=True)
    for ax, metric, letter in zip(axes, ["能耗", "航程", "总任务时间"], ["a", "b", "c"]):
        dd = d[d.metric == metric].set_index("model").reindex([m for m in MODEL_ORDER if m in set(d.model)]).reset_index()
        y = np.arange(len(dd))
        ax.barh(y, dd.utilization, color=[color(m) for m in dd.model])
        ax.axvline(1, color="#333333", lw=0.75, ls="--")
        ax.set_yticks(y); ax.set_yticklabels([label(m) for m in dd.model])
        ax.set_title(metric_map[metric]); ax.set_xlabel("Median budget utilization")
        ax.grid(axis="x", color="#E1E5E9", lw=0.5); panel(ax, letter)
    fig.tight_layout(w_pad=1.0)
    return base.save_all(fig, MAIN / "M04_resource_use_english")


def m06():
    d = pd.read_csv(SRC / "M06_source_data.csv")
    fig, axes = plt.subplots(1, 3, figsize=(base.FULL_WIDTH_MM / 25.4, 3.65), sharex=True, sharey=True)
    for ax, m, letter in zip(axes, ["full", "a2c_pointer", "traditional_ppo"], ["a", "b", "c"]):
        dd = d[(d.model == m) & (d.record_type == "seed")]
        for _, seed in dd.groupby("seed"):
            ax.plot(seed.episodes_seen, seed.safe_weighted_coverage, color=color(m), alpha=.24, lw=.65)
        summary = d[(d.model == m) & (d.record_type == "summary")].sort_values("episodes_seen")
        if len(summary):
            ax.fill_between(summary.episodes_seen, summary.q25, summary.q75, color=color(m), alpha=.18)
            ax.plot(summary.episodes_seen, summary["median"], color=color(m), lw=1.55)
        ax.set_title(label(m)); ax.set_xlabel("Training episodes")
        ax.grid(color="#E2E6EA", lw=.45); panel(ax, letter)
    axes[0].set_ylabel("Training-batch priority-weighted coverage")
    fig.tight_layout(w_pad=.8)
    return base.save_all(fig, MAIN / "M06_training_curves_english")


def m07():
    d = pd.read_csv(SRC / "M07_source_data.csv")
    dd = d[d.metric.isin(["D6训练稳定性", "D7样本效率"])].copy()
    dd.metric = dd.metric.map({"D6训练稳定性": "D6 training stability", "D7样本效率": "D7 sample efficiency"})
    fig, ax = sized(3.6)
    x = np.arange(2); width=.24
    for j, m in enumerate(["full", "a2c_pointer", "traditional_ppo"]):
        vals = dd[dd.model == m].set_index("metric").reindex(["D6 training stability", "D7 sample efficiency"]).score
        ax.bar(x+(j-1)*width, vals, width, label=label(m), color=color(m))
    ax.set_xticks(x); ax.set_xticklabels(["D6 training stability", "D7 sample efficiency"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Normalized score"); ax.grid(axis="y", color="#E0E5E9", lw=.5)
    ax.legend(ncol=3, loc="upper center"); panel(ax, "a"); fig.tight_layout()
    return base.save_all(fig, MAIN / "M07_training_stability_efficiency_english")


def m08():
    d = pd.read_csv(SRC / "M08_source_data.csv")
    fig, axes = plt.subplots(1, 2, figsize=(base.FULL_WIDTH_MM / 25.4, 3.5), sharex=True)
    for ax, dom, letter in zip(axes, ["synthetic", "real"], ["a", "b"]):
        dd = d[d.domain == dom]
        y = np.arange(len(dd))[::-1]
        for yi, (_, r) in zip(y, dd.iterrows()):
            ax.errorbar(r.estimate, yi, xerr=[[r.estimate-r.ci_low], [r.ci_high-r.estimate]], fmt="o", color=color(r.model), ms=6, capsize=2.5)
        ax.set_yticks(y); ax.set_yticklabels([label(m) for m in dd.model])
        ax.set_title(f"{DOMAIN[dom]} (n = {int(dd.n_maps.iloc[0])} maps)"); ax.set_xlabel("Map-level D1")
        ax.grid(axis="x", color="#E0E5E9", lw=.5); panel(ax, letter)
    fig.tight_layout(w_pad=1.2)
    return base.save_all(fig, MAIN / "M08_unseen_maps_and_DSM_transfer_english")


def m09():
    d = pd.read_csv(SRC / "M09_source_data.csv")
    fig, axes = plt.subplots(1, 2, figsize=(base.FULL_WIDTH_MM / 25.4, 4.25), sharex=True)
    for ax, fam, title, letter in zip(axes, ["known_domain_shift", "hidden_model_perception_mismatch"], ["Known domain shift", "Hidden mismatch"], ["a", "b"]):
        dd = d[d.family == fam]
        models = [m for m in ["full", "a2c_pointer", "traditional_ppo", "no_domain_randomization", "no_return_reserve"] if m in set(dd.model)]
        conditions = sorted(dd.condition.unique())
        y = np.arange(len(conditions)); width=.14
        for j, m in enumerate(models):
            vals = dd[dd.model == m].set_index("condition").reindex(conditions).retention
            ax.scatter(vals, y+(j-(len(models)-1)/2)*width, s=26, color=color(m), label=label(m))
        ax.axvline(1, color="#555555", lw=.75, ls="--")
        ax.set_yticks(y); ax.set_yticklabels([c.replace("_", " ").title() for c in conditions])
        ax.set_title(title); ax.set_xlabel("D1 retention relative to nominal")
        ax.grid(axis="x", color="#E0E5E9", lw=.5); panel(ax, letter)
    axes[1].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(w_pad=1.0)
    return base.save_all(fig, MAIN / "M09_two_layer_robustness_english")


def m10():
    d = pd.read_csv(SRC / "M10_source_data.csv")
    fig, axes = plt.subplots(1, 2, figsize=(base.FULL_WIDTH_MM / 25.4, 3.8), sharex=True)
    order = ["no_priority_bias", "no_domain_randomization", "no_resource_shaping", "no_return_reserve"]
    for ax, dom, letter in zip(axes, ["未见合成", "真实DSM"], ["a", "b"]):
        dd = d[d.domain == dom].set_index("comparator").reindex(order).reset_index()
        y=np.arange(len(dd))[::-1]
        for yi,(_,r) in zip(y,dd.iterrows()):
            ax.errorbar(r.mean_difference, yi, xerr=[[r.mean_difference-r.bootstrap_ci_low],[r.bootstrap_ci_high-r.mean_difference]],
                        fmt="o", color=color(r.comparator), ms=6, capsize=2.5)
        ax.axvline(0,color="#444",lw=.75); ax.set_yticks(y); ax.set_yticklabels([label(m) for m in dd.comparator])
        ax.set_title(DOMAIN[dom]); ax.set_xlabel("Full model minus ablation in D1")
        ax.grid(axis="x",color="#E0E5E9",lw=.5); panel(ax,letter)
    fig.tight_layout(w_pad=1.0)
    return base.save_all(fig, MAIN / "M10_ablation_effects_english")


def s01():
    d=pd.read_csv(SRC/"S01_source_data.csv"); fig,ax=sized(3.6)
    for m in sorted(d.model.unique()):
        dd=d[d.model==m].sort_values("regret"); ax.step(dd.regret,dd.ecdf,where="post",lw=1.15,color=color(m),label=label(m))
    ax.set_xlabel("Oracle regret"); ax.set_ylabel("Empirical cumulative probability"); ax.grid(color="#E1E5E9",lw=.5)
    ax.legend(ncol=3,fontsize=6.8); panel(ax,"a"); fig.tight_layout(); return base.save_all(fig,SUPP/"S01_performance_profile_english")


def s02():
    d=pd.read_csv(SRC/"S02_source_data.csv"); fig,axes=plt.subplots(1,2,figsize=(base.FULL_WIDTH_MM/25.4,3.55),sharex=True,sharey=True)
    for ax,scope,title,letter in zip(axes,["synthetic_all","real_all"],["Unseen synthetic","DSM transfer"],["a","b"]):
        dd=d[d.scope==scope]
        for _,r in dd.iterrows():
            ax.scatter(r.planning_time_p95_s,r.D1,s=42,color=color(r.model),edgecolor="white",lw=.5,label=label(r.model))
        ax.set_xscale("log");ax.set_title(title);ax.set_xlabel("95th-percentile planning time (s; log scale)");ax.grid(color="#E1E5E9",lw=.5);panel(ax,letter)
    axes[0].set_ylabel("D1");axes[1].legend(fontsize=6.1,ncol=2,loc="lower right");fig.tight_layout(w_pad=.9)
    return base.save_all(fig,SUPP/"S02_quality_time_tradeoff_english")


def s03():
    d=pd.read_csv(SRC/"S03_source_data.csv"); fig,ax=sized(3.8)
    offsets={"priority_resource_greedy":(4,7),"pso":(4,-9),"nearest_feasible":(4,-2),"ga":(4,4),"sa":(4,3),"aco":(4,3),"milp":(4,4),"exact_pareto_dp":(4,4),"a_star":(4,4)}
    for _,r in d.iterrows():
        mid=(r.regret_low+r.regret_high)/2
        ax.errorbar(r.planning_time_s,mid,yerr=[[mid-r.regret_low],[r.regret_high-mid]],fmt="o",ms=5,color=color(r.model),capsize=2)
        ax.annotate(label(r.model),(r.planning_time_s,mid),xytext=offsets.get(r.model,(4,3)),textcoords="offset points",fontsize=5.8)
    ax.set_xscale("log"); ax.set_xlabel("Planning time (s; log scale)"); ax.set_ylabel("Oracle-regret interval")
    ax.grid(color="#E1E5E9",lw=.5); panel(ax,"a"); fig.tight_layout(); return base.save_all(fig,SUPP/"S03_oracle_regret_cost_english")


def s04():
    d=pd.read_csv(SRC/"S04_source_data.csv"); piv=d.pivot_table(index="scenario",columns="model",values="mean")
    cols=[m for m in ["full","a2c_pointer","traditional_ppo"] if m in piv.columns]; piv=piv[cols]
    fig,ax=plt.subplots(figsize=(base.FULL_WIDTH_MM/25.4,5.4)); im=ax.imshow(piv.values,aspect="auto",cmap="viridis")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([label(m) for m in cols]); ax.set_yticks(range(len(piv))); ax.set_yticklabels([s.replace("｜"," | ") for s in piv.index],fontsize=6.2)
    fig.colorbar(im,ax=ax,label="Mean D1"); panel(ax,"a"); fig.tight_layout(); return base.save_all(fig,SUPP/"S04_scenario_heatmap_english")


def s05():
    d=pd.read_csv(SRC/"S05_source_data.csv"); dd=d[d.metric.isin(["safe_rate","return_rate","stranded_rate"])].copy()
    dd["row"]=dd.family.str.replace("_"," ")+" | "+dd.condition.str.replace("_"," ")+" | "+dd.metric.str.replace("_"," ")
    piv=dd.pivot_table(index="row",columns="model",values="raw_value"); cols=[m for m in ["full","a2c_pointer","traditional_ppo","no_domain_randomization","no_return_reserve"] if m in piv.columns];piv=piv[cols]
    fig,ax=plt.subplots(figsize=(base.FULL_WIDTH_MM/25.4,6.2)); im=ax.imshow(piv.values,aspect="auto",vmin=0,vmax=1,cmap="RdYlGn")
    ax.set_xticks(range(len(cols)));ax.set_xticklabels([label(m) for m in cols],rotation=25,ha="right");ax.set_yticks(range(len(piv)));ax.set_yticklabels(piv.index,fontsize=5.7)
    fig.colorbar(im,ax=ax,label="Rate");panel(ax,"a");fig.tight_layout();return base.save_all(fig,SUPP/"S05_failure_modes_english")


def s06():
    d=pd.read_csv(SRC/"S06_source_data.csv");fig,ax=sized(4.2)
    for m in sorted(d.model.unique()):
        dd=d[d.model==m].sort_values("episodes_seen");ax.plot(dd.episodes_seen,dd.weighted_coverage,lw=1.0,color=color(m),label=label(m))
        if "q25" in dd and dd.q25.notna().any():ax.fill_between(dd.episodes_seen,dd.q25,dd.q75,color=color(m),alpha=.08)
    ax.set_xlabel("Training episodes");ax.set_ylabel("Median training-batch weighted coverage");ax.grid(color="#E1E5E9",lw=.5);ax.legend(ncol=2,fontsize=6.5);panel(ax,"a");fig.tight_layout();return base.save_all(fig,SUPP/"S06_seven_model_training_english")


def s07():
    d=pd.read_csv(SRC/"S07_source_data.csv");metrics=[m for m in ["D1","D2","D3","D4","D5","D6","D7","综合得分"] if m in set(d.metric)]
    fig,ax=sized(3.9);x=np.arange(len(metrics));w=.24
    for j,m in enumerate(["full","a2c_pointer","traditional_ppo"]):
        vals=d[d.model==m].set_index("metric").reindex(metrics).value_100;ax.bar(x+(j-1)*w,vals,w,color=color(m),label=label(m))
    ax.set_xticks(x);ax.set_xticklabels(["Composite" if m=="综合得分" else m for m in metrics]);ax.set_ylabel("Post-hoc 0–100 score");ax.grid(axis="y",color="#E1E5E9",lw=.5);ax.legend(ncol=3);panel(ax,"a");fig.tight_layout();return base.save_all(fig,SUPP/"S07_posthoc_score_english")


def s08():
    d=pd.read_csv(SRC/"S08_source_data.csv");p=d.pivot(index="operational_floor",columns="training_weight",values="first_share")
    fig,ax=sized(3.9);im=ax.imshow(p.values,aspect="auto",origin="lower",vmin=0,vmax=1,cmap="Blues")
    ax.set_xticks(range(len(p.columns)));ax.set_xticklabels(p.columns);ax.set_yticks(range(len(p.index)));ax.set_yticklabels(p.index)
    ax.set_xlabel("Training weight");ax.set_ylabel("Operational floor");fig.colorbar(im,ax=ax,label="PPO–Pointer first-place share");panel(ax,"a");fig.tight_layout();return base.save_all(fig,SUPP/"S08_weight_sensitivity_english")


def v01():
    d=pd.read_csv(SRC/"V01_source_data.csv");fig,ax=sized(4.65)
    for _,road in d[d.record_type=="road"].groupby("group"):
        ax.plot(road.x,road.y,color="#777",lw=.6,alpha=.65)
    for m in ["milp","traditional_ppo","a2c_pointer","full"]:
        route=d[(d.record_type=="route")&(d.model==m)].sort_values("sequence");ax.plot(route.x,route.y,lw=1.4,color=color(m),label=label(m))
    pts=d[d.record_type=="inspection"];ax.scatter(pts.x,pts.y,s=22,c=pts.priority,cmap="YlOrRd",edgecolor="white",lw=.4,zorder=5)
    dep=d[d.record_type=="airport"];ax.scatter(dep.x,dep.y,s=85,marker="*",facecolor="white",edgecolor="#111",zorder=6,label="Depot")
    ax.set_aspect("equal",adjustable="box");ax.set_xlabel("x coordinate");ax.set_ylabel("y coordinate");ax.legend(ncol=3,fontsize=6.6);panel(ax,"a");fig.tight_layout();return base.save_all(fig,SHOW/"V01_fixed_synthetic_route_english")


def main():
    base.setup_style()
    outputs=[]
    for fn in [m01,m02,m03,m04,m06,m07,m08,m09,m10,s01,s02,s03,s04,s05,s06,s07,s08,v01]:
        outputs.extend(fn())
    report={"figures":18,"files":len(outputs),"rule":"Frozen Source Data only; English relayout without new statistics"}
    (ROOT/"figures"/"submission"/"qa"/"english_relayout_v2.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()

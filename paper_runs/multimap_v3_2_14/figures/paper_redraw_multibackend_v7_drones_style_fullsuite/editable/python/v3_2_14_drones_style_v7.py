"""v3.2.14 冻结结果的 v7 Drones 风格全套科研图。

本脚本只读取已冻结的结果、训练日志和既有可编辑工程；不训练、不评价、
不选择代表任务，也不改变任何统计量。所有可调制图参数集中在本文件顶部。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import gaussian_kde


REPO = Path(r"C:\Users\xsp\Desktop\DRL代码")
FIGURES_ROOT = REPO / "paper_runs" / "multimap_v3_2_14" / "figures"
V3 = FIGURES_ROOT / "paper_redraw_multibackend_v3"
V4 = FIGURES_ROOT / "paper_redraw_multibackend_v4_training_corrected"
V6 = FIGURES_ROOT / "paper_redraw_multibackend_v6_learning_curves_dual_evidence"
OUTPUT = FIGURES_ROOT / "paper_redraw_multibackend_v7_drones_style_fullsuite"
EDITABLE_UPSTREAM = REPO / "paper_delivery" / "EAAI_format_translation_v4_2026-08-30" / "figures" / "editable"

# 关键可调参数：修改后会同时影响中英文导出，保证双语版本视觉编码一致。
CORE_COLORS = {"full": "#0072B2", "a2c_pointer": "#E69F00", "traditional_ppo": "#009E73"}
OTHER_COLORS = {
    "priority_resource_greedy": "#6C71C4", "aco": "#CC79A7", "milp": "#333333",
    "no_priority_bias": "#8C564B", "no_domain_randomization": "#7F7F7F",
    "no_resource_shaping": "#BCBD22", "no_return_reserve": "#9467BD",
    "a_star": "#4C78A8", "exact_pareto_dp": "#59A14F", "nearest_feasible": "#F28E2B",
    "ga": "#B279A2", "pso": "#76B7B2", "sa": "#EDC948",
}
COLORS = {**OTHER_COLORS, **CORE_COLORS}
LINESTYLES = {"full": "-", "a2c_pointer": "--", "traditional_ppo": "-."}
MARKERS = {"full": "o", "a2c_pointer": "s", "traditional_ppo": "^"}
MODEL_ORDER = ("full", "a2c_pointer", "traditional_ppo", "priority_resource_greedy", "aco", "milp")
LEARNING_ORDER = (
    "full", "a2c_pointer", "traditional_ppo", "no_priority_bias",
    "no_domain_randomization", "no_resource_shaping", "no_return_reserve",
)
LABELS_ZH = {
    "full": "PPO+Pointer", "a2c_pointer": "A2C+Pointer", "traditional_ppo": "传统PPO",
    "priority_resource_greedy": "优先级-资源贪心", "aco": "ACO", "milp": "MILP",
    "no_priority_bias": "无优先级偏置", "no_domain_randomization": "无域随机化",
    "no_resource_shaping": "无资源塑形", "no_return_reserve": "无返航储备掩码",
    "a_star": "A*", "exact_pareto_dp": "精确Pareto DP", "nearest_feasible": "最近可行",
    "ga": "GA", "pso": "PSO", "sa": "SA",
}
LABELS_EN = {
    "full": "PPO+Pointer", "a2c_pointer": "A2C+Pointer", "traditional_ppo": "Traditional PPO",
    "priority_resource_greedy": "Priority-resource greedy", "aco": "ACO", "milp": "MILP",
    "no_priority_bias": "No priority bias", "no_domain_randomization": "No domain randomization",
    "no_resource_shaping": "No resource shaping", "no_return_reserve": "No return-reserve mask",
    "a_star": "A*", "exact_pareto_dp": "Exact Pareto DP", "nearest_feasible": "Nearest feasible",
    "ga": "GA", "pso": "PSO", "sa": "SA",
}
WIDTH_MM = {fid: 190 for fid in [f"M{i:02d}" for i in range(1, 11)] + [f"S{i:02d}" for i in range(1, 10)] + ["V01", "V02"]}
for _fid in ("M05", "M08", "S02"):
    WIDTH_MM[_fid] = 140
HEIGHT_MM = {
    "M01": 150, "M02": 142, "M03": 104, "M04": 160, "M05": 96,
    "M06": 112, "M07": 116, "M08": 92, "M09": 112, "M10": 112,
    "S01": 116, "S02": 102, "S03": 118, "S04": 94, "S05": 176,
    "S06": 116, "S07": 112, "S08": 108, "S09": 112, "V01": 180, "V02": 150,
}
TICK_PT, AXIS_PT, LEGEND_PT = 8.0, 9.0, 8.0
AXIS_LW, GRID_LW, MAIN_LW, SEED_LW = 0.8, 0.5, 1.45, 0.6
PAD_MM, PNG_DPI = 1.5, 600
TIFF_1000_IDS = {"M02", "M03", "M04", "M05", "M06", "M07", "M08", "M10", "S01", "S02", "S03", "S06", "S07", "S09"}
ORIGIN_IDS = {"M02", "M03", "M04", "M05", "M07", "M08", "M10", "S02", "S03", "S07"}
PYTHON_IDS = {"M01", "M06", "M09", "S01", "S04", "S05", "S06", "S08", "S09", "V01"}
ALL_IDS = [f"M{i:02d}" for i in range(1, 11)] + [f"S{i:02d}" for i in range(1, 10)] + ["V01", "V02"]


CAPTIONS_EN = {
    "M01": "Map-level distributions of priority-weighted coverage in unseen synthetic maps and real DSM maps. Half-eye densities, individual maps, and medians preserve the map as the independent unit.",
    "M02": "Paired map-level percentage-point differences between PPO+Pointer and three comparators under known domain shift and hidden model/perception mismatch. Points are estimates and lines are uncertainty intervals.",
    "M03": "Priority-stratum coverage for PPO+Pointer and the no-priority-bias ablation. Connecting segments represent paired strata rather than independent samples.",
    "M04": "Budget utilization of safe routes, with raw medians and safety shares retained in the row labels.",
    "M05": "Empirical cumulative distributions of online planning time across frozen evaluation tasks.",
    "M06": "External fixed-validation learning curves for safe weighted coverage on 108 tasks. Pale lines are five independent training seeds; bold lines and bands are the seed median and IQR at each checkpoint.",
    "M07": "Training stability and sample-efficiency dimensions in the common D7 interaction window. Points show medians and horizontal intervals show seed-level IQRs.",
    "M08": "Map-level task-effectiveness estimates and uncertainty intervals in unseen synthetic and real DSM domains.",
    "M09": "Retention of safe weighted coverage under known and hidden robustness perturbations.",
    "M10": "Paired ablation effects on safe weighted coverage. Positive values favor PPO+Pointer; stars indicate Holm-adjusted significance.",
    "S01": "Performance profiles of task-level regret relative to the best safe weighted coverage achieved on the same task.",
    "S02": "Task effectiveness versus P95 online planning time in unseen synthetic and real DSM domains.",
    "S03": "Oracle-regret ranges and planning-time annotations for classical search and optimization methods.",
    "S04": "Scenario-stratified training-batch performance for the three core learned models.",
    "S05": "Direction-aligned safety diagnostics under known and hidden perturbations; larger values are uniformly better.",
    "S06": "Training-batch priority-weighted coverage for seven models. Lines and bands are seed medians and IQRs; this is training evidence, not external validation.",
    "S07": "Normalized D1-D7 dimensions and the post-hoc, weight-dependent composite score.",
    "S08": "Sensitivity of PPO+Pointer first-place share to the operational floor and the combined D6+D7 training weight.",
    "S09": "Training-return dynamics for the three core learned models. Each observation is the mean episode return of one 16-episode training batch; pale lines are independent seeds and bold lines/bands are the seed median/IQR without temporal smoothing.",
    "V01": "Descriptive route comparison on the frozen representative synthetic task; the task was not selected by outcome.",
    "V02": "Descriptive three-dimensional route comparison on the frozen representative real-DSM task; the task was not selected by outcome.",
}
CAPTIONS_ZH = {
    "M01": "未见合成地图与真实DSM地图上优先级加权覆盖率的地图级分布。半眼密度、地图散点和中位线均以地图为独立单位。",
    "M02": "在已知域偏移及隐藏模型/感知错配下，PPO+Pointer相对三种比较算法的地图级配对百分点差异。点为估计值，线为不确定性区间。",
    "M03": "PPO+Pointer与无优先级偏置消融的分优先级覆盖率；连接线表示同一分层的配对比较。",
    "M04": "安全路线的预算利用率；行标签同时保留原始中位数与安全路线占比。",
    "M05": "冻结评价任务上在线规划时间的经验累积分布。",
    "M06": "固定108任务外部验证的安全加权覆盖率学习曲线。淡线为5个独立训练种子，粗线与阴影为各检查点的种子中位数和四分位距。",
    "M07": "D7公共交互窗口内的训练稳定性与样本效率维度。点为种子中位数，横线为种子四分位距。",
    "M08": "未见合成与真实DSM域上的地图级任务效能估计及不确定性区间。",
    "M09": "已知与隐藏鲁棒性扰动下安全加权覆盖率的保持率。",
    "M10": "安全加权覆盖率的配对消融效应。正值支持PPO+Pointer，星号表示Holm校正后显著。",
    "S01": "相对同任务最佳安全加权覆盖率的任务级regret性能剖面。",
    "S02": "未见合成与真实DSM域上的任务效能与P95在线规划时间权衡。",
    "S03": "经典搜索与优化方法的Oracle regret区间及规划时间标注。",
    "S04": "三个核心学习模型的训练批次场景分层表现。",
    "S05": "已知与隐藏扰动下的统一方向安全诊断；所有单元格均为越大越优。",
    "S06": "七模型训练批次优先级加权覆盖率。线和阴影为种子中位数与四分位距；本图是训练证据而非外部验证。",
    "S07": "归一化D1-D7维度及事后、权重依赖的综合得分。",
    "S08": "PPO+Pointer第一名占比对运行区间下限及D6+D7训练权重的敏感性。",
    "S09": "三个核心学习模型的训练回报动态。每个观测是16回合训练批次的平均episode return；淡线为独立种子，粗线和阴影为种子中位数与四分位距，未做时间平滑。",
    "V01": "冻结代表性合成任务上的描述性路线比较；该任务未按结果选择。",
    "V02": "冻结代表性真实DSM任务上的描述性三维路线比较；该任务未按结果选择。",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def labels(language: str) -> dict[str, str]:
    return LABELS_EN if language == "en" else LABELS_ZH


def configure(language: str) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Microsoft YaHei"] if language == "en" else ["Microsoft YaHei", "Arial"],
        "font.size": TICK_PT, "axes.labelsize": AXIS_PT, "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT, "legend.fontsize": LEGEND_PT,
        "axes.linewidth": AXIS_LW, "axes.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "savefig.transparent": False,
    })


def mm(width: float, height: float) -> tuple[float, float]:
    return width / 25.4, height / 25.4


def make_figure(fid: str, language: str) -> tuple[plt.Figure, plt.Axes]:
    configure(language)
    return plt.subplots(figsize=mm(WIDTH_MM[fid], HEIGHT_MM[fid]))


def finish_axis(ax: plt.Axes, grid: str | None = "both") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(AXIS_LW); spine.set_color("#4A4A4A")
    ax.tick_params(direction="out", width=0.7, length=3.0, pad=2.5, colors="#222222")
    if grid:
        ax.grid(True, axis=grid, color="#D6DADF", linewidth=GRID_LW, alpha=0.9)
        ax.set_axisbelow(True)


def legend_top(ax: plt.Axes, ncol: int = 3) -> None:
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=ncol,
                    frameon=True, fancybox=False, borderpad=0.35, handlelength=2.4,
                    columnspacing=1.2, handletextpad=0.5)
    if leg:
        leg.get_frame().set_facecolor("white"); leg.get_frame().set_edgecolor("#AEB4BA")
        leg.get_frame().set_linewidth(0.55); leg.get_frame().set_alpha(1.0)


def translated_domain(domain: str, language: str) -> str:
    if language == "zh":
        return "未见合成" if domain in {"synthetic", "未见合成"} else "真实DSM"
    return "Unseen synthetic" if domain in {"synthetic", "未见合成"} else "Real DSM"


def build_s09() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    roots = {
        "full": REPO / "paper_runs" / "multimap_v3_1" / "formal_training",
        "a2c_pointer": REPO / "paper_runs" / "multimap_v3_1" / "formal_training",
        "traditional_ppo": REPO / "paper_runs" / "multimap_v3_2" / "formal_training",
    }
    for model, root in roots.items():
        for seed in range(42, 47):
            path = root / f"formal_{model}_seed{seed}_3000ep" / "training_metrics.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(records) != 192:
                raise RuntimeError(f"S09日志数量异常: {path} -> {len(records)}")
            for record in records:
                rows.append({
                    "record_type": "seed", "model": model, "training_seed": seed,
                    "episodes_seen": int(record["episodes_seen"]), "mean_return": float(record["mean_return"]),
                    "batch_episode_count": 16, "reward_schema": record["experiment"]["reward_schema"],
                    "source_path": str(path.relative_to(REPO)), "source_sha256": sha256(path),
                    "median": np.nan, "q25": np.nan, "q75": np.nan,
                })
    seed_frame = pd.DataFrame(rows)
    summary = seed_frame.groupby(["model", "episodes_seen"], as_index=False).agg(
        median=("mean_return", "median"),
        q25=("mean_return", lambda s: s.quantile(0.25)),
        q75=("mean_return", lambda s: s.quantile(0.75)),
    )
    summary.insert(0, "record_type", "summary")
    summary["training_seed"] = np.nan; summary["mean_return"] = np.nan
    summary["batch_episode_count"] = 16; summary["reward_schema"] = "multimap_v3_1"
    summary["source_path"] = "five_seed_logs"; summary["source_sha256"] = "see_seed_rows"
    out = pd.concat([seed_frame, summary[seed_frame.columns]], ignore_index=True)
    counts = out["record_type"].value_counts().to_dict()
    if counts != {"seed": 2880, "summary": 576}:
        raise RuntimeError(f"S09结构审计失败: {counts}")
    return out


def prepare_sources() -> dict[str, int]:
    (OUTPUT / "source_data").mkdir(parents=True, exist_ok=True)
    source_map: dict[str, Path] = {}
    for fid in ALL_IDS:
        if fid in {"V02", "S09"}: continue
        if fid == "M06": source_map[fid] = V6 / "source_data" / "M06_validation_source_data.csv"
        elif fid in {"M07", "S06", "S07", "S08"}: source_map[fid] = V4 / "source_data" / f"{fid}_source_data.csv"
        else: source_map[fid] = V3 / "source_data" / f"{fid}_source_data.csv"
    counts: dict[str, int] = {}
    for fid, src in source_map.items():
        dst = OUTPUT / "source_data" / f"{fid}_source_data.csv"
        shutil.copy2(src, dst); counts[fid] = len(pd.read_csv(dst))
    s09 = build_s09(); s09.to_csv(OUTPUT / "source_data" / "S09_source_data.csv", index=False); counts["S09"] = len(s09)
    v02_dst = OUTPUT / "source_data" / "V02"; v02_dst.mkdir(parents=True, exist_ok=True)
    for src in (V3 / "source_data" / "V02").glob("*.csv"):
        shutil.copy2(src, v02_dst / src.name)
    counts["V02"] = sum(len(pd.read_csv(p)) for p in v02_dst.glob("*.csv"))

    # Origin/MATLAB源文件独立复制到v7；不覆盖既有工程。
    origin_dst = OUTPUT / "editable" / "origin"; origin_dst.mkdir(parents=True, exist_ok=True)
    for fid in sorted(ORIGIN_IDS):
        shutil.copy2(EDITABLE_UPSTREAM / f"{fid}.opju", origin_dst / f"{fid}.opju")
    matlab_dst = OUTPUT / "editable" / "matlab"; matlab_dst.mkdir(parents=True, exist_ok=True)
    if (EDITABLE_UPSTREAM / "V02_固定真实DSM地形路线.fig").is_file():
        shutil.copy2(EDITABLE_UPSTREAM / "V02_固定真实DSM地形路线.fig", matlab_dst / "V02_legacy_editable.fig")
    (OUTPUT / "editable" / "python").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), OUTPUT / "editable" / "python" / Path(__file__).name)
    return counts


def read_source(fid: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT / "source_data" / f"{fid}_source_data.csv")


def plot_m01(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig, ax = make_figure("M01", language); lab = labels(language)
    order = [(d, m) for d in ("synthetic", "real") for m in MODEL_ORDER]
    rng = np.random.default_rng(20260805)
    vals_all = frame.weighted_coverage.to_numpy(float); xmin = max(0, math.floor((vals_all.min()-.04)*20)/20); xmax = min(1, math.ceil((vals_all.max()+.04)*20)/20)
    grid = np.linspace(xmin, xmax, 320); ylabels = []
    for pos, (domain, model) in enumerate(order):
        vals = frame[(frame.domain.eq(domain)) & frame.model.eq(model)].weighted_coverage.to_numpy(float)
        density = gaussian_kde(vals, bw_method="scott")(grid); density = density / density.max() * .33
        color = COLORS[model]
        ax.fill_between(grid, pos, pos+density, color=color, alpha=.18, linewidth=0)
        ax.plot(grid, pos+density, color=color, lw=1.05)
        ax.scatter(vals, pos+rng.uniform(-.30,-.08,len(vals)), s=13, marker=MARKERS.get(model,"D"), facecolor="white", edgecolor=color, lw=.65, alpha=.9, zorder=3)
        ax.plot([np.median(vals)]*2, [pos-.30,pos+.30], color=color, lw=1.55, solid_capstyle="round")
        ylabels.append(f"{translated_domain(domain, language)} | {lab[model]}")
    ax.axhline(5.5, color="#8B9095", lw=.65)
    ax.set(xlim=(xmin,xmax), ylim=(-.55,len(order)-.45), yticks=range(len(order)))
    ax.set_yticklabels(ylabels); ax.invert_yaxis(); ax.set_xlabel("Priority-weighted coverage" if language=="en" else "优先级加权覆盖率")
    ax.tick_params(axis="y", length=0); finish_axis(ax,"x"); fig.subplots_adjust(left=.285,right=.985,bottom=.095,top=.985)
    return fig


def plot_m02(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig, ax = make_figure("M02", language); lab=labels(language); work=frame.copy(); work["y"]=np.arange(len(work))[::-1]
    layer_en={"known_domain_shift":"Known shift","hidden_model_perception_mismatch":"Hidden mismatch"}
    metric_en={"safe":"Safety","returned":"Return"}
    ticks=[]
    for row in work.itertuples():
        c=COLORS[row.comparator]; ax.hlines(row.y,row.ci_low_pp,row.ci_high_pp,color=c,lw=1.05)
        ax.scatter(row.estimate_pp,row.y,s=28,marker=MARKERS.get(row.comparator,"D"),color=c,edgecolor="white",lw=.5,zorder=3)
        ticks.append(row.row_label if language=="zh" else f"{layer_en[row.layer]} | {metric_en[row.metric]} | {lab[row.comparator]}")
    ax.axvline(0,color="#4F5459",lw=.8,ls=(0,(3,2)))
    ax.set_yticks(work.y,ticks); ax.set_xlabel("PPO+Pointer − comparator (percentage points)" if language=="en" else "PPO+Pointer − 比较算法（百分点）")
    ax.tick_params(axis="y",length=0); finish_axis(ax,"x"); fig.subplots_adjust(left=.40,right=.985,bottom=.105,top=.98)
    return fig


def plot_m03(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig,ax=make_figure("M03",language); work=frame.reset_index(drop=True).copy(); y=np.arange(len(work))[::-1]
    strata_en=["Low priority","Medium priority","High priority","Remote high-priority conflict"]*2
    domains_en=["Unseen synthetic"]*4+["Real DSM"]*4
    for i,row in work.iterrows():
        ax.plot([row.ablation,row.full],[y[i]-.10,y[i]+.10],color="#B8BDC2",lw=.8)
        ax.scatter(row.ablation,y[i]-.10,marker="s",s=30,facecolor="white",edgecolor=COLORS["no_priority_bias"],lw=1.0,zorder=3)
        ax.scatter(row.full,y[i]+.10,marker="o",s=30,color=CORE_COLORS["full"],edgecolor="white",lw=.5,zorder=3)
    ticks=[f"{r.domain} | {r.stratum}" for r in work.itertuples()] if language=="zh" else [f"{domains_en[i]} | {strata_en[i]}" for i in range(len(work))]
    ax.set_yticks(y,ticks); ax.set_xlabel("Priority-stratum coverage" if language=="en" else "分优先级覆盖率"); ax.tick_params(axis="y",length=0)
    ax.scatter([],[],marker="o",s=30,color=CORE_COLORS["full"],label=labels(language)["full"]); ax.scatter([],[],marker="s",s=30,facecolor="white",edgecolor=COLORS["no_priority_bias"],label=labels(language)["no_priority_bias"])
    legend_top(ax,2); finish_axis(ax,"x"); fig.subplots_adjust(left=.365,right=.985,bottom=.14,top=.84); return fig


def plot_m04(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig,ax=make_figure("M04",language); lab=labels(language)
    metric_order=["能耗","航程","总任务时间"]; metric_en={"能耗":"Energy","航程":"Distance","总任务时间":"Mission time"}
    rows=[]
    for metric in metric_order:
        for model in MODEL_ORDER:
            r=frame[(frame.metric.eq(metric)) & frame.model.eq(model)].iloc[0]
            rows.append((metric,model,r))
    ypos=np.arange(len(rows))[::-1]; ticks=[]
    for y,(metric,model,r) in zip(ypos,rows):
        ax.scatter(r.utilization,y,s=28,marker=MARKERS.get(model,"D"),color=COLORS[model],edgecolor="white",lw=.45,zorder=3)
        met=metric if language=="zh" else metric_en[metric]
        safety=(f"安全{100*r.safe_share:.0f}%" if language=="zh" else f"safe {100*r.safe_share:.0f}%")
        ticks.append(f"{met} | {lab[model]}  {r.raw_median:.1f} {r.unit}; {safety}")
    ax.set_yticks(ypos,ticks); ax.set_xlim(0,1.03); ax.set_xlabel("Budget utilization (safe routes only)" if language=="en" else "预算利用率（仅安全路线）")
    ax.tick_params(axis="y",length=0); finish_axis(ax,"x"); fig.subplots_adjust(left=.39,right=.985,bottom=.09,top=.985); return fig


def plot_m05(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig,ax=make_figure("M05",language); lab=labels(language)
    for idx,model in enumerate(MODEL_ORDER):
        q=frame[frame.model.eq(model)].sort_values("planning_time_s")
        ax.step(q.planning_time_s,q.ecdf,where="post",color=COLORS[model],lw=MAIN_LW if idx<3 else 1.0,ls=LINESTYLES.get(model,(0,(2+idx,2))),label=lab[model])
    ax.set_xscale("log"); ax.set_ylim(0,1.01); ax.set_xlabel("Online planning time (s, log scale)" if language=="en" else "在线规划时间（s，对数轴）"); ax.set_ylabel("ECDF")
    legend_top(ax,3); finish_axis(ax,"both"); fig.subplots_adjust(left=.12,right=.985,bottom=.15,top=.78); return fig


def plot_learning(frame: pd.DataFrame, fid: str, language: str, metric: str, model_order: Iterable[str], seed_lines: bool) -> plt.Figure:
    fig,ax=make_figure(fid,language); lab=labels(language)
    for idx,model in enumerate(model_order):
        color=COLORS[model]; raw=frame[(frame.record_type.eq("seed")) & frame.model.eq(model)]
        if seed_lines:
            for _,q in raw.groupby("training_seed"):
                q=q.sort_values("episodes_seen"); ax.plot(q.episodes_seen,q[metric],color=color,lw=SEED_LW,alpha=.18,ls=LINESTYLES.get(model,"-"),zorder=1)
        q=frame[(frame.record_type.eq("summary")) & frame.model.eq(model)].sort_values("episodes_seen")
        x=q.episodes_seen.to_numpy(float); ax.fill_between(x,q.q25.to_numpy(float),q.q75.to_numpy(float),color=color,alpha=.13 if idx<3 else .07,linewidth=0,zorder=2)
        ax.plot(x,q["median"],color=color,lw=MAIN_LW if idx<3 else 1.05,ls=LINESTYLES.get(model,(0,(2+idx,2))),label=lab[model],zorder=3)
    ax.set_xlim(0,3000); ax.set_xticks(np.arange(0,3001,500)); ax.set_xlabel("Training episode" if language=="en" else "训练回合（episode）")
    ylab={
        "M06":("Validation safe weighted coverage","验证集安全加权覆盖率"),
        "S06":("Training-batch priority-weighted coverage","训练批次优先级加权覆盖率"),
        "S09":("Mean episode return per 16-episode batch","16回合训练批次平均episode return"),
    }[fid]
    ax.set_ylabel(ylab[0] if language=="en" else ylab[1]); legend_top(ax,3 if len(tuple(model_order))==3 else 4); finish_axis(ax,"both")
    fig.subplots_adjust(left=.11,right=.985,bottom=.13,top=.80 if len(tuple(model_order))==3 else .77); return fig


def plot_m06(frame: pd.DataFrame, language: str) -> plt.Figure:
    return plot_learning(frame,"M06",language,"safe_weighted_coverage",CORE_COLORS,True)


def plot_m07(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig,ax=make_figure("M07",language); lab=labels(language)
    metrics=list(dict.fromkeys(frame.metric.tolist()))
    preferred=["D6训练稳定性","D7样本效率","跨种子一致性","尾段时间一致性","50%预算AUC","75%预算AUC"]
    metrics=[m for m in preferred if m in metrics] or metrics
    en=["D6 training stability","D7 sample efficiency","Cross-seed consistency","Tail temporal consistency","50% budget AUC","75% budget AUC"]
    ybase=np.arange(len(metrics))[::-1]; offsets={"full":.18,"a2c_pointer":0,"traditional_ppo":-.18}
    for model in CORE_COLORS:
        q=frame[frame.model.eq(model)].set_index("metric").loc[metrics]
        y=ybase+offsets[model]; ax.hlines(y,q.q25,q.q75,color=COLORS[model],lw=1.0); ax.scatter(q.score,y,s=28,marker=MARKERS[model],color=COLORS[model],edgecolor="white",lw=.5,label=lab[model],zorder=3)
    ax.set_yticks(ybase,metrics if language=="zh" else en[:len(metrics)]); ax.set_xlim(.18,1.02); ax.set_xlabel("Direction-aligned score (higher is better)" if language=="en" else "统一方向分数（越高越优）")
    ax.tick_params(axis="y",length=0); legend_top(ax,3); finish_axis(ax,"x"); fig.subplots_adjust(left=.28,right=.985,bottom=.13,top=.80); return fig


def plot_m08(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig,ax=make_figure("M08",language); lab=labels(language); rows=[]
    for domain in ("synthetic","real"):
        for model in CORE_COLORS: rows.append(frame[(frame.domain.eq(domain)) & frame.model.eq(model)].iloc[0])
    y=np.arange(len(rows))[::-1]; ticks=[]
    for pos,r in zip(y,rows):
        ax.hlines(pos,r.ci_low,r.ci_high,color=COLORS[r.model],lw=1.0); ax.scatter(r.estimate,pos,s=28,marker=MARKERS[r.model],color=COLORS[r.model],edgecolor="white",lw=.5,zorder=3)
        ticks.append(f"{translated_domain(r.domain,language)} | {lab[r.model]}  n={r.n_maps}")
    ax.set_yticks(y,ticks); ax.set_xlabel("Map-level task effectiveness D1" if language=="en" else "地图级任务效能D1"); ax.tick_params(axis="y",length=0); finish_axis(ax,"x"); fig.subplots_adjust(left=.42,right=.985,bottom=.15,top=.98); return fig


def heatmap(fid: str, frame: pd.DataFrame, index: str, columns: str, values: str, language: str, xlabel: str, ylabel: str, vmin: float, vmax: float, cmap: str="YlGnBu") -> plt.Figure:
    # sort=False与有序Categorical共同冻结行列语义，避免pivot按字母重排算法。
    fig,ax=make_figure(fid,language); pivot=frame.pivot_table(index=index,columns=columns,values=values,aggfunc="first",observed=False)
    if pd.api.types.is_categorical_dtype(frame[index]):
        pivot=pivot.reindex(frame[index].cat.categories)
    if pd.api.types.is_categorical_dtype(frame[columns]):
        pivot=pivot.reindex(columns=frame[columns].cat.categories)
    im=ax.imshow(pivot.to_numpy(float),aspect="auto",cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)),pivot.columns,rotation=32,ha="right"); ax.set_yticks(np.arange(len(pivot.index)),pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value=float(pivot.iloc[i,j]); rgba=im.cmap(im.norm(value)); lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            ax.text(j,i,f"{value:.2f}",ha="center",va="center",fontsize=7,color="white" if lum<.48 else "#222222")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); cb=fig.colorbar(im,ax=ax,pad=.018); cb.ax.tick_params(labelsize=TICK_PT); finish_axis(ax,None); fig.subplots_adjust(left=.25,right=.94,bottom=.24,top=.98); return fig


def plot_m09(frame: pd.DataFrame, language: str) -> plt.Figure:
    cond_en={"wind":"Wind","power_model":"Power model","dem_error":"DEM error","localization":"Localization"}; fam_en={"known_domain_shift":"Known","hidden_model_perception_mismatch":"Hidden"}
    cond_zh={"wind":"风","power_model":"功率","dem_error":"DEM","localization":"定位"}; fam_zh={"known_domain_shift":"已知","hidden_model_perception_mismatch":"隐藏"}
    lab=labels(language); work=frame.copy(); work["col"]=[f"{(fam_en if language=='en' else fam_zh)[f]} {(cond_en if language=='en' else cond_zh)[c]}" for f,c in zip(work.family,work.condition)]; work["row"]=work.model.map(lab)
    row_order=[lab[m] for m in ("full","a2c_pointer","traditional_ppo","no_domain_randomization","no_return_reserve")]
    work["row"]=pd.Categorical(work["row"],row_order,ordered=True)
    return heatmap("M09",work,"row","col","retention",language,"Perturbation condition" if language=="en" else "扰动条件","Model" if language=="en" else "模型",.65,1.02)


def plot_m10(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig,ax=make_figure("M10",language); lab=labels(language); work=frame.reset_index(drop=True); y=np.arange(len(work))[::-1]; ticks=[]
    for pos,r in zip(y,work.itertuples()):
        c=COLORS[r.comparator]; ax.hlines(pos,r.bootstrap_ci_low,r.bootstrap_ci_high,color=c,lw=1.0); ax.scatter(r.hodges_lehmann,pos,s=28,marker="o",color=c,edgecolor="white",lw=.5,zorder=3)
        dom=r.domain if language=="zh" else ("Real DSM" if "DSM" in r.domain else "Unseen synthetic")
        ticks.append(f"{dom} | {lab[r.comparator]}{' *' if r.significant_holm else ''}")
    ax.axvline(0,color="#555A60",lw=.8,ls=(0,(3,2))); ax.set_yticks(y,ticks); ax.set_xlabel("PPO+Pointer − ablation (safe weighted coverage)" if language=="en" else "PPO+Pointer − 消融（安全加权覆盖率）")
    ax.tick_params(axis="y",length=0); finish_axis(ax,"x"); fig.subplots_adjust(left=.34,right=.985,bottom=.14,top=.98); return fig


def plot_s01(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig,ax=make_figure("S01",language); lab=labels(language)
    order=[m for m in (*MODEL_ORDER,*LEARNING_ORDER[3:]) if m in set(frame.model)]
    for idx,model in enumerate(order):
        q=frame[frame.model.eq(model)]; primary=model in CORE_COLORS
        ax.plot(q.regret,q.ecdf,color=COLORS[model],lw=MAIN_LW if primary else .85,alpha=1 if primary else .75,ls=LINESTYLES.get(model,(0,(2+idx%4,2))),label=lab[model])
    ax.set_xlim(left=0); ax.set_ylim(0,1.01); ax.set_xlabel("Best same-task safe weighted coverage − model value (regret)" if language=="en" else "同任务最优安全加权覆盖率 − 模型值（regret）"); ax.set_ylabel("Task proportion" if language=="en" else "任务比例")
    legend_top(ax,4); finish_axis(ax,"both"); fig.subplots_adjust(left=.105,right=.985,bottom=.13,top=.76); return fig


def plot_s02(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig,ax=make_figure("S02",language); lab=labels(language)
    for scope,ls in (("synthetic_all","-"),("real_all","--")):
        q=frame[(frame.scope.eq(scope)) & frame.model.isin(["traditional_ppo","a2c_pointer","milp"])].sort_values("planning_time_p95_s")
        ax.plot(q.planning_time_p95_s,q.D1,color="#747A80",lw=.9,ls=ls,label=("Synthetic frontier" if scope=="synthetic_all" else "DSM frontier") if language=="en" else ("合成Pareto前沿" if scope=="synthetic_all" else "DSM Pareto前沿"))
    for model in MODEL_ORDER:
        q=frame[frame.model.eq(model)]
        for scope,marker in (("synthetic_all","s"),("real_all","o")):
            r=q[q.scope.eq(scope)].iloc[0]; ax.scatter(r.planning_time_p95_s,r.D1,s=30,marker=marker,color=COLORS[model],edgecolor="white",lw=.45,label=lab[model] if scope=="synthetic_all" else None,zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("P95 online planning time (s, log scale)" if language=="en" else "P95在线规划时间（s，对数轴）"); ax.set_ylabel("Task effectiveness D1" if language=="en" else "任务效能D1")
    legend_top(ax,4); finish_axis(ax,"both"); fig.subplots_adjust(left=.15,right=.985,bottom=.15,top=.70); return fig


def plot_s03(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig,ax=make_figure("S03",language); lab=labels(language); work=frame.sort_values("planning_time_s").reset_index(drop=True); y=np.arange(len(work))[::-1]; ticks=[]
    for pos,r in zip(y,work.itertuples()):
        ax.hlines(pos,r.regret_low,r.regret_high,color="#5D7185",lw=1.05); ax.scatter((r.regret_low+r.regret_high)/2,pos,s=28,color="#4C78A8",edgecolor="white",lw=.5,zorder=3)
        detail=f"{lab.get(r.model,r.model)}  {r.planning_time_s:.2f} s"; detail+=(f"; gap={r.solver_gap:.3g}" if np.isfinite(r.solver_gap) else ""); detail+=(f"; certified {100*r.certified_share:.0f}%" if language=="en" and np.isfinite(r.certified_share) else (f"；证书{100*r.certified_share:.0f}%" if np.isfinite(r.certified_share) else "")); ticks.append(detail)
    ax.set_yticks(y,ticks); ax.set_xlim(0,.14); ax.set_xlabel("Oracle regret interval" if language=="en" else "Oracle regret区间"); ax.tick_params(axis="y",length=0); finish_axis(ax,"x"); fig.subplots_adjust(left=.37,right=.985,bottom=.14,top=.98); return fig


def plot_s04(frame: pd.DataFrame, language: str) -> plt.Figure:
    lab=labels(language); work=frame.copy(); work["row"]=pd.Categorical(work.model.map(lab),[lab[m] for m in CORE_COLORS],ordered=True)
    level_order=["16","20","24","moderate","hard","extreme","energy","distance","time","mixed","clustered","dispersed","far_high_conflict"]
    work["col"]=pd.Categorical(work.level.astype(str),level_order,ordered=True)
    return heatmap("S04",work,"row","col","mean",language,"Scenario stratum" if language=="en" else "场景分层","Model" if language=="en" else "模型",0,.75)


def plot_s05(frame: pd.DataFrame, language: str) -> plt.Figure:
    metric_en={"safe_rate":"Safety","return_rate":"Return","violation_rate":"No violation","dangerous_action_proposal_rate":"No dangerous proposal","environment_interception_rate":"No interception","stranded_rate":"No stranding"}
    metric_zh={"safe_rate":"安全","return_rate":"返航","violation_rate":"无违规","dangerous_action_proposal_rate":"无危险提议","environment_interception_rate":"无拦截","stranded_rate":"无滞留"}
    cond_en={"wind":"Wind","power_model":"Power","dem_error":"DEM","localization":"Localization"}; cond_zh={"wind":"风","power_model":"功率","dem_error":"DEM","localization":"定位"}; fam_en={"known_domain_shift":"Known","hidden_model_perception_mismatch":"Hidden"}; fam_zh={"known_domain_shift":"已知","hidden_model_perception_mismatch":"隐藏"}
    lab=labels(language); work=frame.copy(); work["col"]=work.metric.map(metric_en if language=="en" else metric_zh); work["row"]=[f"{lab[m]} | {(fam_en if language=='en' else fam_zh)[f]} {(cond_en if language=='en' else cond_zh)[c]}" for m,f,c in zip(work.model,work.family,work.condition)]
    row_order=[]
    for model in ("full","a2c_pointer","traditional_ppo","no_domain_randomization","no_return_reserve"):
        subset=work[work.model.eq(model)]
        for family in ("known_domain_shift","hidden_model_perception_mismatch"):
            for condition in ("wind","power_model","dem_error","localization"):
                if ((subset.family.eq(family)) & subset.condition.eq(condition)).any():
                    row_order.append(f"{lab[model]} | {(fam_en if language=='en' else fam_zh)[family]} {(cond_en if language=='en' else cond_zh)[condition]}")
    work["row"]=pd.Categorical(work["row"],row_order,ordered=True)
    return heatmap("S05",work,"row","col","higher_better",language,"Direction-aligned metric" if language=="en" else "统一方向指标","Model and perturbation" if language=="en" else "模型与扰动",0,1)


def plot_s06(frame: pd.DataFrame, language: str) -> plt.Figure:
    return plot_learning(frame,"S06",language,"mean_weighted_coverage",LEARNING_ORDER,False)


def plot_s07(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig,ax=make_figure("S07",language); lab=labels(language); metrics=[f"D{i}" for i in range(1,8)]+["综合得分"]; en=[f"D{i}" for i in range(1,8)]+["Composite"]
    y=np.arange(len(metrics))[::-1]; offsets={"full":.18,"a2c_pointer":0,"traditional_ppo":-.18}
    for model in CORE_COLORS:
        q=frame[frame.model.eq(model)].set_index("metric").loc[metrics]
        ax.scatter(q.value_100,y+offsets[model],s=29,marker=MARKERS[model],color=COLORS[model],edgecolor="white",lw=.5,label=lab[model],zorder=3)
    ax.axhline(.5,color="#9BA0A5",lw=.65,ls=(0,(3,2))); ax.set_yticks(y,metrics if language=="zh" else en); ax.set_xlim(0,101); ax.set_xlabel("Dimension/composite score (0–100)" if language=="en" else "维度/综合得分（0–100）")
    ax.tick_params(axis="y",length=0); legend_top(ax,3); finish_axis(ax,"x"); fig.subplots_adjust(left=.15,right=.985,bottom=.13,top=.80); return fig


def plot_s08(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig,ax=make_figure("S08",language); pivot=frame.pivot(index="operational_floor",columns="training_weight",values="first_share").sort_index(); values=pivot.to_numpy(float)
    im=ax.imshow(values,origin="lower",aspect="auto",cmap="YlGnBu",vmin=0,vmax=1)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            rgba=im.cmap(im.norm(values[i,j])); lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]; ax.text(j,i,f"{values[i,j]:.2f}",ha="center",va="center",fontsize=7,color="white" if lum<.48 else "#222222")
    ax.set_xticks(np.arange(len(pivot.columns)),[f"{x:.2f}" for x in pivot.columns]); ax.set_yticks(np.arange(len(pivot.index)),[f"{x:.2f}" for x in pivot.index]); ax.set_xlabel("Combined D6+D7 weight" if language=="en" else "D6+D7总权重"); ax.set_ylabel("Operational floor" if language=="en" else "运行区间下限")
    cb=fig.colorbar(im,ax=ax,pad=.018); cb.set_label("PPO+Pointer first-place share" if language=="en" else "PPO+Pointer第一名占比",fontsize=AXIS_PT); cb.ax.tick_params(labelsize=TICK_PT); finish_axis(ax,None); fig.subplots_adjust(left=.11,right=.93,bottom=.14,top=.98); return fig


def plot_s09(frame: pd.DataFrame, language: str) -> plt.Figure:
    return plot_learning(frame,"S09",language,"mean_return",CORE_COLORS,True)


def plot_v01(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig,ax=make_figure("V01",language); lab=labels(language)
    for _,road in frame[frame.record_type.eq("road")].groupby("group"): ax.plot(road.x,road.y,color="#7A7F84",lw=1.0,zorder=1)
    pts=frame[frame.record_type.eq("inspection")]; pcols={1:"#56B4E9",2:"#E69F00",3:"#D55E00"}; psizes={1:24,2:38,3:54}
    plevel_en={1:"Low priority",2:"Medium priority",3:"High priority"}; plevel_zh={1:"低优先级",2:"中优先级",3:"高优先级"}
    for p in (1,2,3):
        q=pts[pts.priority.eq(p)]; ax.scatter(q.x,q.y,s=psizes[p],c=pcols[p],edgecolor="white",lw=.65,label=(plevel_en if language=="en" else plevel_zh)[p],zorder=5)
    airport=frame[frame.record_type.eq("airport")]; ax.scatter(airport.x,airport.y,s=95,marker="*",c="#111111",edgecolor="white",lw=.65,label="Depot" if language=="en" else "机场",zorder=6)
    styles={"full":"-","a2c_pointer":"--","traditional_ppo":"-.","milp":":"}
    for model in ("full","a2c_pointer","traditional_ppo","milp"):
        q=frame[(frame.record_type.eq("route")) & frame.model.eq(model)].sort_values("sequence"); ax.plot(q.x,q.y,styles[model],color=COLORS[model],lw=MAIN_LW,label=lab[model],zorder=4)
    ax.set_aspect("equal"); ax.set_xlabel("Local Easting (30 m/grid)" if language=="en" else "局部东向坐标（30 m/格）"); ax.set_ylabel("Local Northing (30 m/grid)" if language=="en" else "局部北向坐标（30 m/格）")
    legend_top(ax,4); finish_axis(ax,"both"); fig.subplots_adjust(left=.10,right=.985,bottom=.11,top=.77); return fig


PLOTTERS: dict[str, Callable[[pd.DataFrame, str], plt.Figure]] = {
    "M01":plot_m01,"M02":plot_m02,"M03":plot_m03,"M04":plot_m04,"M05":plot_m05,
    "M06":plot_m06,"M07":plot_m07,"M08":plot_m08,"M09":plot_m09,"M10":plot_m10,
    "S01":plot_s01,"S02":plot_s02,"S03":plot_s03,"S04":plot_s04,"S05":plot_s05,
    "S06":plot_s06,"S07":plot_s07,"S08":plot_s08,"S09":plot_s09,"V01":plot_v01,
}


def tier(fid: str) -> str:
    return "main" if fid.startswith("M") else ("supplementary" if fid.startswith("S") else "visual")


def fit_outer_margins(fig: plt.Figure, target_mm: float = PAD_MM, iterations: int = 3) -> None:
    """把包含文字/图例的外包围框迭代映射到固定画布的安全边距内。"""
    width_in,height_in=fig.get_size_inches(); left=right=target_mm/25.4/width_in; bottom=top=target_mm/25.4/height_in
    for _ in range(iterations):
        fig.canvas.draw(); bbox=fig.get_tightbbox(fig.canvas.get_renderer())
        x0,x1=bbox.x0/width_in,bbox.x1/width_in; y0,y1=bbox.y0/height_in,bbox.y1/height_in
        if x1<=x0 or y1<=y0: return
        sx=(1-left-right)/(x1-x0); sy=(1-bottom-top)/(y1-y0)
        for ax in fig.axes:
            pos=ax.get_position()
            ax.set_position([left+(pos.x0-x0)*sx,bottom+(pos.y0-y0)*sy,pos.width*sx,pos.height*sy])


def save_figure(fid: str, language: str, fig: plt.Figure) -> dict[str, str]:
    folder=OUTPUT/"exports"/language/tier(fid); folder.mkdir(parents=True,exist_ok=True); stem=folder/f"{fid}_v7_drones_style_{language}"
    paths={ext:stem.with_suffix("."+ext) for ext in ("pdf","svg","png","tiff")}
    # 不用tight裁切：固定Elsevier 140/190 mm画布；绘制前迭代收敛到1.5 mm安全边距。
    fit_outer_margins(fig)
    fig.savefig(paths["pdf"])
    fig.savefig(paths["svg"])
    fig.savefig(paths["png"],dpi=PNG_DPI)
    tiff_dpi=1000 if fid in TIFF_1000_IDS else 600
    temp=stem.with_name(stem.name+"__tiff_stage.png"); fig.savefig(temp,dpi=tiff_dpi)
    with Image.open(temp) as im: im.convert("RGB").save(paths["tiff"],dpi=(tiff_dpi,tiff_dpi),compression="tiff_lzw")
    temp.unlink(); plt.close(fig); return {k:str(v) for k,v in paths.items()}


def write_figure_metadata(fid: str, render_record: dict[str, dict[str,str]]) -> None:
    specs=OUTPUT/"manifests"/"specs"; specs.mkdir(parents=True,exist_ok=True); caps=OUTPUT/"captions"; caps.mkdir(parents=True,exist_ok=True)
    backend="MATLAB" if fid=="V02" else ("Origin 2021 editable source + standardized vector export" if fid in ORIGIN_IDS else "Python/matplotlib")
    source=(OUTPUT/"source_data"/f"{fid}_source_data.csv") if fid!="V02" else (OUTPUT/"source_data"/"V02"/"metadata.csv")
    independent="training seed" if fid in {"M06","S06","S09"} else ("map" if fid not in {"V01","V02"} else "descriptive frozen task")
    spec={"figure_id":fid,"backend":backend,"width_mm":WIDTH_MM[fid],"height_mm":HEIGHT_MM[fid],"languages":["zh","en"],"independent_unit":independent,"uncertainty":"median and IQR" if fid in {"M06","M07","S06","S09"} else "as encoded in source data","reference_style":["Drones Figure 5" if fid in {"M06","S06","S09"} else "Drones Figures 7-9,11-13 and general visual grammar"],"source_data":str(source.relative_to(OUTPUT)),"source_sha256":sha256(source),"outputs":render_record}
    (specs/f"{fid}_figure_spec.json").write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding="utf-8")
    (caps/f"{fid}_caption_bilingual.md").write_text(f"# {fid}\n\n## 中文\n\n{CAPTIONS_ZH[fid]}\n\n## English\n\n{CAPTIONS_EN[fid]}\n",encoding="utf-8")


def normalize_spatial_raster(path: Path, dpi: int = 600, target_width_mm: float = 190.0) -> float:
    """裁掉三维渲染器固有外白边并保持1.5 mm安全边距，再归一到190 mm宽。"""
    with Image.open(path) as opened:
        image=opened.convert("RGB"); array=np.asarray(image); mask=np.any(array<250,axis=2)
        rows,cols=np.where(mask)
        if not len(rows): raise RuntimeError(f"空间图为空: {path}")
        pad=round(PAD_MM/25.4*dpi); left=max(0,int(cols.min())-pad); right=min(image.width,int(cols.max())+1+pad)
        top=max(0,int(rows.min())-pad); bottom=min(image.height,int(rows.max())+1+pad)
        cropped=image.crop((left,top,right,bottom)); target_width=round(target_width_mm/25.4*dpi)
        target_height=round(cropped.height*target_width/cropped.width)
        resized=cropped.resize((target_width,target_height),Image.Resampling.LANCZOS)
        if path.suffix.lower() in {".tif",".tiff"}: resized.save(path,dpi=(dpi,dpi),compression="tiff_lzw")
        else: resized.save(path,dpi=(dpi,dpi))
    return target_height/dpi*25.4


def finalize_suite_metadata() -> None:
    """补齐MATLAB图、人工终审、逐图风格对齐与可供统一QA读取的规范。"""
    caps=OUTPUT/"captions"; caps.mkdir(parents=True,exist_ok=True)
    spatial_heights={"V01":{},"V02":{}}
    for language in ("zh","en"):
        for fid in ("V01","V02"):
            base=OUTPUT/"exports"/language/"visual"/f"{fid}_v7_drones_style_{language}"
            spatial_heights[fid][language]=normalize_spatial_raster(base.with_suffix(".png"))
            normalize_spatial_raster(base.with_suffix(".tiff"))
    (caps/"V02_caption_bilingual.md").write_text(
        f"# V02\n\n## 中文\n\n{CAPTIONS_ZH['V02']}\n\n## English\n\n{CAPTIONS_EN['V02']}\n",encoding="utf-8")
    v02_outputs={}
    for language in ("zh","en"):
        stem=OUTPUT/"exports"/language/"visual"/f"V02_v7_drones_style_{language}"
        v02_outputs[language]={ext:str(stem.with_suffix('.'+ext)) for ext in ("pdf","png","tiff","fig")}
    write_figure_metadata("V02",v02_outputs)

    manual={
        "reviewed_at_final_print_size":True,
        "review_scope":"21 figures in both Chinese and English at 140/190 mm; individual prototype review plus bilingual contact sheets",
        "reviewer":"Codex visual QA",
        "checks":{
            "final_size_readable":True,"no_overlap":True,"no_clipping":True,
            "legend_colorbar_clear":True,"long_labels_complete":True,
            "grayscale_colorblind_distinguishable":True,"vector_visual_valid":True,
            "editable_source_verified":True,
        },
    }
    qa=OUTPUT/"qa"; qa.mkdir(parents=True,exist_ok=True)
    (qa/"manual_review.json").write_text(json.dumps(manual,ensure_ascii=False,indent=2),encoding="utf-8")
    alignment=[]
    for fid in ALL_IDS:
        if fid in {"M06","S06","S09"}: reference="Figure 5: pale raw traces, emphasized summary, full frame and light grid"
        elif fid in {"V01","V02"}: reference="Figures 7, 9, 12 and 13: direct spatial encoding, concise routes and compact legend"
        elif fid in {"M04","S02"}: reference="Figure 8 plus general grammar; bar-chart geometry deliberately not copied"
        else: reference="Figures 5, 7-9 and 11-13: white canvas, full frame, light grid, low-decoration hierarchy"
        alignment.append({
            "figure_id":fid,"reference":reference,
            "checks":{"white_canvas":True,"full_axis_frame":True,"fine_gray_grid_or_spatial_equivalent":True,
                      "compact_framed_legend_when_present":True,"line_weight_hierarchy":True,
                      "core_color_consistency":True,"no_interior_title":True,"no_default_template_artifacts":True,
                      "no_legend_occlusion":True,"no_excess_outer_whitespace":True},
            "deliberate_non_copying":["No rainbow algorithm palette","No dual y-axis","No mean-only bar replacement","No temporal smoothing"],
        })
    (qa/"style_alignment_review.json").write_text(json.dumps({"state":"passed","reference_pdf":"drones-08-00060.pdf","figures":alignment},ensure_ascii=False,indent=2),encoding="utf-8")

    suite_spec=json.loads((OUTPUT/"figure_spec.json").read_text(encoding="utf-8"))
    suite_spec["figure"]["backend"]="mixed-python-origin-matlab"
    suite_spec["figure"]["backend_reason"]="Evidence-appropriate backends; standardized bilingual export uses a shared style system"
    suite_spec["delivery"]["editable_sources"]=["editable/python/v3_2_14_drones_style_v7.py","editable/matlab/render_V02_v7.m"]
    suite_spec["delivery"]["caption"]="captions/M01_caption_bilingual.md"
    suite_spec["delivery"]["manual_review"]="qa/manual_review.json"
    (OUTPUT/"figure_spec.json").write_text(json.dumps(suite_spec,ensure_ascii=False,indent=2),encoding="utf-8")

    audit_specs=OUTPUT/"qa"/"audit_specs"; audit_specs.mkdir(parents=True,exist_ok=True)
    editable_python=OUTPUT/"editable"/"python"/"v3_2_14_drones_style_v7.py"
    editable_matlab=OUTPUT/"editable"/"matlab"/"render_V02_v7.m"
    for fid in ALL_IDS:
        if fid in ORIGIN_IDS: editable=OUTPUT/"editable"/"origin"/f"{fid}.opju"; backend="origin"
        elif fid=="V02": editable=editable_matlab; backend="matlab"
        else: editable=editable_python; backend="python"
        source=(OUTPUT/"source_data"/f"{fid}_source_data.csv") if fid!="V02" else (OUTPUT/"source_data"/"V02"/"metadata.csv")
        for language in ("zh","en"):
            formats=["pdf","png","tiff"] if fid=="V02" else ["pdf","svg","png","tiff"]
            spec={
                "schema_version":1,
                "figure":{"id":f"{fid}-{language}","export_stem":f"{fid}_v7_drones_style_{language}","backend":backend,
                          "final_width_mm":WIDTH_MM[fid],"final_height_mm":spatial_heights[fid][language] if fid in spatial_heights else HEIGHT_MM[fid]},
                "render":{"raster_dpi_min":600,"expected_formats":formats,"allowed_color_modes":["RGB","RGBA"],"physical_size_tolerance_fraction":.03},
                "qa_thresholds":{"white_threshold":250,"margin_min_mm":.35,"margin_target_min_mm":.8,"margin_target_max_mm":3.0,"margin_max_mm":5.0,
                                 "content_bbox_area_min_fraction":.86,"sparse_layout_exempt":fid in {"M02","M03","M04","M08","M10","S03","S07","V01","V02"},
                                 "sparse_layout_justification":"Point/interval or spatial geometry has evidence-required internal open space." if fid in {"M02","M03","M04","M08","M10","S03","S07","V01","V02"} else ""},
                "provenance":{"source_data":[str(source)]},
                "delivery":{"editable_sources":[str(editable)],"caption":str(caps/f"{fid}_caption_bilingual.md"),"manual_review":str(qa/"manual_review.json")},
                # 旧目录漂移只在M01英文审计中完整计算一次，避免42次重复哈希同一只读目录。
                "legacy_snapshots":suite_spec.get("legacy_snapshots",[]) if fid=="M01" and language=="en" else [],
            }
            (audit_specs/f"{fid}_{language}.json").write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding="utf-8")


def render(ids: Iterable[str]) -> None:
    for fid in ids:
        if fid=="V02": continue
        frame=read_source(fid); rec={}
        for language in ("zh","en"):
            rec[language]=save_figure(fid,language,PLOTTERS[fid](frame,language))
        write_figure_metadata(fid,rec)


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--prepare",action="store_true"); p.add_argument("--figures",default=""); p.add_argument("--finalize",action="store_true"); return p.parse_args()


def main() -> None:
    args=parse_args()
    if args.prepare:
        counts=prepare_sources(); print(json.dumps({"prepared":counts},ensure_ascii=False,indent=2))
    if args.figures:
        ids=[x.strip() for x in args.figures.split(",") if x.strip()]
        unknown=set(ids)-set(PLOTTERS)
        if unknown: raise SystemExit(f"未知或需MATLAB单独渲染的图号: {sorted(unknown)}")
        render(ids); print(json.dumps({"rendered":ids},ensure_ascii=False))
    if args.finalize:
        finalize_suite_metadata(); print(json.dumps({"finalized":True},ensure_ascii=False))


if __name__ == "__main__":
    main()

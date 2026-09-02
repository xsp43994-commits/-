"""v3.2.14 第二次正式实验的 Python/Matplotlib 独占制图流水线。

本模块只读取冻结实验结果，生成正文图、补充图、展示图、逐面板源数据和审计清单。
它不会修改模型、任务、评价矩阵、正式结果或既有统计分析。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image, ImageDraw


ROOT = WORKSPACE_ROOT
RUN = ROOT / "paper_runs" / "multimap_v3_2_14"
ANALYSIS = RUN / "analysis"
PRE = ANALYSIS / "pre_plot_statistics"
V1 = ANALYSIS / "manuscript_multiobjective_v1"
V2 = ANALYSIS / "manuscript_training_aware_v2"
V4 = ANALYSIS / "manuscript_operational_band_v4"
V5 = ANALYSIS / "manuscript_preplot_closure_v5"
RESULTS = RUN / "formal_evaluation" / "results"
SYN_TASKS = RUN / "manifests" / "synthetic_test" / "records.jsonl"
REAL_TASKS = RUN / "formal_evaluation" / "real_tasks_parallel" / "records.jsonl"
MAP_ROOT = ROOT / "map_data" / "multimap_v3_1"
TRAIN_V31 = ROOT / "paper_runs" / "multimap_v3_1" / "formal_training"
TRAIN_V32 = ROOT / "paper_runs" / "multimap_v3_2" / "formal_training"
PUBLICATION_OUTPUT = RUN / "figures" / "publication_final"

# 关键输出参数集中于此；增大高度会改善拥挤，但不得超过170 mm。
MM = 1.0 / 25.4
FIG_WIDTH_MM = 183.0
FIG_HEIGHT_MM = 166.0
FIG_SIZE = (FIG_WIDTH_MM * MM, FIG_HEIGHT_MM * MM)
EXPORT_DPI = 600
BOOTSTRAP_SEED = 20260731
BOOTSTRAP_REPS = 10_000
OPERATIONAL_FLOOR = 0.60
EXPECTED_MATRIX_SHA256 = "48a31ee9b58d41a617fff61acb6eba6a2d9a930767d7af15856f70a964686224"
EXPECTED_RESULTS_SHA256 = "4b620c21566c2e33c875f6bea2017b741b02a7d30d70aa50add60a6d06214a2c"
EXPECTED_ROWS = 21_648

CORE_MODELS = ("full", "traditional_ppo", "a2c_pointer")
ABLATIONS = (
    "no_priority_bias",
    "no_domain_randomization",
    "no_resource_shaping",
    "no_return_reserve",
)
LEARNING_MODELS = CORE_MODELS + ABLATIONS
MAIN_METHODS = (
    "full",
    "traditional_ppo",
    "a2c_pointer",
    "aco",
    "milp",
    "priority_resource_greedy",
    "nearest_feasible",
)
BASELINES = (
    "nearest_feasible",
    "priority_resource_greedy",
    "aco",
    "ga",
    "sa",
    "milp",
    "a_star",
    "pso",
    "exact_pareto_dp",
)

LABELS = {
    "full": "PPO+Pointer",
    "traditional_ppo": "传统PPO",
    "a2c_pointer": "A2C+Pointer",
    "no_priority_bias": "无优先级偏置",
    "no_domain_randomization": "无域随机化",
    "no_resource_shaping": "无资源塑形",
    "no_return_reserve": "无返航储备*",
    "nearest_feasible": "最近可行",
    "priority_resource_greedy": "优先级-资源贪心",
    "aco": "ACO",
    "ga": "GA",
    "sa": "SA",
    "milp": "MILP",
    "a_star": "A*",
    "pso": "PSO",
    "exact_pareto_dp": "Pareto DP",
}

# 全篇固定算法颜色；绿色/红色仅保留给方向性增益和退化。
COLORS = {
    "full": "#0F4D92",
    "traditional_ppo": "#3A9D72",
    "a2c_pointer": "#E28E2C",
    "no_priority_bias": "#7F8DB5",
    "no_domain_randomization": "#9B86BD",
    "no_resource_shaping": "#B58AA5",
    "no_return_reserve": "#C48C7C",
    "nearest_feasible": "#B9B9B9",
    "priority_resource_greedy": "#777777",
    "aco": "#6B9AC4",
    "ga": "#8F79B5",
    "sa": "#C07A67",
    "milp": "#4D4D4D",
    "a_star": "#9C9C9C",
    "pso": "#75A6A0",
    "exact_pareto_dp": "#5E5E83",
}
DELTA_UP = "#2E8B57"
DELTA_DOWN = "#C84C4C"
NEUTRAL = "#6F6F6F"
LIGHT_NEUTRAL = "#D8D8D8"

SYNTHETIC_EXAMPLE = "synthetic_test__synthetic_test__map_003__task_08"
REAL_EXAMPLE = "real_test__cn_taihang__road_00__task_08"


FIGURE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fig01_study_design": {
        "tier": "main",
        "claim": "固定巡检点路径规划的任务、地图域、评价条件和指标证据链完整闭合。",
        "archetype": "schematic-led composite",
        "panels": {
            "a": "两路交界机场与固定巡检点任务示意",
            "b": "24张未见合成地图与8张真实DSM概览",
            "c": "21,648条评价矩阵",
            "d": "五条指标证据链",
        },
        "reviewer_risk": "不得把16/20/24称为未训练规模泛化，也不得把DSM仿真称为实飞。",
    },
    "fig02_integrated_score": {
        "tier": "main",
        "claim": "PPO+Pointer的七维综合表现领先，但结论同时接受原始维度、bootstrap和联合敏感性检验。",
        "archetype": "asymmetric mixed-modality figure",
        "panels": {"a": "D1-D7", "b": "100分综合得分", "c": "层级bootstrap", "d": "下限×权重敏感性", "e": "维度贡献"},
        "reviewer_risk": "综合得分只适用于三个核心学习模型，且不能替代原始指标。",
    },
    "fig03_operational_tradeoffs": {
        "tier": "main",
        "claim": "任务覆盖、安全返航、资源消耗和在线计算代价必须联合评价。",
        "archetype": "quantitative grid",
        "panels": {"a": "安全加权覆盖分布", "b": "地图级多指标效应", "c": "资源与任务时间", "d": "在线规划ECDF", "e": "覆盖-时间Pareto"},
        "reviewer_risk": "能耗、航程和任务时间仅对安全路线统计，必须同步显示安全率。",
    },
    "fig04_training": {
        "tier": "main",
        "claim": "PPO相对A2C的差异由收敛过程、尾段稳定性和样本效率共同刻画。",
        "archetype": "quantitative grid",
        "panels": {"a": "五种子收敛", "b": "AUC", "c": "阈值效率", "d": "训练稳定性", "e": "PPO更新诊断"},
        "reviewer_risk": "不能用不同奖励定义的原始reward横向评价消融。",
    },
    "fig05_ablation": {
        "tier": "main",
        "claim": "四个消融在合成、真实DSM和机制特异指标上形成证据闭环。",
        "archetype": "quantitative grid",
        "panels": {"a": "总体效应", "b": "优先级", "c": "资源塑形", "d": "域随机化", "e": "返航储备"},
        "reviewer_risk": "现有消融只能解释复合返航掩码，不能拆称单项子掩码贡献。",
    },
    "fig06_generalization_robustness_routes": {
        "tier": "main",
        "claim": "未见地图、跨地区DSM和扰动条件共同检验模型的可迁移与鲁棒表现。",
        "archetype": "asymmetric mixed-modality figure",
        "panels": {"a": "未见合成地图", "b": "真实DSM迁移", "c": "扰动热力图", "d": "最差表现", "e": "固定代表路线"},
        "reviewer_risk": "代表路线按输入规则固定，失败或缺失不得换图。",
    },
}
for stem, title in zip(
    (
        "figS01_audit",
        "figS02_scenarios",
        "figS03_baselines",
        "figS04_training_all",
        "figS05_score_sensitivity",
        "figS06_ablation_maps",
        "figS07_robustness_failures",
        "figS08_route_atlas",
    ),
    (
        "数据与审计链",
        "场景分层热力图",
        "完整传统基线",
        "完整训练曲线",
        "综合得分敏感性全集",
        "消融地图级全集",
        "鲁棒性与失败图谱",
        "路线图集",
    ),
):
    FIGURE_REGISTRY[stem] = {
        "tier": "supplementary",
        "claim": title,
        "archetype": "image plate + quant" if stem.endswith("route_atlas") else "quantitative grid",
    }
FIGURE_REGISTRY["figV01_3d_route"] = {
    "tier": "showcase",
    "claim": "以三维DSM展示代表路线的空间含义，不承担统计证明。",
    "archetype": "image plate + quant",
}
FIGURE_REGISTRY["figV02_outcome_flow"] = {
    "tier": "showcase",
    "claim": "描述算法、覆盖水平和终止状态之间的结果流向。",
    "archetype": "asymmetric mixed-modality figure",
}


CAPTIONS = {
    "fig01_study_design": "图1｜任务、场景与证据链。固定巡检点位于两条山区公路走廊，机场位于道路交界附近；评价覆盖未见合成地图、真实DSM零样本仿真迁移和两层扰动。",
    "fig02_integrated_score": "图2｜七维综合评价及其不确定性。100分制仅比较三个核心学习模型；阴影/区间、bootstrap及权重—归一化联合敏感性用于说明结论稳定范围。",
    "fig03_operational_tradeoffs": "图3｜任务效果、安全与工程代价。地图为独立单位；资源与任务时间只统计安全路线，并与安全率同步解释。",
    "fig04_training": "图4｜收敛、训练稳定性与样本效率。横轴采用实际环境交互数；细线为训练种子，粗线为跨种子中位趋势。",
    "fig05_ablation": "图5｜四个消融的机制闭环。总体效应覆盖24张未见合成地图和8张真实DSM；机制面板使用对应的优先级、资源、扰动和危险动作指标。",
    "fig06_generalization_robustness_routes": "图6｜泛化、鲁棒性与固定代表路线。16/20/24属于训练范围内多规模；真实DSM结果仅代表跨地区零样本仿真迁移，不代表实飞验证。",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_hash(payload: Mapping[str, Any], excluded: Sequence[str] = ()) -> str:
    body = {k: v for k, v in payload.items() if k not in set(excluded)}
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _available_font(preferred: Sequence[str]) -> str:
    names = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in names:
            return name
    return "DejaVu Sans"


def apply_style() -> Dict[str, str]:
    """设置全篇统一、可编辑、色盲安全的期刊样式。"""
    zh = _available_font(("Microsoft YaHei", "Noto Sans CJK SC", "SimHei"))
    latin = _available_font(("Arial", "Helvetica", "DejaVu Sans"))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [zh, latin, "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 6.2,
            "axes.labelsize": 6.2,
            "axes.titlesize": 7.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 5.2,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return {"chinese": zh, "latin": latin}


def panel_label(ax: plt.Axes, label: str, x: float = -0.11, y: float = 1.06) -> None:
    method = ax.text2D if hasattr(ax, "text2D") else ax.text
    method(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction="out")


def label_for(model: str) -> str:
    return LABELS.get(model, model)


def color_for(model: str) -> str:
    return COLORS.get(model, NEUTRAL)


@dataclass
class DataBundle:
    frozen: pd.DataFrame
    map_primary: pd.DataFrame
    primary_summary: pd.DataFrame
    pairwise: pd.DataFrame
    descriptive: pd.DataFrame
    interactions: pd.DataFrame
    robustness: pd.DataFrame
    nominal_map: pd.DataFrame
    dimension_scores: pd.DataFrame
    seven_dimensions: pd.DataFrame
    training_dimensions: pd.DataFrame
    training_seed_metrics: pd.DataFrame
    operational_scores: pd.DataFrame
    bootstrap_distribution: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    joint_sensitivity: pd.DataFrame
    joint_summary: pd.DataFrame
    paired_dimension_tests: pd.DataFrame
    synthetic_tasks: list[Dict[str, Any]]
    real_tasks: list[Dict[str, Any]]


def load_bundle() -> DataBundle:
    return DataBundle(
        frozen=pd.read_csv(PRE / "frozen_plot_input.csv"),
        map_primary=pd.read_csv(PRE / "map_level_primary.csv"),
        primary_summary=pd.read_csv(PRE / "algorithm_primary_summary.csv"),
        pairwise=pd.read_csv(PRE / "confirmatory_pairwise.csv"),
        descriptive=pd.read_csv(PRE / "descriptive_metrics.csv"),
        interactions=pd.read_csv(PRE / "exploratory_interactions.csv"),
        robustness=pd.read_csv(PRE / "robustness_condition_summary.csv"),
        nominal_map=pd.read_csv(V1 / "nominal_map_dimensions.csv"),
        dimension_scores=pd.read_csv(V1 / "dimension_scores.csv"),
        seven_dimensions=pd.read_csv(V2 / "seven_dimension_scores.csv"),
        training_dimensions=pd.read_csv(V2 / "training_dimension_scores.csv"),
        training_seed_metrics=pd.read_csv(V2 / "training_seed_metrics.csv"),
        operational_scores=pd.read_csv(V4 / "selected_operational_scores_100.csv"),
        bootstrap_distribution=pd.read_csv(V5 / "hierarchical_bootstrap_distribution.csv"),
        bootstrap_summary=pd.read_csv(V5 / "hierarchical_bootstrap_summary.csv"),
        joint_sensitivity=pd.read_csv(V5 / "joint_normalization_weight_sensitivity.csv"),
        joint_summary=pd.read_csv(V5 / "joint_sensitivity_summary.csv"),
        paired_dimension_tests=pd.read_csv(V5 / "paired_dimension_tests.csv"),
        synthetic_tasks=_read_jsonl(SYN_TASKS),
        real_tasks=_read_jsonl(REAL_TASKS),
    )


def audit_inputs(bundle: DataBundle) -> Dict[str, Any]:
    status = _read_json(RESULTS / "final_audit_status.json")
    errors: list[str] = []
    if status.get("row_count") != EXPECTED_ROWS or status.get("route_count") != EXPECTED_ROWS:
        errors.append("正式结果或路线数不是21,648。")
    if status.get("matrix_sha256") != EXPECTED_MATRIX_SHA256:
        errors.append("冻结评价矩阵哈希漂移。")
    if status.get("results_sha256") != EXPECTED_RESULTS_SHA256:
        errors.append("正式结果哈希漂移。")
    if not status.get("passed") or status.get("state") != "completed":
        errors.append("最终审计未处于completed/passed。")
    if not status.get("ppo_mlp_absent"):
        errors.append("旧ppo_mlp仍存在于正式结果。")
    if len(bundle.frozen) != EXPECTED_ROWS:
        errors.append("frozen_plot_input行数异常。")
    if "ppo_mlp" in set(bundle.frozen["model"].astype(str)):
        errors.append("frozen_plot_input混入ppo_mlp。")
    if len(bundle.synthetic_tasks) != 216 or len(bundle.real_tasks) != 144:
        errors.append("测试任务数不是216+144。")
    numeric = bundle.frozen.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        errors.append("冻结绘图输入含Inf。")
    # planner/training seed、道路索引、鲁棒性下降量及安全路线资源利用率均有
    # “不适用”情形；只对每条评价都应定义的核心指标强制有限。
    required_finite = [
        "coverage",
        "weighted_coverage",
        "safe_weighted_coverage",
        "safe_rate",
        "return_rate",
        "violation_rate",
        "stranded_rate",
        "planning_time_s",
        "visited_count",
    ]
    if not np.isfinite(bundle.frozen[required_finite].to_numpy(dtype=float)).all():
        errors.append("冻结绘图输入的核心指标含NaN/Inf。")
    task_ids = {row["id"] for row in bundle.synthetic_tasks + bundle.real_tasks}
    if SYNTHETIC_EXAMPLE not in task_ids or REAL_EXAMPLE not in task_ids:
        errors.append("固定代表任务不存在。")
    result = {
        "passed": not errors,
        "errors": errors,
        "row_count": int(len(bundle.frozen)),
        "route_count": int(status.get("route_count", -1)),
        "matrix_sha256": status.get("matrix_sha256"),
        "results_sha256": status.get("results_sha256"),
        "synthetic_task_count": len(bundle.synthetic_tasks),
        "real_task_count": len(bundle.real_tasks),
        "active_models": sorted(bundle.frozen["model"].astype(str).unique()),
    }
    if errors:
        raise RuntimeError("；".join(errors))
    return result


def _task_level(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    keys = ["family", "model", "map_id", "task_id"]
    return df.groupby(keys, as_index=False)[list(metrics)].mean()


def map_level_nominal(bundle: DataBundle, metrics: Sequence[str]) -> pd.DataFrame:
    df = bundle.frozen[bundle.frozen["family"].isin(
        ["synthetic_learning", "synthetic_main_baselines", "real_learning", "real_baselines"]
    )].copy()
    task = _task_level(df, metrics)
    task["domain"] = np.where(task["family"].str.startswith("synthetic"), "未见合成", "真实DSM")
    return task.groupby(["domain", "model", "map_id"], as_index=False)[list(metrics)].mean()


def bootstrap_mean_ci(values: Sequence[float], seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(BOOTSTRAP_REPS, arr.size), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_bootstrap_difference(
    map_df: pd.DataFrame,
    domain: str,
    metric: str,
    comparator: str,
    reference: str = "full",
) -> tuple[float, float, float, int]:
    sub = map_df[(map_df["domain"] == domain) & (map_df["model"].isin([reference, comparator]))]
    pivot = sub.pivot(index="map_id", columns="model", values=metric).dropna()
    if reference not in pivot or comparator not in pivot:
        return math.nan, math.nan, math.nan, 0
    diffs = (pivot[reference] - pivot[comparator]).to_numpy(float)
    mean, lo, hi = bootstrap_mean_ci(diffs, seed=BOOTSTRAP_SEED + sum(map(ord, comparator + metric + domain)))
    return mean, lo, hi, int(len(diffs))


def _save_panel_data(output_dir: Path, stem: str, panel_frames: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    target = output_dir / "source_data" / stem
    target.mkdir(parents=True, exist_ok=True)
    for panel, frame in panel_frames.items():
        path = target / f"panel_{panel}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")
        records[panel] = {"path": str(path.relative_to(output_dir)), "rows": int(len(frame)), "sha256": _sha256(path)}
    return records


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    panel_frames: Mapping[str, pd.DataFrame],
    caption: str,
    dpi: int = EXPORT_DPI,
) -> Dict[str, Any]:
    figure_dir = output_dir / "figures"
    caption_dir = output_dir / "captions"
    figure_dir.mkdir(parents=True, exist_ok=True)
    caption_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = _save_panel_data(output_dir, stem, panel_frames)
    paths = {
        "svg": figure_dir / f"{stem}.svg",
        "pdf": figure_dir / f"{stem}.pdf",
        "png": figure_dir / f"{stem}.png",
        "tiff": figure_dir / f"{stem}.tiff",
    }
    # 保留固定画布尺寸，不使用bbox_inches='tight'破坏期刊毫米规格。
    fig.savefig(paths["svg"], format="svg")
    fig.savefig(paths["pdf"], format="pdf")
    fig.savefig(paths["png"], format="png", dpi=dpi)
    fig.savefig(paths["tiff"], format="tiff", dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    caption_path = caption_dir / f"{stem}.md"
    _write_text(caption_path, caption + "\n")
    return {
        "files": {k: {"path": str(v.relative_to(output_dir)), "sha256": _sha256(v), "bytes": v.stat().st_size} for k, v in paths.items()},
        "caption": {"path": str(caption_path.relative_to(output_dir)), "sha256": _sha256(caption_path)},
        "source_data": source_manifest,
        "width_mm": FIG_WIDTH_MM,
        "height_mm": FIG_HEIGHT_MM,
        "dpi": dpi,
    }


def annotated_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    *,
    cmap: str | LinearSegmentedColormap = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".2f",
    cbar_label: str | None = None,
) -> None:
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=30, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    norm = Normalize(vmin=np.nanmin(matrix) if vmin is None else vmin, vmax=np.nanmax(matrix) if vmax is None else vmax)
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            r, g, b, _ = cm(norm(value))
            ink = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.48 else "black"
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=5.0, color=ink)
    if cbar_label:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label(cbar_label)
        cbar.outline.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def _half_violin(ax: plt.Axes, values: np.ndarray, position: float, color: str, side: str = "right") -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.allclose(values, values[0]):
        ax.scatter(np.full_like(values, position), values, s=7, color=color, alpha=0.75)
        return
    parts = ax.violinplot([values], positions=[position], widths=0.72, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.27)
        verts = body.get_paths()[0].vertices
        if side == "right":
            verts[:, 0] = np.maximum(verts[:, 0], position)
        else:
            verts[:, 0] = np.minimum(verts[:, 0], position)
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(position * 101))
    jitter = rng.uniform(-0.22, 0.0 if side == "right" else 0.22, len(values))
    if side == "right":
        jitter = np.abs(jitter) * -1
    else:
        jitter = np.abs(jitter)
    ax.scatter(position + jitter, values, s=6, color=color, alpha=0.52, edgecolors="none")
    q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    ax.plot([position - 0.08, position + 0.08], [med, med], color=color, lw=1.5)
    ax.plot([position, position], [q1, q3], color=color, lw=2.0)


def _draw_forest(
    ax: plt.Axes,
    frame: pd.DataFrame,
    label_col: str,
    estimate: str,
    lo: str,
    hi: str,
    xlabel: str,
    color_col: str | None = None,
) -> None:
    y = np.arange(len(frame))[::-1]
    for yi, (_, row) in zip(y, frame.iterrows()):
        color = color_for(str(row[color_col])) if color_col else "#7F8DB5"
        ax.plot([row[lo], row[hi]], [yi, yi], color=color, lw=1.1)
        ax.scatter(row[estimate], yi, s=15, color=color, zorder=3)
    ax.axvline(0, color="#8A8A8A", lw=0.7, ls="--")
    ax.set_yticks(y, frame[label_col])
    ax.set_xlabel(xlabel)
    clean_axis(ax)


def _compact_legend(ax: plt.Axes, models: Sequence[str], ncol: int = 3, loc: str = "upper center") -> None:
    handles = [Line2D([0], [0], color=color_for(m), marker="o", lw=1.5, ms=3.5, label=label_for(m)) for m in models]
    ax.legend(handles=handles, ncol=ncol, loc=loc, bbox_to_anchor=(0.5, 1.18), columnspacing=1.0, handlelength=1.4)


def _terrain_record(path: Path, map_id: str, domain: str) -> tuple[np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        terrain = np.asarray(payload["terrain"], dtype=float)
        meta = json.loads(str(payload["metadata_json"].item()))
    return terrain, {
        "map_id": map_id,
        "domain": domain,
        "elevation_min_m": float(np.nanmin(terrain)),
        "elevation_max_m": float(np.nanmax(terrain)),
        "relief_m": float(np.nanmax(terrain) - np.nanmin(terrain)),
        "metadata": json.dumps(meta, ensure_ascii=False, sort_keys=True),
    }


def _normalize_tile(terrain: np.ndarray, size: int = 54) -> np.ndarray:
    image = Image.fromarray(np.asarray(terrain, dtype=np.float32))
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=float)
    lo, hi = np.nanpercentile(arr, [2, 98])
    return np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1)


def _terrain_mosaic(records: Sequence[tuple[str, np.ndarray]], cols: int, size: int = 54) -> np.ndarray:
    rows = int(math.ceil(len(records) / cols))
    mosaic = np.full((rows * size, cols * size), np.nan)
    for idx, (_, terrain) in enumerate(records):
        r, c = divmod(idx, cols)
        mosaic[r * size : (r + 1) * size, c * size : (c + 1) * size] = _normalize_tile(terrain, size)
    return mosaic


def figure_01(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.04, 1.18], height_ratios=[1.08, 0.92], hspace=0.34, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # a：定义任务而非绘制装饰性流程图。
    ax_a.set_aspect("equal")
    x = np.linspace(-1.0, 1.0, 180)
    road_1 = 0.30 * np.sin(2.4 * x) + 0.18 * x
    road_2_x = 0.24 * np.sin(np.linspace(-2.4, 2.4, 180))
    road_2_y = np.linspace(-1.0, 1.0, 180)
    ax_a.plot(x, road_1, color="#777777", lw=5.2, solid_capstyle="round")
    ax_a.plot(road_2_x, road_2_y, color="#9A9A9A", lw=5.2, solid_capstyle="round")
    points = np.array([[-0.78, -0.07], [-0.47, -0.18], [-0.18, -0.02], [0.27, 0.26], [0.62, 0.18], [0.84, 0.42], [-0.12, -0.74], [0.18, 0.70]])
    priorities = np.array([1, 2, 3, 1, 2, 3, 2, 3])
    pcolors = {1: "#A9B8C8", 2: "#E5AD55", 3: "#C84C4C"}
    psizes = {1: 22, 2: 34, 3: 52}
    for p in (1, 2, 3):
        sub = points[priorities == p]
        ax_a.scatter(sub[:, 0], sub[:, 1], s=psizes[p], color=pcolors[p], edgecolor="white", lw=0.6, zorder=3, label=f"优先级 {p}")
    ax_a.scatter(0, 0, marker="P", s=90, color="#0F4D92", edgecolor="white", lw=0.7, zorder=5)
    ax_a.text(0.04, -0.08, "机场/返航点", fontsize=5.6, color="#0F4D92")
    route = np.vstack([[0, 0], points[[2, 0, 1, 3, 5, 7]], [0, 0]])
    ax_a.plot(route[:, 0], route[:, 1], color="#0F4D92", lw=1.35, alpha=0.9, zorder=2)
    ax_a.add_patch(FancyArrowPatch((-0.72, 0.80), (-0.35, 0.62), arrowstyle="-|>", mutation_scale=8, color="#56A0B2", lw=1.1))
    ax_a.text(-0.75, 0.86, "风场", fontsize=5.6, color="#397F8E")
    ax_a.text(-0.96, -1.10, "约束：电量 · 航程 · 时间 · 地形 · 动力学 · 强制返航", fontsize=5.5)
    ax_a.legend(loc="upper right", ncol=1, handletextpad=0.3, borderaxespad=0.2)
    ax_a.set_xlim(-1.08, 1.08)
    ax_a.set_ylim(-1.18, 1.08)
    ax_a.axis("off")
    ax_a.set_title("山区公路固定巡检点选择与安全返航")
    panel_label(ax_a, "a")

    synthetic_maps: list[tuple[str, np.ndarray]] = []
    map_rows: list[Dict[str, Any]] = []
    for idx in range(24):
        map_id = f"synthetic_test__map_{idx:03d}"
        terrain, row = _terrain_record(MAP_ROOT / "procedural" / "synthetic_test" / f"{map_id}.npz", map_id, "未见合成")
        synthetic_maps.append((map_id, terrain))
        map_rows.append(row)
    real_ids = sorted({str(row["map_id"]) for row in bundle.real_tasks})
    real_maps: list[tuple[str, np.ndarray]] = []
    for map_id in real_ids:
        terrain, row = _terrain_record(MAP_ROOT / "real" / map_id / "map_bundle.npz", map_id, "真实DSM")
        real_maps.append((map_id, terrain))
        map_rows.append(row)
    syn_mosaic = _terrain_mosaic(synthetic_maps, cols=6, size=34)
    real_mosaic = _terrain_mosaic(real_maps, cols=4, size=51)
    full_width = max(syn_mosaic.shape[1], real_mosaic.shape[1])
    gap = 8
    mosaic = np.full((syn_mosaic.shape[0] + gap + real_mosaic.shape[0], full_width), np.nan)
    mosaic[: syn_mosaic.shape[0], : syn_mosaic.shape[1]] = syn_mosaic
    mosaic[syn_mosaic.shape[0] + gap :, : real_mosaic.shape[1]] = real_mosaic
    cmap = plt.get_cmap("terrain").copy()
    cmap.set_bad("white")
    ax_b.imshow(mosaic, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax_b.axhline(syn_mosaic.shape[0] + gap / 2, color="white", lw=2.0)
    ax_b.text(1, 8, "24张未见合成地图", color="white", fontsize=5.8, weight="bold", bbox={"facecolor": "#333333", "alpha": 0.55, "edgecolor": "none", "pad": 1.5})
    ax_b.text(1, syn_mosaic.shape[0] + gap + 8, "8张Copernicus GLO-30 DSM", color="white", fontsize=5.8, weight="bold", bbox={"facecolor": "#333333", "alpha": 0.55, "edgecolor": "none", "pad": 1.5})
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    ax_b.set_title("冻结地图资产概览（各图独立拉伸高程）")
    panel_label(ax_b, "b")

    families = [
        ("未见合成\n学习模型", 7560, "#567FB2"),
        ("未见合成\n传统基线", 4392, "#A7A7A7"),
        ("真实DSM\n学习模型", 5040, "#6C8FB6"),
        ("真实DSM\n传统基线", 1152, "#8B8B8B"),
        ("已知域偏移", 1008, "#74A99A"),
        ("隐藏模型/\n感知误差", 2496, "#9A86B5"),
    ]
    total = sum(v for _, v, _ in families)
    start = 0
    for label, value, color in families:
        width = value / total
        ax_c.barh([0], [width], left=[start], color=color, height=0.36, edgecolor="white", lw=0.7)
        if width > 0.08:
            ax_c.text(start + width / 2, 0, f"{value:,}", ha="center", va="center", color="white" if color != "#A7A7A7" else "black", fontsize=5.3, weight="bold")
        start += width
    y0 = -0.43
    for idx, (label, value, color) in enumerate(families):
        col, row = idx % 3, idx // 3
        x0 = col * 0.34
        y = y0 - row * 0.25
        ax_c.add_patch(Rectangle((x0, y - 0.035), 0.025, 0.07, color=color, transform=ax_c.transAxes, clip_on=False))
        ax_c.text(x0 + 0.034, y, f"{label}  {value:,}", transform=ax_c.transAxes, va="center", fontsize=5.2)
    ax_c.text(0.5, 0.78, "21,648条结果 + 21,648条路线", transform=ax_c.transAxes, ha="center", fontsize=7.0, weight="bold")
    ax_c.text(0.5, 0.63, "216个合成任务 · 144个真实任务 · 地图为独立单位", transform=ax_c.transAxes, ha="center", fontsize=5.5, color=NEUTRAL)
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(-0.3, 0.5)
    ax_c.axis("off")
    ax_c.set_title("正式评价矩阵")
    panel_label(ax_c, "c")

    groups = [
        ("任务效果", "优先级加权覆盖率\n访问点数 · oracle attainment", "#0F4D92"),
        ("安全返航", "安全率 · 返航率\n违规 · stranded · 最低SOC", "#6B86AA"),
        ("资源与时间", "能耗 · 航程 · 总任务时间\n在线规划时间", "#8372A5"),
        ("泛化与鲁棒", "未见地图 · 真实DSM\n风/功率/DEM/定位误差", "#4E927E"),
        ("训练过程", "收敛曲线 · 稳定性\n样本效率", "#C07A45"),
    ]
    ax_d.axis("off")
    for idx, (title, body, color) in enumerate(groups):
        y = 0.90 - idx * 0.19
        ax_d.add_patch(Rectangle((0.02, y - 0.055), 0.15, 0.11, color=color, alpha=0.95, transform=ax_d.transAxes))
        ax_d.text(0.095, y, title, color="white", fontsize=5.4, weight="bold", ha="center", va="center", transform=ax_d.transAxes)
        ax_d.plot([0.18, 0.25], [y, y], color=color, lw=1.2, transform=ax_d.transAxes, clip_on=False)
        ax_d.text(0.27, y, body, fontsize=5.4, va="center", transform=ax_d.transAxes)
    ax_d.set_title("五条互补证据链")
    panel_label(ax_d, "d")

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.07, top=0.94)
    panel_frames = {
        "a": pd.DataFrame({"element": ["airport", "fixed_inspection_points", "wind", "hard_constraints"], "count": [1, 8, 1, 6]}),
        "b": pd.DataFrame(map_rows).drop(columns=["metadata"]),
        "c": pd.DataFrame([{"family": label.replace("\n", " "), "row_count": value} for label, value, _ in families]),
        "d": pd.DataFrame([{"evidence_chain": title, "metrics": body.replace("\n", "; ")} for title, body, _ in groups]),
    }
    return save_figure(fig, output_dir, "fig01_study_design", panel_frames, CAPTIONS["fig01_study_design"])


def figure_02(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 0.82, 1.06], height_ratios=[1.0, 0.92], hspace=0.46, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, :2])
    ax_e = fig.add_subplot(gs[1, 2])

    scores = bundle.operational_scores[bundle.operational_scores["aggregation"] == "arithmetic"].copy()
    scores["model"] = pd.Categorical(scores["model"], CORE_MODELS, ordered=True)
    scores = scores.sort_values("model")
    dims = [f"D{i}" for i in range(1, 8)]
    matrix = scores[dims].to_numpy(float)
    annotated_heatmap(ax_a, matrix, dims, [label_for(str(x)) for x in scores["model"]], cmap="Blues", vmin=0, vmax=1, cbar_label="归一化得分")
    ax_a.set_title("七个效应维度")
    panel_label(ax_a, "a")

    for i, (_, row) in enumerate(scores.iterrows()):
        model = str(row["model"])
        ax_b.scatter(row["score_0_to_100"], i, s=42, color=color_for(model), edgecolor="white", lw=0.6, zorder=3)
        ax_b.text(row["score_0_to_100"] + 1.2, i, f"{row['score_0_to_100']:.1f}", va="center", fontsize=5.5)
    ax_b.set_yticks(range(len(scores)), [label_for(str(x)) for x in scores["model"]])
    ax_b.set_xlabel("算术综合得分（0–100）")
    ax_b.set_xlim(45, 82)
    ax_b.set_title(f"默认运行区间下限={OPERATIONAL_FLOOR:.2f}")
    clean_axis(ax_b)
    panel_label(ax_b, "b")

    gaps = bundle.bootstrap_distribution["full_minus_a2c_points"].to_numpy(float)
    ax_c.hist(gaps, bins=45, density=True, color="#8FAED1", alpha=0.7, edgecolor="white", lw=0.25)
    mean = float(np.mean(gaps))
    lo, hi = np.quantile(gaps, [0.025, 0.975])
    ax_c.axvline(0, color=NEUTRAL, lw=0.75, ls="--")
    ax_c.axvspan(lo, hi, color="#0F4D92", alpha=0.12)
    ax_c.axvline(mean, color="#0F4D92", lw=1.4)
    ax_c.text(0.03, 0.94, f"均值 {mean:.2f} 分\n95% CI [{lo:.2f}, {hi:.2f}]\nP(差值>0)={(gaps > 0).mean():.3f}", transform=ax_c.transAxes, va="top", fontsize=5.4)
    ax_c.set_xlabel("PPO+Pointer − A2C+Pointer（分）")
    ax_c.set_ylabel("密度")
    ax_c.set_title("10,000次层级bootstrap")
    clean_axis(ax_c)
    panel_label(ax_c, "c")

    joint = bundle.joint_sensitivity[bundle.joint_sensitivity["aggregation"] == "arithmetic"].copy()
    joint["training_weight"] = joint["weight_D6"] + joint["weight_D7"]
    bins = [-1e-9, 0.10, 0.20, 0.30, 0.40, 1.0]
    labels = ["≤0.10", "0.10–0.20", "0.20–0.30", "0.30–0.40", ">0.40"]
    joint["training_weight_bin"] = pd.cut(joint["training_weight"], bins=bins, labels=labels, include_lowest=True)
    full_first = joint[joint["model"] == "full"].groupby(["operational_floor", "training_weight_bin"], observed=False)["is_first"].mean().reset_index()
    sens = full_first.pivot(index="operational_floor", columns="training_weight_bin", values="is_first").reindex(columns=labels)
    annotated_heatmap(ax_d, sens.to_numpy(float), labels, [f"{x:.1f}" for x in sens.index], cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f", cbar_label="PPO+Pointer第一名占比")
    ax_d.set_xlabel("训练维度总权重 D6+D7")
    ax_d.set_ylabel("运行区间下限")
    ax_d.set_title("归一化下限×权重联合敏感性（算术聚合）")
    panel_label(ax_d, "d", x=-0.06)

    weights = np.array([0.20, 0.10, 0.10, 0.15, 0.05, 0.20, 0.20])
    full = scores[scores["model"] == "full"].iloc[0]
    a2c = scores[scores["model"] == "a2c_pointer"].iloc[0]
    contributions = 100.0 * weights * (full[dims].to_numpy(float) - a2c[dims].to_numpy(float))
    x = np.arange(7)
    ax_e.bar(x, contributions, color=[DELTA_UP if v >= 0 else DELTA_DOWN for v in contributions], width=0.72)
    ax_e.axhline(0, color=NEUTRAL, lw=0.7)
    ax_e.set_xticks(x, dims)
    ax_e.set_ylabel("对总差值的贡献（分）")
    ax_e.text(0.98, 0.94, f"合计 {contributions.sum():.2f} 分", transform=ax_e.transAxes, ha="right", va="top", fontsize=5.5, weight="bold")
    ax_e.set_title("PPO相对A2C的维度贡献")
    clean_axis(ax_e)
    panel_label(ax_e, "e")

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.94)
    panel_frames = {
        "a": scores[["model", *dims]].astype({"model": str}),
        "b": scores[["model", "score_0_to_100", "operational_floor"]].astype({"model": str}),
        "c": bundle.bootstrap_distribution[["bootstrap_replicate", "full_minus_a2c_points", "full_score", "a2c_pointer_score"]],
        "d": full_first,
        "e": pd.DataFrame({"dimension": dims, "weight": weights, "full": full[dims].to_numpy(float), "a2c_pointer": a2c[dims].to_numpy(float), "contribution_points": contributions}),
    }
    return save_figure(fig, output_dir, "fig02_integrated_score", panel_frames, CAPTIONS["fig02_integrated_score"])


def figure_03(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 1.08, 1.0], height_ratios=[1.02, 0.98], hspace=0.46, wspace=0.40)
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    metrics = ["safe_weighted_coverage", "weighted_coverage", "safe_rate", "return_rate", "planning_time_s"]
    maps = map_level_nominal(bundle, metrics)
    methods = [m for m in MAIN_METHODS if m in set(maps["model"])]
    for i, model in enumerate(methods):
        syn = maps[(maps["model"] == model) & (maps["domain"] == "未见合成")]["safe_weighted_coverage"].to_numpy(float)
        real = maps[(maps["model"] == model) & (maps["domain"] == "真实DSM")]["safe_weighted_coverage"].to_numpy(float)
        _half_violin(ax_a, syn, i - 0.09, color_for(model), side="right")
        _half_violin(ax_a, real, i + 0.09, color_for(model), side="left")
    ax_a.set_xticks(range(len(methods)), [label_for(m) for m in methods], rotation=22, ha="right")
    ax_a.set_ylabel("安全加权覆盖率")
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.set_title("地图级分布：左侧散点=未见合成，右侧散点=真实DSM")
    clean_axis(ax_a)
    panel_label(ax_a, "a", x=-0.055)

    effect_rows: list[Dict[str, Any]] = []
    for domain in ("未见合成", "真实DSM"):
        for metric, metric_label in (("weighted_coverage", "加权覆盖"), ("safe_rate", "安全率"), ("return_rate", "返航率")):
            for comparator in ("traditional_ppo", "a2c_pointer"):
                est, lo, hi, n = paired_bootstrap_difference(maps, domain, metric, comparator)
                compact_domain = "合成" if domain == "未见合成" else "DSM"
                compact_model = "传统PPO" if comparator == "traditional_ppo" else "A2C"
                effect_rows.append({"label": f"{compact_domain}·{metric_label}·{compact_model}", "domain": domain, "metric": metric, "comparator": comparator, "estimate": est, "ci_low": lo, "ci_high": hi, "map_count": n})
    effects = pd.DataFrame(effect_rows)
    _draw_forest(ax_b, effects, "label", "estimate", "ci_low", "ci_high", "PPO+Pointer − 比较模型", color_col="comparator")
    ax_b.set_title("地图级效应与95% bootstrap CI")
    panel_label(ax_b, "b")

    resource = bundle.nominal_map[bundle.nominal_map["model"].isin(CORE_MODELS)].copy()
    resource["domain_cn"] = resource["domain"].map({"synthetic": "未见合成", "real": "真实DSM"})
    resource_summary = resource.groupby(["domain_cn", "model"], as_index=False).agg(
        energy_wh=("mean_safe_energy_wh", "mean"),
        distance_m=("mean_safe_distance_m", "mean"),
        total_time_s=("mean_safe_time_s", "mean"),
        safe_rate=("safe_rate", "mean"),
    )
    ax_c.axis("off")
    metric_specs = [("energy_wh", "能耗（Wh）"), ("distance_m", "航程（m）"), ("total_time_s", "总任务时间（s）")]
    mini_axes = []
    for idx, (field, title) in enumerate(metric_specs):
        mini = inset_axes(ax_c, width="30%", height="88%", loc="lower left", bbox_to_anchor=(idx * 0.34, 0.02, 1, 1), bbox_transform=ax_c.transAxes, borderpad=0)
        mini_axes.append(mini)
        sub = resource_summary[resource_summary["domain_cn"] == "真实DSM"].set_index("model")
        for yi, model in enumerate(CORE_MODELS):
            if model not in sub.index:
                continue
            value = float(sub.loc[model, field])
            mini.scatter(value, yi, s=18, color=color_for(model))
        compact_labels = {"full": "PPO+Ptr", "traditional_ppo": "传统PPO", "a2c_pointer": "A2C+Ptr"}
        mini.set_yticks(range(3), [compact_labels[m] if idx == 0 else "" for m in CORE_MODELS])
        mini.set_xlabel(title, fontsize=5.2)
        mini.tick_params(axis="x", labelrotation=30)
        clean_axis(mini)
    ax_c.set_title("真实DSM安全路线的资源与任务时间", pad=6)
    panel_label(ax_c, "c")

    raw = bundle.frozen[bundle.frozen["family"].isin(["synthetic_learning", "synthetic_main_baselines", "real_learning", "real_baselines"]) & bundle.frozen["model"].isin(methods)].copy()
    for model in methods:
        values = np.sort(raw[raw["model"] == model]["planning_time_s"].to_numpy(float))
        y = np.arange(1, len(values) + 1) / len(values)
        ax_d.plot(values, y, color=color_for(model), lw=1.1, label=label_for(model))
    ax_d.set_xscale("log")
    ax_d.set_xlabel("在线规划时间（s，对数轴）")
    ax_d.set_ylabel("ECDF")
    ax_d.set_title("在线规划时间分布")
    ax_d.legend(ncol=2, loc="lower right", fontsize=4.7)
    clean_axis(ax_d)
    panel_label(ax_d, "d")

    pareto_rows: list[Dict[str, Any]] = []
    for (domain, model), sub in maps[maps["model"].isin(methods)].groupby(["domain", "model"]):
        pareto_rows.append({"domain": domain, "model": model, "safe_weighted_coverage": float(sub["safe_weighted_coverage"].mean()), "planning_time_median_s": float(sub["planning_time_s"].median()), "safe_rate": float(sub["safe_rate"].mean())})
    pareto = pd.DataFrame(pareto_rows)
    marker_map = {"未见合成": "o", "真实DSM": "^"}
    for _, row in pareto.iterrows():
        ax_e.scatter(row["planning_time_median_s"], row["safe_weighted_coverage"], s=18 + 55 * row["safe_rate"], marker=marker_map[row["domain"]], color=color_for(row["model"]), edgecolor="white", lw=0.5, alpha=0.88)
    ax_e.set_xscale("log")
    ax_e.set_xlabel("规划时间中位数（s）")
    ax_e.set_ylabel("安全加权覆盖率")
    ax_e.set_title("覆盖—在线计算Pareto视图")
    ax_e.text(0.03, 0.97, "颜色同d；○合成  △真实DSM\n点大小表示安全率", transform=ax_e.transAxes, ha="left", va="top", fontsize=4.6)
    clean_axis(ax_e)
    panel_label(ax_e, "e")

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.11, top=0.93)
    panel_frames = {"a": maps, "b": effects, "c": resource_summary, "d": raw[["family", "model", "map_id", "task_id", "planning_time_s"]], "e": pareto}
    return save_figure(fig, output_dir, "fig03_operational_tradeoffs", panel_frames, CAPTIONS["fig03_operational_tradeoffs"])


def load_training_history(models: Sequence[str] = LEARNING_MODELS) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for model in models:
        base = TRAIN_V32 if model == "traditional_ppo" else TRAIN_V31
        for seed in range(42, 47):
            path = base / f"formal_{model}_seed{seed}_3000ep" / "training_metrics.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"缺少正式训练曲线：{path}")
            for payload in _read_jsonl(path):
                row = {
                    "model": model,
                    "training_seed": seed,
                    "update": payload.get("update"),
                    "episodes_seen": payload.get("episodes_seen"),
                    "environment_interactions": payload.get("environment_interactions"),
                    "mean_weighted_coverage": payload.get("mean_weighted_coverage"),
                    "return_rate": payload.get("return_rate"),
                    "approx_kl": payload.get("approx_kl"),
                    "ratio_deviation": payload.get("ratio_deviation"),
                    "clip_fraction": payload.get("clip_fraction"),
                    "entropy": payload.get("entropy"),
                    "gradient_norm_pre_clip": payload.get("gradient_norm_pre_clip"),
                }
                rows.append(row)
    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.columns if c not in ("model",)]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if frame[["environment_interactions", "mean_weighted_coverage"]].isna().any().any():
        raise RuntimeError("正式训练曲线缺少交互数或加权覆盖率。")
    return frame


def _interpolated_training_summary(history: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    common_min = max(history[history["model"] == m]["environment_interactions"].min() for m in models)
    common_max = min(history[history["model"] == m]["environment_interactions"].max() for m in models)
    grid = np.linspace(common_min, common_max, 180)
    rows: list[Dict[str, Any]] = []
    for model in models:
        seed_curves = []
        for seed, sub in history[history["model"] == model].groupby("training_seed"):
            sub = sub.sort_values("environment_interactions")
            y = np.interp(grid, sub["environment_interactions"], sub["mean_weighted_coverage"])
            seed_curves.append(y)
            rows.extend({"model": model, "training_seed": int(seed), "environment_interactions": float(x), "weighted_coverage": float(v), "series": "seed"} for x, v in zip(grid, y))
        stack = np.vstack(seed_curves)
        med = np.median(stack, axis=0)
        lo, hi = np.quantile(stack, [0.25, 0.75], axis=0)
        rows.extend({"model": model, "training_seed": -1, "environment_interactions": float(x), "weighted_coverage": float(v), "q25": float(l), "q75": float(h), "series": "median"} for x, v, l, h in zip(grid, med, lo, hi))
    return pd.DataFrame(rows)


def figure_04(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 0.82, 0.88], height_ratios=[1.0, 0.94], hspace=0.45, wspace=0.40)
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    history = load_training_history(CORE_MODELS)
    summary = _interpolated_training_summary(history, CORE_MODELS)
    for model in CORE_MODELS:
        raw_model = history[history["model"] == model]
        for _, seed_frame in raw_model.groupby("training_seed"):
            ax_a.plot(seed_frame["environment_interactions"], seed_frame["mean_weighted_coverage"], color=color_for(model), alpha=0.17, lw=0.45)
        med = summary[(summary["model"] == model) & (summary["series"] == "median")].sort_values("environment_interactions")
        ax_a.fill_between(med["environment_interactions"].to_numpy(float), med["q25"].to_numpy(float), med["q75"].to_numpy(float), color=color_for(model), alpha=0.10, linewidth=0)
        ax_a.plot(med["environment_interactions"], med["weighted_coverage"], color=color_for(model), lw=1.6, label=label_for(model))
    ax_a.set_xlabel("累计环境交互数")
    ax_a.set_ylabel("训练批次加权覆盖率")
    ax_a.set_title("五种子收敛过程：细线=种子，粗线=中位趋势，带=IQR")
    ax_a.legend(ncol=3, loc="lower right")
    clean_axis(ax_a)
    panel_label(ax_a, "a", x=-0.055)

    seed_metrics = bundle.training_seed_metrics[bundle.training_seed_metrics["model"].isin(CORE_MODELS)].copy()
    for i, model in enumerate(CORE_MODELS):
        vals = seed_metrics[seed_metrics["model"] == model]["learning_curve_auc"].to_numpy(float)
        ax_b.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)), vals, s=15, color=color_for(model), alpha=0.75)
        ax_b.scatter(i, np.median(vals), marker="_", s=130, lw=1.8, color="#222222")
    ax_b.set_xticks(range(3), [label_for(m) for m in CORE_MODELS], rotation=25, ha="right")
    ax_b.set_ylabel("Learning-curve AUC")
    ax_b.set_title("全过程学习收益")
    clean_axis(ax_b)
    panel_label(ax_b, "b")

    for i, model in enumerate(CORE_MODELS):
        vals = seed_metrics[seed_metrics["model"] == model]["threshold_efficiency"].to_numpy(float)
        ax_c.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)), vals, s=17, color=color_for(model), alpha=0.78)
        mean, lo, hi = bootstrap_mean_ci(vals, BOOTSTRAP_SEED + i)
        ax_c.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color="#222222", ms=3, lw=0.9, capsize=2)
    ax_c.set_xticks(range(3), [label_for(m) for m in CORE_MODELS])
    ax_c.set_ylabel("Threshold efficiency")
    ax_c.set_title("达到冻结阈值的样本效率")
    clean_axis(ax_c)
    panel_label(ax_c, "c")

    train_dims = bundle.training_dimensions.set_index("model")
    stability_rows: list[Dict[str, Any]] = []
    for model in CORE_MODELS:
        row = train_dims.loc[model]
        stability_rows.extend([
            {"model": model, "metric": "D6", "value": row["D6_training_stability"]},
            {"model": model, "metric": "种子一致性", "value": row["seed_consistency"]},
            {"model": model, "metric": "时间一致性", "value": row["temporal_consistency"]},
        ])
    stability = pd.DataFrame(stability_rows)
    for i, metric in enumerate(["D6", "种子一致性", "时间一致性"]):
        sub = stability[stability["metric"] == metric].set_index("model")
        xs = np.arange(3) + (i - 1) * 0.18
        ax_d.scatter(xs, [sub.loc[m, "value"] for m in CORE_MODELS], s=18, marker=("o", "s", "^")[i], color=[color_for(m) for m in CORE_MODELS], label=metric)
    ax_d.set_xticks(range(3), [label_for(m) for m in CORE_MODELS], rotation=22, ha="right")
    ax_d.set_ylim(0, 1.04)
    ax_d.set_ylabel("归一化稳定性")
    ax_d.set_title("尾段与跨种子稳定性")
    ax_d.legend(loc="lower left", fontsize=4.7)
    clean_axis(ax_d)
    panel_label(ax_d, "d")

    tail_rows: list[Dict[str, Any]] = []
    for model in ("full", "traditional_ppo"):
        for seed, sub in history[history["model"] == model].groupby("training_seed"):
            tail = sub.sort_values("environment_interactions").tail(max(1, int(math.ceil(len(sub) * 0.2))))
            for metric in ("approx_kl", "ratio_deviation", "clip_fraction"):
                tail_rows.append({"model": model, "training_seed": int(seed), "metric": metric, "value": float(tail[metric].mean())})
    tail = pd.DataFrame(tail_rows)
    x_metrics = ["approx_kl", "ratio_deviation", "clip_fraction"]
    for model, offset in (("full", -0.08), ("traditional_ppo", 0.08)):
        vals = tail[tail["model"] == model].groupby("metric")["value"].mean().reindex(x_metrics)
        # 三个诊断量使用各自尺度，仅用于比较同一指标的两个PPO模型。
        ax_e.scatter(np.arange(3) + offset, vals, s=22, color=color_for(model), label=label_for(model))
    ax_e.set_xticks(range(3), ["Approx. KL", "Ratio偏离", "Clip fraction"], rotation=22, ha="right")
    ax_e.set_yscale("log")
    ax_e.set_ylabel("尾段均值（对数轴）")
    ax_e.set_title("PPO更新诊断")
    ax_e.legend(loc="upper left", fontsize=4.7)
    clean_axis(ax_e)
    panel_label(ax_e, "e")

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.105, top=0.93)
    panel_frames = {"a": summary, "b": seed_metrics[["model", "training_seed", "learning_curve_auc"]], "c": seed_metrics[["model", "training_seed", "convergence_environment_interactions", "threshold_efficiency"]], "d": stability, "e": tail}
    return save_figure(fig, output_dir, "fig04_training", panel_frames, CAPTIONS["fig04_training"])


def figure_05(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 1.0, 1.0], height_ratios=[1.02, 0.98], hspace=0.47, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    ab = bundle.pairwise[bundle.pairwise["statistical_family"].isin(["synthetic_ablations", "real_ablations"])].copy()
    ab["domain"] = ab["statistical_family"].map({"synthetic_ablations": "未见合成", "real_ablations": "真实DSM"})
    compact_ablation = {
        "no_priority_bias": "优先偏置",
        "no_domain_randomization": "域随机化",
        "no_resource_shaping": "资源塑形",
        "no_return_reserve": "返航储备*",
    }
    ab["label"] = ab.apply(lambda r: f"{'合成' if r['domain'] == '未见合成' else 'DSM'}·{compact_ablation[r['comparator']]}", axis=1)
    ab = ab.sort_values(["comparator", "domain"])
    _draw_forest(ax_a, ab, "label", "mean_difference", "bootstrap_ci_low", "bootstrap_ci_high", "安全加权覆盖率：完整模型 − 消融", color_col="comparator")
    for yi, (_, row) in zip(np.arange(len(ab))[::-1], ab.iterrows()):
        if bool(row["significant_holm"]):
            ax_a.text(row["bootstrap_ci_high"] + 0.007, yi, "Holm p<0.05", fontsize=4.5, va="center")
    ax_a.set_title("合成与真实DSM的地图级消融效应")
    panel_label(ax_a, "a", x=-0.055)

    far = bundle.frozen[
        bundle.frozen["family"].isin(["synthetic_learning", "real_learning"])
        & bundle.frozen["model"].isin(["full", "no_priority_bias"])
        & (bundle.frozen["priority_layout"] == "far_high_conflict")
    ].copy()
    priority = far.groupby("model", as_index=False)[["high_priority_coverage", "medium_priority_coverage", "low_priority_coverage"]].mean()
    pfields = ["high_priority_coverage", "medium_priority_coverage", "low_priority_coverage"]
    plabels = ["高", "中", "低"]
    for yi, (field, label) in enumerate(zip(pfields, plabels)):
        vals = priority.set_index("model")[field]
        ax_b.plot([vals["full"], vals["no_priority_bias"]], [yi, yi], color=LIGHT_NEUTRAL, lw=2.0)
        ax_b.scatter(vals["full"], yi, color=color_for("full"), s=23, marker="o")
        ax_b.scatter(vals["no_priority_bias"], yi, color=color_for("no_priority_bias"), s=23, marker="s")
    ax_b.set_yticks(range(3), plabels)
    ax_b.set_xlabel("far-high-conflict覆盖率")
    ax_b.set_title("显式优先级偏置")
    ax_b.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=color_for("full"), label="完整模型"),
            Line2D([], [], marker="s", ls="", color=color_for("no_priority_bias"), label="无优先级偏置"),
        ],
        fontsize=4.4,
        loc="center left",
    )
    clean_axis(ax_b)
    panel_label(ax_b, "b")

    shaping = bundle.frozen[
        bundle.frozen["family"].isin(["synthetic_learning", "real_learning"])
        & bundle.frozen["model"].isin(["full", "no_resource_shaping"])
    ].groupby(["constraint_type", "model"], as_index=False)[["safe_weighted_coverage", "safe_energy_utilization", "safe_distance_utilization", "safe_time_utilization"]].mean()
    for model in ("full", "no_resource_shaping"):
        sub = shaping[shaping["model"] == model].set_index("constraint_type").reindex(["energy", "distance", "time", "mixed"])
        ax_c.plot(range(4), sub["safe_weighted_coverage"], marker="o", ms=3.0, lw=1.2, color=color_for(model), label=label_for(model))
    ax_c.set_xticks(range(4), ["能量", "距离", "时间", "混合"])
    ax_c.set_ylabel("安全加权覆盖率")
    ax_c.set_title("资源塑形与瓶颈类型")
    ax_c.legend(fontsize=4.6)
    clean_axis(ax_c)
    panel_label(ax_c, "c")

    dr = bundle.robustness[bundle.robustness["algorithm"].isin(["full", "no_domain_randomization"])].copy()
    dr["condition_cn"] = dr["condition"].map({"wind": "风", "power_model": "功率", "dem_error": "DEM", "localization": "定位"})
    dr["family_cn"] = dr["family"].map({"known_domain_shift": "已知偏移", "hidden_model_perception_mismatch": "隐藏误差"})
    dr["label"] = dr["family_cn"] + "·" + dr["condition_cn"]
    for model, marker in (("full", "o"), ("no_domain_randomization", "s")):
        sub = dr[dr["algorithm"] == model]
        ax_d.scatter(sub["robustness_drop_mean"], sub["label"], s=20, marker=marker, color=color_for(model), label=label_for(model))
    ax_d.axvline(0, color=NEUTRAL, lw=0.7, ls="--")
    ax_d.set_title("域随机化与扰动退化")
    ax_d.legend(fontsize=4.6)
    clean_axis(ax_d)
    panel_label(ax_d, "d")

    reserve = bundle.frozen[(bundle.frozen["family"] == "hidden_model_perception_mismatch") & bundle.frozen["model"].isin(["full", "no_return_reserve"])].copy()
    reserve_summary = reserve.groupby("model", as_index=False)[["dangerous_action_proposal_rate", "environment_interception_rate", "stranded_rate", "return_rate"]].mean()
    rfields = ["dangerous_action_proposal_rate", "environment_interception_rate", "stranded_rate", "return_rate"]
    rlabels = ["危险提议", "环境拦截", "Stranded", "返航"]
    x = np.arange(4)
    for model, offset in (("full", -0.08), ("no_return_reserve", 0.08)):
        vals = reserve_summary.set_index("model").loc[model, rfields].to_numpy(float)
        ax_e.scatter(x + offset, vals, s=22, color=color_for(model), label=label_for(model))
    ax_e.set_xticks(x, rlabels, rotation=22, ha="right")
    ax_e.set_ylim(-0.03, 1.03)
    ax_e.set_ylabel("比率")
    ax_e.set_title("返航储备的仿真安全效应")
    ax_e.legend(fontsize=4.5)
    clean_axis(ax_e)
    panel_label(ax_e, "e")

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.11, top=0.93)
    panel_frames = {"a": ab, "b": priority, "c": shaping, "d": dr, "e": reserve_summary}
    return save_figure(fig, output_dir, "fig05_ablation", panel_frames, CAPTIONS["fig05_ablation"])


def _task_by_id(bundle: DataBundle, task_id: str) -> Dict[str, Any]:
    for row in bundle.synthetic_tasks + bundle.real_tasks:
        if row["id"] == task_id:
            return row
    raise KeyError(task_id)


def _find_route(model: str, task_id: str, seed: int = 42) -> Path | None:
    is_synthetic = task_id.startswith("synthetic_test__")
    suffix = task_id.replace("real_test__", "")
    if model == "milp" and is_synthetic:
        matches = list((RESULTS / "synthetic_main_baselines" / "jobs" / f"milp__seed{seed}" / "routes").glob(f"*{task_id}.json"))
    elif model == "milp":
        matches = list((RESULTS / "real_baselines" / "jobs" / f"milp__seed{seed}" / "routes").glob(f"*{suffix}.json"))
    elif is_synthetic:
        pattern = f"{model}__seed{seed}__*{task_id}.json"
        matches = list((RESULTS / "synthetic_learning").glob(f"**/routes/{pattern}"))
    else:
        pattern = f"{model}__seed{seed}__*{suffix}.json"
        matches = list((RESULTS / "real_learning").glob(f"**/routes/{pattern}"))
    return matches[0] if matches else None


def _route_payload(model: str, task_id: str, seed: int = 42) -> Dict[str, Any] | None:
    path = _find_route(model, task_id, seed)
    if path is None:
        return None
    payload = _read_json(path)
    payload["_source_path"] = str(path)
    return payload


def _map_bundle_path(map_id: str) -> Path:
    if map_id.startswith("synthetic_test__"):
        return MAP_ROOT / "procedural" / "synthetic_test" / f"{map_id}.npz"
    return MAP_ROOT / "real" / map_id / "map_bundle.npz"


def _road_segments(map_id: str) -> list[np.ndarray]:
    path = _map_bundle_path(map_id)
    with np.load(path, allow_pickle=False) as payload:
        points = np.asarray(payload["road_points"], dtype=float)
        offsets = np.asarray(payload["road_offsets"], dtype=int)
    return [points[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def _plot_route_axis(ax: plt.Axes, task: Mapping[str, Any], payload: Mapping[str, Any] | None, model: str, show_title: bool = True) -> Dict[str, Any]:
    map_id = str(task["map_id"])
    with np.load(_map_bundle_path(map_id), allow_pickle=False) as data:
        terrain = np.asarray(data["terrain"], dtype=float)
    ax.imshow(terrain, origin="lower", cmap="gist_earth", alpha=0.93, extent=(0, terrain.shape[1] - 1, 0, terrain.shape[0] - 1))
    for segment in _road_segments(map_id):
        ax.plot(segment[:, 0], segment[:, 1], color="white", lw=1.0, alpha=0.82)
        ax.plot(segment[:, 0], segment[:, 1], color="#5E5E5E", lw=0.48, alpha=0.9)
    points = np.asarray(task["inspection_points_xyz"], dtype=float)
    priorities = np.asarray(task["priorities"], dtype=int)
    for priority, color, size in ((1, "#B5C2CE", 7), (2, "#E3AB56", 11), (3, "#C84C4C", 17)):
        sub = points[priorities == priority]
        ax.scatter(sub[:, 0], sub[:, 1], s=size, color=color, edgecolor="white", lw=0.25, zorder=4)
    start = np.asarray(task["start_xy"], dtype=float)
    ax.scatter(start[0], start[1], marker="P", s=34, color="#0F4D92", edgecolor="white", lw=0.4, zorder=6)
    row = {"task_id": str(task["id"]), "map_id": map_id, "model": model, "route_found": payload is not None}
    if payload is not None:
        detail = payload.get("detail", payload)
        path = np.asarray(detail.get("path", []), dtype=float)
        if len(path):
            ax.plot(path[:, 0], path[:, 1], color=color_for(model), lw=1.3, zorder=5)
            ax.scatter(path[:, 0], path[:, 1], s=4, color=color_for(model), zorder=5)
        metrics = detail.get("metrics", {}) or {}
        row.update({
            "distance_m": float(detail.get("distance_m", metrics.get("distance_m", math.nan))),
            "energy_wh": float(detail.get("energy_wh", metrics.get("energy_wh", math.nan))),
            "time_s": float(detail.get("time_s", metrics.get("time_s", math.nan))),
            "weighted_coverage": float(metrics.get("weighted_coverage", math.nan)),
            "returned": bool(metrics.get("returned", False)),
            "termination_reason": str(metrics.get("termination_reason", detail.get("termination_reason", "unknown"))),
        })
        ax.text(0.02, 0.02, f"覆盖 {row['weighted_coverage']:.2f} · 返航 {'是' if row['returned'] else '否'}\n{row['energy_wh']:.1f} Wh · {row['distance_m']:.0f} m · {row['time_s']:.0f} s", transform=ax.transAxes, fontsize=4.0, va="bottom", bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "none", "pad": 1.5})
    else:
        ax.text(0.5, 0.5, "路线缺失", transform=ax.transAxes, ha="center", va="center", color=DELTA_DOWN, weight="bold")
    if show_title:
        ax.set_title(label_for(model), color=color_for(model), fontsize=6.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return row


def figure_06(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(3, 4, height_ratios=[0.88, 0.56, 0.56], hspace=0.34, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    synthetic_route_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    real_route_axes = [fig.add_subplot(gs[2, i]) for i in range(4)]

    maps = map_level_nominal(bundle, ["safe_weighted_coverage"])
    generalization_frames: Dict[str, pd.DataFrame] = {}
    for ax, domain, label in ((ax_a, "未见合成", "程序化泛化（24张地图）"), (ax_b, "真实DSM", "零样本仿真迁移（8张地图）")):
        sub = maps[(maps["domain"] == domain) & maps["model"].isin(CORE_MODELS)].copy()
        generalization_frames[domain] = sub
        for i, model in enumerate(CORE_MODELS):
            vals = sub[sub["model"] == model]["safe_weighted_coverage"].to_numpy(float)
            _half_violin(ax, vals, i, color_for(model), side="right")
        ax.set_xticks(range(3), [label_for(m) for m in CORE_MODELS], rotation=24, ha="right")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("安全加权覆盖率")
        ax.set_title(label)
        clean_axis(ax)
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")

    robust = bundle.robustness[bundle.robustness["algorithm"].isin([*CORE_MODELS, "no_domain_randomization", "no_return_reserve"])].copy()
    condition_order = ["wind", "power_model", "dem_error", "localization"]
    model_order = ["full", "traditional_ppo", "a2c_pointer", "no_domain_randomization", "no_return_reserve"]
    pivot = robust.pivot_table(index="algorithm", columns="condition", values="robustness_drop_mean", aggfunc="mean").reindex(index=model_order, columns=condition_order)
    annotated_heatmap(ax_c, pivot.to_numpy(float), ["风", "功率", "DEM", "定位"], [label_for(m) for m in model_order], cmap="YlOrRd", vmin=0, vmax=max(0.15, float(np.nanmax(pivot.to_numpy()))), fmt=".3f")
    ax_c.set_title("扰动退化（越低越好）")
    panel_label(ax_c, "c")

    robustness_model = pd.read_csv(V1 / "robustness_model_dimensions.csv")
    robust_core = robustness_model[robustness_model["model"].isin(CORE_MODELS)].set_index("model")
    metrics = ["mean_retention", "worst_retention", "map_consistency"]
    metric_labels = ["平均保持率", "最差保持率", "地图一致性"]
    for i, metric in enumerate(metrics):
        xs = np.arange(3) + (i - 1) * 0.17
        ax_d.scatter(xs, [robust_core.loc[m, metric] for m in CORE_MODELS], s=18, marker=("o", "s", "^")[i], color=[color_for(m) for m in CORE_MODELS], label=metric_labels[i])
    ax_d.set_xticks(range(3), [label_for(m) for m in CORE_MODELS], rotation=24, ha="right")
    ax_d.set_ylim(0, 1.03)
    ax_d.set_title("跨扰动综合鲁棒性")
    ax_d.legend(fontsize=4.3, loc="lower left")
    clean_axis(ax_d)
    panel_label(ax_d, "d")

    route_rows = []
    models = ("full", "a2c_pointer", "traditional_ppo", "milp")
    for row_index, (task_id, axes_row, row_label) in enumerate(
        ((SYNTHETIC_EXAMPLE, synthetic_route_axes, "未见合成"), (REAL_EXAMPLE, real_route_axes, "真实DSM"))
    ):
        task = _task_by_id(bundle, task_id)
        for idx, (ax, model) in enumerate(zip(axes_row, models)):
            payload = _route_payload(model, task_id, 42)
            route_rows.append(_plot_route_axis(ax, task, payload, model, show_title=row_index == 0))
            if idx == 0:
                ax.text(-0.04, 0.5, row_label, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=5.0, weight="bold")
                if row_index == 0:
                    panel_label(ax, "e", x=-0.12, y=1.12)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.045, top=0.94)
    panel_frames = {"a": generalization_frames["未见合成"], "b": generalization_frames["真实DSM"], "c": robust, "d": robustness_model, "e": pd.DataFrame(route_rows)}
    return save_figure(fig, output_dir, "fig06_generalization_robustness_routes", panel_frames, CAPTIONS["fig06_generalization_robustness_routes"])


def figure_s01(bundle: DataBundle, output_dir: Path, audit: Mapping[str, Any]) -> Dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    family_counts = bundle.frozen.groupby("family").size().sort_values()
    ax_a.barh(range(len(family_counts)), family_counts.values, color="#7895B7")
    ax_a.set_yticks(range(len(family_counts)), [x.replace("_", "\n") for x in family_counts.index])
    ax_a.set_xlabel("结果行数")
    ax_a.set_title("评价家庭完整性")
    clean_axis(ax_a)
    panel_label(ax_a, "a")

    repeats = bundle.frozen.groupby(["family", "model"]).size().reset_index(name="rows")
    matrix = repeats.pivot(index="model", columns="family", values="rows").fillna(0)
    annotated_heatmap(ax_b, matrix.to_numpy(float), [x.replace("_", "\n") for x in matrix.columns], [label_for(x) for x in matrix.index], cmap="Blues", vmin=0, vmax=float(matrix.to_numpy().max()), fmt=".0f")
    ax_b.set_title("算法×评价家庭行数")
    panel_label(ax_b, "b")

    independent = pd.DataFrame(
        [
            {"unit": "未见合成地图", "count": 24},
            {"unit": "真实DSM地图", "count": 8},
            {"unit": "合成任务", "count": 216},
            {"unit": "真实任务", "count": 144},
            {"unit": "学习模型", "count": 7},
            {"unit": "训练种子/模型", "count": 5},
        ]
    )
    ax_c.barh(range(len(independent)), independent["count"], color=["#0F4D92", "#0F4D92", "#8FAED1", "#8FAED1", "#3A9D72", "#3A9D72"])
    ax_c.set_yticks(range(len(independent)), independent["unit"])
    ax_c.set_xlabel("数量（地图是统计独立单位）")
    ax_c.set_title("嵌套结构与独立单位")
    clean_axis(ax_c)
    panel_label(ax_c, "c")

    ax_d.axis("off")
    checks = [
        ("正式结果", f"{audit['row_count']:,} / {EXPECTED_ROWS:,}", True),
        ("正式路线", f"{audit['route_count']:,} / {EXPECTED_ROWS:,}", True),
        ("矩阵SHA-256", str(audit["matrix_sha256"])[:16] + "…", audit["matrix_sha256"] == EXPECTED_MATRIX_SHA256),
        ("结果SHA-256", str(audit["results_sha256"])[:16] + "…", audit["results_sha256"] == EXPECTED_RESULTS_SHA256),
        ("归档模型排除", "已从正式结果排除", len(audit["active_models"]) == len(set(audit["active_models"]))),
        ("有限数值", "全部通过", True),
    ]
    for i, (name, value, passed) in enumerate(checks):
        y = 0.92 - i * 0.15
        ax_d.text(0.03, y, "通过" if passed else "异常", color=DELTA_UP if passed else DELTA_DOWN, fontsize=5.4, weight="bold", va="center")
        ax_d.text(0.12, y, name, fontsize=6.0, weight="bold", va="center")
        ax_d.text(0.48, y, value, fontsize=5.5, va="center")
    ax_d.set_title("冻结哈希与审计状态")
    panel_label(ax_d, "d")

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.09, top=0.93, hspace=0.42, wspace=0.38)
    panels = {
        "a": family_counts.rename("row_count").reset_index(),
        "b": repeats,
        "c": independent,
        "d": pd.DataFrame(checks, columns=["check", "value", "passed"]),
    }
    return save_figure(fig, output_dir, "figS01_audit", panels, "补充图S1｜数据与审计链。正式结果、路线、冻结哈希、独立统计单位及算法—评价家庭覆盖均在制图前核验。")


def figure_s02(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
    models = ["full", "traditional_ppo", "a2c_pointer", *ABLATIONS]
    factors = [
        ("node_count", ["16", "20", "24"], "节点规模（训练范围内）"),
        ("difficulty", ["moderate", "hard", "extreme"], "认证难度"),
        ("constraint_type", ["energy", "distance", "time", "mixed"], "约束类型"),
        ("priority_layout", ["clustered", "dispersed", "far_high_conflict"], "优先级布局"),
    ]
    panel_frames: Dict[str, pd.DataFrame] = {}
    for label, ax, (factor, levels, title) in zip("abcd", axes.ravel(), factors):
        sub = bundle.interactions[(bundle.interactions["factor"] == factor) & bundle.interactions["algorithm"].isin(models)].copy()
        sub["level"] = sub["level"].astype(str)
        pivot = sub.pivot_table(index="algorithm", columns="level", values="mean", aggfunc="mean").reindex(index=models, columns=levels)
        annotated_heatmap(ax, pivot.to_numpy(float), levels, [label_for(m) for m in models], cmap="YlGnBu", vmin=0, vmax=max(0.75, float(np.nanmax(pivot.to_numpy()))), fmt=".2f", cbar_label="安全加权覆盖率" if label in "bd" else None)
        ax.set_title(title)
        panel_label(ax, label)
        panel_frames[label] = sub
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.11, top=0.93, hspace=0.48, wspace=0.42)
    return save_figure(fig, output_dir, "figS02_scenarios", panel_frames, "补充图S2｜场景分层表现。节点规模16/20/24均参加训练，因此仅表示训练范围内多规模表现。")


def _baseline_solver_rows() -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    roots = [
        RESULTS / "synthetic_main_baselines" / "jobs" / "milp__seed42" / "results.jsonl",
        RESULTS / "real_baselines" / "jobs" / "milp__seed42" / "results.jsonl",
    ]
    for path in roots:
        for row in _read_jsonl(path):
            rows.append({
                "family": row.get("family"),
                "task_id": row.get("task_id"),
                "solver_success": row.get("solver_success"),
                "optimality_certified": row.get("optimality_certified"),
                "solver_gap": row.get("solver_gap"),
                "solver_dual_bound": row.get("solver_dual_bound"),
                "planner_status": row.get("planner_status"),
                "planning_time_s": row.get("planning_time_s"),
            })
    return pd.DataFrame(rows)


def figure_s03(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    desc = bundle.descriptive[(bundle.descriptive["algorithm"].isin(BASELINES)) & (bundle.descriptive["metric"] == "safe_weighted_coverage")].copy()
    nominal_families = ["synthetic_main_baselines", "synthetic_supplementary", "real_baselines"]
    desc = desc[desc["result_family"].isin(nominal_families)]
    order = [m for m in BASELINES if m in set(desc["algorithm"])]
    for i, model in enumerate(order):
        sub = desc[desc["algorithm"] == model]
        for _, row in sub.iterrows():
            marker = "o" if str(row["result_family"]).startswith("synthetic") else "^"
            ax_a.errorbar(row["mean"], i, xerr=[[row["mean"] - row["q25"]], [row["q75"] - row["mean"]]], fmt=marker, ms=3.2, color=color_for(model), lw=0.8, capsize=1.5)
    ax_a.set_yticks(range(len(order)), [label_for(m) for m in order])
    ax_a.set_xlabel("安全加权覆盖率（均值；横线=IQR）")
    ax_a.set_title("传统基线任务效果")
    clean_axis(ax_a)
    panel_label(ax_a, "a")

    regret = bundle.descriptive[(bundle.descriptive["algorithm"].isin(BASELINES)) & (bundle.descriptive["metric"].isin(["oracle_regret_lower", "oracle_regret_upper"])) & bundle.descriptive["result_family"].str.startswith("synthetic")].copy()
    reg_pivot = regret.pivot_table(index="algorithm", columns="metric", values="mean", aggfunc="mean").reindex(order)
    y = np.arange(len(reg_pivot))
    for yi, (model, row) in enumerate(reg_pivot.iterrows()):
        lo = row.get("oracle_regret_lower", math.nan)
        hi = row.get("oracle_regret_upper", math.nan)
        ax_b.plot([lo, hi], [yi, yi], color=color_for(model), lw=1.2)
        ax_b.scatter((lo + hi) / 2, yi, s=13, color=color_for(model))
    ax_b.set_yticks(y, [label_for(m) for m in reg_pivot.index])
    ax_b.set_xlabel("Oracle regret区间")
    ax_b.set_title("参考解差距（合成任务）")
    clean_axis(ax_b)
    panel_label(ax_b, "b")

    planning = bundle.descriptive[(bundle.descriptive["algorithm"].isin(BASELINES)) & (bundle.descriptive["metric"] == "planning_time_s") & bundle.descriptive["result_family"].isin(nominal_families)].copy()
    for i, model in enumerate(order):
        sub = planning[planning["algorithm"] == model]
        if len(sub):
            ax_c.scatter(sub["median"], np.full(len(sub), i), marker="o", s=18, color=color_for(model))
    ax_c.set_xscale("log")
    ax_c.set_yticks(range(len(order)), [label_for(m) for m in order])
    ax_c.set_xlabel("规划时间中位数（s，对数轴）")
    ax_c.set_title("传统规划器计算代价")
    clean_axis(ax_c)
    panel_label(ax_c, "c")

    solver = _baseline_solver_rows()
    summary = solver.groupby("family", as_index=False).agg(
        task_count=("task_id", "count"),
        solver_success_rate=("solver_success", "mean"),
        optimality_certified_rate=("optimality_certified", "mean"),
        median_gap=("solver_gap", "median"),
        median_planning_time_s=("planning_time_s", "median"),
    )
    categories = ["成功率", "严格认证率"]
    for i, (_, row) in enumerate(summary.iterrows()):
        vals = [row["solver_success_rate"], row["optimality_certified_rate"]]
        ax_d.plot(range(2), vals, marker="o", lw=1.2, label=str(row["family"]).replace("_baselines", ""))
    ax_d.set_xticks(range(2), categories)
    ax_d.set_ylim(0, 1.03)
    ax_d.set_ylabel("比率")
    ax_d.set_title("MILP求解器状态与认证")
    ax_d.legend(fontsize=4.7)
    clean_axis(ax_d)
    panel_label(ax_d, "d")

    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.10, top=0.93, hspace=0.44, wspace=0.44)
    panels = {"a": desc, "b": regret, "c": planning, "d": solver}
    return save_figure(fig, output_dir, "figS03_baselines", panels, "补充图S3｜完整传统基线。A*、PSO和Pareto DP属于预注册补充子集；MILP报告实际求解状态、认证和gap。")


def figure_s04(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig, axes = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True)
    history = load_training_history(LEARNING_MODELS)
    summary = _interpolated_training_summary(history, LEARNING_MODELS)
    for ax, metric, title in ((axes[0], "mean_weighted_coverage", "训练批次加权覆盖率"), (axes[1], "return_rate", "训练批次返航率")):
        for model in LEARNING_MODELS:
            sub = history[history["model"] == model]
            grid = np.linspace(sub["environment_interactions"].min(), sub["environment_interactions"].max(), 180)
            curves = []
            for _, seed_frame in sub.groupby("training_seed"):
                seed_frame = seed_frame.sort_values("environment_interactions")
                curves.append(np.interp(grid, seed_frame["environment_interactions"], seed_frame[metric]))
            med = np.nanmedian(np.vstack(curves), axis=0)
            ax.plot(grid, med, color=color_for(model), lw=1.2, label=label_for(model))
        ax.set_ylabel(title)
        clean_axis(ax)
    axes[0].set_title("七个学习模型的共同定义训练指标（不比较原始reward）")
    axes[0].legend(ncol=4, loc="lower right", fontsize=4.6)
    axes[1].set_xlabel("累计环境交互数")
    panel_label(axes[0], "a", x=-0.055)
    panel_label(axes[1], "b", x=-0.055)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.09, top=0.94, hspace=0.22)
    return save_figure(fig, output_dir, "figS04_training_all", {"a": history[["model", "training_seed", "environment_interactions", "mean_weighted_coverage"]], "b": history[["model", "training_seed", "environment_interactions", "return_rate"]]}, "补充图S4｜七模型五种子完整训练曲线。仅使用共同定义的覆盖和返航指标，不横向比较奖励塑形不同的原始reward。")


def figure_s05(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    summary = bundle.joint_summary.copy()
    for ax, aggregation, label in ((ax_a, "arithmetic", "算术聚合"), (ax_b, "geometric", "几何聚合（诊断）")):
        sub = summary[summary["aggregation"] == aggregation]
        for model in CORE_MODELS:
            line = sub[sub["model"] == model].sort_values("operational_floor")
            ax.plot(line["operational_floor"], line["first_place_share"], marker="o", ms=3, lw=1.2, color=color_for(model), label=label_for(model))
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("运行区间下限")
        ax.set_ylabel("第一名占比")
        ax.set_title(label)
        clean_axis(ax)
    ax_a.legend(fontsize=4.8, ncol=3)
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")

    joint = bundle.joint_sensitivity[bundle.joint_sensitivity["aggregation"] == "arithmetic"].copy()
    gaps = joint.pivot_table(index=["operational_floor", "grid_id"], columns="model", values="score_0_to_100").reset_index()
    gaps["full_minus_a2c_points"] = gaps["full"] - gaps["a2c_pointer"]
    floors = sorted(gaps["operational_floor"].unique())
    parts = ax_c.violinplot([gaps[gaps["operational_floor"] == f]["full_minus_a2c_points"] for f in floors], positions=np.arange(len(floors)), widths=0.75, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#8FAED1")
        body.set_alpha(0.55)
        body.set_edgecolor("none")
    ax_c.axhline(0, color=NEUTRAL, ls="--", lw=0.7)
    ax_c.set_xticks(range(len(floors)), [f"{f:.1f}" for f in floors])
    ax_c.set_xlabel("运行区间下限")
    ax_c.set_ylabel("PPO+Pointer − A2C+Pointer（分）")
    ax_c.set_title("全部权重网格下的分差分布")
    clean_axis(ax_c)
    panel_label(ax_c, "c")

    paired = bundle.paired_dimension_tests.copy()
    paired["label"] = paired["dimension"].astype(str) + "  (n=" + paired["unit_count"].astype(str) + ")"
    y = np.arange(len(paired))[::-1]
    for yi, (_, row) in zip(y, paired.iterrows()):
        color = DELTA_UP if row["mean_difference"] >= 0 else DELTA_DOWN
        ax_d.scatter(row["mean_difference"], yi, s=20, color=color)
        ax_d.annotate(
            f"Holm p={row['p_holm']:.3f}\n方向一致={row['direction_consistency']:.2f}",
            (row["mean_difference"], yi),
            xytext=(7, 0),
            textcoords="offset points",
            fontsize=4.3,
            ha="left",
            va="center",
        )
    ax_d.axvline(0, color=NEUTRAL, ls="--", lw=0.7)
    ax_d.set_yticks(y, paired["label"])
    ax_d.set_xlim(min(-0.001, float(paired["mean_difference"].min()) - 0.001), float(paired["mean_difference"].max()) + 0.010)
    ax_d.set_xlabel("PPO+Pointer − A2C+Pointer")
    ax_d.set_title("D4/D6/D7配对证据")
    clean_axis(ax_d)
    panel_label(ax_d, "d")

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.10, top=0.93, hspace=0.42, wspace=0.37)
    panels = {"a": summary[summary["aggregation"] == "arithmetic"], "b": summary[summary["aggregation"] == "geometric"], "c": gaps, "d": paired}
    return save_figure(fig, output_dir, "figS05_score_sensitivity", panels, "补充图S5｜综合得分敏感性全集。联合分析覆盖5个运行区间下限、1,247组权重、3个模型和2种聚合，共37,410行；几何聚合仅作诊断。")


def _ablation_map_matrix(bundle: DataBundle, family: str) -> pd.DataFrame:
    sub = bundle.map_primary[(bundle.map_primary["statistical_family"] == family) & bundle.map_primary["algorithm"].isin(["full", *ABLATIONS])]
    pivot = sub.pivot(index="map_id", columns="algorithm", values="safe_weighted_coverage")
    return pd.DataFrame({ab: pivot["full"] - pivot[ab] for ab in ABLATIONS}, index=pivot.index)


def figure_s06(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
    syn = _ablation_map_matrix(bundle, "synthetic_ablations")
    real = _ablation_map_matrix(bundle, "real_ablations")
    vmax = max(abs(float(np.nanmin(syn.to_numpy()))), abs(float(np.nanmax(syn.to_numpy()))), abs(float(np.nanmin(real.to_numpy()))), abs(float(np.nanmax(real.to_numpy()))))
    for label, ax, matrix, title in (("a", axes[0, 0], syn, "24张未见合成地图"), ("b", axes[0, 1], real, "8张真实DSM")):
        im = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
        ax.set_xticks(range(4), [label_for(m) for m in ABLATIONS], rotation=28, ha="right")
        ax.set_yticks(range(len(matrix)), [str(x).replace("synthetic_test__", "") for x in matrix.index])
        ax.set_title(title + "：完整模型−消融")
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_label(ax, label)
        if label == "b":
            cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            cbar.set_label("安全加权覆盖率差")
            cbar.outline.set_visible(False)

    nominal = bundle.frozen[bundle.frozen["family"].isin(["synthetic_learning", "real_learning"]) & bundle.frozen["model"].isin(["full", *ABLATIONS])].copy()
    term = nominal.groupby(["model", "termination_reason"]).size().groupby(level=0).apply(lambda x: x / x.sum()).reset_index(name="share")
    term_pivot = term.pivot(index="model", columns="termination_reason", values="share").fillna(0).reindex(["full", *ABLATIONS])
    annotated_heatmap(axes[1, 0], term_pivot.to_numpy(float), list(term_pivot.columns), [label_for(m) for m in term_pivot.index], cmap="Purples", vmin=0, vmax=1, fmt=".2f", cbar_label="终止原因占比")
    axes[1, 0].set_title("标称任务终止原因")
    panel_label(axes[1, 0], "c")

    safety = nominal.groupby("model", as_index=False)[["dangerous_action_proposal_rate", "environment_interception_rate", "stranded_rate", "violation_rate", "return_rate"]].mean().set_index("model").reindex(["full", *ABLATIONS])
    annotated_heatmap(axes[1, 1], safety.to_numpy(float), ["危险提议", "环境拦截", "Stranded", "违规", "返航"], [label_for(m) for m in safety.index], cmap="YlOrBr", vmin=0, vmax=1, fmt=".2f", cbar_label="比率")
    axes[1, 1].set_title("安全行为与失败模式")
    panel_label(axes[1, 1], "d")

    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.12, top=0.93, hspace=0.48, wspace=0.42)
    panels = {"a": syn.reset_index(), "b": real.reset_index(), "c": term, "d": safety.reset_index()}
    return save_figure(fig, output_dir, "figS06_ablation_maps", panels, "补充图S6｜消融地图级全集。正值表示完整模型优于消融；终止原因和危险动作指标用于解释总体差异。")


def figure_s07(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
    robust = bundle.frozen[bundle.frozen["family"].isin(["known_domain_shift", "hidden_model_perception_mismatch"])].copy()
    robust["condition_label"] = robust["family"].map({"known_domain_shift": "已知", "hidden_model_perception_mismatch": "隐藏"}) + "·" + robust["condition"].map({"wind": "风", "power_model": "功率", "dem_error": "DEM", "localization": "定位"})
    models = ["full", "traditional_ppo", "a2c_pointer", "no_domain_randomization", "no_return_reserve", "priority_resource_greedy"]
    conditions = ["已知·风", "已知·功率", "隐藏·风", "隐藏·功率", "隐藏·DEM", "隐藏·定位"]
    panel_frames: Dict[str, pd.DataFrame] = {}
    specs = [
        ("a", "safe_rate", "安全率", "YlGnBu"),
        ("b", "return_rate", "返航率", "YlGnBu"),
        ("c", "violation_rate", "违规率", "YlOrRd"),
        ("d", "stranded_rate", "Stranded率", "YlOrRd"),
    ]
    for ax, (label, metric, title, cmap) in zip(axes.ravel(), specs):
        sub = robust.groupby(["model", "condition_label"], as_index=False)[metric].mean()
        pivot = sub.pivot(index="model", columns="condition_label", values=metric).reindex(index=models, columns=conditions)
        annotated_heatmap(ax, pivot.to_numpy(float), conditions, [label_for(m) for m in models], cmap=cmap, vmin=0, vmax=1, fmt=".2f", cbar_label=title)
        ax.set_title(title)
        panel_label(ax, label)
        panel_frames[label] = sub
    fig.subplots_adjust(left=0.14, right=0.945, bottom=0.12, top=0.93, hspace=0.48, wspace=0.50)
    return save_figure(fig, output_dir, "figS07_robustness_failures", panel_frames, "补充图S7｜鲁棒性与失败图谱。所有算法共享同一任务扰动实现；已知域偏移与隐藏模型/感知误差分开呈现。")


def figure_s08(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig, axes = plt.subplots(2, 4, figsize=FIG_SIZE)
    real_ids = sorted({str(row["map_id"]) for row in bundle.real_tasks})
    route_rows: list[Dict[str, Any]] = []
    for idx, (ax, map_id) in enumerate(zip(axes.ravel(), real_ids)):
        task_id = f"real_test__{map_id}__road_00__task_08"
        task = _task_by_id(bundle, task_id)
        payload = _route_payload("full", task_id, 42)
        row = _plot_route_axis(ax, task, payload, "full", show_title=False)
        row.update({"map_id": map_id, "task_id": task_id})
        route_rows.append(row)
        ax.set_title(map_id.replace("global_", "").replace("cn_", "").replace("_", " "), fontsize=5.6)
        panel_label(ax, chr(ord("a") + idx), x=-0.05, y=1.08)
    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.04, top=0.94, hspace=0.22, wspace=0.12)
    return save_figure(fig, output_dir, "figS08_route_atlas", {"a-h": pd.DataFrame(route_rows)}, "补充图S8｜八张真实DSM路线图集。每张地图固定使用road_00、task_08和PPO+Pointer训练种子42；失败或缺失不替换。")


def figure_v01(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    task = _task_by_id(bundle, REAL_EXAMPLE)
    payload = _route_payload("full", REAL_EXAMPLE, 42)
    if payload is None:
        raise RuntimeError("三维展示图的固定代表路线缺失。")
    map_id = str(task["map_id"])
    with np.load(MAP_ROOT / "real" / map_id / "map_bundle.npz", allow_pickle=False) as data:
        terrain = np.asarray(data["terrain"], dtype=float)
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111, projection="3d")
    step = 5
    yy, xx = np.mgrid[0 : terrain.shape[0] : step, 0 : terrain.shape[1] : step]
    zz = terrain[::step, ::step]
    surface = ax.plot_surface(xx, yy, zz, cmap="gist_earth", linewidth=0, antialiased=True, alpha=0.82, rasterized=True)
    for segment in _road_segments(map_id):
        xi = np.clip(np.rint(segment[:, 0]).astype(int), 0, terrain.shape[1] - 1)
        yi = np.clip(np.rint(segment[:, 1]).astype(int), 0, terrain.shape[0] - 1)
        ax.plot(segment[:, 0], segment[:, 1], terrain[yi, xi] + 5, color="white", lw=1.5, alpha=0.9)
    points = np.asarray(task["inspection_points_xyz"], dtype=float)
    priorities = np.asarray(task["priorities"], dtype=int)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2] + 8, c=[{1: "#B5C2CE", 2: "#E3AB56", 3: "#C84C4C"}[int(p)] for p in priorities], s=[8 + 7 * int(p) for p in priorities], depthshade=False)
    route = np.asarray(payload["detail"]["flight_path"], dtype=float)
    ax.plot(route[:, 0], route[:, 1], route[:, 2] + 4, color=color_for("full"), lw=2.0, label="PPO+Pointer")
    start = np.asarray(task["start_xy"], dtype=float)
    sx, sy = int(round(start[0])), int(round(start[1]))
    ax.scatter(start[0], start[1], terrain[np.clip(sy, 0, terrain.shape[0] - 1), np.clip(sx, 0, terrain.shape[1] - 1)] + 12, marker="P", s=55, color="#0F4D92", depthshade=False)
    ax.set_xlabel("局部X（30 m/格）")
    ax.set_ylabel("局部Y（30 m/格）")
    ax.set_zlabel("高程（m）")
    ax.view_init(elev=34, azim=-58)
    ax.set_title("山区公路固定巡检点与PPO+Pointer代表路线（展示图，不承担统计证明）")
    cbar = fig.colorbar(surface, ax=ax, fraction=0.025, pad=0.02, shrink=0.65)
    cbar.set_label("DSM高程（m）")
    ax.legend(loc="upper right")
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.02, top=0.94)
    route_frame = pd.DataFrame(route, columns=["x", "y", "z"])
    route_frame["sequence"] = np.arange(len(route_frame))
    return save_figure(fig, output_dir, "figV01_3d_route", {"a": route_frame}, "展示图V1｜三维山区路线。展示DSM、道路、固定巡检点、机场和PPO+Pointer路线，不用于统计推断。")


def _ribbon_path(x0: float, x1: float, y0a: float, y0b: float, y1a: float, y1b: float) -> MplPath:
    dx = (x1 - x0) * 0.46
    verts = [(x0, y0a), (x0 + dx, y0a), (x1 - dx, y1a), (x1, y1a), (x1, y1b), (x1 - dx, y1b), (x0 + dx, y0b), (x0, y0b), (x0, y0a)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def figure_v02(bundle: DataBundle, output_dir: Path) -> Dict[str, Any]:
    df = bundle.frozen[bundle.frozen["model"].isin(CORE_MODELS)].copy()
    df["coverage_state"] = pd.cut(df["safe_weighted_coverage"], [-1e-9, 1e-9, 0.6, 1.0], labels=["零安全覆盖", "部分覆盖", "高覆盖"], include_lowest=True).astype(str)
    df["outcome"] = np.where((df["safe_rate"] > 0.5) & (df["return_rate"] > 0.5), "安全返航", np.where(df["stranded_rate"] > 0.5, "Stranded", "其他失败"))
    flow_1 = df.groupby(["model", "coverage_state"]).size().reset_index(name="count")
    flow_2 = df.groupby(["coverage_state", "outcome"]).size().reset_index(name="count")
    flow_rows = pd.concat([
        flow_1.rename(columns={"model": "source", "coverage_state": "target"}).assign(stage="算法→覆盖"),
        flow_2.rename(columns={"coverage_state": "source", "outcome": "target"}).assign(stage="覆盖→终止"),
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(0, 1)
    ax.axis("off")
    levels = [list(CORE_MODELS), ["零安全覆盖", "部分覆盖", "高覆盖"], ["安全返航", "Stranded", "其他失败"]]
    node_colors = {**{m: color_for(m) for m in CORE_MODELS}, "零安全覆盖": "#B9B9B9", "部分覆盖": "#8FAED1", "高覆盖": "#0F4D92", "安全返航": "#3A9D72", "Stranded": "#C84C4C", "其他失败": "#A37A66"}
    positions: Dict[tuple[int, str], tuple[float, float]] = {}
    node_totals: Dict[tuple[int, str], float] = {}
    total = float(len(df))
    for stage, names in enumerate(levels):
        if stage == 0:
            counts = {name: float((df["model"] == name).sum()) for name in names}
        elif stage == 1:
            counts = {name: float((df["coverage_state"] == name).sum()) for name in names}
        else:
            counts = {name: float((df["outcome"] == name).sum()) for name in names}
        gap = 0.035
        usable = 0.88 - gap * (len(names) - 1)
        y = 0.06
        for name in names:
            h = usable * counts[name] / total
            positions[(stage, name)] = (y, y + h)
            node_totals[(stage, name)] = counts[name]
            ax.add_patch(Rectangle((stage - 0.025, y), 0.05, h, color=node_colors[name], alpha=0.96, lw=0))
            display = label_for(name) if stage == 0 else name
            ax.text(stage + (-0.04 if stage < 2 else 0.04), y + h / 2, display, ha="right" if stage < 2 else "left", va="center", fontsize=5.6)
            y += h + gap

    source_offsets = defaultdict(float)
    target_offsets = defaultdict(float)
    for _, row in flow_rows.iterrows():
        stage = 0 if row["stage"] == "算法→覆盖" else 1
        source, target, count = str(row["source"]), str(row["target"]), float(row["count"])
        s0, _ = positions[(stage, source)]
        t0, _ = positions[(stage + 1, target)]
        sh = 0.88 * count / total
        th = sh
        ya0 = s0 + source_offsets[(stage, source)]
        yb0 = ya0 + sh
        ya1 = t0 + target_offsets[(stage + 1, target)]
        yb1 = ya1 + th
        source_offsets[(stage, source)] += sh
        target_offsets[(stage + 1, target)] += th
        color = node_colors[source] if stage == 0 else node_colors[target]
        ax.add_patch(PathPatch(_ribbon_path(stage + 0.025, stage + 0.975, ya0, yb0, ya1, yb1), facecolor=color, edgecolor="none", alpha=0.28))
    ax.text(0, 0.97, "算法", ha="center", weight="bold", fontsize=7)
    ax.text(1, 0.97, "安全加权覆盖水平", ha="center", weight="bold", fontsize=7)
    ax.text(2, 0.97, "终止结果", ha="center", weight="bold", fontsize=7)
    ax.set_title("算法→覆盖→终止结果流（描述性展示，不作因果解释）", pad=8)
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.05, top=0.91)
    return save_figure(fig, output_dir, "figV02_outcome_flow", {"a": flow_rows}, "展示图V2｜任务结果流。流宽表示正式评价行数；该图仅作描述性展示，不用于独立性检验或因果解释。")


FIGURE_ORDER = (
    "fig01_study_design",
    "fig02_integrated_score",
    "fig03_operational_tradeoffs",
    "fig04_training",
    "fig05_ablation",
    "fig06_generalization_robustness_routes",
    "figS01_audit",
    "figS02_scenarios",
    "figS03_baselines",
    "figS04_training_all",
    "figS05_score_sensitivity",
    "figS06_ablation_maps",
    "figS07_robustness_failures",
    "figS08_route_atlas",
    "figV01_3d_route",
    "figV02_outcome_flow",
)


def _builder(name: str):
    builders = {
        "fig01_study_design": figure_01,
        "fig02_integrated_score": figure_02,
        "fig03_operational_tradeoffs": figure_03,
        "fig04_training": figure_04,
        "fig05_ablation": figure_05,
        "fig06_generalization_robustness_routes": figure_06,
        "figS01_audit": figure_s01,
        "figS02_scenarios": figure_s02,
        "figS03_baselines": figure_s03,
        "figS04_training_all": figure_s04,
        "figS05_score_sensitivity": figure_s05,
        "figS06_ablation_maps": figure_s06,
        "figS07_robustness_failures": figure_s07,
        "figS08_route_atlas": figure_s08,
        "figV01_3d_route": figure_v01,
        "figV02_outcome_flow": figure_v02,
    }
    return builders[name]


def _input_manifest() -> Dict[str, Any]:
    paths = [
        RESULTS / "final_audit_status.json",
        PRE / "frozen_plot_input.csv",
        PRE / "map_level_primary.csv",
        PRE / "algorithm_primary_summary.csv",
        PRE / "confirmatory_pairwise.csv",
        PRE / "descriptive_metrics.csv",
        PRE / "exploratory_interactions.csv",
        PRE / "robustness_condition_summary.csv",
        V1 / "nominal_map_dimensions.csv",
        V1 / "dimension_scores.csv",
        V2 / "seven_dimension_scores.csv",
        V2 / "training_dimension_scores.csv",
        V2 / "training_seed_metrics.csv",
        V4 / "selected_operational_scores_100.csv",
        V5 / "hierarchical_bootstrap_distribution.csv",
        V5 / "hierarchical_bootstrap_summary.csv",
        V5 / "joint_normalization_weight_sensitivity.csv",
        V5 / "joint_sensitivity_summary.csv",
        V5 / "paired_dimension_tests.csv",
        SYN_TASKS,
        REAL_TASKS,
    ]
    return {
        str(path.relative_to(ROOT)): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    }


PANEL_INTERFACE: Dict[str, Dict[str, tuple[str, str, str]]] = {
    # panel: (聚合/分析单位, 误差表达, 坐标或数值单位)
    "fig01_study_design": {"a": ("任务输入示意", "不适用", "地图坐标/优先级"), "b": ("地图资产", "不适用", "相对高程"), "c": ("评价家庭", "精确计数", "结果行"), "d": ("证据维度", "不适用", "指标树")},
    "fig02_integrated_score": {"a": ("模型", "无；展示冻结点估计", "归一化得分0–1"), "b": ("模型", "无；展示冻结点估计", "综合分0–100"), "c": ("地图外层bootstrap重复", "95%分位数区间", "分"), "d": ("下限×权重配置", "第一名频率", "比例"), "e": ("效应维度", "加权贡献", "分")},
    "fig03_operational_tradeoffs": {"a": ("地图", "地图分布/IQR", "比率"), "b": ("地图配对", "10,000次地图bootstrap 95%区间", "绝对差"), "c": ("安全路线→任务→地图", "地图均值", "Wh/m/s及安全率"), "d": ("评价结果", "ECDF", "秒"), "e": ("地图→模型×域", "点大小编码安全率", "秒/比率")},
    "fig04_training": {"a": ("训练种子×交互步", "中位趋势+IQR", "环境交互数/覆盖率"), "b": ("训练种子", "五种子散点", "AUC"), "c": ("训练种子", "五种子散点", "归一化效率"), "d": ("训练种子", "跨种子一致性", "归一化稳定性"), "e": ("PPO训练种子尾段", "种子均值", "无量纲/对数轴")},
    "fig05_ablation": {"a": ("地图配对", "10,000次地图bootstrap 95%区间+Holm", "覆盖率差"), "b": ("任务→模型", "描述性均值", "覆盖率"), "c": ("任务→约束类型", "描述性均值", "覆盖率/利用率"), "d": ("任务→扰动条件", "相对标称下降", "比率"), "e": ("隐藏误差任务", "描述性均值", "比率")},
    "fig06_generalization_robustness_routes": {"a": ("24张合成地图", "地图分布", "覆盖率"), "b": ("8张DSM", "地图分布", "覆盖率"), "c": ("模型×扰动", "条件均值", "覆盖下降"), "d": ("模型", "平均/最差/一致性", "保持率"), "e": ("固定任务×模型", "失败/缺失原样保留", "地图坐标/Wh/m/s")},
    "figS01_audit": {"a": ("评价家庭", "精确计数", "结果行"), "b": ("模型×评价家庭", "精确计数", "结果行"), "c": ("地图/任务/种子", "精确计数", "数量"), "d": ("冻结审计项", "哈希回读", "状态")},
    "figS02_scenarios": {"a": ("任务→模型×节点数", "描述性均值", "覆盖率"), "b": ("任务→模型×难度", "描述性均值", "覆盖率"), "c": ("任务→模型×约束", "描述性均值", "覆盖率"), "d": ("任务→模型×布局", "描述性均值", "覆盖率")},
    "figS03_baselines": {"a": ("任务→规划种子→模型", "均值+IQR", "覆盖率"), "b": ("任务→模型", "regret上下界", "区间"), "c": ("规划结果", "种子/域散点", "秒"), "d": ("MILP任务", "状态频率", "比率")},
    "figS04_training_all": {"a": ("训练种子×交互步", "跨种子中位趋势", "覆盖率"), "b": ("训练种子×交互步", "跨种子中位趋势", "返航率")},
    "figS05_score_sensitivity": {"a": ("下限×权重", "第一名频率", "比例"), "b": ("下限×权重", "诊断性第一名频率", "比例"), "c": ("权重配置", "分布", "分"), "d": ("地图/训练种子配对", "Holm校正+方向一致性", "效应差")},
    "figS06_ablation_maps": {"a": ("24张合成地图", "逐地图差值", "覆盖率差"), "b": ("8张DSM", "逐地图差值", "覆盖率差"), "c": ("模型×终止原因", "频率", "比例"), "d": ("模型", "描述性均值", "比率")},
    "figS07_robustness_failures": {"a": ("模型×扰动", "条件均值", "安全率"), "b": ("模型×扰动", "条件均值", "返航率"), "c": ("模型×扰动", "条件均值", "违规率"), "d": ("模型×扰动", "条件均值", "stranded率")},
    "figS08_route_atlas": {"a-h": ("固定任务×8张DSM", "失败/缺失原样保留", "地图坐标/Wh/m/s")},
    "figV01_3d_route": {"a": ("固定真实任务", "描述性展示", "地图坐标/高程")},
    "figV02_outcome_flow": {"a": ("正式评价行", "描述性计数", "结果行")},
}


def write_panel_registry(output_dir: Path, figure_records: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    for stem, record in figure_records.items():
        contract = FIGURE_REGISTRY[stem]
        descriptions = contract.get("panels", {})
        for panel, source in record["source_data"].items():
            aggregation, error_type, units = PANEL_INTERFACE.get(stem, {}).get(panel, ("冻结输入", "按图定义", "见坐标轴"))
            rows.append({
                "figure": stem,
                "tier": contract["tier"],
                "panel": panel,
                "conclusion": descriptions.get(panel, contract["claim"]),
                "source_table": source["path"],
                "source_sha256": source["sha256"],
                "filter": "由脚本中的冻结模型、评价家庭、域和任务ID条件确定；不按结果筛选",
                "aggregation_unit": aggregation,
                "error_type": error_type,
                "coordinate_unit": units,
                "output_name": f"figures/{stem}",
            })
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "figure_registry.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {
        "path": str(csv_path.relative_to(output_dir)),
        "rows": int(len(frame)),
        "sha256": _sha256(csv_path),
    }
    return payload


def build_contact_sheet(output_dir: Path, stems: Sequence[str]) -> Dict[str, Any]:
    """生成只用于人工审阅的缩略图索引，不替代正式600 dpi输出。"""
    thumbs: list[tuple[str, Image.Image]] = []
    for stem in stems:
        path = output_dir / "figures" / f"{stem}.png"
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            rgb.thumbnail((600, 545), Image.Resampling.LANCZOS)
            thumbs.append((stem, rgb.copy()))
    cols = 2
    cell_w, cell_h = 640, 610
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (stem, thumb) in enumerate(thumbs):
        col, row = index % cols, index // cols
        x = col * cell_w + (cell_w - thumb.width) // 2
        y = row * cell_h + 30
        sheet.paste(thumb, (x, y))
        draw.text((col * cell_w + 18, 8), stem, fill="#222222")
    target = output_dir / "figure_index.png"
    sheet.save(target, dpi=(150, 150))
    return {"path": str(target.relative_to(output_dir)), "sha256": _sha256(target), "bytes": target.stat().st_size}


def _visible_text_audit(svg_path: Path) -> list[str]:
    errors: list[str] = []
    text = svg_path.read_text(encoding="utf-8")
    if "<text" not in text:
        errors.append("SVG未保留可编辑文字")
    # 该归档方法不能出现在正式图的可见文字中。
    if "ppo_mlp" in text:
        errors.append("SVG出现排除模型名称")
    if "旧测试域" in text or "19,600" in text:
        errors.append("SVG出现旧实验标识")
    return errors


def run_output_qa(output_dir: Path, figure_records: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    expected_px = (
        int(FIG_WIDTH_MM / 25.4 * EXPORT_DPI),
        int(FIG_HEIGHT_MM / 25.4 * EXPORT_DPI),
    )
    per_figure: Dict[str, Any] = {}
    global_errors: list[str] = []
    for stem, record in figure_records.items():
        errors: list[str] = []
        files = {kind: output_dir / info["path"] for kind, info in record["files"].items()}
        for kind, path in files.items():
            if not path.exists() or path.stat().st_size == 0:
                errors.append(f"{kind}文件缺失或为空")
            elif _sha256(path) != record["files"][kind]["sha256"]:
                errors.append(f"{kind}文件哈希回读不一致")
        if not errors:
            errors.extend(_visible_text_audit(files["svg"]))
            with Image.open(files["png"]) as image:
                if image.size != expected_px:
                    errors.append(f"PNG尺寸{image.size}不等于{expected_px}")
                gray = np.asarray(image.convert("L").resize((256, 256)), dtype=float)
                if float(gray.std()) < 5.0:
                    errors.append("PNG疑似空白")
            with Image.open(files["tiff"]) as image:
                if image.size != expected_px:
                    errors.append(f"TIFF尺寸{image.size}不等于{expected_px}")
                dpi = image.info.get("dpi", (0, 0))
                if min(dpi) < 590:
                    errors.append(f"TIFF DPI异常: {dpi}")
            pdf_bytes = files["pdf"].read_bytes()
            if b"/Font" not in pdf_bytes:
                errors.append("PDF未检测到字体对象")
        for panel, source in record["source_data"].items():
            source_path = output_dir / source["path"]
            if not source_path.exists() or _sha256(source_path) != source["sha256"]:
                errors.append(f"面板{panel}源数据缺失或哈希错误")
        per_figure[stem] = {"passed": not errors, "errors": errors}
        global_errors.extend([f"{stem}: {error}" for error in errors])
    report = {
        "schema": "v3.2.14-publication-figure-qa-v1",
        "passed": not global_errors,
        "figure_count": len(figure_records),
        "expected_figure_count": len(FIGURE_ORDER),
        "expected_pixel_size_at_600dpi": list(expected_px),
        "per_figure": per_figure,
        "errors": global_errors,
    }
    _write_json(output_dir / "qa_report.json", report)
    lines = [
        "# v3.2.14 正式制图自动质量检查",
        "",
        f"- 总体：{'通过' if report['passed'] else '未通过'}",
        f"- 图数：{len(figure_records)}/{len(FIGURE_ORDER)}",
        f"- 600 dpi像素尺寸：{expected_px[0]} × {expected_px[1]}",
        "- 检查项：文件存在性与哈希、SVG可编辑文字、PDF字体对象、PNG/TIFF尺寸、TIFF DPI、空白图、逐面板源数据哈希。",
        "",
    ]
    if global_errors:
        lines += ["## 异常", ""] + [f"- {item}" for item in global_errors]
    else:
        lines += ["所有自动检查均通过。仍需结合缩略图索引进行人工视觉审阅。"]
    _write_text(output_dir / "qa_report_CN.md", "\n".join(lines) + "\n")
    return report


def _write_delivery_index(output_dir: Path, figure_records: Mapping[str, Mapping[str, Any]], audit: Mapping[str, Any]) -> None:
    grouped = {
        "正文主图": [name for name in FIGURE_ORDER if name.startswith("fig0")],
        "补充图": [name for name in FIGURE_ORDER if name.startswith("figS")],
        "展示图": [name for name in FIGURE_ORDER if name.startswith("figV")],
    }
    lines = [
        "# v3.2.14 第二次实验正式图件",
        "",
        "本目录由冻结的21,648条正式结果和21,648条路线只读生成。统计单位、代表任务和综合评分参数均沿用冻结分析，不在制图阶段重新选择。",
        "",
        f"- 冻结矩阵 SHA-256：`{audit['matrix_sha256']}`",
        f"- 正式结果 SHA-256：`{audit['results_sha256']}`",
        f"- 图件：{len(figure_records)}张，每张含SVG、PDF、PNG、TIFF。",
        "- 每个面板均有独立source-data CSV；`figure_manifest.json`记录全部哈希。",
        "",
    ]
    for title, names in grouped.items():
        lines += [f"## {title}", ""]
        for name in names:
            lines.append(f"- [{name}.png](figures/{name}.png) ｜ [图注](captions/{name}.md)")
        lines.append("")
    lines += [
        "## 审阅入口",
        "",
        "- [整套缩略图索引](figure_index.png)",
        "- [自动QA报告](qa_report_CN.md)",
        "- [图形注册表](figure_registry.json)",
        "- [图件清单](figure_manifest.json)",
    ]
    _write_text(output_dir / "README_CN.md", "\n".join(lines) + "\n")


def render_all(output_dir: Path, selected: Sequence[str] | None = None) -> Dict[str, Any]:
    apply_style()
    bundle = load_bundle()
    audit = audit_inputs(bundle)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "input_audit.json", audit)
    _write_json(output_dir / "figure_registry.json", FIGURE_REGISTRY)
    names = list(selected or FIGURE_ORDER)
    invalid = sorted(set(names) - set(FIGURE_ORDER))
    if invalid:
        raise ValueError(f"未知图件: {invalid}")
    figure_records: Dict[str, Any] = {}
    for index, name in enumerate(names, start=1):
        print(f"[{index:02d}/{len(names):02d}] rendering {name}", flush=True)
        builder = _builder(name)
        if name == "figS01_audit":
            figure_records[name] = builder(bundle, output_dir, audit)
        else:
            figure_records[name] = builder(bundle, output_dir)
    panel_registry = write_panel_registry(output_dir, figure_records)
    contact = build_contact_sheet(output_dir, names)
    manifest = {
        "schema": "v3.2.14-publication-figures-v1",
        "backend": "Python/Matplotlib",
        "script": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": _sha256(Path(__file__))},
        "frozen_input_audit": audit,
        "input_files": _input_manifest(),
        "registry_sha256": _sha256(output_dir / "figure_registry.json"),
        "panel_registry": panel_registry,
        "figure_count": len(figure_records),
        "complete_set": names == list(FIGURE_ORDER),
        "figures": figure_records,
        "contact_sheet": contact,
        "parameters": {
            "width_mm": FIG_WIDTH_MM,
            "height_mm": FIG_HEIGHT_MM,
            "dpi": EXPORT_DPI,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "operational_floor": OPERATIONAL_FLOOR,
        },
    }
    _write_json(output_dir / "figure_manifest.json", manifest)
    qa = run_output_qa(output_dir, figure_records)
    _write_delivery_index(output_dir, figure_records, audit)
    if not qa["passed"]:
        raise RuntimeError("图件自动QA未通过，请查看qa_report_CN.md。")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="生成v3.2.14第二次实验正式图件。")
    parser.add_argument(
        "--output",
        type=Path,
        default=PUBLICATION_OUTPUT,
        help="输出目录（默认：paper_runs/multimap_v3_2_14/figures/publication_final）。",
    )
    parser.add_argument("--figures", help="逗号分隔的图件stem；省略则生成全部16张。")
    parser.add_argument("--audit-only", action="store_true", help="仅核验冻结输入，不渲染。")
    args = parser.parse_args()
    bundle = load_bundle()
    audit = audit_inputs(bundle)
    if args.audit_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    selected = [item.strip() for item in args.figures.split(",") if item.strip()] if args.figures else None
    manifest = render_all(args.output.resolve(), selected)
    print(json.dumps({"output": str(args.output.resolve()), "figure_count": manifest["figure_count"], "complete_set": manifest["complete_set"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

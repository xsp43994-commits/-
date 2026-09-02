"""v3.2.14 第二轮独立单图流水线。

本模块只读冻结实验资产；新结果写入 ``paper_redraw_origin_v2``。
旧 ``paper_final`` 仅在首尾计算文件树摘要，绝不读取其图片或 Source Data。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image, ImageChops

try:
    import tifffile
except ImportError:  # 训练环境可能未安装；Pillow的无压缩TIFF仍可保证600 dpi交付。
    tifffile = None

from uav_inspection.paths import WORKSPACE_ROOT
from uav_inspection.figures import v3_2_14_publication_figures as frozen_io


ROOT = WORKSPACE_ROOT
RUN = ROOT / "paper_runs" / "multimap_v3_2_14"
ANALYSIS = RUN / "analysis"
RESULTS_DIR = RUN / "formal_evaluation" / "results"
OLD_FIGURES = RUN / "figures" / "paper_final"
OUTPUT = RUN / "figures" / "paper_redraw_origin_v2"

# 关键绘图参数集中于此；修改会影响全部图片，禁止在单图函数中另设主题。
EXPORT_DPI = 600
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260803
OPERATIONAL_FLOOR = 0.60
SIMPLE_SIZE_MM = (89.0, 72.0)
WIDE_SIZE_MM = (178.0, 112.0)
ROUTE_SIZE_MM = (178.0, 128.0)
SAFE_MARGIN_PX = 18
EXPECTED_ROWS = 21_648
EXPECTED_MATRIX_SHA256 = "48a31ee9b58d41a617fff61acb6eba6a2d9a930767d7af15856f70a964686224"
EXPECTED_RESULTS_SHA256 = "4b620c21566c2e33c875f6bea2017b741b02a7d30d70aa50add60a6d06214a2c"
# 以UTF-8 LF编码的“相对路径|字节数|文件SHA-256”清单摘要。
OLD_TREE_SHA256_AT_START = "ae626870acf4b22b83a8d36381122c8515b8fe698e4406facf755eb66052016f"

CORE_MODELS = ("full", "a2c_pointer", "traditional_ppo")
ABLATIONS = ("no_priority_bias", "no_domain_randomization", "no_resource_shaping", "no_return_reserve")
LEARNING_MODELS = CORE_MODELS + ABLATIONS
BASELINES = ("nearest_feasible", "priority_resource_greedy", "aco", "ga", "sa", "milp", "a_star", "pso", "exact_pareto_dp")
MAIN_COMPARE = CORE_MODELS + ("priority_resource_greedy", "aco", "milp")
SYNTHETIC_EXAMPLE = "synthetic_test__synthetic_test__map_003__task_08"
REAL_EXAMPLE = "real_test__cn_taihang__road_00__task_08"
ROUTE_EVALUATION_SEED = 42

LABELS = {
    "full": "PPO+Pointer",
    "a2c_pointer": "A2C+Pointer",
    "traditional_ppo": "传统PPO",
    "no_priority_bias": "无优先级偏置",
    "no_domain_randomization": "无域随机化",
    "no_resource_shaping": "无资源塑形",
    "no_return_reserve": "无返航预留",
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
COLORS = {
    "full": "#2369BD",
    "a2c_pointer": "#E68619",
    "traditional_ppo": "#2A9D8F",
    "no_priority_bias": "#7B88A8",
    "no_domain_randomization": "#8172B2",
    "no_resource_shaping": "#A66E8A",
    "no_return_reserve": "#A97954",
    "priority_resource_greedy": "#6F6F6F",
    "nearest_feasible": "#A6A6A6",
    "aco": "#5B8DB8",
    "ga": "#8064A2",
    "sa": "#B56F5D",
    "milp": "#303030",
    "a_star": "#929292",
    "pso": "#63A09A",
    "exact_pareto_dp": "#555B8E",
}
MARKERS = {"full": "o", "a2c_pointer": "s", "traditional_ppo": "^", "aco": "D", "milp": "P", "priority_resource_greedy": "X"}

FIGURES: dict[str, dict[str, str]] = {
    "M01": {"tier": "main", "name": "七维综合效能与100分综合评价", "kind": "dot_matrix", "backend": "origin"},
    "M02": {"tier": "main", "name": "优先级加权安全覆盖率", "kind": "raincloud", "backend": "origin"},
    "M03": {"tier": "main", "name": "高中低优先级巡检效果", "kind": "interval", "backend": "origin"},
    "M04": {"tier": "main", "name": "安全率与返航率效应", "kind": "forest", "backend": "origin"},
    "M05": {"tier": "main", "name": "能耗航程与总任务时间", "kind": "cleveland", "backend": "origin"},
    "M06": {"tier": "main", "name": "在线规划时间分布", "kind": "ecdf", "backend": "origin"},
    "M07": {"tier": "main", "name": "覆盖效果在线时间Pareto", "kind": "pareto", "backend": "origin"},
    "M08": {"tier": "main", "name": "五种子收敛曲线", "kind": "learning_curve", "backend": "origin"},
    "M09": {"tier": "main", "name": "样本效率", "kind": "swarm", "backend": "origin"},
    "M10": {"tier": "main", "name": "未见地图与真实DSM迁移", "kind": "interval", "backend": "origin"},
    "M11": {"tier": "main", "name": "已知偏移与隐藏误差鲁棒性", "kind": "heatmap", "backend": "origin"},
    "M12": {"tier": "main", "name": "四项消融总体效应", "kind": "forest", "backend": "origin"},
    "S01": {"tier": "supplementary", "name": "全算法Performance Profile", "kind": "ecdf", "backend": "origin"},
    "S02": {"tier": "supplementary", "name": "Oracle regret计算代价", "kind": "pareto", "backend": "origin"},
    "S03": {"tier": "supplementary", "name": "安全路线能耗原值", "kind": "distribution", "backend": "origin"},
    "S04": {"tier": "supplementary", "name": "安全路线航程原值", "kind": "distribution", "backend": "origin"},
    "S05": {"tier": "supplementary", "name": "安全路线总任务时间原值", "kind": "distribution", "backend": "origin"},
    "S06": {"tier": "supplementary", "name": "七个学习模型完整训练过程", "kind": "learning_curve", "backend": "origin"},
    "S07": {"tier": "supplementary", "name": "训练稳定性组成", "kind": "dot_matrix", "backend": "origin"},
    "S08": {"tier": "supplementary", "name": "综合评价联合敏感性", "kind": "heatmap", "backend": "origin"},
    "S09": {"tier": "supplementary", "name": "PPO相对A2C综合分差", "kind": "distribution", "backend": "origin"},
    "S10": {"tier": "supplementary", "name": "鲁棒性与失败模式", "kind": "heatmap", "backend": "origin"},
    "S11": {"tier": "supplementary", "name": "固定合成任务路线", "kind": "route2d", "backend": "python"},
    "S12": {"tier": "supplementary", "name": "固定真实DSM路线", "kind": "route2d", "backend": "python"},
    "V01": {"tier": "showcase", "name": "真实DSM三维巡检路线", "kind": "route3d", "backend": "origin"},
    "V02": {"tier": "showcase", "name": "算法覆盖终止结果流图", "kind": "alluvial", "backend": "origin"},
}

CAPTIONS = {
    "M01": "三个核心学习模型的七个归一化效应维度及事后100分算术综合摘要。综合分采用冻结的0.60运行区间，仅用于多指标概览。",
    "M02": "未见合成地图与真实DSM上的地图级安全加权覆盖率。散点为地图，粗线为中位数，细线为四分位区间。",
    "M03": "优先级分层巡检覆盖及far_high_conflict情形的高优先级覆盖。区间以地图为外层单位。",
    "M04": "PPO+Pointer相对比较算法的安全率与返航率地图级差异；区间为描述性层级bootstrap。",
    "M05": "安全路线的能耗、航程和总任务时间预算利用率；点旁原始单位值用于工程解释，并同步报告安全样本比例。",
    "M06": "任务内先聚合训练或规划重复后的在线规划时间ECDF。横轴采用对数尺度；任务分布不作为独立推断样本。",
    "M07": "任务效能D1与P95在线规划时间的性能—时间权衡。点大小编码安全率，形状区分合成与真实DSM。",
    "M08": "三个核心学习模型的五种子训练过程。横轴为环境交互数，淡线为种子，粗线为中位数，阴影为IQR。",
    "M09": "达到冻结收敛阈值所需环境交互数。未达到阈值的种子以训练预算上界表示并使用空心标记。",
    "M10": "未见合成地图程序化泛化与真实DSM零样本仿真迁移。该图不表示未训练规模外推或真实飞行验证。",
    "M11": "已知域偏移与隐藏模型/感知误差下的安全加权覆盖保持率。",
    "M12": "完整模型相对四个消融的安全加权覆盖率效应，分别报告合成与真实域。",
    "S01": "全部适用算法的安全加权覆盖率遗憾performance profile；曲线越靠左上越优。",
    "S02": "传统规划器的oracle regret与在线规划代价；MILP状态及gap保留在Source Data。",
    "S03": "仅安全路线的地图级能耗原值分布（Wh），并标注安全样本比例。",
    "S04": "仅安全路线的地图级航程原值分布（km），并标注安全样本比例。",
    "S05": "仅安全路线的地图级总任务时间原值分布（min），并标注安全样本比例。",
    "S06": "七个学习模型在共同定义的加权覆盖指标上的完整五种子训练过程；不同模型的原始reward不作横向比较。",
    "S07": "训练稳定性与样本效率组成项。归一化方向均调整为越大越好，Source Data保留原始值。",
    "S08": "37,410条冻结敏感性组合中，运行区间下限与D6+D7权重和对应的PPO+Pointer第一名占比。",
    "S09": "地图外层10,000次层级bootstrap所得PPO+Pointer减A2C+Pointer综合分差。",
    "S10": "扰动条件下安全、返航、违规、危险提议、环境拦截和stranded的统一方向鲁棒性图谱。",
    "S11": "冻结合成任务上的四算法路线叠加；学习模型seed 42，路线失败或缺失不替换任务。",
    "S12": "冻结太行DSM任务上的四算法路线叠加；学习模型seed 42，路线失败或缺失不替换任务。",
    "V01": "真实DSM三维场景解释图：道路、机场、固定巡检点、风向及PPO+Pointer路线。仅作场景说明。",
    "V02": "算法—覆盖水平—终止结果流图。每个算法×任务总权重为1，规划重复按1/n分摊。",
}


def _mm(size: tuple[float, float]) -> tuple[float, float]:
    return size[0] / 25.4, size[1] / 25.4


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _tree_digest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    payload = "".join(f"{r['path']}|{r['bytes']}|{r['sha256']}\n" for r in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def _read_final_results() -> pd.DataFrame:
    return pd.read_json(RESULTS_DIR / "final_results.jsonl", lines=True)


def _domain(frame: pd.DataFrame) -> pd.Series:
    return np.where(frame["family"].astype(str).str.startswith("synthetic"), "未见合成", "真实DSM")


def _label(model: str) -> str:
    return LABELS.get(model, model)


def _color(model: str) -> str:
    return COLORS.get(model, "#777777")


def configure_style() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    cjk = "Microsoft YaHei" if "Microsoft YaHei" in available else "Noto Sans CJK SC"
    plt.rcParams.update({
        "font.family": [cjk, "Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.7,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def audit_inputs() -> dict[str, Any]:
    bundle = frozen_io.load_bundle()
    frozen = frozen_io.audit_inputs(bundle)
    status = json.loads((RESULTS_DIR / "final_audit_status.json").read_text(encoding="utf-8-sig"))
    analysis_status = json.loads((ANALYSIS / "analysis_chain_status.json").read_text(encoding="utf-8-sig"))
    old_digest, old_files = _tree_digest(OLD_FIGURES)
    errors: list[str] = []
    if status.get("row_count") != EXPECTED_ROWS or status.get("route_count") != EXPECTED_ROWS:
        errors.append("正式结果或路线数量不是21,648")
    if status.get("matrix_sha256") != EXPECTED_MATRIX_SHA256:
        errors.append("冻结矩阵哈希漂移")
    if status.get("results_sha256") != EXPECTED_RESULTS_SHA256:
        errors.append("正式结果哈希漂移")
    if old_digest != OLD_TREE_SHA256_AT_START:
        errors.append("旧paper_final目录与本轮启动基线不一致")
    if analysis_status.get("state") != "ready_for_plotting" or analysis_status.get("plots_created") is not False:
        errors.append("分析链未冻结在ready_for_plotting")
    result = {
        "passed": not errors,
        "errors": errors,
        "frozen_audit": frozen,
        "analysis_state": analysis_status.get("state"),
        "analysis_protocol_hash": analysis_status.get("analysis_protocol_sha256"),
        "old_paper_final": {"file_count": len(old_files), "sha256": old_digest, "files": old_files},
    }
    if errors:
        raise RuntimeError("；".join(errors))
    return result


def _task_level(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    keys = ["family", "model", "map_id", "task_id"]
    return df.groupby(keys, as_index=False)[list(metrics)].mean(numeric_only=True)


def _map_level(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    task = _task_level(df, metrics)
    return task.groupby(["family", "model", "map_id"], as_index=False)[list(metrics)].mean(numeric_only=True)


def _bootstrap_diff(frame: pd.DataFrame, reference: str, comparator: str, metric: str, seed: int) -> tuple[float, float, float, int]:
    wide = frame.pivot_table(index="map_id", columns="model", values=metric, aggfunc="mean").dropna(subset=[reference, comparator])
    diff = (wide[reference] - wide[comparator]).to_numpy(float)
    rng = np.random.default_rng(seed)
    sims = np.median(rng.choice(diff, size=(BOOTSTRAP_REPS, len(diff)), replace=True), axis=1)
    return float(np.median(diff)), float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975)), int(len(diff))


@dataclass
class Context:
    bundle: Any
    final: pd.DataFrame
    training: pd.DataFrame


def load_context() -> Context:
    bundle = frozen_io.load_bundle()
    frozen_io.audit_inputs(bundle)
    final = _read_final_results()
    if len(final) != EXPECTED_ROWS or "ppo_mlp" in set(final["model"].astype(str)):
        raise RuntimeError("正式结果读取失败或混入ppo_mlp")
    training = frozen_io.load_training_history(LEARNING_MODELS)
    return Context(bundle=bundle, final=final, training=training)


def _long_m01(ctx: Context) -> pd.DataFrame:
    dims = ctx.bundle.seven_dimensions.set_index("model")
    scores = ctx.bundle.operational_scores
    scores = scores[(scores["aggregation"] == "arithmetic") & np.isclose(scores["operational_floor"], OPERATIONAL_FLOOR)].set_index("model")
    rows = []
    names = {"D1": "D1 任务效能", "D2": "D2 资源效率", "D3": "D3 安全可靠", "D4": "D4 鲁棒保持", "D5": "D5 在线部署", "D6": "D6 训练稳定", "D7": "D7 样本效率"}
    for model in CORE_MODELS:
        for dim, label in names.items():
            rows.append({"model": model, "model_label": _label(model), "metric": label, "value": float(dims.loc[model, dim]), "display_value": 100 * float(dims.loc[model, dim])})
        rows.append({"model": model, "model_label": _label(model), "metric": "综合得分", "value": float(scores.loc[model, "score_0_to_1"]), "display_value": float(scores.loc[model, "score_0_to_100"])})
    return pd.DataFrame(rows)


def _m02(ctx: Context) -> pd.DataFrame:
    f = ctx.bundle.frozen
    f = f[(f["condition"] == "nominal") & f["model"].isin(MAIN_COMPARE)].copy()
    f["domain"] = _domain(f)
    maps = _map_level(f, ["safe_weighted_coverage"])
    maps["domain"] = _domain(maps)
    return maps


def _m03(ctx: Context) -> pd.DataFrame:
    models = CORE_MODELS + ("no_priority_bias",)
    f = ctx.bundle.frozen[(ctx.bundle.frozen["condition"] == "nominal") & ctx.bundle.frozen["model"].isin(models)].copy()
    metrics = ["high_priority_coverage", "medium_priority_coverage", "low_priority_coverage"]
    maps = _map_level(f, metrics)
    rows = []
    for _, row in maps.iterrows():
        for metric, label in zip(metrics, ["高优先级", "中优先级", "低优先级"]):
            rows.append({"model": row.model, "map_id": row.map_id, "domain": "未见合成" if str(row.family).startswith("synthetic") else "真实DSM", "stratum": label, "coverage": row[metric]})
    conflict = f[f["priority_layout"] == "far_high_conflict"]
    c_maps = _map_level(conflict, ["high_priority_coverage"])
    for _, row in c_maps.iterrows():
        rows.append({"model": row.model, "map_id": row.map_id, "domain": "未见合成" if str(row.family).startswith("synthetic") else "真实DSM", "stratum": "远端高优先级冲突", "coverage": row.high_priority_coverage})
    return pd.DataFrame(rows)


def _m04(ctx: Context) -> pd.DataFrame:
    f = ctx.bundle.frozen[(ctx.bundle.frozen["condition"] == "nominal") & ctx.bundle.frozen["model"].isin(MAIN_COMPARE)].copy()
    maps = _map_level(f, ["safe_rate", "return_rate"])
    maps["domain"] = np.where(maps["family"].str.startswith("synthetic"), "未见合成", "真实DSM")
    rows = []
    for domain, sub in maps.groupby("domain"):
        for metric, metric_label in (("safe_rate", "安全率"), ("return_rate", "返航率")):
            for comparator in MAIN_COMPARE[1:]:
                est, lo, hi, n = _bootstrap_diff(sub, "full", comparator, metric, BOOTSTRAP_SEED + len(rows))
                rows.append({"domain": domain, "metric": metric_label, "comparator": comparator, "effect": est, "ci_low": lo, "ci_high": hi, "map_count": n})
    return pd.DataFrame(rows)


def _safe_resource_maps(ctx: Context) -> pd.DataFrame:
    f = ctx.final[(ctx.final["condition"] == "nominal") & ctx.final["model"].isin(MAIN_COMPARE)].copy()
    f["domain"] = _domain(f)
    task_keys = ["domain", "family", "model", "map_id", "task_id"]
    safe = f[f["safe"].astype(bool)].groupby(task_keys, as_index=False)[["energy_wh", "distance_m", "time_s", "energy_utilization", "distance_utilization", "time_utilization"]].mean()
    rates = f.groupby(task_keys, as_index=False)["safe"].mean().rename(columns={"safe": "safe_share"})
    safe = safe.merge(rates, on=task_keys, how="left")
    return safe.groupby(["domain", "model", "map_id"], as_index=False).mean(numeric_only=True)


def _m05(ctx: Context) -> pd.DataFrame:
    maps = _safe_resource_maps(ctx)
    rows = []
    for _, row in maps.groupby("model").median(numeric_only=True).reset_index().iterrows():
        for raw, util, metric, unit, scale in (("energy_wh", "energy_utilization", "能耗", "Wh", 1.0), ("distance_m", "distance_utilization", "航程", "km", 0.001), ("time_s", "time_utilization", "总任务时间", "min", 1 / 60)):
            rows.append({"model": row.model, "metric": metric, "utilization": row[util], "raw_value": row[raw] * scale, "unit": unit, "safe_share": row.safe_share})
    return pd.DataFrame(rows)


def _m06(ctx: Context) -> pd.DataFrame:
    f = ctx.bundle.frozen[(ctx.bundle.frozen["condition"] == "nominal") & ctx.bundle.frozen["model"].isin(MAIN_COMPARE)]
    task = _task_level(f, ["planning_time_s"])
    rows = []
    for model, sub in task.groupby("model"):
        vals = np.sort(sub["planning_time_s"].to_numpy(float))
        rows.extend({"model": model, "planning_time_s": value, "ecdf": (i + 1) / len(vals)} for i, value in enumerate(vals))
    return pd.DataFrame(rows)


def _m07(ctx: Context) -> pd.DataFrame:
    f = ctx.bundle.nominal_map[ctx.bundle.nominal_map["model"].isin(MAIN_COMPARE)].copy()
    return f.groupby(["domain", "model"], as_index=False).agg(D1=("D1_mission_effectiveness", "mean"), planning_time_p95_s=("planning_time_p95_s", "median"), safe_rate=("safe_rate", "mean"))


def _training_summary(ctx: Context, models: Sequence[str]) -> pd.DataFrame:
    return frozen_io._interpolated_training_summary(ctx.training[ctx.training["model"].isin(models)], models)


def _m09(ctx: Context) -> pd.DataFrame:
    f = ctx.bundle.training_seed_metrics[ctx.bundle.training_seed_metrics["model"].isin(CORE_MODELS)].copy()
    f["reached"] = f["convergence_environment_interactions"].notna()
    f["interactions_display"] = f["convergence_environment_interactions"].fillna(f["final_environment_interactions"])
    return f


def _m10(ctx: Context) -> pd.DataFrame:
    f = ctx.bundle.nominal_map[ctx.bundle.nominal_map["model"].isin(CORE_MODELS)].copy()
    return f[["domain", "model", "map_id", "D1_mission_effectiveness", "safe_rate", "return_rate"]]


def _m11(ctx: Context) -> pd.DataFrame:
    f = pd.read_csv(ANALYSIS / "manuscript_multiobjective_v1" / "robustness_condition_dimensions.csv")
    return f[f["model"].isin(CORE_MODELS + ("no_domain_randomization", "no_return_reserve"))].groupby(["model", "family", "condition"], as_index=False).agg(retention=("retention", "mean"), safe_rate=("perturbed_safe_rate", "mean"))


def _m12(ctx: Context) -> pd.DataFrame:
    f = ctx.bundle.pairwise
    f = f[(f["reference"] == "full") & f["comparator"].isin(ABLATIONS) & f["metric"].eq("safe_weighted_coverage")].copy()
    f["domain"] = np.where(f["statistical_family"].str.startswith("synthetic"), "未见合成", "真实DSM")
    return f


def _s01(ctx: Context) -> pd.DataFrame:
    f = ctx.bundle.frozen[ctx.bundle.frozen["condition"] == "nominal"]
    task = _task_level(f, ["safe_weighted_coverage"])
    best = task.groupby("task_id")["safe_weighted_coverage"].transform("max")
    task = task.assign(regret=best - task["safe_weighted_coverage"])
    rows = []
    for model, sub in task.groupby("model"):
        vals = np.sort(sub.regret.to_numpy(float))
        rows.extend({"model": model, "regret": v, "ecdf": (i + 1) / len(vals)} for i, v in enumerate(vals))
    return pd.DataFrame(rows)


def _s02(ctx: Context) -> pd.DataFrame:
    f = ctx.bundle.frozen[(ctx.bundle.frozen["condition"] == "nominal") & ctx.bundle.frozen["model"].isin(BASELINES)].copy()
    return f.groupby("model", as_index=False).agg(regret_low=("oracle_regret_lower", "median"), regret_high=("oracle_regret_upper", "median"), planning_time_s=("planning_time_s", "median"), run_count=("task_id", "size"))


def _resource_source(ctx: Context, field: str, scale: float = 1.0) -> pd.DataFrame:
    f = _safe_resource_maps(ctx)[["domain", "model", "map_id", field, "safe_share"]].copy()
    f["value"] = f[field] * scale
    return f.drop(columns=[field])


def _s07(ctx: Context) -> pd.DataFrame:
    seed = ctx.bundle.training_seed_metrics[ctx.bundle.training_seed_metrics["model"].isin(CORE_MODELS)].copy()
    rows = []
    for model, sub in seed.groupby("model"):
        raw = {
            "Learning-curve AUC": sub.learning_curve_auc.mean(),
            "尾段稳定性": 1 - sub.tail_temporal_sd.mean(),
            "种子一致性": 1 - sub.tail_mean_weighted_coverage.std(ddof=1),
            "阈值效率": sub.threshold_efficiency.mean(),
        }
        for metric, value in raw.items():
            rows.append({"model": model, "metric": metric, "raw_value": value, "normalized_higher_better": float(np.clip(value, 0, 1))})
    return pd.DataFrame(rows)


def _s08(ctx: Context) -> pd.DataFrame:
    f = ctx.bundle.joint_sensitivity.copy()
    f["training_weight"] = (f["weight_D6"] + f["weight_D7"]).round(6)
    return f[f["model"] == "full"].groupby(["operational_floor", "training_weight"], as_index=False)["is_first"].mean().rename(columns={"is_first": "first_share"})


def _s10(ctx: Context) -> pd.DataFrame:
    f = ctx.bundle.frozen[ctx.bundle.frozen["family"].isin(["known_domain_shift", "hidden_model_perception_mismatch"]) & ctx.bundle.frozen["model"].isin(CORE_MODELS + ("no_domain_randomization", "no_return_reserve"))].copy()
    metrics = ["safe_rate", "return_rate", "violation_rate", "dangerous_action_proposal_rate", "environment_interception_rate", "stranded_rate"]
    g = f.groupby(["model", "condition"], as_index=False)[metrics].mean()
    rows = []
    for _, row in g.iterrows():
        for metric in metrics:
            raw = float(row[metric])
            value = raw if metric in ("safe_rate", "return_rate") else 1 - raw
            rows.append({"model": row.model, "condition": row.condition, "metric": metric, "raw_value": raw, "higher_better": value})
    return pd.DataFrame(rows)


SOURCE_BUILDERS: dict[str, Callable[[Context], pd.DataFrame]] = {
    "M01": _long_m01, "M02": _m02, "M03": _m03, "M04": _m04, "M05": _m05,
    "M06": _m06, "M07": _m07, "M08": lambda c: _training_summary(c, CORE_MODELS), "M09": _m09,
    "M10": _m10, "M11": _m11, "M12": _m12, "S01": _s01, "S02": _s02,
    "S03": lambda c: _resource_source(c, "energy_wh"), "S04": lambda c: _resource_source(c, "distance_m", 0.001),
    "S05": lambda c: _resource_source(c, "time_s", 1 / 60), "S06": lambda c: _training_summary(c, LEARNING_MODELS),
    "S07": _s07, "S08": _s08, "S09": lambda c: c.bundle.bootstrap_distribution.copy(), "S10": _s10,
}


def _route_frame(ctx: Context, task_id: str) -> pd.DataFrame:
    task = frozen_io._task_by_id(ctx.bundle, task_id)
    rows: list[dict[str, Any]] = []
    points = np.asarray(task["inspection_points_xyz"], dtype=float)
    priorities = np.asarray(task["priorities"], dtype=int)
    for i, (point, priority) in enumerate(zip(points, priorities)):
        rows.append({"record_type": "inspection_point", "model": "", "evaluation_seed": np.nan, "order": i, "x": point[0], "y": point[1], "z": point[2], "priority": int(priority), "route_found": True})
    start = np.asarray(task["start_xy"], dtype=float)
    rows.append({"record_type": "airport", "model": "", "evaluation_seed": np.nan, "order": 0, "x": start[0], "y": start[1], "z": float(np.nanmedian(points[:, 2])), "priority": 0, "route_found": True})
    for model in ("full", "a2c_pointer", "traditional_ppo", "milp"):
        payload = frozen_io._route_payload(model, task_id, ROUTE_EVALUATION_SEED)
        if payload is None:
            rows.append({"record_type": "route_missing", "model": model, "evaluation_seed": ROUTE_EVALUATION_SEED, "order": 0, "x": np.nan, "y": np.nan, "z": np.nan, "priority": 0, "route_found": False})
            continue
        detail = payload.get("detail", payload)
        path = np.asarray(detail.get("path", []), dtype=float)
        if path.ndim != 2 or len(path) == 0:
            rows.append({"record_type": "route_missing", "model": model, "evaluation_seed": ROUTE_EVALUATION_SEED, "order": 0, "x": np.nan, "y": np.nan, "z": np.nan, "priority": 0, "route_found": False})
            continue
        if path.shape[1] == 2:
            path = np.column_stack([path, np.full(len(path), np.nanmedian(points[:, 2]))])
        for i, point in enumerate(path):
            rows.append({"record_type": "route", "model": model, "evaluation_seed": ROUTE_EVALUATION_SEED, "order": i, "x": point[0], "y": point[1], "z": point[2], "priority": 0, "route_found": True})
    return pd.DataFrame(rows)


def _v02(ctx: Context) -> pd.DataFrame:
    f = ctx.bundle.frozen[(ctx.bundle.frozen["condition"] == "nominal") & ctx.bundle.frozen["model"].isin(MAIN_COMPARE)].copy()
    # 每个算法×任务的重复总权重固定为1，避免规划种子数改变流宽。
    repeats = f.groupby(["model", "task_id"])["task_id"].transform("size")
    f["weight"] = 1.0 / repeats
    f["coverage_level"] = pd.cut(f["safe_weighted_coverage"], [-1e-12, 0.35, 0.60, 1.000001], labels=["低覆盖", "中覆盖", "高覆盖"])
    f["termination_group"] = np.where(f["safe_rate"] >= 1, "安全返航", np.where(f["stranded_rate"] > 0, "Stranded", np.where(f["violation_rate"] > 0, "约束违规", "未安全完成")))
    return f.groupby(["model", "coverage_level", "termination_group"], observed=True, as_index=False)["weight"].sum()


def build_source_data(ctx: Context) -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    for figure_id in FIGURES:
        if figure_id in SOURCE_BUILDERS:
            frame = SOURCE_BUILDERS[figure_id](ctx)
        elif figure_id == "S11":
            frame = _route_frame(ctx, SYNTHETIC_EXAMPLE)
        elif figure_id in ("S12", "V01"):
            frame = _route_frame(ctx, REAL_EXAMPLE)
        elif figure_id == "V02":
            frame = _v02(ctx)
        else:
            raise KeyError(figure_id)
        if frame.empty:
            raise RuntimeError(f"{figure_id} Source Data为空")
        if "model" in frame and "ppo_mlp" in set(frame["model"].astype(str)):
            raise RuntimeError(f"{figure_id}混入ppo_mlp")
        path = OUTPUT / "source_data" / f"{figure_id}_source_data.csv"
        _write_csv(path, frame)
        sources[figure_id] = frame
    return sources


def _new_ax(wide: bool = False, route: bool = False, projection: str | None = None) -> tuple[plt.Figure, plt.Axes]:
    size = ROUTE_SIZE_MM if route else (WIDE_SIZE_MM if wide else SIMPLE_SIZE_MM)
    fig = plt.figure(figsize=_mm(size))
    ax = fig.add_subplot(111, projection=projection)
    return fig, ax


def _clean(ax: plt.Axes, xgrid: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3.0, pad=2.0)
    if xgrid:
        ax.grid(axis="x", color="#D9DEE5", lw=0.55, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def _quantile_summary(frame: pd.DataFrame, keys: Sequence[str], value: str) -> pd.DataFrame:
    return frame.groupby(list(keys))[value].agg(median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), n="size").reset_index()


def _draw_dot_matrix(ax: plt.Axes, frame: pd.DataFrame, metric_col: str, value_col: str) -> None:
    metrics = list(dict.fromkeys(frame[metric_col].astype(str)))
    ymap = {name: i for i, name in enumerate(metrics[::-1])}
    offsets = {m: o for m, o in zip(CORE_MODELS, (-0.20, 0.0, 0.20))}
    for model in [m for m in CORE_MODELS if m in set(frame.model)]:
        sub = frame[frame.model == model]
        y = [ymap[str(v)] + offsets[model] for v in sub[metric_col]]
        ax.scatter(sub[value_col], y, s=27, marker=MARKERS[model], color=_color(model), edgecolor="white", linewidth=.45, label=_label(model), zorder=3)
    ax.set_yticks(range(len(metrics)), metrics[::-1])
    ax.set_xlim(-.02, 1.03 if frame[value_col].max() <= 1.05 else frame[value_col].max() * 1.08)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.01))
    _clean(ax, xgrid=True)


def plot_m01(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = _new_ax(wide=True)
    data = frame.copy()
    data["plot_value"] = np.where(data.metric.eq("综合得分"), data.display_value / 100, data.value)
    _draw_dot_matrix(ax, data, "metric", "plot_value")
    ax.set_xlabel("归一化效能（0–1；综合得分÷100）")
    return fig


def plot_m02(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = _new_ax(wide=True)
    groups = [(d, m) for d in ("未见合成", "真实DSM") for m in MAIN_COMPARE if len(frame[(frame.domain == d) & (frame.model == m)])]
    labels = []
    for i, (domain, model) in enumerate(groups):
        values = frame[(frame.domain == domain) & (frame.model == model)].safe_weighted_coverage.to_numpy(float)
        if len(values) >= 2:
            vio = ax.violinplot(values, positions=[i], vert=False, widths=.72, showextrema=False)
            for body in vio["bodies"]:
                body.set_facecolor(_color(model)); body.set_edgecolor("none"); body.set_alpha(.18)
        jitter = np.linspace(-.20, .20, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(values, i + jitter, s=9, color=_color(model), alpha=.62, edgecolor="white", linewidth=.2, zorder=3)
        q25, med, q75 = np.quantile(values, [.25, .5, .75])
        ax.plot([q25, q75], [i, i], color="#20252B", lw=1.8, zorder=4)
        ax.scatter([med], [i], color="#20252B", s=15, zorder=5)
        labels.append(f"{domain}｜{_label(model)}")
    ax.set_yticks(range(len(groups)), labels)
    ax.set_xlim(-.03, 1.03)
    ax.set_xlabel("安全加权覆盖率")
    _clean(ax, xgrid=True)
    return fig


def plot_interval(frame: pd.DataFrame, category: str, value: str, xlabel: str, model_order: Sequence[str] = CORE_MODELS) -> plt.Figure:
    fig, ax = _new_ax(wide=True)
    summary = _quantile_summary(frame, [category, "model"], value)
    cats = list(dict.fromkeys(frame[category].astype(str)))
    positions, labels = [], []
    idx = 0
    for cat in cats:
        for model in model_order:
            row = summary[(summary[category].astype(str) == cat) & (summary.model == model)]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.errorbar(row["median"], idx, xerr=[[row["median"] - row.q25], [row.q75 - row["median"]]], fmt=MARKERS.get(model, "o"), ms=4.2, color=_color(model), capsize=2.2, lw=1.0)
            positions.append(idx); labels.append(f"{cat}｜{_label(model)}"); idx += 1
        idx += .45
    ax.set_yticks(positions, labels)
    ax.set_xlabel(xlabel)
    _clean(ax, xgrid=True)
    return fig


def plot_forest(frame: pd.DataFrame, effect: str, low: str, high: str, label_cols: Sequence[str], xlabel: str) -> plt.Figure:
    fig, ax = _new_ax(wide=True)
    data = frame.reset_index(drop=True)
    y = np.arange(len(data))[::-1]
    for i, (_, row) in enumerate(data.iterrows()):
        model = str(row.get("comparator", row.get("model", "full")))
        ax.errorbar(row[effect], y[i], xerr=[[row[effect] - row[low]], [row[high] - row[effect]]], fmt=MARKERS.get(model, "o"), ms=4.4, color=_color(model), ecolor=_color(model), capsize=2.2, lw=1.05)
    ax.axvline(0, color="#5F6670", lw=.75, ls="--")
    labels = ["｜".join(_label(str(row[c])) if c in ("comparator", "model") else str(row[c]) for c in label_cols) for _, row in data.iterrows()]
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    _clean(ax, xgrid=True)
    return fig


def plot_m05(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = _new_ax(wide=True)
    metrics = ["能耗", "航程", "总任务时间"]
    positions, labels = [], []
    idx = 0
    for metric in metrics:
        for model in MAIN_COMPARE:
            row = frame[(frame.metric == metric) & (frame.model == model)]
            if row.empty: continue
            r = row.iloc[0]
            ax.scatter(r.utilization, idx, s=27, color=_color(model), marker=MARKERS.get(model, "o"), edgecolor="white", linewidth=.35)
            ax.text(r.utilization + .018, idx, f"{r.raw_value:.1f} {r.unit}｜安全{100*r.safe_share:.0f}%", va="center", fontsize=6.1)
            positions.append(idx); labels.append(f"{metric}｜{_label(model)}"); idx += 1
        idx += .5
    ax.set_yticks(positions, labels)
    ax.set_xlabel("预算利用率（仅安全路线）")
    ax.set_xlim(left=0)
    _clean(ax, xgrid=True)
    return fig


def plot_ecdf(frame: pd.DataFrame, x: str, xlabel: str) -> plt.Figure:
    fig, ax = _new_ax(wide=True)
    models = [m for m in tuple(CORE_MODELS) + BASELINES if m in set(frame.model)]
    for model in models:
        sub = frame[frame.model == model].sort_values(x)
        ax.plot(sub[x], sub.ecdf, color=_color(model), lw=1.15 if model in CORE_MODELS else .85, label=_label(model), alpha=.95)
    ax.set_xlabel(xlabel); ax.set_ylabel("累计比例")
    if (frame[x] > 0).all() and frame[x].max() / max(frame[x].min(), 1e-12) > 30:
        ax.set_xscale("log")
    ax.set_ylim(0, 1.015)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    _clean(ax, xgrid=True)
    return fig


def plot_m07(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = _new_ax(wide=True)
    for _, row in frame.iterrows():
        marker = "o" if row.domain == "synthetic" else "^"
        ax.scatter(row.planning_time_p95_s, row.D1, s=28 + 78 * row.safe_rate, marker=marker, color=_color(row.model), edgecolor="white", linewidth=.55)
        ax.annotate(_label(row.model), (row.planning_time_p95_s, row.D1), xytext=(4, 3), textcoords="offset points", fontsize=6.2)
    ax.set_xscale("log")
    ax.set_xlabel("P95在线规划时间（s，对数轴）"); ax.set_ylabel("任务效能 D1")
    _clean(ax, xgrid=True)
    return fig


def plot_learning(frame: pd.DataFrame, all_models: bool = False) -> plt.Figure:
    fig, ax = _new_ax(wide=True)
    models = LEARNING_MODELS if all_models else CORE_MODELS
    for model in models:
        raw = frame[(frame.model == model) & (frame.series == "seed")]
        if not all_models:
            for _, sub in raw.groupby("training_seed"):
                ax.plot(sub.environment_interactions, sub.weighted_coverage, color=_color(model), alpha=.13, lw=.45)
        med = frame[(frame.model == model) & (frame.series == "median")].sort_values("environment_interactions")
        ax.fill_between(med.environment_interactions.to_numpy(float), med.q25.to_numpy(float), med.q75.to_numpy(float), color=_color(model), alpha=.10, linewidth=0)
        ax.plot(med.environment_interactions, med.weighted_coverage, color=_color(model), lw=1.45 if model in CORE_MODELS else .95, label=_label(model))
    ax.set_xlabel("累计环境交互数"); ax.set_ylabel("共同定义的加权覆盖率")
    ax.set_ylim(-.02, 1.03)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    _clean(ax, xgrid=True)
    return fig


def plot_m09(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = _new_ax(wide=False)
    for i, model in enumerate(CORE_MODELS):
        sub = frame[frame.model == model].sort_values("training_seed")
        x = i + np.linspace(-.12, .12, len(sub))
        for xx, (_, row) in zip(x, sub.iterrows()):
            ax.scatter(xx, row.interactions_display, s=29, marker="o", facecolor=_color(model) if row.reached else "white", edgecolor=_color(model), linewidth=.9)
        med = sub.interactions_display.median()
        ax.plot([i-.20, i+.20], [med, med], color="#222222", lw=1.5)
    ax.set_xticks(range(3), [_label(m) for m in CORE_MODELS], rotation=18, ha="right")
    ax.set_ylabel("达到阈值所需环境交互数")
    _clean(ax, xgrid=False)
    return fig


def plot_heatmap(frame: pd.DataFrame, row: str, col: str, value: str, cmap: str = "YlGnBu", center: float | None = None) -> plt.Figure:
    fig, ax = _new_ax(wide=True)
    pivot = frame.pivot_table(index=row, columns=col, values=value, aggfunc="mean")
    norm = TwoSlopeNorm(vmin=np.nanmin(pivot.values), vcenter=center, vmax=np.nanmax(pivot.values)) if center is not None and np.nanmin(pivot.values) < center < np.nanmax(pivot.values) else None
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(pivot.columns)), [str(v) for v in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), [_label(str(v)) if row == "model" else str(v) for v in pivot.index])
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.iloc[i, j]
            if np.isfinite(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.8, color="white" if v > np.nanmedian(pivot.values) else "#20252B")
    fig.colorbar(im, ax=ax, fraction=.028, pad=.02)
    _clean(ax)
    return fig


def plot_s02(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = _new_ax(wide=True)
    for _, row in frame.iterrows():
        xerr = [[row.regret_low], [row.regret_high]]
        ax.scatter(row.planning_time_s, (row.regret_low + row.regret_high) / 2, s=37, marker=MARKERS.get(row.model, "o"), color=_color(row.model), edgecolor="white", linewidth=.45)
        ax.vlines(row.planning_time_s, row.regret_low, row.regret_high, color=_color(row.model), lw=1.0)
        ax.annotate(_label(row.model), (row.planning_time_s, (row.regret_low + row.regret_high) / 2), xytext=(4, 3), textcoords="offset points", fontsize=6.2)
    ax.set_xscale("log")
    ax.set_xlabel("在线规划时间中位数（s，对数轴）"); ax.set_ylabel("Oracle regret 区间")
    _clean(ax, xgrid=True)
    return fig


def plot_resource(frame: pd.DataFrame, ylabel: str) -> plt.Figure:
    fig, ax = _new_ax(wide=True)
    models = [m for m in MAIN_COMPARE if m in set(frame.model)]
    for i, model in enumerate(models):
        vals = frame[frame.model == model].value.to_numpy(float)
        if len(vals) > 1:
            vio = ax.violinplot(vals, positions=[i], widths=.72, showextrema=False)
            for body in vio["bodies"]:
                body.set_facecolor(_color(model)); body.set_edgecolor("none"); body.set_alpha(.22)
        jitter = np.linspace(-.16, .16, len(vals))
        ax.scatter(i+jitter, vals, s=8, color=_color(model), alpha=.58, edgecolor="white", linewidth=.18)
        ax.scatter(i, np.median(vals), marker="_", s=100, color="#20252B", linewidth=1.4)
        ax.text(i, np.nanmax(vals)*1.02, f"安全{100*frame[frame.model==model].safe_share.mean():.0f}%", ha="center", fontsize=5.7)
    ax.set_xticks(range(len(models)), [_label(m) for m in models], rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    _clean(ax)
    return fig


def plot_s07(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = _new_ax(wide=True)
    _draw_dot_matrix(ax, frame.rename(columns={"normalized_higher_better": "plot_value"}), "metric", "plot_value")
    ax.set_xlabel("归一化得分（越大越好）")
    return fig


def plot_s09(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = _new_ax(wide=False)
    values = frame.full_minus_a2c_points.to_numpy(float)
    ax.hist(values, bins=38, density=True, color="#2369BD", alpha=.72, edgecolor="white", linewidth=.35)
    mean, med = values.mean(), np.median(values)
    lo, hi = np.quantile(values, [.025, .975])
    ax.axvline(0, color="#636A73", ls="--", lw=.85)
    ax.axvline(med, color="#112D4E", lw=1.25)
    ax.axvspan(lo, hi, color="#8DB7E2", alpha=.22)
    ax.text(.03, .96, f"均值 {mean:.2f} 分\n中位数 {med:.2f} 分\n95%区间 [{lo:.2f}, {hi:.2f}]\nP(差值>0)={(values>0).mean():.3f}", transform=ax.transAxes, va="top", fontsize=6.7)
    ax.set_xlabel("PPO+Pointer − A2C+Pointer（分）"); ax.set_ylabel("概率密度")
    _clean(ax)
    return fig


def _terrain_and_routes(ax: plt.Axes, ctx: Context, frame: pd.DataFrame, task_id: str) -> None:
    task = frozen_io._task_by_id(ctx.bundle, task_id)
    map_id = str(task["map_id"])
    with np.load(frozen_io._map_bundle_path(map_id), allow_pickle=False) as data:
        terrain = np.asarray(data["terrain"], dtype=float)
    ax.imshow(terrain, origin="lower", cmap="terrain", alpha=.90, extent=(0, terrain.shape[1]-1, 0, terrain.shape[0]-1))
    for segment in frozen_io._road_segments(map_id):
        ax.plot(segment[:,0], segment[:,1], color="white", lw=1.8, alpha=.85)
        ax.plot(segment[:,0], segment[:,1], color="#5F6268", lw=.65, alpha=.95)
    pts = frame[frame.record_type == "inspection_point"]
    pcolors = {1: "#A8B5C2", 2: "#E9A23B", 3: "#D1495B"}
    for priority, sub in pts.groupby("priority"):
        ax.scatter(sub.x, sub.y, s={1:16,2:25,3:38}[int(priority)], color=pcolors[int(priority)], edgecolor="white", linewidth=.45, label=f"优先级{int(priority)}", zorder=5)
    airport = frame[frame.record_type == "airport"].iloc[0]
    ax.scatter(airport.x, airport.y, marker="P", s=72, color="#111827", edgecolor="white", linewidth=.7, label="机场", zorder=7)
    for model in ("full", "a2c_pointer", "traditional_ppo", "milp"):
        route = frame[(frame.record_type == "route") & (frame.model == model)].sort_values("order")
        if route.empty:
            ax.plot([], [], color=_color(model), ls="--", label=f"{_label(model)}（缺失/失败）")
        else:
            ax.plot(route.x, route.y, color=_color(model), lw=1.35 if model == "full" else 1.0, alpha=.95, label=_label(model), zorder=6)
    ax.set_aspect("equal"); ax.set_xlabel("局部东向坐标"); ax.set_ylabel("局部北向坐标")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", ncol=3, loc="upper right", fontsize=6.1)


def plot_route2d(ctx: Context, frame: pd.DataFrame, task_id: str) -> plt.Figure:
    fig, ax = _new_ax(wide=True, route=True)
    _terrain_and_routes(ax, ctx, frame, task_id)
    return fig


def plot_v01(ctx: Context, frame: pd.DataFrame) -> plt.Figure:
    fig, ax = _new_ax(wide=True, route=True, projection="3d")
    task = frozen_io._task_by_id(ctx.bundle, REAL_EXAMPLE)
    map_id = str(task["map_id"])
    with np.load(frozen_io._map_bundle_path(map_id), allow_pickle=False) as data:
        terrain = np.asarray(data["terrain"], dtype=float)
    sy = max(1, terrain.shape[0] // 70); sx = max(1, terrain.shape[1] // 70)
    yy, xx = np.mgrid[0:terrain.shape[0]:sy, 0:terrain.shape[1]:sx]
    zz = terrain[::sy, ::sx]
    ax.plot_surface(xx[:zz.shape[0], :zz.shape[1]], yy[:zz.shape[0], :zz.shape[1]], zz, cmap="terrain", alpha=.72, linewidth=0, antialiased=True)
    route = frame[(frame.record_type == "route") & (frame.model == "full")].sort_values("order")
    if not route.empty:
        ax.plot(route.x, route.y, route.z, color=_color("full"), lw=2.0, label="PPO+Pointer路线")
    pts = frame[frame.record_type == "inspection_point"]
    ax.scatter(pts.x, pts.y, pts.z, c=pts.priority, cmap="YlOrRd", s=20, edgecolor="white", linewidth=.25, label="固定巡检点")
    airport = frame[frame.record_type == "airport"].iloc[0]
    ax.scatter(airport.x, airport.y, airport.z, marker="P", s=65, color="#111827", label="机场")
    ax.set_xlabel("东向"); ax.set_ylabel("北向"); ax.set_zlabel("高程")
    ax.view_init(elev=34, azim=-58)
    ax.legend(frameon=False, loc="upper left")
    return fig


def plot_v02(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = _new_ax(wide=True, route=True)
    algorithms = [m for m in MAIN_COMPARE if m in set(frame.model)]
    coverages = ["低覆盖", "中覆盖", "高覆盖"]
    terms = ["安全返航", "未安全完成", "约束违规", "Stranded"]
    left_y = {m: i for i, m in enumerate(algorithms)}
    mid_y = {c: i * max(1, len(algorithms)-1) / max(1, len(coverages)-1) for i, c in enumerate(coverages)}
    right_y = {t: i * max(1, len(algorithms)-1) / max(1, len(terms)-1) for i, t in enumerate(terms)}
    maxw = frame.weight.max()
    for _, row in frame.iterrows():
        y0, y1, y2 = left_y[row.model], mid_y[str(row.coverage_level)], right_y[row.termination_group]
        verts = [(0, y0), (.42, y0), (.58, y1), (1, y1)]
        path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=_color(row.model), lw=.35 + 5*row.weight/maxw, alpha=.25))
        verts2 = [(1, y1), (1.42, y1), (1.58, y2), (2, y2)]
        ax.add_patch(PathPatch(MplPath(verts2, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]), facecolor="none", edgecolor=_color(row.model), lw=.35 + 5*row.weight/maxw, alpha=.25))
    for x, mapping, formatter in ((0,left_y,_label),(1,mid_y,str),(2,right_y,str)):
        for key, y in mapping.items():
            ax.scatter(x, y, s=26, color="#F8FAFC", edgecolor="#374151", linewidth=.75, zorder=4)
            ax.text(x + (-.04 if x==2 else .04), y, formatter(key), ha="right" if x==2 else "left", va="center", fontsize=6.5)
    ax.text(0, len(algorithms)-.35, "算法", ha="center", weight="bold")
    ax.text(1, len(algorithms)-.35, "覆盖水平", ha="center", weight="bold")
    ax.text(2, len(algorithms)-.35, "终止结果", ha="center", weight="bold")
    ax.set_xlim(-.35, 2.35); ax.set_ylim(-.55, len(algorithms)-.05); ax.axis("off")
    return fig


def render_matplotlib(figure_id: str, frame: pd.DataFrame, ctx: Context) -> plt.Figure:
    if figure_id == "M01": return plot_m01(frame)
    if figure_id == "M02": return plot_m02(frame)
    if figure_id == "M03": return plot_interval(frame, "stratum", "coverage", "覆盖率", CORE_MODELS + ("no_priority_bias",))
    if figure_id == "M04": return plot_forest(frame, "effect", "ci_low", "ci_high", ["domain", "metric", "comparator"], "PPO+Pointer − 比较算法（比例点）")
    if figure_id == "M05": return plot_m05(frame)
    if figure_id == "M06": return plot_ecdf(frame, "planning_time_s", "在线规划时间（s）")
    if figure_id == "M07": return plot_m07(frame)
    if figure_id == "M08": return plot_learning(frame)
    if figure_id == "M09": return plot_m09(frame)
    if figure_id == "M10": return plot_interval(frame.rename(columns={"domain":"stratum", "D1_mission_effectiveness":"value"}), "stratum", "value", "任务效能 D1", CORE_MODELS)
    if figure_id == "M11": return plot_heatmap(frame.assign(condition_label=frame.family.str.replace("known_domain_shift","已知").str.replace("hidden_model_perception_mismatch","隐藏")+"｜"+frame.condition), "model", "condition_label", "retention", "YlGnBu")
    if figure_id == "M12": return plot_forest(frame.rename(columns={"hodges_lehmann":"effect", "bootstrap_ci_low":"ci_low", "bootstrap_ci_high":"ci_high"}), "effect", "ci_low", "ci_high", ["domain", "comparator"], "完整模型 − 消融（安全加权覆盖率）")
    if figure_id == "S01": return plot_ecdf(frame, "regret", "安全加权覆盖率遗憾")
    if figure_id == "S02": return plot_s02(frame)
    if figure_id == "S03": return plot_resource(frame, "能耗（Wh，仅安全路线）")
    if figure_id == "S04": return plot_resource(frame, "航程（km，仅安全路线）")
    if figure_id == "S05": return plot_resource(frame, "总任务时间（min，仅安全路线）")
    if figure_id == "S06": return plot_learning(frame, all_models=True)
    if figure_id == "S07": return plot_s07(frame)
    if figure_id == "S08": return plot_heatmap(frame, "training_weight", "operational_floor", "first_share", "YlGnBu")
    if figure_id == "S09": return plot_s09(frame)
    if figure_id == "S10": return plot_heatmap(frame.assign(row_label=frame.model.map(_label)+"｜"+frame.metric), "row_label", "condition", "higher_better", "YlGnBu")
    if figure_id == "S11": return plot_route2d(ctx, frame, SYNTHETIC_EXAMPLE)
    if figure_id == "S12": return plot_route2d(ctx, frame, REAL_EXAMPLE)
    if figure_id == "V01": return plot_v01(ctx, frame)
    if figure_id == "V02": return plot_v02(frame)
    raise KeyError(figure_id)


def _trim_raster(path: Path) -> None:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        bg = Image.new("RGB", rgb.size, (255, 255, 255))
        bbox = ImageChops.difference(rgb, bg).getbbox()
        if bbox is None:
            return
        left = max(0, bbox[0] - SAFE_MARGIN_PX); top = max(0, bbox[1] - SAFE_MARGIN_PX)
        right = min(rgb.width, bbox[2] + SAFE_MARGIN_PX); bottom = min(rgb.height, bbox[3] + SAFE_MARGIN_PX)
        cropped = rgb.crop((left, top, right, bottom))
        if path.suffix.lower() in (".tif", ".tiff"):
            cropped.save(path, dpi=(EXPORT_DPI, EXPORT_DPI), compression="tiff_lzw")
        else:
            cropped.save(path, dpi=(EXPORT_DPI, EXPORT_DPI))


def _output_stem(figure_id: str) -> Path:
    meta = FIGURES[figure_id]
    folder = {"main":"main", "supplementary":"supplementary", "showcase":"showcase"}[meta["tier"]]
    return OUTPUT / folder / f"{figure_id}_{meta['name']}"


def export_matplotlib(figure_id: str, fig: plt.Figure) -> dict[str, str]:
    stem = _output_stem(figure_id)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for ext in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{ext}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": .045, "facecolor": "white"}
        if ext == "png": kwargs["dpi"] = EXPORT_DPI
        fig.savefig(path, **kwargs)
        if ext == "png": _trim_raster(path)
        outputs[ext] = str(path)
    # 先生成已裁切PNG再转TIFF，规避部分Pillow/Matplotlib组合直接写LZW TIFF时的崩溃。
    tiff = stem.with_suffix(".tiff")
    with Image.open(stem.with_suffix(".png")) as image:
        rgb = image.convert("RGB")
        if tifffile is None:
            rgb.save(tiff, format="TIFF", compression="raw", dpi=(EXPORT_DPI, EXPORT_DPI))
        else:
            array = np.asarray(rgb)
            tifffile.imwrite(tiff, array, photometric="rgb", resolution=(EXPORT_DPI, EXPORT_DPI), resolutionunit="INCH")
    outputs["tiff"] = str(tiff)
    plt.close(fig)
    return outputs


LITERATURE = [
    ("POMO", "Kwon et al.", "NeurIPS 2020", "https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html", ["验证学习曲线", "解质量与gap表图", "推理时间比较"]),
    ("Deep RL at the Edge of the Statistical Precipice", "Agarwal et al.", "NeurIPS 2021", "https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html", ["区间估计", "Performance profile", "稳健聚合比较"]),
    ("Procgen Benchmark", "Cobbe et al.", "ICML 2020", "https://proceedings.mlr.press/v119/cobbe20a.html", ["训练测试曲线", "未见环境泛化", "难度分层"]),
    ("Zero-shot Coverage Path Planning", "IEEE/CAA JAS authors", "IEEE/CAA JAS 2024", "https://www.ieee-jas.com/en/article/doi/10.1109/JAS.2024.125064", ["未见地图案例", "消融点区间", "覆盖路线解释"]),
    ("NHDE", "Zhang et al.", "NeurIPS 2023", "https://proceedings.neurips.cc/paper_files/paper/2023/file/7b5ae891000049b91b3b62de596b1560-Paper-Conference.pdf", ["Pareto前沿", "性能时间权衡", "多目标消融"]),
    ("Attention, Learn to Solve Routing Problems", "Kool et al.", "ICLR 2019", "https://openreview.net/forum?id=ByxBFsRqYm", ["训练曲线", "最优gap比较", "规模性能曲线"]),
    ("Reinforcement Learning for Solving the VRP", "Nazari et al.", "NeurIPS 2018", "https://proceedings.neurips.cc/paper/2018/hash/9fb4651c05b2ed70fba5afe0b039a550-Abstract.html", ["收敛过程", "路线案例", "基线性能比较"]),
    ("Neural Combinatorial Optimization", "Bello et al.", "ICLR 2017", "https://openreview.net/forum?id=Bk9mxlSFx", ["学习曲线", "主动搜索过程", "解质量分布"]),
    ("Pointer Networks", "Vinyals et al.", "NeurIPS 2015", "https://proceedings.neurips.cc/paper/2015/hash/29921001f2f04bd3baee84a12e98098f-Abstract.html", ["模型示意", "规模误差曲线", "排序案例"]),
    ("Sym-NCO", "Kim et al.", "NeurIPS 2022", "https://proceedings.neurips.cc/paper_files/paper/2022/hash/0cddb777d3441326544e21b67f41bdc8-Abstract-Conference.html", ["训练曲线", "对称增强消融", "gap比较"]),
    ("Efficient Active Search", "Hottung et al.", "ICLR 2022", "https://openreview.net/forum?id=nO5caZwFwYu", ["测试时改进曲线", "时间质量权衡", "规模比较"]),
    ("DACT", "Ma et al.", "NeurIPS 2021", "https://proceedings.neurips.cc/paper_files/paper/2021/hash/5c53292c032b6cb8510041c54274e65f-Abstract.html", ["迭代改进曲线", "消融比较", "推理时间"]),
    ("NeuroLKH", "Xin et al.", "NeurIPS 2021", "https://proceedings.neurips.cc/paper/2021/hash/3d863b367aa379f71c7afc0c9cdca41d-Abstract.html", ["gap性能表图", "规模扩展", "运行时间"]),
    ("AMDKD", "Bi et al.", "NeurIPS 2022", "https://proceedings.neurips.cc/paper_files/paper/2022/hash/ca70528fb11dc8086c6a623da9f3fee6-Abstract-Conference.html", ["跨分布曲线", "知识蒸馏消融", "泛化比较"]),
    ("PIP-D", "Bi et al.", "NeurIPS 2024", "https://proceedings.neurips.cc/paper_files/paper/2024/hash/a9d2a5fd12d34250c21b5e4fa8d906b0-Abstract-Conference.html", ["可行率比较", "约束消融", "性能时间图"]),
    ("RL4CO", "Berto et al.", "KDD 2025", "https://doi.org/10.1145/3711896.3737433", ["算法基准矩阵", "学习曲线", "计算代价"]),
    ("DeepACO", "Ye et al.", "NeurIPS 2023", "https://proceedings.neurips.cc/paper_files/paper/2023/hash/883105b282fe15275991b411e6b200c5-Abstract-Conference.html", ["迭代收敛曲线", "跨问题泛化", "消融热力图"]),
    ("DIFUSCO", "Sun and Yang", "NeurIPS 2023", "https://proceedings.neurips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html", ["扩散过程示意", "解质量时间", "规模比较"]),
    ("Learning to Perform Local Rewriting", "Chen and Tian", "NeurIPS 2019", "https://proceedings.neurips.cc/paper/2019/hash/131f383b434fdf48079bff1e44e2d9a5-Abstract.html", ["局部改写过程", "学习曲线", "路线质量比较"]),
    ("Learning Combinatorial Optimization Algorithms over Graphs", "Khalil et al.", "NeurIPS 2017", "https://proceedings.neurips.cc/paper_files/paper/2017/hash/d9896106ca98d3d05b8cbdf4fd8b13a1-Abstract.html", ["训练过程", "规模泛化", "基线性能"]),
]


def build_literature_audit() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for title, authors, venue, url, archetypes in LITERATURE:
        for i, archetype in enumerate(archetypes, start=1):
            rows.append({
                "paper": title, "authors": authors, "venue": venue, "primary_url": url,
                # 不臆造可能随出版版本变化的图号；标题、图型目的和主来源共同构成稳定定位键。
                "figure_locator": f"{title}｜实验图型条目{i}", "figure_type": archetype,
                "evidence_purpose": archetype, "palette": "克制、色盲安全、算法色固定",
                "line_and_marker": "细线+点形冗余编码", "legend": "图内或上方紧凑图例",
                "adopted_principle": "只借鉴证据组织与视觉语法，不复制原图",
            })
    frame = pd.DataFrame(rows)
    if frame.paper.nunique() < 20 or len(frame) < 50:
        raise RuntimeError("文献图型审计未达到20篇/50幅")
    _write_csv(OUTPUT / "literature_audit" / "literature_style_audit.csv", frame)
    return frame


def build_registry(sources: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    records = []
    for figure_id, meta in FIGURES.items():
        source_path = OUTPUT / "source_data" / f"{figure_id}_source_data.csv"
        records.append({
            "figure_id": figure_id, **meta, "claim_question": CAPTIONS[figure_id],
            "source_data": str(source_path.relative_to(OUTPUT)), "source_sha256": _sha256(source_path),
            "row_count": int(len(sources[figure_id])), "independent_unit": "地图（任务与种子嵌套）",
            "aggregation_order": "重复/训练种子→任务→地图", "error_definition": "按图注；确认性结果读取冻结统计",
            "colors": {m: COLORS[m] for m in COLORS}, "output_stem": _output_stem(figure_id).name,
        })
    payload = {"schema_version": "v3.2.14-origin-redraw-v2", "figure_count": len(records), "figures": records}
    _write_json(OUTPUT / "manifests" / "figure_registry.json", payload)
    return payload


def _origin_plot_data(figure_id: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str]:
    """把各图整理成 Origin 可编辑的 XY/XYZ 表；不改变原始 SourceData。"""
    kind = FIGURES[figure_id]["kind"]
    if figure_id == "M01":
        f = frame.copy(); f["x"] = np.where(f.metric.eq("综合得分"), f.display_value / 100, f.value); f["y"] = f.metric.astype("category").cat.codes
    elif figure_id == "M02":
        f = frame.copy(); f["x"] = f.safe_weighted_coverage; f["y"] = (f.domain+"｜"+f.model).astype("category").cat.codes
    elif figure_id == "M03":
        f = frame.copy(); f["x"] = f.coverage; f["y"] = f.stratum.astype("category").cat.codes
    elif figure_id in ("M04",):
        f = frame.copy(); f["x"] = f.effect; f["y"] = np.arange(len(f))
    elif figure_id == "M05":
        f = frame.copy(); f["x"] = f.utilization; f["y"] = (f.metric+"｜"+f.model).astype("category").cat.codes
    elif figure_id in ("M06", "S01"):
        xcol = "planning_time_s" if figure_id == "M06" else "regret"; f = frame.rename(columns={xcol:"x", "ecdf":"y"}).copy()
    elif figure_id == "M07":
        f = frame.rename(columns={"planning_time_p95_s":"x", "D1":"y"}).copy()
    elif figure_id in ("M08", "S06"):
        f = frame[frame.series == "median"].rename(columns={"environment_interactions":"x", "weighted_coverage":"y"}).copy()
    elif figure_id == "M09":
        f = frame.copy(); f["x"] = f.model.astype("category").cat.codes; f["y"] = f.interactions_display
    elif figure_id == "M10":
        f = frame.copy(); f["x"] = f.D1_mission_effectiveness; f["y"] = f.domain.astype("category").cat.codes
    elif figure_id == "M12":
        f = frame.copy(); f["x"] = f.hodges_lehmann; f["y"] = np.arange(len(f))
    elif figure_id == "S02":
        f = frame.copy(); f["x"] = f.planning_time_s; f["y"] = (f.regret_low+f.regret_high)/2
    elif figure_id in ("S03", "S04", "S05"):
        f = frame.copy(); f["x"] = f.model.astype("category").cat.codes; f["y"] = f.value
    elif figure_id == "S07":
        f = frame.copy(); f["x"] = f.normalized_higher_better; f["y"] = f.metric.astype("category").cat.codes
    elif figure_id == "S09":
        vals = frame.full_minus_a2c_points.to_numpy(float); hist, edges = np.histogram(vals, bins=50, density=True); f = pd.DataFrame({"model":"full", "x":(edges[:-1]+edges[1:])/2, "y":hist})
    elif figure_id == "V02":
        f = frame.copy(); f["x"] = f.coverage_level.astype("category").cat.codes; f["y"] = f.weight
    elif figure_id == "V01":
        f = frame[(frame.record_type == "route") & (frame.model == "full")].rename(columns={"x":"x0", "y":"y0", "z":"z0"}).copy(); f["x"], f["y"], f["z"] = f.x0, f.y0, f.z0
    elif kind == "heatmap":
        if figure_id == "M11":
            f = frame.copy(); f["x"] = (f.family+"｜"+f.condition).astype("category").cat.codes; f["y"] = f.model.astype("category").cat.codes; f["z"] = f.retention
        elif figure_id == "S08":
            f = frame.copy(); f["x"] = f.operational_floor; f["y"] = f.training_weight; f["z"] = f.first_share
        else:
            f = frame.copy(); f["x"] = f.condition.astype("category").cat.codes; f["y"] = (f.model+"｜"+f.metric).astype("category").cat.codes; f["z"] = f.higher_better
    else:
        raise KeyError(figure_id)
    if kind == "heatmap": return f, "heatmap", "条件", "模型/指标"
    if figure_id == "V01": return f, "3d", "东向", "北向"
    mode = "line" if kind in ("ecdf", "learning_curve") or figure_id == "S09" else "scatter"
    return f, mode, "指标值", "分组/效能"


def _wide_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    series_col = "model" if "model" in frame else None
    groups = [("series", frame)] if series_col is None else list(frame.groupby(series_col, sort=False))
    columns: dict[str, pd.Series] = {}; names = []
    for i, (name, sub) in enumerate(groups, start=1):
        sub = sub[["x", "y"]].dropna().reset_index(drop=True)
        columns[f"x_{i}"] = sub.x; columns[f"y_{i}"] = sub.y; names.append(str(name))
    return pd.DataFrame(columns), names


def _origin_category_labels(figure_id: str, frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    xlabels: list[str] = []; ylabels: list[str] = []
    if figure_id == "M01": ylabels = list(frame.metric.astype("category").cat.categories)
    elif figure_id == "M02": ylabels = list((frame.domain+"｜"+frame.model.map(_label)).astype("category").cat.categories)
    elif figure_id == "M03": ylabels = list(frame.stratum.astype("category").cat.categories)
    elif figure_id == "M04": ylabels = [f"{r.domain}｜{r.metric}｜{_label(r.comparator)}" for _, r in frame.iterrows()]
    elif figure_id == "M05": ylabels = list((frame.metric+"｜"+frame.model.map(_label)).astype("category").cat.categories)
    elif figure_id == "M09": xlabels = list(frame.model.astype("category").cat.categories.map(_label))
    elif figure_id == "M10": ylabels = list(frame.domain.astype("category").cat.categories)
    elif figure_id == "M12": ylabels = [f"{r.domain}｜{_label(r.comparator)}" for _, r in frame.iterrows()]
    elif figure_id in ("S03","S04","S05"): xlabels = list(frame.model.astype("category").cat.categories.map(_label))
    elif figure_id == "S07": ylabels = list(frame.metric.astype("category").cat.categories)
    elif figure_id == "M11":
        xlabels = list((frame.family+"｜"+frame.condition).astype("category").cat.categories)
        ylabels = list(frame.model.astype("category").cat.categories.map(_label))
    elif figure_id == "S10":
        xlabels = list(frame.condition.astype("category").cat.categories)
        ylabels = list((frame.model.map(_label)+"｜"+frame.metric).astype("category").cat.categories)
    return xlabels, ylabels


def _origin_violin_data(figure_id: str, source: pd.DataFrame) -> pd.DataFrame:
    groups: list[tuple[str, pd.Series]] = []
    if figure_id == "M02":
        for (domain, model), sub in source.groupby(["domain", "model"], sort=False):
            groups.append((f"{domain}｜{_label(model)}", sub.safe_weighted_coverage.reset_index(drop=True)))
    else:
        for model, sub in source.groupby("model", sort=False):
            groups.append((_label(model), sub.value.reset_index(drop=True)))
    return pd.DataFrame({name: values for name, values in groups})


def _origin_raincloud_data(figure_id: str, source: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """生成Origin原生XY半小提琴轮廓与地图级散点，避免分组模板的不可控彩虹配色。"""
    groups: list[tuple[str, str, np.ndarray]] = []
    if figure_id == "M02":
        for (domain, model), sub in source.groupby(["domain", "model"], sort=False):
            groups.append((f"{domain}｜{_label(model)}", str(model), sub.safe_weighted_coverage.to_numpy(float)))
    else:
        for model, sub in source.groupby("model", sort=False):
            groups.append((_label(str(model)), str(model), sub.value.to_numpy(float)))

    columns: dict[str, pd.Series] = {}
    names: list[str] = []
    plot_modes: list[str] = []
    labels: list[str] = []
    for i, (label, model, raw) in enumerate(groups, start=1):
        values = raw[np.isfinite(raw)]
        labels.append(label)
        spread = max(float(np.ptp(values)), max(abs(float(np.nanmedian(values))), 1.0) * 0.02)
        bandwidth = max(1.06 * float(np.nanstd(values, ddof=1)) * max(len(values), 2) ** (-0.2), spread / 30, 1e-6)
        grid = np.linspace(float(np.nanmin(values)) - 0.08 * spread, float(np.nanmax(values)) + 0.08 * spread, 90)
        density = np.exp(-0.5 * ((grid[:, None] - values[None, :]) / bandwidth) ** 2).mean(axis=1)
        density = density / max(float(density.max()), 1e-12)
        columns[f"x_{2*i-1}"] = pd.Series(i + 0.30 * density)
        columns[f"y_{2*i-1}"] = pd.Series(grid)
        names.append(f"{model}｜density"); plot_modes.append("line")
        jitter = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else np.zeros(1)
        columns[f"x_{2*i}"] = pd.Series(i - 0.13 + jitter)
        columns[f"y_{2*i}"] = pd.Series(values)
        names.append(f"{model}｜maps"); plot_modes.append("scatter")
    return pd.DataFrame(columns), names, plot_modes, labels


def _origin_safe(values: Iterable[Any]) -> list[Any]:
    result = []
    for value in values:
        if pd.isna(value): result.append("")
        elif isinstance(value, (np.integer,)): result.append(int(value))
        elif isinstance(value, (np.floating,)): result.append(float(value))
        else: result.append(str(value))
    return result


def _lt_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', "'")


def _put_frame(app: Any, book: str, frame: pd.DataFrame) -> None:
    for col, name in enumerate(frame.columns):
        app.PutWorksheet(book, _origin_safe(frame[name].tolist()), 0, col)
        app.Execute(f"win -a {book}; wks.col{col+1}.lname$=\"{_lt_escape(name)}\";")


def _rgb_tuple(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))


def render_origin_project(figure_id: str, source: pd.DataFrame) -> dict[str, Any]:
    import win32com.client

    raincloud_modes: list[str] = []
    raincloud_labels: list[str] = []
    if figure_id in ("M02", "S03", "S04", "S05"):
        plot, names, raincloud_modes, raincloud_labels = _origin_raincloud_data(figure_id, source)
        mode = "raincloud"; xlabel = "算法/测试域"; ylabel = "安全加权覆盖率" if figure_id == "M02" else "指标值"
    else:
        plot, mode, xlabel, ylabel = _origin_plot_data(figure_id, source)
    project_path = (OUTPUT / "origin_projects" / f"{figure_id}.opju").resolve()
    native_dir = (OUTPUT / "qa" / "origin_native_exports" / figure_id).resolve()
    project_path.parent.mkdir(parents=True, exist_ok=True); native_dir.mkdir(parents=True, exist_ok=True)
    app = win32com.client.Dispatch("Origin.ApplicationSI")
    app.Visible = 0
    app.Execute("doc -s; doc -n;")
    source_book = app.CreatePage(2, "SourceData", "Origin", 2); _put_frame(app, source_book, source)
    meta_book = app.CreatePage(2, "Metadata", "Origin", 2)
    metadata = pd.DataFrame({"key":["figure_id","caption","source_sha256","row_count","renderer"], "value":[figure_id,CAPTIONS[figure_id],_sha256(OUTPUT/'source_data'/f'{figure_id}_source_data.csv'),len(source),"Origin 2021 COM+LabTalk"]})
    _put_frame(app, meta_book, metadata)
    plot_book = app.CreatePage(2, "PlotData", "Origin", 2)
    axis_label_book = None
    if mode == "raincloud":
        pframe = plot
        # 长类别名通过独立数据集驱动刻度，避免Origin 2021字符串属性长度截断。
        axis_label_book = app.CreatePage(2, "AxisLabels", "Origin", 2)
        _put_frame(app, axis_label_book, pd.DataFrame({"position": np.arange(1, len(raincloud_labels) + 1), "label": raincloud_labels}))
    elif mode in ("heatmap", "3d"):
        pframe = plot[["x","y","z"]].dropna().reset_index(drop=True); names = ["surface"]
    else:
        pframe, names = _wide_xy(plot)
    _put_frame(app, plot_book, pframe)
    if mode not in ("heatmap", "3d", "raincloud"):
        for i, name in enumerate(names, start=1):
            app.Execute(f'win -a {plot_book}; wks.col{2*i}.lname$="{_lt_escape(_label(name))}";')
    app.Execute(f"win -a {plot_book};")
    if mode == "raincloud":
        ok = True
        for i, series_mode in enumerate(raincloud_modes):
            target = "[<new template:=origin name:=Graph1>]" if i == 0 else "[Graph1]1!"
            plot_type = 200 if series_mode == "line" else 201
            ok = bool(app.Execute(f"win -a {plot_book}; plotxy iy:=({2*i+1},{2*i+2}) plot:={plot_type} ogl:={target};")) and ok
    elif mode == "heatmap":
        ok = app.Execute("wks.col1.type=4; wks.col2.type=4; wks.col3.type=6; plot_heatmapxyz iz:=(1,2,3) template:=Heat_Map;")
    elif mode == "3d":
        ok = app.Execute("wks.col1.type=4; wks.col2.type=4; wks.col3.type=6; plotxyz iz:=(1,2,3) plot:=240 ogl:=<new template:=glTraject name:=Graph1>;")
    else:
        plot_type = 200 if mode == "line" else 201
        ok = True
        for i in range(len(names)):
            target = "[<new template:=origin name:=Graph1>]" if i == 0 else "[Graph1]1!"
            ok = bool(app.Execute(f"win -a {plot_book}; plotxy iy:=({2*i+1},{2*i+2}) plot:={plot_type} ogl:={target};")) and ok
    if not ok:
        app.Execute("doc -s;"); app.Exit(); raise RuntimeError(f"Origin创建{figure_id}图页失败")
    app.Execute(f'win -a Graph1; xb.text$="{_lt_escape(xlabel)}"; yl.text$="{_lt_escape(ylabel)}"; xb.fsize=9; yl.fsize=9; layer.x.label.pt=8; layer.y.label.pt=8; layer -a; legendupdate mode:=lname; legend.fsize=8;')
    if mode not in ("heatmap", "3d"):
        for i, name in enumerate(names, start=1):
            model = str(name).split("｜", 1)[0]
            rgb = _rgb_tuple(COLORS.get(model, "#6F6F6F")); symbol = 3 + (i % 8)
            if mode == "raincloud" and raincloud_modes[i - 1] == "line":
                app.Execute(f"win -a Graph1; layer.plot={i}; set %C -cl color({rgb[0]},{rgb[1]},{rgb[2]}); set %C -k 0; set %C -wp 1.1;")
            elif mode == "raincloud":
                app.Execute(f"win -a Graph1; layer.plot={i}; set %C -cse color({rgb[0]},{rgb[1]},{rgb[2]}); set %C -csf color({rgb[0]},{rgb[1]},{rgb[2]}); set %C -k 3; set %C -z 4;")
            else:
                app.Execute(f"win -a Graph1; layer.plot={i}; set %C -cl color({rgb[0]},{rgb[1]},{rgb[2]}); set %C -cse color({rgb[0]},{rgb[1]},{rgb[2]}); set %C -csf color({rgb[0]},{rgb[1]},{rgb[2]}); set %C -k {symbol}; set %C -w 2;")
    if mode == "heatmap":
        app.Execute('win -a Graph1; set %C -cpal Heatmap4ColorBlind; layer.cmap.numMajorLevels=7; layer.cmap.numMinorLevels=0;')
    xlabels, ylabels = ([], []) if mode == "raincloud" else _origin_category_labels(figure_id, source)
    if mode == "raincloud":
        app.Execute(f'range axisLabels=[{axis_label_book}]Sheet1!col(B); win -a Graph1; layer.x.from=0; layer.x.to={len(raincloud_labels)+1}; layer.x.inc=1; axis -ps X T axisLabels; layer.x.label.rotate=45; legend.show=0;')
    if xlabels:
        joined = " ".join(str(v).replace(" ", "·") for v in xlabels)
        app.Execute(f'layer.x.from=-0.5; layer.x.to={len(xlabels)-0.5}; layer.x.inc=1; layer.x.label.type=10; layer.x.label.string$="{_lt_escape(joined)}";')
    if ylabels:
        joined = " ".join(str(v).replace(" ", "·") for v in ylabels)
        app.Execute(f'layer.y.from=-0.5; layer.y.to={len(ylabels)-0.5}; layer.y.inc=1; layer.y.label.type=10; layer.y.label.string$="{_lt_escape(joined)}";')
    app.Execute("page -fit -m 1 -b 3 -u; doc -uw;")
    saved = bool(app.Save(str(project_path)))
    export_state: dict[str, bool] = {}
    for typ in ("png", "pdf", "tif", "svg"):
        command = f'expGraph type:={typ} path:="{_lt_escape(str(native_dir))}" filename:="{figure_id}" overwrite:=replace tr.Margin:=2 tr1.Unit:=2 tr1.Width:=2400 tr2.TIF.DotsPerInch:=600 tr2.TIF.Compression:=LZW;'
        export_state[typ] = bool(app.Execute(command))
    app.Execute("doc -s;"); app.Exit()
    return {"saved": saved, "project": str(project_path), "native_exports": export_state, "series": names, "mode": mode}


def _copy_origin_exports_to_delivery(figure_id: str) -> dict[str, str]:
    if figure_id == "V01":
        # Origin 2021可稳定保存三维轨迹OPJU，但无法在同一自动化图层叠加DSM表面与道路；
        # 按预注册规则使用Python完整场景图作为最终导出，OPJU仍保留可编辑轨迹与数据。
        stem = _output_stem(figure_id)
        return {ext: str(stem.with_suffix(f".{ext}")) for ext in ("png","pdf","svg","tiff")}
    native = OUTPUT / "qa" / "origin_native_exports" / figure_id
    stem = _output_stem(figure_id); stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for src_ext, dst_ext in (("png","png"),("pdf","pdf"),("tif","tiff")):
        src = native / f"{figure_id}.{src_ext}"
        if not src.exists(): raise FileNotFoundError(src)
        dst = stem.with_suffix(f".{dst_ext}"); shutil.copy2(src, dst); outputs[dst_ext] = str(dst)
    # Origin 2021在该安装环境不能直接导出SVG；优先保留已生成的Python可编辑文本SVG。
    svg = stem.with_suffix(".svg")
    if not svg.exists():
        raise FileNotFoundError(f"缺少SVG兼容回退：{svg}")
    outputs["svg"] = str(svg)
    return outputs


def write_captions() -> None:
    root = OUTPUT / "captions_CN"; root.mkdir(parents=True, exist_ok=True)
    for figure_id, caption in CAPTIONS.items():
        (root / f"{figure_id}.md").write_text(f"# {figure_id} {FIGURES[figure_id]['name']}\n\n{caption}\n", encoding="utf-8")


def render_all(sources: Mapping[str, pd.DataFrame], ctx: Context, use_origin: bool = True) -> list[dict[str, Any]]:
    configure_style()
    records = []
    for figure_id in FIGURES:
        # Python先生成结构预览及SVG兼容母版；24张统计/展示图随后由Origin原生工程导出覆盖PDF/PNG/TIFF。
        fig = render_matplotlib(figure_id, sources[figure_id], ctx)
        py_outputs = export_matplotlib(figure_id, fig)
        origin_record = None
        if use_origin and FIGURES[figure_id]["backend"] == "origin":
            origin_record = render_origin_project(figure_id, sources[figure_id])
            final_outputs = _copy_origin_exports_to_delivery(figure_id)
        else:
            final_outputs = py_outputs
        records.append({"figure_id": figure_id, "backend": FIGURES[figure_id]["backend"], "outputs": final_outputs, "origin": origin_record})
    _write_json(OUTPUT / "manifests" / "render_manifest.json", {"figures": records})
    return records


def _load_sources() -> dict[str, pd.DataFrame]:
    sources = {}
    for figure_id in FIGURES:
        path = OUTPUT / "source_data" / f"{figure_id}_source_data.csv"
        if not path.exists(): raise FileNotFoundError(path)
        sources[figure_id] = pd.read_csv(path, encoding="utf-8-sig")
    return sources


def build_thumbnail_index() -> Path:
    thumbs = []
    for figure_id in FIGURES:
        path = _output_stem(figure_id).with_suffix(".png")
        if not path.exists(): continue
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((420, 300), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (450, 345), "white")
            tile.paste(rgb, ((450-rgb.width)//2, 28+(300-rgb.height)//2))
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(tile)
            draw.text((12, 8), f"{figure_id}  {FIGURES[figure_id]['name']}", fill="#1F2937")
            thumbs.append(tile)
    cols = 3; rows = math.ceil(len(thumbs)/cols)
    board = Image.new("RGB", (cols*450, rows*345), "#EEF1F4")
    for i, tile in enumerate(thumbs): board.paste(tile, ((i%cols)*450, (i//cols)*345))
    path = OUTPUT / "thumbnail_index" / "all_figures_thumbnail_index.png"
    path.parent.mkdir(parents=True, exist_ok=True); board.save(path, dpi=(200,200))
    return path


def qa_all() -> dict[str, Any]:
    errors: list[str] = []; figures = []
    old_digest, _ = _tree_digest(OLD_FIGURES)
    if old_digest != OLD_TREE_SHA256_AT_START:
        errors.append("旧paper_final目录发生变化")
    for figure_id, meta in FIGURES.items():
        source = OUTPUT / "source_data" / f"{figure_id}_source_data.csv"
        if not source.exists(): errors.append(f"{figure_id}缺Source Data"); continue
        frame = pd.read_csv(source, encoding="utf-8-sig")
        if frame.empty: errors.append(f"{figure_id} Source Data为空")
        if "model" in frame and "ppo_mlp" in set(frame.model.astype(str)): errors.append(f"{figure_id}混入ppo_mlp")
        file_info = {}
        for ext in ("pdf","svg","tiff","png"):
            path = _output_stem(figure_id).with_suffix(f".{ext}")
            if not path.exists() or path.stat().st_size == 0:
                errors.append(f"{figure_id}缺少{ext}")
            else:
                file_info[ext] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if meta["backend"] == "origin":
            opju = OUTPUT / "origin_projects" / f"{figure_id}.opju"
            if not opju.exists() or opju.stat().st_size < 10_000: errors.append(f"{figure_id}缺少有效OPJU")
        png = _output_stem(figure_id).with_suffix(".png")
        if png.exists():
            with Image.open(png) as image:
                if image.width < 900 or image.height < 500: errors.append(f"{figure_id} PNG分辨率不足")
                bbox = ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, "white")).getbbox()
                if bbox is None: errors.append(f"{figure_id}为空白图")
        figures.append({"figure_id":figure_id, "source_rows":len(frame), "source_sha256":_sha256(source), "files":file_info})
    if len(figures) != 26: errors.append("图形记录数不是26")
    opju_count = len(list((OUTPUT/"origin_projects").glob("*.opju"))) if (OUTPUT/"origin_projects").exists() else 0
    if opju_count != 24: errors.append(f"OPJU数量应为24，实际{opju_count}")
    report = {"passed":not errors, "errors":errors, "figure_count":len(figures), "opju_count":opju_count, "old_tree_sha256":old_digest, "old_unchanged":old_digest==OLD_TREE_SHA256_AT_START, "figures":figures}
    _write_json(OUTPUT / "qa" / "qa_report.json", report)
    if errors: raise RuntimeError("；".join(errors))
    return report


def write_figure_manifests() -> None:
    for figure_id, meta in FIGURES.items():
        source = OUTPUT / "source_data" / f"{figure_id}_source_data.csv"
        payload = {"figure_id":figure_id, **meta, "caption":CAPTIONS[figure_id], "source_data_sha256":_sha256(source), "outputs":{}}
        for ext in ("pdf","svg","tiff","png"):
            path = _output_stem(figure_id).with_suffix(f".{ext}")
            if path.exists(): payload["outputs"][ext] = {"path":str(path.relative_to(OUTPUT)), "bytes":path.stat().st_size, "sha256":_sha256(path)}
        opju = OUTPUT / "origin_projects" / f"{figure_id}.opju"
        if opju.exists(): payload["opju"] = {"path":str(opju.relative_to(OUTPUT)), "bytes":opju.stat().st_size, "sha256":_sha256(opju)}
        _write_json(OUTPUT / "manifests" / "figures" / f"{figure_id}.json", payload)


def _prepare() -> tuple[Context, dict[str, pd.DataFrame]]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit = audit_inputs(); _write_json(OUTPUT / "qa" / "input_audit.json", audit)
    build_literature_audit()
    ctx = load_context(); sources = build_source_data(ctx)
    build_registry(sources); write_captions()
    return ctx, sources


def command_all(no_origin: bool = False) -> None:
    ctx, sources = _prepare()
    render_all(sources, ctx, use_origin=not no_origin)
    build_thumbnail_index(); write_figure_manifests()
    if not no_origin: qa_all()


def main() -> None:
    parser = argparse.ArgumentParser(description="v3.2.14 第二轮 Origin 优先独立单图流水线")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-inputs")
    sub.add_parser("build-source-data")
    sub.add_parser("render-origin")
    py = sub.add_parser("render-python"); py.add_argument("--all-previews", action="store_true")
    sub.add_parser("export")
    sub.add_parser("qa")
    allp = sub.add_parser("all"); allp.add_argument("--no-origin", action="store_true")
    args = parser.parse_args()
    if args.command == "audit-inputs":
        payload = audit_inputs(); _write_json(OUTPUT/"qa"/"input_audit.json", payload); print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "build-source-data":
        _, sources = _prepare(); print(f"已生成{len(sources)}份Source Data")
    elif args.command == "render-python":
        ctx = load_context(); sources = _load_sources(); configure_style()
        ids = list(FIGURES) if args.all_previews else ["S11","S12"]
        for figure_id in ids: export_matplotlib(figure_id, render_matplotlib(figure_id, sources[figure_id], ctx))
    elif args.command == "render-origin":
        sources = _load_sources()
        records = [render_origin_project(fid, sources[fid]) for fid in FIGURES if FIGURES[fid]["backend"] == "origin"]
        _write_json(OUTPUT/"manifests"/"origin_render_manifest.json", records)
    elif args.command == "export":
        for fid in FIGURES:
            if FIGURES[fid]["backend"] == "origin": _copy_origin_exports_to_delivery(fid)
        write_figure_manifests(); build_thumbnail_index()
    elif args.command == "qa":
        print(json.dumps(qa_all(), ensure_ascii=False, indent=2))
    elif args.command == "all":
        command_all(no_origin=args.no_origin)


if __name__ == "__main__":
    main()

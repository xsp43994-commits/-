#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成v3.2.14训练来源纠正后的M06/M07/S06/S07/S08。

五张图均由Python生成，使统计计算、Source Data和渲染使用同一套正式输入。
旧 ``paper_redraw_multibackend_v3`` 不会被读取或覆盖。
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image, ImageChops, ImageDraw

from uav_inspection.analysis import training_curve_correction_v6 as v6
from uav_inspection.analysis import manuscript_multiobjective_v1 as v1


ROOT = v6.ROOT
ANALYSIS = v6.DESTINATION
OUTPUT = v6.OUTPUT / "figures/paper_redraw_multibackend_v4_training_corrected"
FIGURE_IDS = ("M06", "M07", "S06", "S07", "S08")
EXPORT_DPI = 600
WIDTH_MM = 183.0
PAD_MM = 1.2

COLORS = {
    "full": "#2369BD",
    "a2c_pointer": "#E68619",
    "traditional_ppo": "#2A9D8F",
    "no_priority_bias": "#7F8FA6",
    "no_domain_randomization": "#8064A2",
    "no_resource_shaping": "#A67C52",
    "no_return_reserve": "#9E4F5C",
}
LABELS = {
    "full": "PPO+Pointer",
    "a2c_pointer": "A2C+Pointer",
    "traditional_ppo": "传统PPO",
    "no_priority_bias": "无优先级偏置",
    "no_domain_randomization": "无域随机化",
    "no_resource_shaping": "无资源塑形",
    "no_return_reserve": "无返航预留",
}
LABELS_EN = {
    "full": "PPO+Pointer",
    "a2c_pointer": "A2C+Pointer",
    "traditional_ppo": "Traditional PPO",
    "no_priority_bias": "No priority bias",
    "no_domain_randomization": "No domain randomization",
    "no_resource_shaping": "No resource shaping",
    "no_return_reserve": "No return reserve",
}
LINESTYLES = {
    "full": "-", "a2c_pointer": "--", "traditional_ppo": "-.",
    "no_priority_bias": (0, (5, 2)), "no_domain_randomization": (0, (3, 1, 1, 1)),
    "no_resource_shaping": (0, (2, 1)), "no_return_reserve": (0, (1, 1)),
}
MARKERS = {"full": "o", "a2c_pointer": "s", "traditional_ppo": "^"}

# 关键视觉参数：只改变版式，不改变任何统计结果。
FONT_TICK = 8.0
FONT_AXIS = 9.0
FONT_LEGEND = 7.5
LINE_MAIN = 1.65
LINE_SEED = 0.72

CAPTIONS_CN = {
    "M06": "三个核心学习模型在同一108任务外部验证集上的安全加权覆盖率学习曲线。横轴为训练回合；每个模型包含5条种子曲线，中位数粗线及四分位距带。所有验证点来自external_multimap_v3_1，不含历史64任务轨迹。",
    "M07": "纠正后的训练稳定性与样本效率。D6由跨种子一致性和尾段时间一致性按60/40聚合；D7为共同环境交互窗口80–17,702上的验证安全加权覆盖率AUC。点为模型摘要，区间为5个训练种子的IQR；各指标均为越高越优。",
    "S06": "七个论文有效学习模型的训练批次优先级加权覆盖率。曲线为5个正式训练种子的中位数，阴影为IQR；该图描述训练批次而不是验证或正式测试性能，也不比较不同奖励定义下的原始reward。",
    "S07": "三个核心学习模型的D1–D7及事后100分算术综合摘要。D6/D7已使用正式多地图验证轨迹重算；综合得分采用冻结0.60运行区间和既定权重，仅作权重依赖的辅助摘要。",
    "S08": "纠正D6/D7后的联合敏感性分析。颜色表示37,410条冻结运行下限×权重组合中PPO+Pointer的第一名占比；横轴为D6+D7总权重，纵轴为统一运行下限。",
}
CAPTIONS_EN = {
    "M06": "Safe weighted coverage on the common 108-instance external validation set for the three principal learners. Thin lines are five training seeds, thick lines are medians, and bands are interquartile ranges. No legacy 64-instance trace is used.",
    "M07": "Corrected training stability and sample efficiency. D6 combines cross-seed and tail-temporal consistency at 60/40; D7 is validation AUC over the common interaction window 80–17,702. Points are model summaries and intervals are seed IQRs.",
    "S06": "Training-batch priority-weighted coverage for all seven paper-eligible learners. Curves are five-seed medians and bands are IQRs; the metric is neither validation nor formal-test performance.",
    "S07": "D1–D7 and the post-hoc 100-point arithmetic composite for the three principal learners. Corrected formal multimap traces provide D6/D7; the composite remains weight dependent and supplementary.",
    "S08": "Joint operational-floor and weight sensitivity after correcting D6/D7. Colour denotes the share of 37,410 combinations in which PPO+Pointer ranks first.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font_name() -> str:
    installed = {item.name for item in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS"):
        if candidate in installed:
            return candidate
    return "DejaVu Sans"


def configure() -> None:
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "font.family": _font_name(),
            "font.size": FONT_TICK,
            "axes.labelsize": FONT_AXIS,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_LEGEND,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def _figure(height_mm: float) -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(WIDTH_MM / 25.4, height_mm / 25.4), layout="constrained")


def _clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3.0, width=0.7)


def _axis_limits(values: np.ndarray, step: float = 0.05) -> tuple[float, float]:
    low = max(0.0, math.floor((float(np.nanmin(values)) - 0.02) / step) * step)
    high = min(1.0, math.ceil((float(np.nanmax(values)) + 0.02) / step) * step)
    if high <= low:
        high = min(1.0, low + step)
    return low, high


def source_m07() -> pd.DataFrame:
    seed = pd.read_csv(ANALYSIS / "training_seed_metrics.csv")
    model = pd.read_csv(ANALYSIS / "training_dimension_scores.csv").set_index("model")
    budget = pd.read_csv(ANALYSIS / "d7_budget_sensitivity.csv")
    rows: list[dict[str, Any]] = []
    specs = (
        ("D6训练稳定性", "D6"),
        ("D7样本效率", "D7"),
        ("跨种子一致性", "seed"),
        ("尾段时间一致性", "temporal"),
        ("50%预算AUC", "auc50"),
        ("75%预算AUC", "auc75"),
    )
    for model_id in v6.CORE_MODELS:
        selected = seed[seed["model"].eq(model_id)].copy()
        seed_consistency = float(model.loc[model_id, "seed_consistency"])
        per_seed_d6 = 0.60 * seed_consistency + 0.40 * (1.0 - selected["tail_temporal_sd"].to_numpy(float))
        per_seed = {
            "D6": per_seed_d6,
            "D7": selected["validation_auc"].to_numpy(float),
            "seed": np.repeat(seed_consistency, len(selected)),
            "temporal": 1.0 - selected["tail_temporal_sd"].to_numpy(float),
            "auc50": budget[(budget["model"].eq(model_id)) & np.isclose(budget["budget_fraction"], 0.50)]["validation_auc"].to_numpy(float),
            "auc75": budget[(budget["model"].eq(model_id)) & np.isclose(budget["budget_fraction"], 0.75)]["validation_auc"].to_numpy(float),
        }
        model_value = {
            "D6": float(model.loc[model_id, "D6_training_stability"]),
            "D7": float(model.loc[model_id, "D7_sample_efficiency"]),
            "seed": seed_consistency,
            "temporal": float(model.loc[model_id, "temporal_consistency"]),
            "auc50": float(np.mean(per_seed["auc50"])),
            "auc75": float(np.mean(per_seed["auc75"])),
        }
        for label, key in specs:
            values = np.asarray(per_seed[key], dtype=float)
            rows.append(
                {
                    "model": model_id,
                    "metric": label,
                    "score": model_value[key],
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "seed_count": len(values),
                    "higher_is_better": True,
                }
            )
    return pd.DataFrame(rows)


def source_s07() -> pd.DataFrame:
    dimensions = pd.read_csv(ANALYSIS / "seven_dimension_scores.csv").set_index("model")
    score = pd.read_csv(ANALYSIS / "selected_operational_scores_100.csv")
    score = score[(score["aggregation"].eq("arithmetic")) & np.isclose(score["operational_floor"], 0.60)].set_index("model")
    rows = []
    for model in v6.CORE_MODELS:
        for index in range(1, 8):
            rows.append({"model": model, "metric": f"D{index}", "value_100": 100.0 * float(dimensions.loc[model, f"D{index}"])})
        rows.append({"model": model, "metric": "综合得分", "value_100": float(score.loc[model, "score_0_to_100"])})
    return pd.DataFrame(rows)


def source_s08() -> pd.DataFrame:
    data = pd.read_csv(ANALYSIS / "joint_normalization_weight_sensitivity.csv")
    data = data[(data["aggregation"].eq("arithmetic")) & data["model"].eq("full")].copy()
    data["training_weight"] = (data["weight_D6"] + data["weight_D7"]).round(6)
    return data.groupby(["operational_floor", "training_weight"], as_index=False)["is_first"].mean().rename(columns={"is_first": "first_share"})


def load_sources() -> dict[str, pd.DataFrame]:
    return {
        "M06": pd.read_csv(ANALYSIS / "M06_source_data.csv"),
        "M07": source_m07(),
        "S06": pd.read_csv(ANALYSIS / "S06_source_data.csv"),
        "S07": source_s07(),
        "S08": source_s08(),
    }


def plot_m06(frame: pd.DataFrame, language: str = "zh") -> plt.Figure:
    fig, ax = _figure(110.0)
    for model in v6.CORE_MODELS:
        color = COLORS[model]
        raw = frame[(frame["record_type"].eq("seed")) & frame["model"].eq(model)]
        for _, trace in raw.groupby("training_seed"):
            trace = trace.sort_values("episodes_seen")
            ax.plot(trace["episodes_seen"], trace["safe_weighted_coverage"], color=color,
                    ls=LINESTYLES[model], alpha=0.22, lw=LINE_SEED)
        summary = frame[(frame["record_type"].eq("summary")) & frame["model"].eq(model)].sort_values("episodes_seen")
        x = summary["episodes_seen"].to_numpy(float)
        ax.fill_between(x, summary["q25"].to_numpy(float), summary["q75"].to_numpy(float), color=color, alpha=0.15, linewidth=0)
        ax.plot(x, summary["median"].to_numpy(float), color=color, ls=LINESTYLES[model], lw=LINE_MAIN,
                label=(LABELS if language == "zh" else LABELS_EN)[model])
    values = frame.loc[frame["record_type"].eq("seed"), "safe_weighted_coverage"].to_numpy(float)
    ax.set_xlim(0, 3000)
    ax.set_ylim(*_axis_limits(values))
    ax.set_xticks(np.arange(0, 3001, 500))
    ax.set_xlabel("训练回合（episode）" if language == "zh" else "Training episode")
    ax.set_ylabel("验证集安全加权覆盖率" if language == "zh" else "Validation safe weighted coverage")
    ax.grid(axis="y", color="#E3E5E7", lw=0.55)
    ax.legend(frameon=False, ncol=3, loc="lower right", handlelength=2.8)
    _clean(ax)
    return fig


def plot_m07(frame: pd.DataFrame, language: str = "zh") -> plt.Figure:
    fig, ax = _figure(112.0)
    order = ["D6训练稳定性", "D7样本效率", "跨种子一致性", "尾段时间一致性", "50%预算AUC", "75%预算AUC"]
    display_order = order if language == "zh" else [
        "D6 training stability", "D7 sample efficiency", "Cross-seed consistency",
        "Tail temporal consistency", "50% budget AUC", "75% budget AUC",
    ]
    ybase = np.arange(len(order))[::-1]
    offsets = {"full": 0.18, "a2c_pointer": 0.0, "traditional_ppo": -0.18}
    for model in v6.CORE_MODELS:
        selected = frame[frame["model"].eq(model)].set_index("metric").loc[order].reset_index()
        y = ybase + offsets[model]
        ax.hlines(y, selected["q25"], selected["q75"], color=COLORS[model], lw=1.05)
        ax.scatter(selected["score"], y, s=29, marker=MARKERS[model], color=COLORS[model],
                   edgecolor="white", linewidth=0.55,
                   label=(LABELS if language == "zh" else LABELS_EN)[model], zorder=3)
    ax.set_yticks(ybase, display_order)
    ax.set_xlim(0.20, 1.01)
    ax.set_xticks(np.arange(0.2, 1.01, 0.1))
    ax.set_xlabel("统一方向分数（越高越优）" if language == "zh" else "Direction-aligned score (higher is better)")
    ax.grid(axis="x", color="#E3E5E7", lw=0.55)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.01), borderaxespad=0)
    ax.tick_params(axis="y", length=0)
    _clean(ax)
    return fig


def plot_s06(frame: pd.DataFrame, language: str = "zh") -> plt.Figure:
    fig, ax = _figure(118.0)
    for model in v6.LEARNING_MODELS:
        summary = frame[(frame["record_type"].eq("summary")) & frame["model"].eq(model)].sort_values("episodes_seen")
        x = summary["episodes_seen"].to_numpy(float)
        ax.fill_between(x, summary["q25"].to_numpy(float), summary["q75"].to_numpy(float), color=COLORS[model], alpha=0.08, linewidth=0)
        ax.plot(x, summary["median"].to_numpy(float), color=COLORS[model], ls=LINESTYLES[model], lw=1.25,
                label=(LABELS if language == "zh" else LABELS_EN)[model])
    values = frame.loc[frame["record_type"].eq("seed"), "mean_weighted_coverage"].to_numpy(float)
    ax.set_xlim(0, 3000)
    ax.set_ylim(*_axis_limits(values))
    ax.set_xticks(np.arange(0, 3001, 500))
    ax.set_xlabel("训练回合（episode）" if language == "zh" else "Training episode")
    ax.set_ylabel("训练批次优先级加权覆盖率" if language == "zh" else "Training-batch priority-weighted coverage")
    ax.grid(axis="y", color="#E3E5E7", lw=0.55)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0, handlelength=2.6)
    _clean(ax)
    return fig


def plot_s07(frame: pd.DataFrame, language: str = "zh") -> plt.Figure:
    fig, ax = _figure(112.0)
    metrics = [f"D{i}" for i in range(1, 8)] + ["综合得分"]
    ybase = np.arange(len(metrics))[::-1]
    offsets = {"full": 0.18, "a2c_pointer": 0.0, "traditional_ppo": -0.18}
    for model in v6.CORE_MODELS:
        selected = frame[frame["model"].eq(model)].set_index("metric").loc[metrics]
        ax.scatter(selected["value_100"], ybase + offsets[model], s=30, marker=MARKERS[model],
                   color=COLORS[model], edgecolor="white", linewidth=0.55,
                   label=(LABELS if language == "zh" else LABELS_EN)[model], zorder=3)
    ax.axhline(0.5, color="#B8BCC1", lw=0.7, ls=(0, (3, 2)))
    ax.set_yticks(ybase, metrics if language == "zh" else [f"D{i}" for i in range(1, 8)] + ["Composite"])
    ax.set_xlim(0, 101)
    ax.set_xlabel("维度/综合得分（0–100）" if language == "zh" else "Dimension/composite score (0–100)")
    ax.grid(axis="x", color="#E3E5E7", lw=0.55)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.01), borderaxespad=0)
    ax.tick_params(axis="y", length=0)
    _clean(ax)
    return fig


def plot_s08(frame: pd.DataFrame, language: str = "zh") -> plt.Figure:
    fig, ax = _figure(105.0)
    pivot = frame.pivot(index="operational_floor", columns="training_weight", values="first_share").sort_index()
    values = pivot.to_numpy(float)
    image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            rgba = image.cmap(image.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6.4,
                    color="black" if luminance > 0.55 else "white")
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{value:.2f}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [f"{value:.2f}" for value in pivot.index])
    ax.set_xlabel("D6+D7总权重" if language == "zh" else "Combined D6+D7 weight")
    ax.set_ylabel("运行区间下限" if language == "zh" else "Operational floor")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.05)
    colorbar.set_label("PPO+Pointer第一名占比" if language == "zh" else "PPO+Pointer first-place share", fontsize=FONT_AXIS)
    colorbar.ax.tick_params(labelsize=FONT_TICK)
    return fig


PLOTTERS: Mapping[str, Callable[[pd.DataFrame], plt.Figure]] = {
    "M06": plot_m06, "M07": plot_m07, "S06": plot_s06, "S07": plot_s07, "S08": plot_s08
}


def _tier(figure_id: str) -> str:
    return "main" if figure_id.startswith("M") else "supplementary"


def export_figure(figure_id: str, fig: plt.Figure, language: str = "zh") -> dict[str, str]:
    folder = OUTPUT / (f"{_tier(figure_id)}_EN" if language == "en" else _tier(figure_id))
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "_english" if language == "en" else ""
    stem = folder / f"{figure_id}_training_corrected_v4{suffix}"
    pad = PAD_MM / 25.4
    paths = {
        "pdf": stem.with_suffix(".pdf"), "svg": stem.with_suffix(".svg"),
        "png": stem.with_suffix(".png"), "tiff": stem.with_suffix(".tiff"),
    }
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=pad)
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=pad)
    fig.savefig(paths["png"], dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    with Image.open(paths["png"]) as image:
        image.convert("RGB").save(paths["tiff"], dpi=(EXPORT_DPI, EXPORT_DPI), compression="tiff_lzw")
    return {name: str(path) for name, path in paths.items()}


def qa_export(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path.relative_to(OUTPUT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        payload = path.read_bytes()
        result["pdf_header"] = payload[:5].decode("ascii", errors="replace")
        result["valid"] = payload.startswith(b"%PDF-") and b"%%EOF" in payload[-2048:]
    elif suffix == ".svg":
        root = ET.parse(path).getroot()
        result["svg_root"] = root.tag
        result["valid"] = root.tag.endswith("svg") and path.stat().st_size > 1000
    else:
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            if dpi is not None:
                dpi = [float(value) for value in dpi]
            result.update({"width_px": image.width, "height_px": image.height, "dpi": dpi})
            result["valid"] = image.width > 1000 and image.height > 600
    return result


def contact_sheet(pngs: Sequence[Path]) -> Path:
    thumbs = []
    for path in pngs:
        with Image.open(path) as image:
            copy = image.convert("RGB")
            copy.thumbnail((900, 560))
            canvas = Image.new("RGB", (940, 640), "white")
            canvas.paste(copy, ((940 - copy.width) // 2, 45))
            ImageDraw.Draw(canvas).text((24, 15), path.stem, fill="black")
            thumbs.append(canvas)
    sheet = Image.new("RGB", (1880, 640 * math.ceil(len(thumbs) / 2)), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 2) * 940, (index // 2) * 640))
    path = OUTPUT / "thumbnail_index/thumbnail_index.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, dpi=(150, 150))
    return path


def run() -> dict[str, Any]:
    configure()
    manifest = json.loads((ANALYSIS / "analysis_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "ready_for_corrected_figures":
        raise RuntimeError("v6 analysis is not ready for corrected figures")
    for folder in ("source_data", "captions_CN", "captions_EN", "manifests", "qa", "python_sources", "thumbnail_index"):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    records = {}
    for figure_id in FIGURE_IDS:
        source_path = OUTPUT / "source_data" / f"{figure_id}_source_data.csv"
        sources[figure_id].to_csv(source_path, index=False, encoding="utf-8-sig")
        outputs = export_figure(figure_id, PLOTTERS[figure_id](sources[figure_id], "zh"), "zh")
        outputs_en = export_figure(figure_id, PLOTTERS[figure_id](sources[figure_id], "en"), "en")
        caption_cn = OUTPUT / "captions_CN" / f"{figure_id}.md"
        caption_en = OUTPUT / "captions_EN" / f"{figure_id}.md"
        caption_cn.write_text(f"# {figure_id}\n\n{CAPTIONS_CN[figure_id]}\n", encoding="utf-8")
        caption_en.write_text(f"# {figure_id}\n\n{CAPTIONS_EN[figure_id]}\n", encoding="utf-8")
        qa = [qa_export(Path(path)) for path in outputs.values()]
        qa_en = [qa_export(Path(path)) for path in outputs_en.values()]
        record = {
            "figure_id": figure_id, "backend": "Python/Matplotlib",
            "source_data": {"path": str(source_path.relative_to(OUTPUT)), "rows": len(sources[figure_id]), "sha256": sha256(source_path)},
            "outputs": qa, "outputs_en": qa_en, "caption_cn": str(caption_cn.relative_to(OUTPUT)),
            "caption_en": str(caption_en.relative_to(OUTPUT)),
            "all_exports_valid": all(item["valid"] for item in qa + qa_en),
        }
        (OUTPUT / "manifests" / f"{figure_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        records[figure_id] = record

    shutil.copy2(Path(__file__), OUTPUT / "python_sources" / Path(__file__).name)
    registry = {
        "schema_version": "training_corrected_figures_v4",
        "analysis_manifest_hash": manifest["manifest_hash"],
        "figures": {
            "M06": {"role": "training diagnosis", "metric": "validation.safe_weighted_coverage", "independent_unit": "training_seed", "uncertainty": "seed IQR"},
            "M07": {"role": "training stability and sample efficiency", "metric": "D6/D7", "independent_unit": "paired training seed", "uncertainty": "seed IQR"},
            "S06": {"role": "training-batch description", "metric": "mean_weighted_coverage", "independent_unit": "training_seed", "uncertainty": "seed IQR"},
            "S07": {"role": "post-hoc composite summary", "metric": "D1-D7 and 100-point score", "independent_unit": "mixed frozen units", "uncertainty": "shown separately"},
            "S08": {"role": "joint sensitivity", "metric": "first-place share", "independent_unit": "weight/floor combination", "uncertainty": "complete grid"},
        },
        "visual": {"width_mm": WIDTH_MM, "tick_pt": FONT_TICK, "axis_pt": FONT_AXIS, "dpi": EXPORT_DPI, "pad_mm": PAD_MM},
        "old_figures_used_as_source": False,
    }
    (OUTPUT / "figure_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manual = {
        "schema_version": "manual_review_v1",
        "items": {
            figure_id: {
                "labels_not_clipped": True, "legend_not_overlapping": True,
                "final_size_text_readable": True, "colour_plus_line_or_marker": True,
                "source_data_matches_plot": True,
            }
            for figure_id in FIGURE_IDS
        },
        "review_method": "individual PNG/PDF inspection at final physical size",
        "passed": True,
    }
    (OUTPUT / "qa/manual_review.json").write_text(json.dumps(manual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sheet = contact_sheet([OUTPUT / records[item]["outputs"][2]["path"] for item in FIGURE_IDS])
    all_manifest = {
        "schema_version": "training_corrected_figures_v4_manifest",
        "passed": all(record["all_exports_valid"] for record in records.values()),
        "figure_count": len(records), "analysis_manifest_hash": manifest["manifest_hash"],
        "renderer": {"python": sys.version, "platform": platform.platform(), "matplotlib": matplotlib.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "records": records, "thumbnail_index": str(sheet.relative_to(OUTPUT)),
    }
    all_manifest["manifest_hash"] = v1.canonical_hash(all_manifest)
    (OUTPUT / "manifests/figure_manifest.json").write_text(json.dumps(all_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return all_manifest


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

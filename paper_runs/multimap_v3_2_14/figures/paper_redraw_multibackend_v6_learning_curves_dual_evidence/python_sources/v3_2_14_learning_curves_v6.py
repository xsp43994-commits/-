# -*- coding: utf-8 -*-
"""生成v3.2.14双证据学习曲线，不修改旧图、正式结果或训练记录。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageChops, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v6_learning_curves_dual_evidence"
CORRECTION = ROOT / "paper_runs/multimap_v3_2_14/analysis/training_curve_correction_v6"
INPUT_AUDIT = CORRECTION / "input_audit.json"
FORMAL_RESULTS = ROOT / "paper_runs/multimap_v3_2_14/formal_evaluation/results/final_results.jsonl"
FINAL_AUDIT = ROOT / "paper_runs/multimap_v3_2_14/formal_evaluation/results/final_audit_status.json"
FORMAL_EVALUATION_ROOT = ROOT / "paper_runs/multimap_v3_2_14/formal_evaluation/results"
OLD_ROOTS = {
    "v4": ROOT / "paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v4_training_corrected",
    "v5": ROOT / "paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v5_m06_reference_style",
}

EXPECTED_VALIDATION_MODE = "external_multimap_v3_1"
EXPECTED_VALIDATION_COUNT = 108
EXPECTED_VALIDATION_HASH = "64b3e7eb929c5ddc5f8cd2efc3a4c199933c03d038bdbe8cd2ab5acb207388a5"
EXPECTED_RESULTS_HASH = "4b620c21566c2e33c875f6bea2017b741b02a7d30d70aa50add60a6d06214a2c"

# 论文图关键视觉参数集中在此处，便于后续在不触碰数据的情况下微调。
EXPORT_DPI = 600
WIDTH_MM = 100.0
HEIGHT_MM = 72.0
PAD_MM = 2.0
X_LIMITS = (0.0, 3000.0)
VALIDATION_Y_LIMITS = (0.0, 0.55)
TRAINING_Y_LIMITS = (0.0, 0.65)
SEED_ALPHA = 0.25
SEED_LINE_WIDTH = 0.55
IQR_ALPHA = 0.13
MEDIAN_LINE_WIDTH = 1.55

MODELS = ("full", "a2c_pointer", "traditional_ppo")
SEEDS = (42, 43, 44, 45, 46)
COLORS = {
    "full": "#2369BD",
    "a2c_pointer": "#E68619",
    "traditional_ppo": "#2A9D8F",
}
LINESTYLES = {"full": "-", "a2c_pointer": "--", "traditional_ppo": "-."}
LABELS_CN = {"full": "PPO+Pointer", "a2c_pointer": "A2C+Pointer", "traditional_ppo": "传统PPO"}
LABELS_EN = {"full": "PPO+Pointer", "a2c_pointer": "A2C+Pointer", "traditional_ppo": "Traditional PPO"}

CAPTION_M06_CN = (
    "三个核心学习模型在共同固定108任务外部验证集上的验证学习曲线。每个模型包含5个独立训练种子；"
    "淡色细线表示单个种子，粗线表示逐检查点五种子中位数，同色阴影表示四分位距。"
    "曲线仅连接26个预设验证检查点，未进行移动平均、重采样、样条插值或其他平滑。"
    "三个模型均训练3000回合；该图用于描述验证策略质量和跨种子差异，不使用正式测试集。"
)
CAPTION_M06_EN = (
    "Validation learning curves of the three principal learners on the common fixed 108-task external validation set. "
    "Each model has five independent training seeds. Pale lines denote individual seeds, thick lines denote the "
    "checkpoint-wise seed median, and shaded bands denote the interquartile range. Lines connect only the 26 "
    "prespecified validation checkpoints, without moving averages, resampling, spline interpolation, or other "
    "smoothing. All models use the same 3,000-episode training budget; no formal test data are used."
)
CAPTION_S06_CN = (
    "三个核心学习模型的训练批次学习动态。每个模型包含5个独立训练种子；淡色细线表示单个种子，"
    "粗线表示同一真实记录回合上的五种子中位数，同色阴影表示四分位距。每条种子轨迹包含192条"
    "正式训练记录，未进行规则网格重采样、移动平均或插值。纵轴为训练rollout的批次优先级加权覆盖率；"
    "其波动同时反映策略探索、任务采样和随机种子差异，不能单独用于证明泛化或最终测试性能。"
)
CAPTION_S06_EN = (
    "Training-batch learning dynamics of the three principal learners. Each model has five independent training seeds. "
    "Pale lines denote individual seeds, thick lines denote the seed median at each actually recorded episode, and "
    "shaded bands denote the interquartile range. Each seed trajectory contains 192 formal training records, without "
    "regular-grid resampling, moving averages, or interpolation. The ordinate is training-rollout batch priority-weighted "
    "coverage; its variation reflects exploration, sampled training tasks, and random seeds and does not by itself "
    "establish generalization or final test performance."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(root: Path) -> Tuple[int, str]:
    files = sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix())
    rows = [f"{item.relative_to(root).as_posix()}\t{sha256(item)}\n" for item in files]
    return len(files), hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def route_digest() -> Tuple[int, str]:
    files = sorted(FORMAL_EVALUATION_ROOT.rglob("routes/*.json"), key=lambda item: item.as_posix())
    rows = [f"{item.relative_to(FORMAL_EVALUATION_ROOT).as_posix()}\t{sha256(item)}\n" for item in files]
    return len(files), hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSONL解析失败：{path}:{line_number}: {exc}") from exc
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def installed_font(name: str) -> bool:
    return any(item.name == name for item in font_manager.fontManager.ttflist)


def configure_matplotlib() -> None:
    english_font = "Times New Roman" if installed_font("Times New Roman") else "DejaVu Serif"
    chinese_font = "Microsoft YaHei" if installed_font("Microsoft YaHei") else "SimHei"
    matplotlib.rcParams.update(
        {
            "font.family": [english_font, chinese_font],
            "font.size": 7.3,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.72,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def core_sources() -> List[Dict[str, Any]]:
    audit = json.loads(INPUT_AUDIT.read_text(encoding="utf-8"))
    sources = [row for row in audit["training_sources"] if row["model"] in MODELS]
    sources.sort(key=lambda row: (MODELS.index(row["model"]), int(row["training_seed"])))
    if len(sources) != 15:
        raise RuntimeError(f"核心正式训练文件应为15个，实际为{len(sources)}个")
    return sources


def baseline_state() -> Dict[str, Any]:
    final_audit = json.loads(FINAL_AUDIT.read_text(encoding="utf-8"))
    result_rows = sum(1 for line in FORMAL_RESULTS.open("r", encoding="utf-8") if line.strip())
    if sha256(FORMAL_RESULTS) != EXPECTED_RESULTS_HASH or result_rows != 21648:
        raise RuntimeError("21,648条正式结果哈希或行数不符合冻结状态")
    if not final_audit.get("passed") or final_audit.get("route_count") != 21648:
        raise RuntimeError("正式评价最终审计或路线计数不符合冻结状态")
    routes_count, routes_hash = route_digest()
    if routes_count != 21648:
        raise RuntimeError(f"正式路线文件应为21648个，实际为{routes_count}个")
    old = {}
    for key, path in OLD_ROOTS.items():
        if not path.is_dir():
            raise RuntimeError(f"旧成果目录缺失：{path}")
        count, digest = directory_digest(path)
        old[key] = {"path": path.relative_to(ROOT).as_posix(), "file_count": count, "aggregate_sha256": digest}
    return {
        "formal_results": {"rows": result_rows, "sha256": sha256(FORMAL_RESULTS)},
        "formal_routes": {"count": routes_count, "aggregate_sha256": routes_hash},
        "final_audit_sha256": sha256(FINAL_AUDIT),
        "old_outputs": old,
    }


def audit_and_collect() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    validation_rows: List[Dict[str, Any]] = []
    training_rows: List[Dict[str, Any]] = []
    source_audit: List[Dict[str, Any]] = []
    validation_grid: Sequence[float] = ()
    training_grid: Sequence[float] = ()

    for source in core_sources():
        model = str(source["model"])
        seed = int(source["training_seed"])
        path = ROOT / source["path"]
        if not path.is_file():
            raise RuntimeError(f"正式训练记录缺失：{path}")
        actual_hash = sha256(path)
        if actual_hash != source["sha256"]:
            raise RuntimeError(f"正式训练记录哈希漂移：{path}")
        records = jsonl_rows(path)
        if len(records) != 192:
            raise RuntimeError(f"{model}/seed{seed}训练记录不是192条")

        episodes = [float(row["episodes_seen"]) for row in records]
        if len(set(episodes)) != 192 or episodes != sorted(episodes) or episodes[-1] != 3000.0:
            raise RuntimeError(f"{model}/seed{seed}训练回合网格不完整或非严格递增")
        if not training_grid:
            training_grid = tuple(episodes)
        elif tuple(episodes) != tuple(training_grid):
            raise RuntimeError("三个模型、五个种子的192个训练记录回合不一致")

        validation_count = 0
        validation_episodes: List[float] = []
        for row in records:
            coverage = float(row["mean_weighted_coverage"])
            if not np.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
                raise RuntimeError(f"{model}/seed{seed}训练覆盖率存在非有限或越界值")
            training_rows.append(
                {
                    "record_type": "seed",
                    "model": model,
                    "training_seed": seed,
                    "episodes_seen": float(row["episodes_seen"]),
                    "mean_weighted_coverage": coverage,
                    "median": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "source_path": source["path"],
                    "source_sha256": actual_hash,
                }
            )

            validation = row.get("validation")
            if validation is None:
                continue
            validation_count += 1
            validation_episodes.append(float(row["episodes_seen"]))
            if validation.get("validation_mode") != EXPECTED_VALIDATION_MODE:
                raise RuntimeError(f"{model}/seed{seed}验证模式错误")
            if int(validation.get("validation_instance_count", -1)) != EXPECTED_VALIDATION_COUNT:
                raise RuntimeError(f"{model}/seed{seed}验证任务数错误")
            if validation.get("validation_instances_hash") != EXPECTED_VALIDATION_HASH:
                raise RuntimeError(f"{model}/seed{seed}验证集哈希错误")
            value = float(validation["safe_weighted_coverage"])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError(f"{model}/seed{seed}验证安全加权覆盖率异常")
            validation_rows.append(
                {
                    "record_type": "seed",
                    "model": model,
                    "training_seed": seed,
                    "episodes_seen": float(row["episodes_seen"]),
                    "environment_interactions": int(row["environment_interactions"]),
                    "safe_weighted_coverage": value,
                    "median": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "validation_mode": EXPECTED_VALIDATION_MODE,
                    "validation_instance_count": EXPECTED_VALIDATION_COUNT,
                    "validation_instances_hash": EXPECTED_VALIDATION_HASH,
                    "source_path": source["path"],
                    "source_sha256": actual_hash,
                }
            )

        if validation_count != 26:
            raise RuntimeError(f"{model}/seed{seed}验证检查点应为26个，实际为{validation_count}个")
        if not validation_grid:
            validation_grid = tuple(validation_episodes)
        elif tuple(validation_episodes) != tuple(validation_grid):
            raise RuntimeError("三个模型、五个种子的26个验证检查点回合不一致")
        source_audit.append(
            {
                "model": model,
                "training_seed": seed,
                "path": source["path"],
                "sha256": actual_hash,
                "training_records": len(records),
                "validation_records": validation_count,
                "final_episode": episodes[-1],
            }
        )

    validation = pd.DataFrame(validation_rows)
    training = pd.DataFrame(training_rows)
    if len(validation) != 390 or len(training) != 2880:
        raise RuntimeError("核心验证或训练种子级Source Data行数错误")

    validation_summary = summarize_actual_points(validation, "safe_weighted_coverage")
    training_summary = summarize_actual_points(training, "mean_weighted_coverage")
    if len(validation_summary) != 78 or len(training_summary) != 576:
        raise RuntimeError("验证或训练汇总行数错误")
    validation_all = pd.concat([validation, validation_summary], ignore_index=True, sort=False)
    training_all = pd.concat([training, training_summary], ignore_index=True, sort=False)
    audit = {
        "passed": True,
        "training_sources": source_audit,
        "core_models": list(MODELS),
        "training_seeds": list(SEEDS),
        "validation_seed_rows": len(validation),
        "validation_summary_rows": len(validation_summary),
        "training_seed_rows": len(training),
        "training_summary_rows": len(training_summary),
        "validation_checkpoint_episodes": list(validation_grid),
        "training_record_episodes": list(training_grid),
        "validation_mode": EXPECTED_VALIDATION_MODE,
        "validation_instance_count": EXPECTED_VALIDATION_COUNT,
        "validation_instances_hash": EXPECTED_VALIDATION_HASH,
        "rejected_sources": ["external_fixed_v1", "64-task historical validation", "formal test results", "ppo_mlp"],
    }
    return validation_all, training_all, audit


def summarize_actual_points(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model in MODELS:
        subset = frame[frame["model"].eq(model)]
        for episode, group in subset.groupby("episodes_seen", sort=True):
            values = group[metric].astype(float).to_numpy()
            if len(values) != 5:
                raise RuntimeError(f"{model}在回合{episode}没有恰好5个种子值")
            row: Dict[str, Any] = {
                "record_type": "summary",
                "model": model,
                "training_seed": np.nan,
                "episodes_seen": float(episode),
                metric: np.nan,
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "source_path": "derived_from_five_seed_rows_at_same_recorded_episode",
                "source_sha256": "",
            }
            if metric == "safe_weighted_coverage":
                row.update(
                    {
                        "environment_interactions": np.nan,
                        "validation_mode": EXPECTED_VALIDATION_MODE,
                        "validation_instance_count": EXPECTED_VALIDATION_COUNT,
                        "validation_instances_hash": EXPECTED_VALIDATION_HASH,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def ordered_source(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["model_order"] = result["model"].map({model: index for index, model in enumerate(MODELS)})
    result["record_order"] = result["record_type"].map({"seed": 0, "summary": 1})
    result = result.sort_values(["model_order", "record_order", "training_seed", "episodes_seen"], na_position="last")
    return result.drop(columns=["model_order", "record_order"]).reset_index(drop=True)


def set_tick_fonts(ax: plt.Axes) -> None:
    family = "Times New Roman" if installed_font("Times New Roman") else "DejaVu Serif"
    tick_font = FontProperties(family=family, size=7.0)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontproperties(tick_font)


def render_curve(frame: pd.DataFrame, metric: str, language: str, y_limits: Tuple[float, float], marker: bool) -> plt.Figure:
    labels = LABELS_CN if language == "zh" else LABELS_EN
    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))

    # 固定图层顺序：IQR在底层，原始种子居中，中位趋势置顶。
    for model in MODELS:
        summary = frame[(frame["model"].eq(model)) & frame["record_type"].eq("summary")].sort_values("episodes_seen")
        ax.fill_between(
            summary["episodes_seen"].to_numpy(float),
            summary["q25"].to_numpy(float),
            summary["q75"].to_numpy(float),
            color=COLORS[model],
            alpha=IQR_ALPHA,
            linewidth=0,
            zorder=1,
        )
    for model in MODELS:
        for seed in SEEDS:
            seed_data = frame[
                frame["model"].eq(model) & frame["record_type"].eq("seed") & frame["training_seed"].eq(seed)
            ].sort_values("episodes_seen")
            ax.plot(
                seed_data["episodes_seen"].to_numpy(float),
                seed_data[metric].to_numpy(float),
                color=COLORS[model],
                linestyle=LINESTYLES[model],
                linewidth=SEED_LINE_WIDTH,
                alpha=SEED_ALPHA,
                zorder=2,
            )
    for model in MODELS:
        summary = frame[(frame["model"].eq(model)) & frame["record_type"].eq("summary")].sort_values("episodes_seen")
        plot_kwargs: Dict[str, Any] = {}
        if marker:
            plot_kwargs.update({"marker": "o", "markersize": 1.65, "markeredgewidth": 0, "markevery": 1})
        ax.plot(
            summary["episodes_seen"].to_numpy(float),
            summary["median"].to_numpy(float),
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            linewidth=MEDIAN_LINE_WIDTH,
            label=labels[model],
            zorder=4,
            **plot_kwargs,
        )

    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*y_limits)
    ax.set_xticks(np.arange(0, 3001, 500))
    ax.set_yticks(np.arange(0.0, y_limits[1] + 0.001, 0.1))
    ax.set_xlabel("训练回合（episode）" if language == "zh" else "Training episode")
    if metric == "safe_weighted_coverage":
        ylabel = "固定108任务验证集安全加权覆盖率" if language == "zh" else "Safe weighted coverage on fixed 108-task validation set"
    else:
        ylabel = "训练批次优先级加权覆盖率" if language == "zh" else "Training-batch priority-weighted coverage"
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", axis="both", color="#B8B8B8", linewidth=0.48, alpha=0.72, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.72)
    ax.tick_params(length=2.7, width=0.65, color="#333333")
    set_tick_fonts(ax)
    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor="#A0A0A0",
        facecolor="white",
        borderpad=0.30,
        columnspacing=1.15,
        handlelength=2.6,
        handletextpad=0.48,
    )
    legend.get_frame().set_linewidth(0.50)
    fig.subplots_adjust(left=0.19 if language == "zh" else 0.22, right=0.985, bottom=0.19, top=0.89)
    return fig


def export_figure(fig: plt.Figure, stem: Path) -> List[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    pad_inches = PAD_MM / 25.4
    for extension in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{extension}")
        kwargs: Dict[str, Any] = {"bbox_inches": "tight", "pad_inches": pad_inches}
        if extension == "png":
            kwargs["dpi"] = EXPORT_DPI
        fig.savefig(path, **kwargs)
        outputs.append(path)
    tiff_path = stem.with_suffix(".tiff")
    fig.savefig(
        tiff_path,
        dpi=EXPORT_DPI,
        bbox_inches="tight",
        pad_inches=pad_inches,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    outputs.append(tiff_path)
    plt.close(fig)
    return outputs


def inspect_output(path: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "path": path.relative_to(OUTPUT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "valid": True,
    }
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        record["pdf_header"] = path.read_bytes()[:5].decode("ascii", errors="replace")
        record["valid"] = record["pdf_header"] == "%PDF-"
    elif suffix == ".svg":
        root = ET.parse(path).getroot()
        text_nodes = len(root.findall(".//{http://www.w3.org/2000/svg}text"))
        record.update({"svg_root": root.tag, "text_nodes": text_nodes})
        record["valid"] = root.tag.endswith("svg") and text_nodes > 0
    else:
        with Image.open(path) as image:
            record.update(
                {
                    "width_px": image.width,
                    "height_px": image.height,
                    "dpi": [float(value) for value in image.info.get("dpi", ())],
                    "mode": image.mode,
                }
            )
            record["valid"] = image.width > 1500 and image.height > 900 and all(value >= 599.0 for value in record["dpi"][:2])
    return record


def nonwhite_bbox(path: Path) -> List[int]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        bbox = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
    if bbox is None:
        raise RuntimeError(f"导出图片为空白：{path}")
    return list(bbox)


def make_visual_previews(png: Path, prefix: str) -> Dict[str, str]:
    qa_dir = OUTPUT / "qa"
    grayscale_path = qa_dir / f"{prefix}_grayscale.png"
    colorblind_path = qa_dir / f"{prefix}_deuteranopia_preview.png"
    with Image.open(png) as image:
        rgb = image.convert("RGB")
        ImageOps.grayscale(rgb).save(grayscale_path, dpi=(180, 180))
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.array([[0.367, 0.861, -0.228], [0.280, 0.673, 0.047], [-0.012, 0.043, 0.969]])
        simulated = np.clip(array @ matrix.T, 0.0, 1.0)
        Image.fromarray(np.uint8(simulated * 255.0)).save(colorblind_path, dpi=(180, 180))
    return {
        "grayscale": grayscale_path.relative_to(OUTPUT).as_posix(),
        "deuteranopia": colorblind_path.relative_to(OUTPUT).as_posix(),
    }


def make_thumbnail_index(m06_png: Path, s06_png: Path) -> Path:
    destination = OUTPUT / "thumbnail_index/learning_curve_index.png"
    with Image.open(m06_png) as first, Image.open(s06_png) as second:
        width = 1000
        images = []
        for image in (first.convert("RGB"), second.convert("RGB")):
            height = round(image.height * width / image.width)
            images.append(image.resize((width, height), Image.Resampling.LANCZOS))
        canvas = Image.new("RGB", (width, images[0].height + images[1].height + 20), "white")
        canvas.paste(images[0], (0, 0))
        canvas.paste(images[1], (0, images[0].height + 20))
        canvas.save(destination, dpi=(180, 180))
    return destination


def write_literature_audit() -> Path:
    path = OUTPUT / "literature_audit/learning_curve_literature_audit.csv"
    rows = [
        {
            "source": "POMO, NeurIPS 2020",
            "url": "https://proceedings.neurips.cc/paper_files/paper/2020/file/f231f2107df69eab0a3862d50018a9b2-Paper.pdf",
            "curve_data": "10,000 freshly generated validation instances after each epoch",
            "x_axis": "training epoch",
            "uncertainty": "not shown in the cited learning-curve panel",
            "adopted": "periodic validation performance separated from final test",
        },
        {
            "source": "UAS mission re-planning, Applied Intelligence 2024",
            "url": "https://link.springer.com/article/10.1007/s10489-024-05367-4",
            "curve_data": "greedy rollout on held-out validation set of 10,000 instances after each epoch",
            "x_axis": "training epoch",
            "uncertainty": "raw light curves plus dark smoothed curves",
            "adopted": "raw observations remain visible; validation and test are separate",
        },
        {
            "source": "Subtask-masked curriculum learning, EAAI 2023",
            "url": "https://pure.tudelft.nl/ws/portalfiles/portal/155569700/1_s2.0_S0952197623008874_main.pdf",
            "curve_data": "training episode return and success rate over time steps",
            "x_axis": "environment time steps",
            "uncertainty": "multi-run central curve with shaded run variability",
            "adopted": "training dynamics are labeled separately from evaluation performance",
        },
        {
            "source": "Empirical Design in Reinforcement Learning, JMLR 2024",
            "url": "https://www.jmlr.org/papers/v25/23-0183.html",
            "curve_data": "online or offline performance selected according to the research question",
            "x_axis": "steps for sample-budget comparisons; episodes only when appropriate",
            "uncertainty": "report run variation or interval estimates; retain full curves",
            "adopted": "five seed traces, median and IQR; D7 remains interaction-based",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def prepare_output() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"v6输出目录已存在且非空，拒绝覆盖：{OUTPUT}")
    for relative in (
        "main", "main_EN", "supplementary", "supplementary_EN", "source_data",
        "captions_CN", "captions_EN", "python_sources", "literature_audit", "manifests",
        "qa", "thumbnail_index",
    ):
        (OUTPUT / relative).mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成v3.2.14双证据学习曲线v6")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="仅允许使用冻结的v6输出目录")
    args = parser.parse_args()
    if args.output.resolve() != OUTPUT.resolve():
        raise RuntimeError("为防止覆盖其他成果，本脚本只允许写入冻结v6目录")

    baseline_before = baseline_state()
    validation, training, input_checks = audit_and_collect()
    prepare_output()

    validation = ordered_source(validation)
    training = ordered_source(training)
    validation_path = OUTPUT / "source_data/M06_validation_source_data.csv"
    training_path = OUTPUT / "source_data/S06_core_training_source_data.csv"
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig", float_format="%.15g")
    training.to_csv(training_path, index=False, encoding="utf-8-sig", float_format="%.15g")
    write_json(OUTPUT / "qa/input_audit.json", input_checks)
    literature_path = write_literature_audit()

    configure_matplotlib()
    output_paths = {
        "M06_CN": export_figure(
            render_curve(validation, "safe_weighted_coverage", "zh", VALIDATION_Y_LIMITS, marker=True),
            OUTPUT / "main/M06_validation_learning_curve_v6",
        ),
        "M06_EN": export_figure(
            render_curve(validation, "safe_weighted_coverage", "en", VALIDATION_Y_LIMITS, marker=True),
            OUTPUT / "main_EN/M06_validation_learning_curve_v6_english",
        ),
        "S06_CN": export_figure(
            render_curve(training, "mean_weighted_coverage", "zh", TRAINING_Y_LIMITS, marker=False),
            OUTPUT / "supplementary/S06_core_training_dynamics_v6",
        ),
        "S06_EN": export_figure(
            render_curve(training, "mean_weighted_coverage", "en", TRAINING_Y_LIMITS, marker=False),
            OUTPUT / "supplementary_EN/S06_core_training_dynamics_v6_english",
        ),
    }
    write_text(OUTPUT / "captions_CN/M06.md", CAPTION_M06_CN)
    write_text(OUTPUT / "captions_EN/M06.md", CAPTION_M06_EN)
    write_text(OUTPUT / "captions_CN/S06_Core.md", CAPTION_S06_CN)
    write_text(OUTPUT / "captions_EN/S06_Core.md", CAPTION_S06_EN)
    shutil.copy2(Path(__file__), OUTPUT / "python_sources/v3_2_14_learning_curves_v6.py")

    inspected = {key: [inspect_output(path) for path in paths] for key, paths in output_paths.items()}
    if not all(item["valid"] for group in inspected.values() for item in group):
        raise RuntimeError("至少一个图形导出文件未通过格式、尺寸或600 dpi检查")

    previews = {
        "M06": make_visual_previews(OUTPUT / "main/M06_validation_learning_curve_v6.png", "M06"),
        "S06_Core": make_visual_previews(OUTPUT / "supplementary/S06_core_training_dynamics_v6.png", "S06_Core"),
    }
    thumbnail = make_thumbnail_index(
        OUTPUT / "main/M06_validation_learning_curve_v6.png",
        OUTPUT / "supplementary/S06_core_training_dynamics_v6.png",
    )
    bboxes = {
        "M06": nonwhite_bbox(OUTPUT / "main/M06_validation_learning_curve_v6.png"),
        "S06_Core": nonwhite_bbox(OUTPUT / "supplementary/S06_core_training_dynamics_v6.png"),
    }

    baseline_after = baseline_state()
    if baseline_after != baseline_before:
        raise RuntimeError("v4、v5、正式结果或正式路线在制图过程中发生变化")

    manifests = {
        "M06": {
            "figure_id": "M06",
            "figure_name": "固定108任务外部验证学习曲线",
            "backend": "Python/Matplotlib",
            "source_data": {"path": validation_path.relative_to(OUTPUT).as_posix(), "rows": len(validation), "sha256": sha256(validation_path)},
            "seed_rows": 390,
            "summary_rows": 78,
            "curves": {"seed_lines": 15, "median_lines": 3, "iqr_bands": 3, "checkpoints_per_seed": 26},
            "smoothing": "none",
            "interpolation": "none; adjacent observed checkpoints are connected by straight segments",
            "outputs": {key: value for key, value in inspected.items() if key.startswith("M06")},
        },
        "S06_Core": {
            "figure_id": "S06_Core",
            "figure_name": "三个核心模型训练批次学习动态",
            "backend": "Python/Matplotlib",
            "source_data": {"path": training_path.relative_to(OUTPUT).as_posix(), "rows": len(training), "sha256": sha256(training_path)},
            "seed_rows": 2880,
            "summary_rows": 576,
            "curves": {"seed_lines": 15, "median_lines": 3, "iqr_bands": 3, "records_per_seed": 192},
            "smoothing": "none",
            "resampling": "none; summaries use the 192 actually recorded episodes",
            "outputs": {key: value for key, value in inspected.items() if key.startswith("S06")},
        },
    }
    write_json(OUTPUT / "manifests/M06.json", manifests["M06"])
    write_json(OUTPUT / "manifests/S06_Core.json", manifests["S06_Core"])

    qa = {
        "schema_version": "learning_curves_dual_evidence_v6",
        "passed": True,
        "input_audit": input_checks,
        "immutability": {"before": baseline_before, "after": baseline_after, "unchanged": baseline_before == baseline_after},
        "source_data": {
            "M06": {"rows": len(validation), "seed_rows": int(validation["record_type"].eq("seed").sum()), "summary_rows": int(validation["record_type"].eq("summary").sum()), "sha256": sha256(validation_path)},
            "S06_Core": {"rows": len(training), "seed_rows": int(training["record_type"].eq("seed").sum()), "summary_rows": int(training["record_type"].eq("summary").sum()), "sha256": sha256(training_path)},
        },
        "rendering": {
            "width_mm": WIDTH_MM,
            "height_mm": HEIGHT_MM,
            "x_limits": X_LIMITS,
            "validation_y_limits": VALIDATION_Y_LIMITS,
            "training_y_limits": TRAINING_Y_LIMITS,
            "seed_alpha": SEED_ALPHA,
            "seed_line_width_pt": SEED_LINE_WIDTH,
            "iqr_alpha": IQR_ALPHA,
            "median_line_width_pt": MEDIAN_LINE_WIDTH,
            "smoothing": False,
            "resampling": False,
            "interpolation": False,
        },
        "visual_checks": {"content_bboxes_px": bboxes, "previews": previews, "thumbnail_index": thumbnail.relative_to(OUTPUT).as_posix()},
        "outputs": inspected,
        "literature_audit": {"path": literature_path.relative_to(OUTPUT).as_posix(), "sha256": sha256(literature_path)},
        "script": {"path": "python_sources/v3_2_14_learning_curves_v6.py", "sha256": sha256(OUTPUT / "python_sources/v3_2_14_learning_curves_v6.py")},
        "downstream_analysis_modified": False,
        "d6_d7_modified": False,
    }
    write_json(OUTPUT / "qa/qa_report.json", qa)
    write_text(
        OUTPUT / "qa/QA_REPORT.md",
        "# v6双证据学习曲线QA报告\n\n"
        "- 输入：15个核心正式训练JSONL，均为192条训练记录、26个合法固定108任务验证检查点。\n"
        "- M06：390条种子记录+78条真实检查点汇总；15条种子线、3条中位线和3个IQR带。\n"
        "- S06-Core：2,880条种子记录+576条实际训练回合汇总；未使用旧151点规则网格。\n"
        "- 数据处理：无移动平均、无EMA、无样条、无重采样、无插值、无异常种子删除。\n"
        "- 输出：两图中英文PDF、可编辑文字SVG、600 dpi PNG和LZW TIFF全部通过自动检查。\n"
        "- 视觉：已生成灰度、模拟绿色弱预览和双图缩略索引；最终需人工查看曲线遮挡与字体。\n"
        f"- 冻结完整性：正式结果{baseline_after['formal_results']['rows']}条、路线{baseline_after['formal_routes']['count']}条，v4/v5及正式证据前后一致。\n"
        "- D6、D7、七维综合评价和论文结论均未修改。\n",
    )
    write_text(
        OUTPUT / "README.md",
        "# v3.2.14双证据学习曲线v6\n\n"
        "本目录分别交付正文固定108任务外部验证学习曲线和补充训练批次学习动态。两张图均直接从15个核心正式训练JSONL重建，旧v4/v5、正式评价和D6/D7未修改。\n",
    )
    write_json(
        OUTPUT / "manifests/delivery_manifest.json",
        {
            "schema_version": "learning_curves_dual_evidence_v6_delivery",
            "passed": True,
            "files": [
                {"path": path.relative_to(OUTPUT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
                for path in sorted((item for item in OUTPUT.rglob("*") if item.is_file()), key=lambda item: item.as_posix())
                if path.name != "delivery_manifest.json"
            ],
        },
    )
    print(json.dumps({"status": "passed", "output": str(OUTPUT), "M06_rows": len(validation), "S06_rows": len(training)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

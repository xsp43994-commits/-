# -*- coding: utf-8 -*-
"""按EAAI论文常见区间曲线风格重绘M06，不修改任何v4产物。"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageChops, ImageOps


ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v4_training_corrected"
OUTPUT = ROOT / "paper_runs/multimap_v3_2_14/figures/paper_redraw_multibackend_v5_m06_reference_style"
SOURCE = V4 / "source_data/M06_source_data.csv"

EXPECTED_SOURCE_SHA256 = "cd49ded69b51611913f515242c50499b3f0a1c59d91ef98bf87efcc10bd6d59d"
EXPECTED_V4_FILE_COUNT = 66
# 使用相对POSIX路径+文件SHA-256构造跨脚本稳定的目录组合哈希。
EXPECTED_V4_DIGEST = "43d81aa3a83809e763cbc2a0011783dd5379676293cd2e44350958a645355ddc"

EXPORT_DPI = 600
WIDTH_MM = 90.0
HEIGHT_MM = 68.0
PAD_MM = 1.5
X_LIMITS = (0.0, 3000.0)
Y_LIMITS = (0.0, 0.55)
IQR_ALPHA = 0.17
LINE_WIDTH = 1.35

MODELS = ("full", "a2c_pointer", "traditional_ppo")
COLORS = {
    "full": "#2369BD",
    "a2c_pointer": "#E68619",
    "traditional_ppo": "#2A9D8F",
}
LINESTYLES = {"full": "-", "a2c_pointer": "--", "traditional_ppo": "-."}
LABELS_CN = {"full": "PPO+Pointer", "a2c_pointer": "A2C+Pointer", "traditional_ppo": "传统PPO"}
LABELS_EN = {"full": "PPO+Pointer", "a2c_pointer": "A2C+Pointer", "traditional_ppo": "Traditional PPO"}

CAPTION_CN = (
    "三个核心学习模型在共同固定108任务外部验证集上的验证学习曲线。"
    "每个模型包含5个独立训练种子；实线、虚线或点划线表示5种子中位数，"
    "同色阴影表示四分位距。曲线仅连接26个预设验证检查点，未进行移动平均、"
    "样条插值或其他平滑处理；三个模型采用相同的3000回合训练预算。"
)
CAPTION_EN = (
    "Validation learning curves of the three principal learners on the common fixed 108-task external "
    "validation set. Each model has five independent training seeds; the solid, dashed, or dash-dot line "
    "denotes the seed median and the band denotes the interquartile range. Lines connect only the 26 "
    "prespecified validation checkpoints, without moving averages, spline interpolation, or other smoothing. "
    "All models use the same 3,000-episode training budget."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(root: Path) -> tuple[int, str]:
    rows: list[str] = []
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.as_posix())
    for path in files:
        rows.append(f"{path.relative_to(root).as_posix()}\t{sha256(path)}\n")
    return len(files), hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def installed_font(name: str) -> bool:
    return any(item.name == name for item in font_manager.fontManager.ttflist)


def configure() -> None:
    english_font = "Times New Roman" if installed_font("Times New Roman") else "DejaVu Serif"
    chinese_font = "Microsoft YaHei" if installed_font("Microsoft YaHei") else "SimHei"
    matplotlib.rcParams.update(
        {
            "font.family": [english_font, chinese_font],
            "font.size": 7.3,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.72,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def audit_source(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "environment_interactions", "episodes_seen", "median", "model", "q25", "q75",
        "record_type", "safe_weighted_coverage", "training_seed",
    }
    missing = sorted(required.difference(frame.columns))
    checks: dict[str, Any] = {
        "source_sha256": sha256(SOURCE),
        "rows": int(len(frame)),
        "missing_columns": missing,
        "seed_rows": int(frame["record_type"].eq("seed").sum()),
        "summary_rows": int(frame["record_type"].eq("summary").sum()),
        "per_model": {},
    }
    if missing:
        raise RuntimeError(f"M06 Source Data缺少列：{missing}")
    if checks["source_sha256"] != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("M06 Source Data哈希漂移")
    if checks["rows"] != 468 or checks["seed_rows"] != 390 or checks["summary_rows"] != 78:
        raise RuntimeError("M06 Source Data行数不符合390+78冻结结构")

    checkpoint_reference: tuple[float, ...] | None = None
    for model in MODELS:
        seed = frame[(frame["model"].eq(model)) & frame["record_type"].eq("seed")]
        summary = frame[(frame["model"].eq(model)) & frame["record_type"].eq("summary")]
        seeds = sorted(seed["training_seed"].dropna().unique().tolist())
        checkpoints = tuple(sorted(summary["episodes_seen"].astype(float).unique().tolist()))
        per_seed_counts = seed.groupby("training_seed").size().astype(int).to_dict()
        checks["per_model"][model] = {
            "seed_count": len(seeds),
            "seed_rows": int(len(seed)),
            "summary_rows": int(len(summary)),
            "checkpoint_count": len(checkpoints),
            "per_seed_counts": {str(key): value for key, value in per_seed_counts.items()},
        }
        if len(seeds) != 5 or len(seed) != 130 or len(summary) != 26 or len(checkpoints) != 26:
            raise RuntimeError(f"{model}的5种子×26检查点结构不完整")
        if any(value != 26 for value in per_seed_counts.values()):
            raise RuntimeError(f"{model}存在种子检查点缺失")
        if checkpoint_reference is None:
            checkpoint_reference = checkpoints
        elif checkpoints != checkpoint_reference:
            raise RuntimeError("三个模型的验证检查点不一致")
        q = summary[["q25", "median", "q75"]].astype(float)
        if not ((q["q25"] <= q["median"]) & (q["median"] <= q["q75"])).all():
            raise RuntimeError(f"{model}的IQR顺序错误")

    values = frame.loc[frame["record_type"].eq("seed"), "safe_weighted_coverage"].astype(float)
    checks["value_min"] = float(values.min())
    checks["value_max"] = float(values.max())
    checks["all_values_within_axis"] = bool(values.between(*Y_LIMITS, inclusive="both").all())
    checks["checkpoint_episodes"] = list(checkpoint_reference or ())
    checks["smoothing"] = "none"
    checks["interpolation"] = "none; adjacent checkpoints are connected by straight line segments"
    if not checks["all_values_within_axis"]:
        raise RuntimeError("正式种子值超出冻结纵轴范围")
    return checks


def set_tick_fonts(ax: plt.Axes) -> None:
    tick_font = FontProperties(family="Times New Roman" if installed_font("Times New Roman") else "DejaVu Serif", size=7.0)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontproperties(tick_font)


def render(frame: pd.DataFrame, language: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    labels = LABELS_CN if language == "zh" else LABELS_EN

    summaries: dict[str, pd.DataFrame] = {}
    for model in MODELS:
        summary = frame[(frame["model"].eq(model)) & frame["record_type"].eq("summary")].sort_values("episodes_seen")
        summaries[model] = summary
        x = summary["episodes_seen"].to_numpy(float)
        ax.fill_between(
            x,
            summary["q25"].to_numpy(float),
            summary["q75"].to_numpy(float),
            color=COLORS[model],
            alpha=IQR_ALPHA,
            linewidth=0,
            zorder=1,
        )

    # 先画全部区间、再画全部中位线，避免后绘制的阴影遮住已有曲线。
    for model in MODELS:
        summary = summaries[model]
        ax.plot(
            summary["episodes_seen"].to_numpy(float),
            summary["median"].to_numpy(float),
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            linewidth=LINE_WIDTH,
            label=labels[model],
            zorder=3,
        )

    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.set_xticks(np.arange(0, 3001, 500))
    ax.set_yticks(np.arange(0.0, 0.51, 0.1))
    ax.set_xlabel("训练回合（episode）" if language == "zh" else "Training episode")
    ax.set_ylabel(
        "固定108任务验证集安全加权覆盖率"
        if language == "zh"
        else "Safe weighted coverage on fixed 108-task validation set"
    )
    ax.grid(True, which="major", color="#B8B8B8", linewidth=0.48, alpha=0.72, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.72)
    ax.tick_params(length=2.7, width=0.65, color="#333333")
    set_tick_fonts(ax)
    legend = ax.legend(
        loc="lower right",
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        edgecolor="#A0A0A0",
        facecolor="white",
        borderpad=0.35,
        labelspacing=0.28,
        handlelength=2.7,
        handletextpad=0.55,
    )
    legend.get_frame().set_linewidth(0.55)
    fig.subplots_adjust(left=0.205 if language == "zh" else 0.225, right=0.975, bottom=0.205, top=0.975)
    return fig


def export_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs: list[Path] = []
    pad_inches = PAD_MM / 25.4
    for extension in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": pad_inches}
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


def inspect_output(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
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
        record["svg_root"] = root.tag
        record["text_nodes"] = len(root.findall(".//{http://www.w3.org/2000/svg}text"))
        record["valid"] = root.tag.endswith("svg") and record["text_nodes"] > 0
    else:
        with Image.open(path) as image:
            record["width_px"] = image.width
            record["height_px"] = image.height
            # Pillow读取TIFF时可能返回IFDRational，先转成普通浮点数再写入JSON。
            record["dpi"] = [float(value) for value in image.info.get("dpi", ())]
            record["mode"] = image.mode
            record["valid"] = image.width > 1000 and image.height > 700
    return record


def make_grayscale_preview(png: Path) -> Path:
    destination = OUTPUT / "qa/M06_grayscale_preview.png"
    with Image.open(png) as image:
        ImageOps.grayscale(image.convert("RGB")).save(destination, dpi=(180, 180))
    return destination


def nonwhite_bbox(path: Path) -> list[int] | None:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        white = Image.new("RGB", rgb.size, "white")
        bbox = ImageChops.difference(rgb, white).getbbox()
    return list(bbox) if bbox else None


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"v5输出目录已存在且非空，拒绝覆盖：{OUTPUT}")
    for relative in (
        "main", "main_EN", "source_data", "captions_CN", "captions_EN",
        "python_sources", "manifests", "qa", "thumbnail_index",
    ):
        (OUTPUT / relative).mkdir(parents=True, exist_ok=True)

    v4_count_before, v4_digest_before = directory_digest(V4)
    if (v4_count_before, v4_digest_before) != (EXPECTED_V4_FILE_COUNT, EXPECTED_V4_DIGEST):
        raise RuntimeError("v4目录基线与实施前记录不一致，停止重绘")

    frame = pd.read_csv(SOURCE)
    input_audit = audit_source(frame)
    shutil.copy2(SOURCE, OUTPUT / "source_data/M06_source_data.csv")
    configure()

    outputs_cn = export_figure(render(frame, "zh"), OUTPUT / "main/M06_validation_learning_curve_v5")
    outputs_en = export_figure(render(frame, "en"), OUTPUT / "main_EN/M06_validation_learning_curve_v5_english")
    write_text(OUTPUT / "captions_CN/M06.md", CAPTION_CN)
    write_text(OUTPUT / "captions_EN/M06.md", CAPTION_EN)
    shutil.copy2(Path(__file__), OUTPUT / "python_sources/v3_2_14_m06_reference_v5.py")

    output_records_cn = [inspect_output(path) for path in outputs_cn]
    output_records_en = [inspect_output(path) for path in outputs_en]
    if not all(item["valid"] for item in [*output_records_cn, *output_records_en]):
        raise RuntimeError("至少一个导出文件未通过格式检查")

    grayscale = make_grayscale_preview(OUTPUT / "main/M06_validation_learning_curve_v5.png")
    bbox = nonwhite_bbox(OUTPUT / "main/M06_validation_learning_curve_v5.png")
    v4_count_after, v4_digest_after = directory_digest(V4)
    if (v4_count_after, v4_digest_after) != (v4_count_before, v4_digest_before):
        raise RuntimeError("v4目录在重绘过程中发生变化")

    manifest = {
        "figure_id": "M06",
        "figure_name": "验证学习曲线",
        "backend": "Python/Matplotlib",
        "style_reference": "EAAI interval-learning-curve grammar; no copied data or layout",
        "source_data": {
            "path": "source_data/M06_source_data.csv",
            "rows": len(frame),
            "sha256": sha256(OUTPUT / "source_data/M06_source_data.csv"),
        },
        "statistics": "five-seed median with interquartile range",
        "checkpoint_count_per_model": 26,
        "smoothing": "none",
        "outputs_cn": output_records_cn,
        "outputs_en": output_records_en,
        "caption_cn": "captions_CN/M06.md",
        "caption_en": "captions_EN/M06.md",
        "script": {
            "path": "python_sources/v3_2_14_m06_reference_v5.py",
            "sha256": sha256(OUTPUT / "python_sources/v3_2_14_m06_reference_v5.py"),
        },
    }
    write_text(OUTPUT / "manifests/M06.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    qa = {
        "passed": True,
        "input_audit": input_audit,
        "old_v4_immutability": {
            "files_before": v4_count_before,
            "files_after": v4_count_after,
            "digest_before": v4_digest_before,
            "digest_after": v4_digest_after,
            "unchanged": v4_digest_before == v4_digest_after and v4_count_before == v4_count_after,
        },
        "rendering": {
            "width_mm": WIDTH_MM,
            "height_mm": HEIGHT_MM,
            "x_limits": X_LIMITS,
            "y_limits": Y_LIMITS,
            "line_width_pt": LINE_WIDTH,
            "iqr_alpha": IQR_ALPHA,
            "seed_lines_drawn": False,
            "median_lines": 3,
            "iqr_bands": 3,
            "smoothing": False,
            "interpolation": False,
            "all_four_spines": True,
            "grid_axes": ["x", "y"],
        },
        "visual_checks": {
            "content_bbox_px": bbox,
            "grayscale_preview": grayscale.relative_to(OUTPUT).as_posix(),
            "grayscale_distinction_basis": "distinct solid, dashed, and dash-dot line styles",
            "legend_location": "lower right, inside empty data region",
            "manual_review_required": True,
        },
        "outputs": {"cn": output_records_cn, "en": output_records_en},
    }
    write_text(OUTPUT / "qa/qa_report.json", json.dumps(qa, ensure_ascii=False, indent=2))
    write_text(
        OUTPUT / "qa/QA_REPORT.md",
        "# M06 v5 QA报告\n\n"
        "- 输入：468行，含390条正式种子记录和78条汇总记录。\n"
        "- 曲线：3条五种子中位数主线和3个IQR带；不显示种子细线。\n"
        "- 检查点：每模型26个，未平滑、未插值、未新增观测点。\n"
        f"- 纵轴：{Y_LIMITS[0]:.2f}–{Y_LIMITS[1]:.2f}，包含全部正式种子值。\n"
        "- 输出：中英文PDF、SVG、600 dpi PNG和600 dpi LZW TIFF均通过格式检查。\n"
        f"- v4不可变审计：{v4_count_after}个文件，目录组合哈希`{v4_digest_after}`，前后一致。\n"
        "- 灰度可读性：三模型同时使用实线、虚线和点划线区分。\n"
        "- 最终人工视觉检查：需检查实际尺寸字体、图例遮挡、IQR重叠和边界裁切。\n",
    )
    write_text(
        OUTPUT / "README.md",
        "# M06权威文献风格重绘v5\n\n"
        "本目录只包含M06中英文验证学习曲线及其可复现证据。旧v4目录和论文包均未修改。"
        "正文曲线使用5种子中位数与IQR，不做任何平滑或插值。\n",
    )
    print(json.dumps({"status": "passed", "output": str(OUTPUT), "qa": qa["old_v4_immutability"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

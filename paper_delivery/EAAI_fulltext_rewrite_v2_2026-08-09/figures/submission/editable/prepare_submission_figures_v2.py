from __future__ import annotations

"""基于冻结 Source Data 生成第二轮投稿图件，不重新计算实验结果。"""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = ROOT / "figures"
SOURCE = FIG_ROOT / "source_data"
OUT = FIG_ROOT / "submission"
MAIN = OUT / "main"
SUPP = OUT / "supplementary"
SHOW = OUT / "showcase"
EDIT = OUT / "editable"
QA = OUT / "qa"

# 重要可调参数：最终双栏宽度、位图分辨率、裁切留白阈值。
FULL_WIDTH_MM = 183.0
DPI = 600
PAD_IN = 0.04
MARGIN_FAIL_MM = 4.0

COLORS = {
    "full": "#1764AB",
    "a2c_pointer": "#E07A1F",
    "traditional_ppo": "#2A9D55",
    "priority_resource_greedy": "#7A7A7A",
    "aco": "#8C4FB7",
    "milp": "#222222",
}
LABELS = {
    "full": "PPO–Pointer",
    "a2c_pointer": "A2C–Pointer",
    "traditional_ppo": "Flat-MLP PPO",
    "priority_resource_greedy": "Priority greedy",
    "aco": "ACO",
    "milp": "MILP",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_all(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("pdf", "svg", "png"):
        path = base.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "pad_inches": PAD_IN}
        if ext == "png":
            kwargs["dpi"] = DPI
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    # Matplotlib 直接写 TIFF 在部分 Windows 环境会触发底层崩溃；由已生成的
    # 600 dpi PNG 无损转换，可保持同一画布和像素内容。
    tiff_path = base.with_suffix(".tiff")
    with Image.open(base.with_suffix(".png")) as im:
        im.save(tiff_path, format="TIFF", compression="raw", dpi=(DPI, DPI))
    outputs.append(tiff_path)
    return outputs


def workflow_figure() -> list[Path]:
    width = FULL_WIDTH_MM / 25.4
    fig, ax = plt.subplots(figsize=(width, 3.05))
    ax.set_xlim(0, 1)
    # 收紧数据坐标范围，避免流程框上下出现与内容无关的大块留白。
    ax.set_ylim(0.10, 0.91)
    ax.axis("off")

    boxes = [
        (0.02, 0.63, 0.19, 0.24, "Mountain-road task", "Fixed inspection points\nDEM/DSM, wind and budgets"),
        (0.27, 0.63, 0.19, 0.24, "State encoding", "Point, route and resource\nfeatures with action masks"),
        (0.52, 0.63, 0.19, 0.24, "PPO–Pointer policy", "Permutation-aware selection\nwith clipped policy updates"),
        (0.77, 0.63, 0.21, 0.24, "Return-aware decision", "Feasible next point or\nexplicit return to depot"),
        (0.14, 0.15, 0.24, 0.24, "Frozen evaluation", "Unseen synthetic maps, DSM\ntransfer and robustness shifts"),
        (0.62, 0.15, 0.24, 0.24, "Evidence outputs", "Coverage, safety, resources,\ntime, training and ablations"),
    ]
    for i, (x, y, w, h, title, body) in enumerate(boxes):
        color = "#EAF2FB" if i < 4 else "#F2F5F1"
        edge = "#1764AB" if i < 4 else "#537A55"
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=1.0, edgecolor=edge, facecolor=color
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", weight="bold", fontsize=8.2)
        ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", fontsize=7.0, linespacing=1.15)

    arrows = [
        ((0.21, 0.75), (0.27, 0.75)),
        ((0.46, 0.75), (0.52, 0.75)),
        ((0.71, 0.75), (0.77, 0.75)),
        ((0.88, 0.63), (0.77, 0.39)),
        ((0.62, 0.27), (0.38, 0.27)),
        ((0.26, 0.39), (0.12, 0.63)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, lw=1.05, color="#44505C"))
    ax.text(0.50, 0.48, "Frozen protocol; no model retraining or result recomputation", ha="center", va="center", fontsize=7.4, color="#5A3333")
    return save_all(fig, MAIN / "F01_method_and_evaluation_workflow")


def m05_figure() -> list[Path]:
    data = pd.read_csv(SOURCE / "M05_source_data.csv")
    width = FULL_WIDTH_MM / 25.4
    fig, ax = plt.subplots(figsize=(width, 3.55))
    order = ["full", "a2c_pointer", "traditional_ppo", "priority_resource_greedy", "aco", "milp"]
    for model in order:
        d = data[data["model"] == model].sort_values("planning_time_s")
        ax.step(d["planning_time_s"], d["ecdf"], where="post", lw=1.55, color=COLORS[model], label=LABELS[model])
    ax.set_xscale("log")
    ax.set_xlim(max(data["planning_time_s"].min() * 0.75, 1e-3), data["planning_time_s"].max() * 1.25)
    ax.set_ylim(0, 1.015)
    ax.set_xlabel("Online planning time per task (s; log scale)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.grid(True, which="major", color="#D7DDE3", lw=0.55, alpha=0.85)
    ax.grid(True, which="minor", axis="x", color="#E9EDF1", lw=0.4, alpha=0.7)
    ax.legend(ncol=3, loc="lower right", frameon=True, framealpha=0.95, borderpad=0.45, handlelength=2.3)
    fig.text(0.01, 0.985, "a", ha="left", va="top", fontsize=10, weight="bold")
    return save_all(fig, MAIN / "M05_online_planning_time_ECDF_repaired")


def v02_figure() -> list[Path]:
    terrain = pd.read_csv(SOURCE / "V02" / "terrain.csv")
    roads = pd.read_csv(SOURCE / "V02" / "roads.csv")
    routes = pd.read_csv(SOURCE / "V02" / "routes.csv")
    points = pd.read_csv(SOURCE / "V02" / "points.csv")

    xvals = np.sort(terrain["x"].unique())
    yvals = np.sort(terrain["y"].unique())
    zgrid = terrain.pivot(index="y", columns="x", values="z").reindex(index=yvals, columns=xvals).to_numpy()
    width = FULL_WIDTH_MM / 25.4
    fig, ax = plt.subplots(figsize=(width, 4.75))
    cf = ax.contourf(xvals, yvals, zgrid, levels=26, cmap="terrain", antialiased=True)
    for _, road in roads.sort_values(["road_id", "sequence"]).groupby("road_id"):
        ax.plot(road["x"], road["y"], color="#4D5052", lw=0.55, alpha=0.70, zorder=2)

    order = ["milp", "traditional_ppo", "a2c_pointer", "full"]
    linestyles = {"milp": ":", "traditional_ppo": "--", "a2c_pointer": "-.", "full": "-"}
    for model in order:
        d = routes[routes["model"] == model].sort_values("sequence")
        ax.plot(d["x"], d["y"], lw=1.65 if model == "full" else 1.25,
                color=COLORS[model], ls=linestyles[model], label=LABELS[model], zorder=5)

    inspect = points[points["point_type"] == "inspection"]
    priority_colors = {1: "#F5C04A", 2: "#F28E2B", 3: "#C43B3B"}
    for priority in (1, 2, 3):
        d = inspect[inspect["priority"] == priority]
        ax.scatter(d["x"], d["y"], s=24, marker="o", facecolor=priority_colors[priority], edgecolor="white",
                   linewidth=0.45, label=f"Priority {priority} point", zorder=7)
    airport = points[points["point_type"] == "airport"]
    ax.scatter(airport["x"], airport["y"], s=90, marker="*", facecolor="white", edgecolor="#111111",
               linewidth=0.9, label="Depot", zorder=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Projected x coordinate")
    ax.set_ylabel("Projected y coordinate")
    ax.set_title("Fixed DSM task: terrain, road network and representative routes", pad=5)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Elevation")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=2, loc="upper left", frameon=True, framealpha=0.93,
              borderpad=0.45, columnspacing=0.9, handlelength=2.3)
    ax.tick_params(direction="out")
    fig.text(0.01, 0.985, "b", ha="left", va="top", fontsize=10, weight="bold")
    return save_all(fig, SHOW / "V02_fixed_DSM_route_repaired")


def copy_frozen_figures() -> list[Path]:
    copied: list[Path] = []
    for src_dir, dst_dir in ((FIG_ROOT / "main", MAIN), (FIG_ROOT / "supplementary", SUPP), (FIG_ROOT / "showcase", SHOW)):
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.iterdir():
            if src.is_file():
                # M05 与 V02 使用可追溯的英文重绘版本，旧件只留在冻结副本中。
                if src.name.startswith("M05_") or src.name.startswith("V02_"):
                    continue
                dst = dst_dir / src.name
                shutil.copy2(src, dst)
                copied.append(dst)
    return copied


def png_margin_mm(path: Path) -> dict[str, float]:
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    nonwhite = np.any(arr < 248, axis=2)
    ys, xs = np.where(nonwhite)
    if len(xs) == 0:
        return {"left": 999.0, "right": 999.0, "top": 999.0, "bottom": 999.0}
    dpi = float(im.info.get("dpi", (DPI, DPI))[0] or DPI)
    mm_per_px = 25.4 / dpi
    return {
        "left": float(xs.min() * mm_per_px),
        "right": float((arr.shape[1] - 1 - xs.max()) * mm_per_px),
        "top": float(ys.min() * mm_per_px),
        "bottom": float((arr.shape[0] - 1 - ys.max()) * mm_per_px),
    }


def main() -> None:
    setup_style()
    for d in (MAIN, SUPP, SHOW, EDIT, QA):
        d.mkdir(parents=True, exist_ok=True)
    copied = copy_frozen_figures()
    created = workflow_figure() + m05_figure() + v02_figure()

    # 可编辑源明确包括本脚本和冻结 CSV；不以位图充当源文件。
    shutil.copy2(__file__, EDIT / Path(__file__).name)
    source_manifest = []
    for p in sorted(SOURCE.rglob("*")):
        if p.is_file():
            source_manifest.append({"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size})

    margin_records = []
    for p in (MAIN / "F01_method_and_evaluation_workflow.png", MAIN / "M05_online_planning_time_ECDF_repaired.png", SHOW / "V02_fixed_DSM_route_repaired.png"):
        margins = png_margin_mm(p)
        margin_records.append({"file": str(p.relative_to(ROOT)), "margins_mm": margins, "pass": max(margins.values()) <= MARGIN_FAIL_MM})

    all_files = sorted([p for p in OUT.rglob("*") if p.is_file()])
    manifest = {
        "purpose": "Submission figure package from frozen third-round figures and frozen Source Data",
        "generation_rule": "No experiment, statistic, or source value was recomputed",
        "full_width_mm": FULL_WIDTH_MM,
        "raster_dpi": DPI,
        "created_count": len(created),
        "copied_count": len(copied),
        "files": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size} for p in all_files],
        "source_data": source_manifest,
        "margin_qa": margin_records,
    }
    (OUT / "submission_figure_manifest_v2.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (QA / "automatic_qa_v2.json").write_text(json.dumps({"margin_qa": margin_records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"created": len(created), "copied": len(copied), "margin_qa": margin_records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

"""准备v3中英文图件；仅复制或重排冻结图件，不重新计算实验结果。"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT.parent / "EAAI_fulltext_rewrite_v2_2026-08-09"
SRC = ROOT / "figures" / "source_data"
EN = ROOT / "figures" / "english"
ZH = ROOT / "figures" / "chinese"
EDIT = ROOT / "figures" / "editable"
QA = ROOT / "figures" / "qa"

# 重要可调参数：EAAI全宽图183 mm、600 dpi、约1 mm安全留白。
FULL_WIDTH_MM = 183.0
DPI = 600
PAD_IN = 1.0 / 25.4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 8.0,
        "axes.labelsize": 9.0,
        "legend.fontsize": 7.3,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_all(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for ext in ("pdf", "svg", "png"):
        path = base.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "pad_inches": PAD_IN}
        if ext == "png":
            kwargs["dpi"] = DPI
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    tiff = base.with_suffix(".tiff")
    # 部分Windows Pillow构建在LZW编码时会触发本机崩溃；使用无损raw TIFF保持600 dpi与像素一致。
    with Image.open(base.with_suffix(".png")) as im:
        im.convert("RGB").save(tiff, format="TIFF", compression="raw", dpi=(DPI, DPI))
    outputs.append(tiff)
    return outputs


def render_f01_zh() -> list[Path]:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_MM / 25.4, 3.05), layout="constrained")
    ax.set_xlim(0, 1); ax.set_ylim(0.10, 0.91); ax.axis("off")
    boxes = [
        (0.02, 0.63, 0.19, 0.24, "山区公路任务", "固定巡检点\nDSM、风场与预算"),
        (0.27, 0.63, 0.19, 0.24, "状态编码", "巡检点、路线与资源特征\n以及动作掩码"),
        (0.52, 0.63, 0.19, 0.24, "PPO–Pointer策略", "带截断策略更新的\n变长候选选择"),
        (0.77, 0.63, 0.21, 0.24, "返航感知决策", "选择可行下一点\n或显式返回基地"),
        (0.14, 0.18, 0.25, 0.23, "冻结评价", "未见合成地图、DSM迁移\n与鲁棒性偏移"),
        (0.61, 0.18, 0.25, 0.23, "证据输出", "覆盖率、安全性、资源、时间\n训练与消融结果"),
    ]
    for x, y, w, h, title, body in boxes:
        face = "#F4F7FA" if y > 0.5 else "#F4F8F3"
        edge = "#4079A7" if y > 0.5 else "#608A67"
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=face, ec=edge, lw=1.0))
        ax.text(x + 0.012, y + h - 0.065, title, weight="bold", va="top", fontsize=8.2)
        ax.text(x + 0.012, y + h - 0.115, body, va="top", fontsize=7.1, linespacing=1.25)
    arrows = [((0.21, .75), (.27, .75)), ((.46, .75), (.52, .75)), ((.71, .75), (.77, .75)),
              ((.875, .63), (.74, .41)), ((.61, .29), (.39, .29)), ((.265, .41), (.12, .63))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color="#46535D", lw=.9))
    ax.text(.5, .50, "冻结协议：不重训模型，不重新计算结果", ha="center", va="center", fontsize=7.4, color="#7A4B3A")
    return save_all(fig, ZH / "main" / "F01_workflow_zh")


def render_m05_zh() -> list[Path]:
    labels = {
        "full": "PPO–Pointer", "a2c_pointer": "A2C–Pointer", "traditional_ppo": "Flat-MLP PPO",
        "priority_resource_greedy": "优先级–资源贪心", "aco": "ACO", "milp": "MILP",
    }
    colors = {"full":"#1764AB", "a2c_pointer":"#E07A1F", "traditional_ppo":"#2A9D55",
              "priority_resource_greedy":"#777777", "aco":"#7B6FA6", "milp":"#222222"}
    styles = {"full":"-", "a2c_pointer":"-", "traditional_ppo":"-", "priority_resource_greedy":"--", "aco":"--", "milp":"--"}
    data = pd.read_csv(SRC / "M05_source_data.csv")
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_MM / 25.4, 4.2), layout="constrained")
    for model in labels:
        part = data[data.model == model].sort_values("planning_time_s")
        if part.empty:
            continue
        ax.step(part.planning_time_s, part.ecdf, where="post", label=labels[model], color=colors[model],
                linestyle=styles[model], lw=1.55)
    ax.set_xscale("log")
    ax.set_xlabel("在线规划时间（s，对数轴）")
    ax.set_ylabel("经验累积概率")
    ax.set_ylim(0, 1.02)
    ax.grid(color="#E0E5E9", lw=.5)
    ax.legend(ncol=3, loc="lower right", frameon=False)
    return save_all(fig, ZH / "main" / "M05_online_planning_time_zh")


def copy_tree_files() -> list[Path]:
    copied: list[Path] = []
    src_en = V2 / "figures" / "submission"
    for sub in ("main", "supplementary", "showcase"):
        dst = EN / sub
        dst.mkdir(parents=True, exist_ok=True)
        for path in (src_en / sub).glob("*"):
            if path.is_file() and path.suffix.lower() in {".pdf", ".svg", ".png", ".tiff"}:
                target = dst / path.name
                shutil.copy2(path, target); copied.append(target)

    mapping = {
        "main": {
            "M01": "M01_优先级加权覆盖率分布", "M02": "M02_安全率与返航率效应",
            "M03": "M03_高中低优先级巡检效果", "M04": "M04_能耗、航程与总任务时间",
            "M06": "M06_五种子收敛曲线", "M07": "M07_训练稳定性与样本效率",
            "M08": "M08_未见地图与真实DSM迁移", "M09": "M09_已知偏移与隐藏误差鲁棒性",
            "M10": "M10_四项消融总体效应",
        },
        "supplementary": {
            "S01": "S01_全算法Performance Profile", "S02": "S02_覆盖效果—在线时间Pareto",
            "S03": "S03_Oracle regret—计算代价", "S04": "S04_场景分层结果热力图",
            "S05": "S05_鲁棒性与失败模式", "S06": "S06_七个学习模型训练过程",
            "S07": "S07_七维指标与100分综合摘要", "S08": "S08_权重与归一化联合敏感性",
        },
        "showcase": {"V01": "V01_固定合成任务路线"},
    }
    src_zh = V2 / "figures" / "submission"
    for sub, entries in mapping.items():
        dst = ZH / sub; dst.mkdir(parents=True, exist_ok=True)
        for new_stem, old_stem in entries.items():
            for ext in (".pdf", ".svg", ".png", ".tiff"):
                source = src_zh / sub / f"{old_stem}{ext}"
                if source.exists():
                    target = dst / f"{new_stem}_zh{ext}"
                    shutil.copy2(source, target); copied.append(target)

    # 保留现有可编辑源；无效SVG不伪造，由PDF/OPJU承担矢量和可编辑交付。
    EDIT.mkdir(parents=True, exist_ok=True)
    for source in (V2 / "figures" / "origin_projects").glob("*.opju"):
        target = EDIT / source.name; shutil.copy2(source, target); copied.append(target)
    for source in (V2 / "figures" / "showcase").glob("V02_*.fig"):
        target = EDIT / source.name; shutil.copy2(source, target); copied.append(target)
    return copied


def main() -> None:
    setup_style(); QA.mkdir(parents=True, exist_ok=True)
    print("[v3-figures] copying validated exports", flush=True)
    outputs = copy_tree_files()
    print("[v3-figures] rendering M05", flush=True)
    outputs += render_m05_zh()
    print("[v3-figures] rendering F01", flush=True)
    outputs += render_f01_zh()
    records = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size} for p in outputs]
    report = {
        "rule": "Frozen Source Data and validated v2 exports only; no experiment or statistic recomputed",
        "full_width_mm": FULL_WIDTH_MM, "dpi": DPI, "files": records,
        "pending_matlab": "V02 Chinese route is rendered by render_V02_zh_v4.m",
    }
    (QA / "figure_manifest_pre_matlab_v4.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(outputs), "pending": "V02"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

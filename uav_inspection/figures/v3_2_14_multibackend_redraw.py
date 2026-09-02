"""v3.2.14 第三轮多后端制图流水线。

当前版本只实施四图样稿闸门：M01、M02、M06、V02。所有统计输入均直接
来自冻结正式结果、训练日志和地图/路线资产；旧制图目录只参与哈希审计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image, ImageChops, ImageDraw, ImageFont

from uav_inspection.figures.v3_2_14_literature_audit import LITERATURE_AUDIT, audit_summary
from scipy.stats import gaussian_kde

from uav_inspection.paths import WORKSPACE_ROOT


ROOT = WORKSPACE_ROOT
RUN = ROOT / "paper_runs" / "multimap_v3_2_14"
RESULTS_DIR = RUN / "formal_evaluation" / "results"
FINAL_RESULTS = RESULTS_DIR / "final_results.jsonl"
FINAL_AUDIT = RESULTS_DIR / "final_audit_status.json"
EVALUATION_MATRIX = RUN / "formal_evaluation" / "evaluation_matrix.jsonl"
REAL_TASKS = RUN / "formal_evaluation" / "real_tasks_parallel" / "records.jsonl"
TRAINING_TRACES = RUN / "analysis" / "training_trace_inputs_v2"
TRADITIONAL_TRAINING = ROOT / "paper_runs" / "multimap_v3_2" / "formal_training"
MAP_ROOT = ROOT / "map_data" / "multimap_v3_1" / "real" / "cn_taihang"

OLD_PAPER_FINAL = RUN / "figures" / "paper_final"
OLD_ORIGIN_V2 = RUN / "figures" / "paper_redraw_origin_v2"
OUTPUT = RUN / "figures" / "paper_redraw_multibackend_v3"

EXPECTED_ROWS = 21_648
EXPECTED_MATRIX_SHA256 = "48a31ee9b58d41a617fff61acb6eba6a2d9a930767d7af15856f70a964686224"
EXPECTED_RESULTS_SHA256 = "4b620c21566c2e33c875f6bea2017b741b02a7d30d70aa50add60a6d06214a2c"
EXPORT_DPI = 600
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260805
REAL_EXAMPLE = "real_test__cn_taihang__road_00__task_08"
ROUTE_SEED = 42

# 关键视觉参数集中在这里；调整后会影响四张样图的整体风格。
COLORS = {
    "full": "#2369BD",
    "a2c_pointer": "#E68619",
    "traditional_ppo": "#2A9D8F",
    "priority_resource_greedy": "#747474",
    "aco": "#7A6E9D",
    "milp": "#222222",
}
MARKERS = {
    "full": "o",
    "a2c_pointer": "s",
    "traditional_ppo": "^",
    "priority_resource_greedy": "X",
    "aco": "D",
    "milp": "P",
}
LABELS = {
    "full": "PPO+Pointer",
    "a2c_pointer": "A2C+Pointer",
    "traditional_ppo": "传统PPO",
    "priority_resource_greedy": "优先级-资源贪心",
    "aco": "ACO",
    "milp": "MILP",
}
DOMAINS = {"synthetic": "未见合成地图", "real": "真实DSM"}
MAIN_COMPARE = (
    "full",
    "a2c_pointer",
    "traditional_ppo",
    "priority_resource_greedy",
    "aco",
    "milp",
)
PROTOTYPE_IDS = ("M01", "M02", "M06", "V02")

FIGURES: dict[str, dict[str, str]] = {
    "M01": {"tier": "main", "name": "优先级加权覆盖率分布", "backend": "python", "template": "custom_half_eye"},
    "M02": {"tier": "main", "name": "安全率与返航率效应", "backend": "origin", "template": "SCATTERINTERVAL.otp"},
    "M03": {"tier": "main", "name": "高中低优先级巡检效果", "backend": "origin", "template": "SCATTER.OTP"},
    "M04": {"tier": "main", "name": "能耗、航程与总任务时间", "backend": "origin", "template": "SCATTERINTERVAL.otp"},
    "M05": {"tier": "main", "name": "在线规划时间ECDF", "backend": "origin", "template": "LINE.OTP"},
    "M06": {"tier": "main", "name": "五种子收敛曲线", "backend": "python", "template": "median_iqr_learning_curve"},
    "M07": {"tier": "main", "name": "训练稳定性与样本效率", "backend": "origin", "template": "SCATTER.OTP"},
    "M08": {"tier": "main", "name": "未见地图与真实DSM迁移", "backend": "origin", "template": "SCATTER.OTP"},
    "M09": {"tier": "main", "name": "已知偏移与隐藏误差鲁棒性", "backend": "python", "template": "annotated_heatmap"},
    "M10": {"tier": "main", "name": "四项消融总体效应", "backend": "origin", "template": "SCATTER.OTP"},
    "S01": {"tier": "supplementary", "name": "全算法Performance Profile", "backend": "python", "template": "performance_profile"},
    "S02": {"tier": "supplementary", "name": "覆盖效果—在线时间Pareto", "backend": "origin", "template": "SCATTER.OTP"},
    "S03": {"tier": "supplementary", "name": "Oracle regret—计算代价", "backend": "origin", "template": "SCATTER.OTP"},
    "S04": {"tier": "supplementary", "name": "场景分层结果热力图", "backend": "python", "template": "annotated_heatmap"},
    "S05": {"tier": "supplementary", "name": "鲁棒性与失败模式", "backend": "python", "template": "directional_heatmap"},
    "S06": {"tier": "supplementary", "name": "七个学习模型训练过程", "backend": "python", "template": "learning_curve"},
    "S07": {"tier": "supplementary", "name": "七维指标与100分综合摘要", "backend": "origin", "template": "SCATTER.OTP"},
    "S08": {"tier": "supplementary", "name": "权重与归一化联合敏感性", "backend": "python", "template": "contour_heatmap"},
    "V01": {"tier": "showcase", "name": "固定合成任务路线", "backend": "python", "template": "route_map"},
    "V02": {"tier": "showcase", "name": "固定真实DSM地形路线", "backend": "matlab", "template": "terrain_route_2p5d"},
}

CAPTIONS = {
    "M01": (
        "未见合成地图与真实DSM上的地图级优先级加权覆盖率。每个散点代表一张独立地图；"
        "密度脊仅描述地图分布，短竖线为地图中位数。任务内先聚合训练/规划种子，再聚合到地图。"
    ),
    "M02": (
        "已知域偏移与隐藏模型/感知误差下，PPO+Pointer相对比较算法的地图级安全率和返航率差异。"
        "点为配对Hodges–Lehmann伪中位效应，横线为10,000次地图外层bootstrap 95%区间。"
        "标称合成域和DSM域中所列六种算法的安全率与返航率均为100%，因此标称域仅构成天花板证据，"
        "不在主森林图中重复绘制全零效应。该图为描述性区间，不追加确认性显著性检验。"
    ),
    "M06": (
        "三个核心学习模型在同一训练任务分布上均训练5个种子、每个种子3000回合。横轴为累计训练回合，"
        "纵轴为每个训练批次的优先级加权覆盖率；所有批次返航率均为100%。细线为单个训练种子，粗线为"
        "跨种子中位数，阴影为四分位距。该图用于比较收敛形态和训练稳定性，最终泛化性能仍由冻结测试集评价。"
    ),
    "V02": (
        f"固定真实DSM任务{REAL_EXAMPLE}上的四算法seed 42路线。地形来自冻结太行DSM资产，"
        "道路、机场、24个固定巡检点和优先级均来自正式任务；失败或缺失路线不得更换任务或种子。"
        "该图为零样本仿真迁移的场景解释，不构成真实飞行验证。"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": _sha256(path),
        })
    payload = "".join(
        f"{row['path']}|{row['bytes']}|{row['mtime_ns']}|{row['sha256']}\n" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def _mm(width: float, height: float) -> tuple[float, float]:
    return width / 25.4, height / 25.4


def configure_matplotlib() -> None:
    available = {font.name for font in fm.fontManager.ttflist}
    cjk = "Microsoft YaHei" if "Microsoft YaHei" in available else "SimHei"
    plt.rcParams.update({
        "font.family": [cjk, "Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _output_stem(figure_id: str) -> Path:
    meta = FIGURES[figure_id]
    folder = OUTPUT / ("main" if meta["tier"] == "main" else "showcase")
    return folder / f"{figure_id}_{meta['name']}"


def _save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for extension, kwargs in {
        "pdf": {"bbox_inches": "tight", "pad_inches": 0.04},
        "svg": {"bbox_inches": "tight", "pad_inches": 0.04},
        "png": {"dpi": EXPORT_DPI, "bbox_inches": "tight", "pad_inches": 0.04},
    }.items():
        target = stem.with_suffix(f".{extension}")
        fig.savefig(target, **kwargs)
        outputs[extension] = str(target)
    plt.close(fig)
    # 本机Pillow的LZW编码器在大幅RGBA图上会原生崩溃；由600 dpi PNG无损转为RGB TIFF。
    # TIFF不压缩只增加文件体积，不改变像素内容或出版质量。
    tiff = stem.with_suffix(".tiff")
    with Image.open(stem.with_suffix(".png")) as image:
        image.convert("RGB").save(tiff, dpi=(EXPORT_DPI, EXPORT_DPI))
    outputs["tiff"] = str(tiff)
    return outputs


def audit_inputs(write_snapshot: bool = True) -> dict[str, Any]:
    status = json.loads(FINAL_AUDIT.read_text(encoding="utf-8-sig"))
    old_final_hash, old_final_files = _tree_digest(OLD_PAPER_FINAL)
    old_v2_hash, old_v2_files = _tree_digest(OLD_ORIGIN_V2)
    errors: list[str] = []
    if status.get("row_count") != EXPECTED_ROWS or status.get("route_count") != EXPECTED_ROWS:
        errors.append("正式结果或路线数量不是21,648")
    if status.get("matrix_sha256") != EXPECTED_MATRIX_SHA256:
        errors.append("冻结评价矩阵哈希漂移")
    if status.get("results_sha256") != EXPECTED_RESULTS_SHA256:
        errors.append("正式结果哈希漂移")
    if _sha256(FINAL_RESULTS) != EXPECTED_RESULTS_SHA256:
        errors.append("final_results.jsonl实际哈希漂移")
    if "ppo_mlp" in FINAL_RESULTS.read_text(encoding="utf-8", errors="ignore"):
        errors.append("正式结果中出现已排除的ppo_mlp")
    report = {
        "passed": not errors,
        "errors": errors,
        "formal": {
            "row_count": status.get("row_count"),
            "route_count": status.get("route_count"),
            "results_sha256": status.get("results_sha256"),
            "matrix_sha256": status.get("matrix_sha256"),
        },
        "old_directories": {
            "paper_final": {"sha256": old_final_hash, "file_count": len(old_final_files), "files": old_final_files},
            "paper_redraw_origin_v2": {"sha256": old_v2_hash, "file_count": len(old_v2_files), "files": old_v2_files},
        },
    }
    if write_snapshot:
        _write_json(OUTPUT / "qa" / "old_directories_at_start.json", report)
    if errors:
        raise RuntimeError("；".join(errors))
    return report


def _read_results() -> pd.DataFrame:
    return pd.read_json(FINAL_RESULTS, lines=True)


def build_m01(results: pd.DataFrame) -> pd.DataFrame:
    frame = results[
        results["condition"].eq("nominal") & results["model"].isin(MAIN_COMPARE)
    ].copy()
    frame["domain"] = np.where(frame["family"].str.startswith("synthetic"), "synthetic", "real")
    frame = frame[frame["family"].isin({
        "synthetic_learning", "synthetic_main_baselines", "real_learning", "real_baselines"
    })]
    # 先在任务内聚合训练/规划种子，再在地图内聚合任务。
    task = (
        frame.groupby(["domain", "map_id", "task_id", "model"], as_index=False)
        .agg(weighted_coverage=("weighted_coverage", "mean"), repetitions=("result_hash", "size"))
    )
    maps = (
        task.groupby(["domain", "map_id", "model"], as_index=False)
        .agg(weighted_coverage=("weighted_coverage", "mean"), task_count=("task_id", "nunique"), repetitions=("repetitions", "sum"))
    )
    maps["domain_label"] = maps["domain"].map(DOMAINS)
    maps["model_label"] = maps["model"].map(LABELS)
    expected = {"synthetic": 24, "real": 8}
    for domain, count in expected.items():
        observed = maps[maps["domain"].eq(domain)].groupby("model")["map_id"].nunique()
        if set(observed.index) != set(MAIN_COMPARE) or not (observed == count).all():
            raise RuntimeError(f"M01 {domain}地图配对不完整: {observed.to_dict()}")
    return maps.sort_values(["domain", "model", "map_id"]).reset_index(drop=True)


def _paired_hl(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    Walsh = (values[:, None] + values[None, :]) / 2.0
    return float(np.median(Walsh[np.triu_indices(values.size)]))


def _bootstrap_hl(values: np.ndarray, seed_offset: int) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    values = np.asarray(values, dtype=float)
    draws = np.empty(BOOTSTRAP_REPS, dtype=float)
    for index in range(BOOTSTRAP_REPS):
        draws[index] = _paired_hl(rng.choice(values, size=values.size, replace=True))
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def build_m02(results: pd.DataFrame) -> pd.DataFrame:
    layers = {
        "known_domain_shift": "已知域偏移",
        "hidden_model_perception_mismatch": "隐藏模型/感知误差",
    }
    comparators = ("a2c_pointer", "traditional_ppo", "priority_resource_greedy")
    frame = results[
        results["family"].isin(layers) & results["model"].isin(("full",) + comparators)
    ].copy()
    for metric in ("safe", "returned"):
        frame[metric] = frame[metric].astype(float)
    # 每个条件在每张地图内等权；训练种子和任务均嵌套在地图内。
    condition_map = (
        frame.groupby(["family", "condition", "map_id", "model"], as_index=False)
        .agg(safe=("safe", "mean"), returned=("returned", "mean"))
    )
    layer_map = (
        condition_map.groupby(["family", "map_id", "model"], as_index=False)
        .agg(safe=("safe", "mean"), returned=("returned", "mean"), condition_count=("condition", "nunique"))
    )
    rows: list[dict[str, Any]] = []
    seed_offset = 0
    for family, layer_label in layers.items():
        subset = layer_map[layer_map["family"].eq(family)]
        for metric, metric_label in (("safe", "安全率"), ("returned", "返航率")):
            pivot = subset.pivot(index="map_id", columns="model", values=metric)
            for comparator in comparators:
                paired = pivot[["full", comparator]].dropna()
                if len(paired) != 8:
                    raise RuntimeError(f"M02 {family}/{metric}/{comparator}只有{len(paired)}张配对地图")
                diff = 100.0 * (paired["full"] - paired[comparator]).to_numpy()
                estimate = _paired_hl(diff)
                low, high = _bootstrap_hl(diff, seed_offset)
                seed_offset += 1
                rows.append({
                    "layer": family,
                    "layer_label": layer_label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "comparator": comparator,
                    "comparator_label": LABELS[comparator],
                    "estimate_pp": estimate,
                    "ci_low_pp": low,
                    "ci_high_pp": high,
                    "n_maps": len(paired),
                    "row_label": f"{layer_label} · {metric_label}｜{LABELS[comparator]}",
                })
    return pd.DataFrame(rows)


def _trace_directories() -> Iterable[tuple[str, int, Path]]:
    for model, prefix in (("full", "formal_full_seed"), ("a2c_pointer", "formal_a2c_pointer_seed")):
        for seed in range(42, 47):
            yield model, seed, TRAINING_TRACES / f"{prefix}{seed}_3000ep" / "metrics.jsonl"
    # 传统PPO位于v3.2补充训练目录；与前两者相同，五个种子均训练3000回合。
    traditional_root = ROOT / "paper_runs" / "multimap_v3_2" / "formal_training"
    for seed in range(42, 47):
        yield (
            "traditional_ppo",
            seed,
            traditional_root / f"formal_traditional_ppo_seed{seed}_3000ep" / "training_metrics.jsonl",
        )


def build_m06() -> pd.DataFrame:
    raw_rows: list[dict[str, Any]] = []
    for model, seed, path in _trace_directories():
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("return_rate") != 1.0:
                raise RuntimeError("M06训练批次并非全部安全返航，不能直接使用加权覆盖率")
            raw_rows.append({
                "record_type": "seed",
                "model": model,
                "model_label": LABELS[model],
                "seed": seed,
                "episodes_seen": float(record["episodes_seen"]),
                "environment_interactions": float(record["environment_interactions"]),
                "safe_weighted_coverage": float(record["mean_weighted_coverage"]),
                "curve_source": "training_batch",
                "q25": np.nan,
                "median": np.nan,
                "q75": np.nan,
            })
    raw = pd.DataFrame(raw_rows)
    if set(raw.groupby("model")["seed"].nunique()) != {5}:
        raise RuntimeError("M06不是每模型五个训练种子")
    if set(raw["model"]) != {"full", "a2c_pointer", "traditional_ppo"}:
        raise RuntimeError("M06三个核心模型不完整")
    if not all(
        math.isclose(value, 3000.0)
        for value in raw.groupby(["model", "seed"])["episodes_seen"].max().to_numpy()
    ):
        raise RuntimeError("M06存在未达到3000回合的正式种子")

    lower = max(raw.groupby(["model", "seed"])["episodes_seen"].min())
    grid = np.linspace(lower, 3000.0, 100)
    summary_rows: list[dict[str, Any]] = []
    for model in ("full", "a2c_pointer", "traditional_ppo"):
        curves: list[np.ndarray] = []
        for seed in range(42, 47):
            trace = raw[(raw["model"].eq(model)) & (raw["seed"].eq(seed))].sort_values("episodes_seen")
            curves.append(np.interp(grid, trace["episodes_seen"], trace["safe_weighted_coverage"]))
        matrix = np.vstack(curves)
        q25, median, q75 = np.quantile(matrix, [0.25, 0.5, 0.75], axis=0)
        for x, lo, mid, hi in zip(grid, q25, median, q75):
            summary_rows.append({
                "record_type": "summary",
                "model": model,
                "model_label": LABELS[model],
                "seed": np.nan,
                "episodes_seen": x,
                "environment_interactions": np.nan,
                "safe_weighted_coverage": np.nan,
                "curve_source": "training_batch_seed_summary",
                "q25": lo,
                "median": mid,
                "q75": hi,
            })
    return pd.concat([raw, pd.DataFrame(summary_rows)], ignore_index=True)


def _read_task(task_id: str) -> dict[str, Any]:
    with REAL_TASKS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("id") == task_id:
                return record
    raise KeyError(task_id)


def _route_path(model: str) -> Path:
    patterns = {
        "full": f"full__seed{ROUTE_SEED}__{REAL_EXAMPLE}.json",
        "a2c_pointer": f"a2c_pointer__seed{ROUTE_SEED}__{REAL_EXAMPLE}.json",
        "traditional_ppo": f"traditional_ppo__seed{ROUTE_SEED}__{REAL_EXAMPLE}.json",
    }
    if model == "milp":
        return RESULTS_DIR / "real_baselines" / "jobs" / "milp__seed42" / "routes" / f"{REAL_EXAMPLE}.json"
    matches = list((RESULTS_DIR / "real_learning" / "shards").rglob(patterns[model]))
    if len(matches) != 1:
        raise RuntimeError(f"{model}固定路线匹配数={len(matches)}")
    return matches[0]


def _sample_terrain(terrain: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xi = np.clip(np.rint(x).astype(int), 0, terrain.shape[1] - 1)
    yi = np.clip(np.rint(y).astype(int), 0, terrain.shape[0] - 1)
    return terrain[yi, xi]


def build_v02() -> dict[str, pd.DataFrame]:
    task = _read_task(REAL_EXAMPLE)
    bundle = np.load(MAP_ROOT / "map_bundle.npz", allow_pickle=True)
    terrain = bundle["terrain"].astype(float)
    roads = bundle["road_points"].astype(float)
    offsets = bundle["road_offsets"].astype(int)
    yy, xx = np.indices(terrain.shape)
    terrain_frame = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "z": terrain.ravel()})

    road_rows: list[dict[str, Any]] = []
    for road_id, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        segment = roads[start:end]
        elevations = _sample_terrain(terrain, segment[:, 0], segment[:, 1])
        for sequence, ((x, y), z) in enumerate(zip(segment, elevations)):
            road_rows.append({"road_id": road_id, "sequence": sequence, "x": x, "y": y, "z": z})

    points = np.asarray(task["inspection_points_xyz"], dtype=float)
    point_z = _sample_terrain(terrain, points[:, 0], points[:, 1])
    point_rows = pd.DataFrame({
        "point_index": np.arange(len(points)),
        "x": points[:, 0],
        "y": points[:, 1],
        "z": point_z,
        "priority": np.asarray(task["priorities"], dtype=int),
        "point_type": "inspection",
    })
    start = np.asarray(task["start_xy"], dtype=float)
    airport = pd.DataFrame({
        "point_index": [24], "x": [start[0]], "y": [start[1]],
        "z": [_sample_terrain(terrain, start[:1], start[1:])[0]],
        "priority": [0], "point_type": ["airport"],
    })
    point_rows = pd.concat([point_rows, airport], ignore_index=True)

    route_rows: list[dict[str, Any]] = []
    route_files: dict[str, str] = {}
    for model in ("full", "a2c_pointer", "traditional_ppo", "milp"):
        path = _route_path(model)
        route_files[model] = str(path.relative_to(ROOT))
        payload = json.loads(path.read_text(encoding="utf-8"))
        detail = payload.get("detail") if model != "milp" else payload.get("result")
        route = detail.get("path") if detail else None
        if not route:
            route_rows.append({"model": model, "sequence": -1, "x": np.nan, "y": np.nan, "z": np.nan, "status": "route_missing"})
            continue
        for sequence, xyz in enumerate(route):
            route_rows.append({
                "model": model, "sequence": sequence,
                "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]), "status": "route",
            })
    routes = pd.DataFrame(route_rows)
    if set(routes["model"]) != {"full", "a2c_pointer", "traditional_ppo", "milp"}:
        raise RuntimeError("V02四条冻结路线不完整")
    metadata = pd.DataFrame({
        "key": ["task_id", "task_hash", "map_hash", "map_file_sha256", "route_seed", "route_files_json"],
        "value": [REAL_EXAMPLE, task["task_hash"], task["map_hash"], task["map_file_sha256"], ROUTE_SEED, json.dumps(route_files, ensure_ascii=False)],
    })
    return {
        "terrain": terrain_frame,
        "roads": pd.DataFrame(road_rows),
        "points": point_rows,
        "routes": routes,
        "metadata": metadata,
    }


def build_source_data() -> dict[str, Any]:
    results = _read_results()
    if len(results) != EXPECTED_ROWS or "ppo_mlp" in set(results["model"].astype(str)):
        raise RuntimeError("正式结果行数或模型集合异常")
    sources: dict[str, Any] = {
        "M01": build_m01(results),
        "M02": build_m02(results),
        "M06": build_m06(),
        "V02": build_v02(),
    }
    for figure_id in ("M01", "M02", "M06"):
        _write_csv(OUTPUT / "source_data" / f"{figure_id}_source_data.csv", sources[figure_id])
    for name, frame in sources["V02"].items():
        _write_csv(OUTPUT / "source_data" / "V02" / f"{name}.csv", frame)
    return sources


def plot_m01(frame: pd.DataFrame) -> plt.Figure:
    configure_matplotlib()
    order = [(domain, model) for domain in ("synthetic", "real") for model in MAIN_COMPARE]
    fig, ax = plt.subplots(figsize=_mm(178, 142))
    rng = np.random.default_rng(20260805)
    all_values = frame["weighted_coverage"].to_numpy()
    xmin = max(0.0, math.floor((all_values.min() - 0.04) * 20) / 20)
    xmax = min(1.0, math.ceil((all_values.max() + 0.04) * 20) / 20)
    grid = np.linspace(xmin, xmax, 300)
    ylabels: list[str] = []
    for position, (domain, model) in enumerate(order):
        values = frame[(frame["domain"].eq(domain)) & (frame["model"].eq(model))]["weighted_coverage"].to_numpy()
        color = COLORS[model]
        density = gaussian_kde(values, bw_method="scott")(grid)
        density = density / density.max() * 0.34
        ax.fill_between(grid, position, position + density, color=color, alpha=0.22, linewidth=0)
        ax.plot(grid, position + density, color=color, lw=1.0)
        jitter = rng.uniform(-0.30, -0.08, size=len(values))
        ax.scatter(values, position + jitter, s=15, marker=MARKERS[model],
                   facecolor="white", edgecolor=color, linewidth=0.75, alpha=0.92, zorder=3)
        median = float(np.median(values))
        ax.plot([median, median], [position - 0.31, position + 0.31], color=color, lw=2.0, solid_capstyle="round")
        ylabels.append(f"{DOMAINS[domain]}｜{LABELS[model]}")
    ax.axhline(5.5, color="#B5B5B5", lw=0.8)
    ax.text(xmin, 5.33, "24张地图", color="#666666", fontsize=6.8, va="bottom")
    ax.text(xmin, 11.33, "8张地图", color="#666666", fontsize=6.8, va="bottom")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.55, len(order) - 0.45)
    ax.set_yticks(range(len(order)), ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("优先级加权覆盖率")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E5E5E5", lw=0.55, zorder=0)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.12, top=0.98)
    return fig


def plot_m02_reference(frame: pd.DataFrame) -> plt.Figure:
    """生成Origin人工排版时的结构参照，不作为最终M02交付。"""
    configure_matplotlib()
    ordered = frame.copy()
    ordered["y"] = np.arange(len(ordered))[::-1]
    fig, ax = plt.subplots(figsize=_mm(178, 132))
    for row in ordered.itertuples():
        color = COLORS[row.comparator]
        ax.plot([row.ci_low_pp, row.ci_high_pp], [row.y, row.y], color=color, lw=1.25)
        ax.scatter(row.estimate_pp, row.y, s=27, marker=MARKERS[row.comparator],
                   facecolor=color, edgecolor="white", linewidth=0.55, zorder=3)
    ax.axvline(0, color="#555555", lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(ordered["y"], ordered["row_label"])
    ax.set_xlabel("PPO+Pointer − 比较算法（百分点）")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E6E6E6", lw=0.55)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.subplots_adjust(left=0.39, right=0.98, bottom=0.13, top=0.98)
    return fig


def plot_m06(frame: pd.DataFrame) -> plt.Figure:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=_mm(178, 108))
    for model in ("full", "a2c_pointer", "traditional_ppo"):
        color = COLORS[model]
        raw = frame[(frame["record_type"].eq("seed")) & (frame["model"].eq(model))]
        for _, trace in raw.groupby("seed"):
            trace = trace.sort_values("episodes_seen")
            ax.plot(trace["episodes_seen"], trace["safe_weighted_coverage"],
                    color=color, alpha=0.20, lw=0.75)
        summary = frame[(frame["record_type"].eq("summary")) & (frame["model"].eq(model))].sort_values("episodes_seen")
        x = summary["episodes_seen"].to_numpy()
        ax.fill_between(x, summary["q25"].to_numpy(), summary["q75"].to_numpy(), color=color, alpha=0.16, linewidth=0)
        ax.plot(x, summary["median"], color=color, lw=2.05, label=LABELS[model])
    ax.set_xlabel("训练回合（episode）")
    ax.set_ylabel("训练批次优先级加权覆盖率")
    ax.set_xlim(0, 3000)
    ymin = max(0.0, math.floor((frame["safe_weighted_coverage"].min() - 0.04) * 20) / 20)
    ax.set_ylim(ymin, 1.005)
    ax.grid(axis="y", color="#E4E4E4", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower right", handlelength=2.5)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.17, top=0.97)
    return fig


def render_python_prototypes(sources: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    records["M01"] = _save_figure(plot_m01(sources["M01"]), _output_stem("M01"))
    records["M06"] = _save_figure(plot_m06(sources["M06"]), _output_stem("M06"))
    reference_stem = OUTPUT / "qa" / "style_references" / "M02_origin_layout_reference"
    records["M02_reference"] = _save_figure(plot_m02_reference(sources["M02"]), reference_stem)
    return records


def _origin_safe(values: Iterable[Any]) -> list[Any]:
    output: list[Any] = []
    for value in values:
        if pd.isna(value):
            output.append("--")
        elif isinstance(value, np.generic):
            output.append(value.item())
        else:
            output.append(value)
    return output


def _lt_escape(text: str) -> str:
    return str(text).replace("\\", "/").replace('"', '\\"')


def _put_frame(app: Any, book: str, frame: pd.DataFrame) -> None:
    for index, column in enumerate(frame.columns):
        app.PutWorksheet(book, _origin_safe(frame[column].tolist()), 0, index)
        app.Execute(f'win -a {book}; wks.col{index + 1}.lname$="{_lt_escape(column)}";')


def render_origin_m02(frame: pd.DataFrame, visible: bool = True) -> dict[str, Any]:
    """使用SCATTERINTERVAL模板创建可编辑M02 Origin项目。"""
    import win32com.client

    output_project = (OUTPUT / "origin_projects" / "M02.opju").resolve()
    native_dir = (OUTPUT / "qa" / "origin_native_exports" / "M02").resolve()
    output_project.parent.mkdir(parents=True, exist_ok=True)
    native_dir.mkdir(parents=True, exist_ok=True)

    display_rows: list[dict[str, Any]] = []
    for layer, heading in (
        ("known_domain_shift", "已知域偏移（2种条件）"),
        ("hidden_model_perception_mismatch", "隐藏模型/感知误差（4种条件）"),
    ):
        display_rows.append({
            "layer": layer, "row_label": heading, "comparator": "heading",
            "estimate_pp": np.nan, "ci_low_pp": np.nan, "ci_high_pp": np.nan,
        })
        subset = frame[frame["layer"].eq(layer)]
        for row in subset.to_dict(orient="records"):
            row["row_label"] = f"{row['metric_label']}｜{row['comparator_label']}"
            display_rows.append(row)
    ordered = pd.DataFrame(display_rows).reset_index(drop=True)
    ordered["y_position"] = np.arange(1, len(ordered) + 1)[::-1]
    plot_columns: dict[str, pd.Series] = {}
    zero_y = np.array([0.5, len(ordered) + 0.5])
    plot_columns["zero_x"] = pd.Series([0.0, 0.0])
    plot_columns["zero_y"] = pd.Series(zero_y)
    for comparator in ("a2c_pointer", "traditional_ppo", "priority_resource_greedy"):
        subset = ordered[ordered["comparator"].eq(comparator)]
        interval_x: list[float] = []
        interval_y: list[float] = []
        for row in subset.itertuples():
            interval_x.extend([row.ci_low_pp, row.ci_high_pp, np.nan])
            interval_y.extend([row.y_position, row.y_position, np.nan])
        plot_columns[f"{comparator}_interval_x"] = pd.Series(interval_x)
        plot_columns[f"{comparator}_interval_y"] = pd.Series(interval_y)
        plot_columns[f"{comparator}_estimate_x"] = pd.Series(subset["estimate_pp"].to_numpy())
        plot_columns[f"{comparator}_estimate_y"] = pd.Series(subset["y_position"].to_numpy())
    plot_data = pd.DataFrame(plot_columns)
    axis_labels = ordered[["y_position", "row_label"]].sort_values("y_position")
    metadata = pd.DataFrame({
        "key": ["figure_id", "template", "source_sha256", "caption", "bootstrap_reps", "bootstrap_seed", "renderer"],
        "value": ["M02", "SCATTERINTERVAL.otp", _sha256(OUTPUT / "source_data" / "M02_source_data.csv"), CAPTIONS["M02"], BOOTSTRAP_REPS, BOOTSTRAP_SEED, "Origin 2021 COM + LabTalk逐项精修"],
    })

    app = win32com.client.Dispatch("Origin.ApplicationSI")
    app.Visible = 1 if visible else 0
    app.Execute("doc -s; doc -n;")
    source_book = app.CreatePage(2, "SourceData", "Origin", 2)
    _put_frame(app, source_book, ordered)
    plot_book = app.CreatePage(2, "PlotData", "Origin", 2)
    _put_frame(app, plot_book, plot_data)
    meta_book = app.CreatePage(2, "Metadata", "Origin", 2)
    _put_frame(app, meta_book, metadata)
    label_book = app.CreatePage(2, "AxisLabels", "Origin", 2)
    _put_frame(app, label_book, axis_labels)
    graph = app.CreatePage(3, "M02", "SCATTERINTERVAL", 2)
    if not graph:
        app.Exit()
        raise RuntimeError("Origin未能从SCATTERINTERVAL.otp创建M02图页")

    # 按比较算法绘制区间线与点；零效应线使用可编辑的Origin图形对象。
    column = 3
    series: list[tuple[str, int, int]] = []
    for comparator in ("a2c_pointer", "traditional_ppo", "priority_resource_greedy"):
        app.Execute(f"win -a {plot_book}; plotxy iy:=({column},{column + 1}) plot:=200 ogl:=[{graph}]1!;")
        interval_plot = 1 + 2 * len(series)
        app.Execute(f"win -a {plot_book}; plotxy iy:=({column + 2},{column + 3}) plot:=201 ogl:=[{graph}]1!;")
        point_plot = interval_plot + 1
        series.append((comparator, interval_plot, point_plot))
        column += 4

    rgb = {
        "a2c_pointer": (230, 134, 25),
        "traditional_ppo": (42, 157, 143),
        "priority_resource_greedy": (116, 116, 116),
    }
    symbols = {"a2c_pointer": 1, "traditional_ppo": 3, "priority_resource_greedy": 2}
    for comparator, interval_plot, point_plot in series:
        r, g, b = rgb[comparator]
        app.Execute(
            f"win -a {graph}; layer.plot={interval_plot}; set %C -cl color({r},{g},{b}); "
            "set %C -k 0; set %C -l 1; set %C -wp 0.35;"
        )
        app.Execute(
            f"win -a {graph}; layer.plot={point_plot}; set %C -cse color({r},{g},{b}); "
            f"set %C -csf color({r},{g},{b}); set %C -k {symbols[comparator]}; set %C -z 2; set %C -kh 1;"
        )

    max_abs = float(np.nanmax(np.abs(ordered[["ci_low_pp", "ci_high_pp"]].to_numpy())))
    bound = max(5.0, math.ceil((max_abs + 1.0) / 5.0) * 5.0)
    xinc = max(2.0, bound / 4.0)
    app.Execute(
        f'range axisLabels=[{label_book}]Sheet1!col(B); win -a {graph}; '
        'page.width=1050; page.height=760; page.aa=1; '
        f'layer.x.from={-bound}; layer.x.to={bound}; layer.x.inc={xinc}; '
        f'layer.y.from=0.5; layer.y.to={len(ordered) + 0.5}; layer.y.inc=1; '
        'axis -ps X A 1; axis -ps X L 1; axis -ps Y A 1; axis -ps Y L 1; '
        'axis -ps Y T axisLabels; layer.x.showlabel=1; layer.y.showlabel=1; '
        'layer.x.labelType=1; layer.x.labelSubtype=1; layer.x.label.decPlaces=1; '
        'layer.x.label.pt=1.5; layer.y.label.pt=1.3; layer.x.label.font=font(Arial); '
        'layer.x.label.color=color(0,0,0); '
        'layer.y.label.color=color(40,40,40); layer.y.label.halign=2; '
        'layer.x.thickness=0.18; layer.y.thickness=0.18; layer.x.tickthickness=0.18; '
        'layer.y.tickthickness=0.18; layer.x.ticklength=1.8; layer.y.ticklength=1.8; '
        'yl.text$=""; xb.text$=""; label -r M02XTitle; '
        'label -p 50 110 -n M02XTitle PPO+Pointer - comparison (percentage points); '
        'M02XTitle.fsize=1.4; M02XTitle.font=font(Arial); M02XTitle.color=color(20,20,20); '
        'legend.show=0; layer.left=39; layer.top=5; layer.width=58; layer.height=80; '
        'draw -n M02Zero -d 1 -w 0.22 -l -v 0; M02Zero.attach=2; '
        'doc -uw;'
    )
    saved = bool(app.Save(str(output_project)))
    export_status: dict[str, bool] = {}
    for extension in ("png", "pdf", "tif", "svg"):
        command = (
            f'expGraph type:={extension} path:="{_lt_escape(str(native_dir))}" filename:="M02" '
            'overwrite:=replace tr.Margin:=2 tr1.Unit:=2 tr1.Width:=4200 '
            'tr2.TIF.DotsPerInch:=600 tr2.TIF.Compression:=LZW;'
        )
        export_status[extension] = bool(app.Execute(command))
    app.Execute("doc -uw;")
    # 保持Origin窗口打开，便于随后进行人工式视觉核验；项目已安全落盘。
    if not visible:
        app.Execute("doc -s;")
        app.Exit()

    stem = _output_stem("M02")
    stem.parent.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for source_extension, target_extension in (("png", "png"), ("pdf", "pdf"), ("tif", "tiff"), ("svg", "svg")):
        source = native_dir / f"M02.{source_extension}"
        if source.is_file() and source.stat().st_size > 0:
            target = stem.with_suffix(f".{target_extension}")
            shutil.copy2(source, target)
            if target_extension == "png":
                # Origin已按4200 px双栏宽导出；补写600 dpi元数据，避免出版系统误读为300 dpi。
                with Image.open(target) as image:
                    image.save(target, dpi=(EXPORT_DPI, EXPORT_DPI))
            copied[target_extension] = str(target)
    return {"project": str(output_project), "saved": saved, "exports": copied, "native_status": export_status, "origin_window_left_open": visible}


def write_matlab_v02_script() -> Path:
    script = OUTPUT / "matlab_sources" / "render_V02.m"
    script.parent.mkdir(parents=True, exist_ok=True)
    content = r'''function render_V02(outputRoot)
% 固定真实DSM任务路线图。关键尺寸、颜色和视角均在本文件顶部集中设置。
if nargin < 1
    outputRoot = fileparts(fileparts(mfilename('fullpath')));
end
dataRoot = fullfile(outputRoot, 'source_data', 'V02');
outRoot = fullfile(outputRoot, 'showcase');
if ~exist(outRoot, 'dir'), mkdir(outRoot); end

terrainT = readtable(fullfile(dataRoot, 'terrain.csv'));
roads = readtable(fullfile(dataRoot, 'roads.csv'));
points = readtable(fullfile(dataRoot, 'points.csv'));
routes = readtable(fullfile(dataRoot, 'routes.csv'));

nx = max(terrainT.x) + 1; ny = max(terrainT.y) + 1;
Z = reshape(terrainT.z, [nx, ny])';
nodes = points(strcmp(points.point_type, 'inspection'), :);
airport = points(strcmp(points.point_type, 'airport'), :);
margin = 20;
xlimTask = [max(0, floor(min(nodes.x)-margin)), min(nx-1, ceil(max(nodes.x)+margin))];
ylimTask = [max(0, floor(min(nodes.y)-margin)), min(ny-1, ceil(max(nodes.y)+margin))];
[X, Y] = meshgrid(0:nx-1, 0:ny-1);
maskX = X(1,:) >= xlimTask(1) & X(1,:) <= xlimTask(2);
maskY = Y(:,1) >= ylimTask(1) & Y(:,1) <= ylimTask(2);

fig = figure('Color','w','Units','centimeters','Position',[2 2 17.8 15.0], 'Renderer','opengl');
ax = axes(fig); hold(ax,'on');
surf(ax, X(maskY,maskX), Y(maskY,maskX), Z(maskY,maskX), Z(maskY,maskX), ...
    'EdgeColor','none','FaceAlpha',0.96);
colormap(ax, parula(256));
contour3(ax, X(maskY,maskX), Y(maskY,maskX), Z(maskY,maskX)+1.5, 12, ...
    'LineColor',[0.42 0.42 0.42],'LineWidth',0.45);

roadIds = unique(roads.road_id)';
for rid = roadIds
    r = roads(roads.road_id==rid,:);
    keep = r.x>=xlimTask(1) & r.x<=xlimTask(2) & r.y>=ylimTask(1) & r.y<=ylimTask(2);
    r = r(keep,:);
    if height(r)>1
        plot3(ax,r.x,r.y,r.z+3.0,'Color',[0.25 0.25 0.25],'LineWidth',1.25);
    end
end

priorityColors = [0.30 0.55 0.85; 0.93 0.66 0.20; 0.78 0.20 0.18];
prioritySizes = [28 40 56];
priorityHandles = gobjects(3,1);
for p = 1:3
    q = nodes(nodes.priority==p,:);
    priorityHandles(p) = scatter3(ax,q.x,q.y,q.z+7,prioritySizes(p),priorityColors(p,:), ...
        'filled','MarkerEdgeColor','w','LineWidth',0.75);
end
airportHandle = scatter3(ax,airport.x,airport.y,airport.z+10,110,'p','filled', ...
    'MarkerFaceColor',[0.05 0.05 0.05],'MarkerEdgeColor','w','LineWidth',1.0);

models = {'full','a2c_pointer','traditional_ppo','milp'};
modelLabels = {'PPO+Pointer','A2C+Pointer','传统PPO','MILP'};
modelColors = [35 105 189; 230 134 25; 42 157 143; 34 34 34]/255;
lineStyles = {'-','--','-.',':'};
routeHandles = gobjects(4,1);
for i = 1:numel(models)
    r = routes(strcmp(routes.model,models{i}) & strcmp(routes.status,'route'),:);
    r = sortrows(r,'sequence');
    routeHandles(i) = plot3(ax,r.x,r.y,r.z+12, 'Color',modelColors(i,:), ...
        'LineStyle',lineStyles{i},'LineWidth',2.15);
end

axis(ax,'tight'); axis(ax,'vis3d');
xlim(ax,xlimTask); ylim(ax,ylimTask);
view(ax,-38,36); camproj(ax,'perspective'); camzoom(ax,0.80);
grid(ax,'off'); box(ax,'on');
xlabel(ax,'Local Easting (30 m/grid)','FontName','Arial');
ylabel(ax,'Local Northing (30 m/grid)','FontName','Arial');
zlabel(ax,'Elevation (m)','FontName','Arial');
set(ax,'FontName','Arial','FontSize',8,'LineWidth',0.8,'TickDir','out');
cb = colorbar(ax); cb.Label.String = 'Terrain elevation (m)'; cb.Label.FontName = 'Arial';
lgd = legend([routeHandles; airportHandle; priorityHandles], ...
    [modelLabels, {'机场','低优先级','中优先级','高优先级'}], ...
    'Location','southoutside','NumColumns',4,'Box','off','FontName','Microsoft YaHei','FontSize',7);
% 为三维坐标框、色标和图例分配互不重叠的固定区域。
set(ax,'Position',[0.055 0.290 0.705 0.600]);
set(cb,'Position',[0.845 0.305 0.026 0.565]);
set(lgd,'Position',[0.115 0.018 0.710 0.105]);

stem = fullfile(outRoot,'V02_固定真实DSM地形路线');
% 固定页面尺寸，避免exportgraphics自动紧裁切导致三维坐标框贴边。
set(fig,'PaperUnits','centimeters','PaperPosition',[0 0 17.8 15.0], ...
    'PaperSize',[17.8 15.0],'PaperPositionMode','manual','InvertHardcopy','off');
print(fig,[stem '.pdf'],'-dpdf','-painters');
print(fig,[stem '.png'],'-dpng','-r600');
print(fig,[stem '.tiff'],'-dtiff','-r600');
print(fig,[stem '.svg'],'-dsvg','-painters');
savefig(fig,[stem '.fig']);
close(fig);
end
'''
    script.write_text(content, encoding="utf-8")
    return script


def render_matlab_v02() -> dict[str, Any]:
    script = write_matlab_v02_script()
    matlab = Path(r"D:\Matlab 2020A\bin\matlab.exe")
    command = (
        f"addpath('{_lt_escape(str(script.parent))}'); "
        f"render_V02('{_lt_escape(str(OUTPUT.resolve()))}');"
    )
    completed = subprocess.run(
        [str(matlab), "-batch", command], cwd=ROOT, text=True,
        capture_output=True, encoding="utf-8", errors="replace", timeout=180,
    )
    (OUTPUT / "qa" / "matlab_V02.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (OUTPUT / "qa" / "matlab_V02.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"MATLAB V02失败: {completed.stderr[-1200:]}")
    stem = _output_stem("V02")
    outputs = {extension: str(stem.with_suffix(f".{extension}")) for extension in ("pdf", "svg", "png", "tiff", "fig")}
    return {"script": str(script), "outputs": outputs}


def write_registry_and_literature() -> None:
    literature_summary = audit_summary()
    registry = {
        "schema_version": 2,
        "output_root": str(OUTPUT),
        "prototype_gate": list(PROTOTYPE_IDS),
        "gate_state": "completed",
        "prototype_review": {
            "approved": ["M01", "M02", "M06"],
            "approved_after_revision": ["V02"],
            "v02_revision": "修复上方坐标轴裁切和图例/标注重叠",
        },
        "figures": FIGURES,
        "backend_counts": {
            "origin": sum(meta["backend"] == "origin" for meta in FIGURES.values()),
            "python": sum(meta["backend"] == "python" for meta in FIGURES.values()),
            "matlab": sum(meta["backend"] == "matlab" for meta in FIGURES.values()),
        },
        "literature_audit": literature_summary,
        "origin_template_revisions": {
            "M03": "BeforeAfter.otpu -> SCATTER.OTP；避免模板默认连接和分组冲突",
            "M07": "SCATTERINTERVAL.otp -> SCATTER.OTP；点区间由绘图数据显式构造",
            "M08": "SCATTERINTERVAL.otp -> SCATTER.OTP；地图域分组和区间由显式图层控制",
            "M10": "SCATTERINTERVAL.otp -> SCATTER.OTP；消融效应与零线人工排版",
            "S02": "bubble.otpu -> SCATTER.OTP；全部安全率为100%，取消无信息量气泡大小",
            "S03": "SCATTERINTERVAL.otp -> SCATTER.OTP；证书区间、零regret点和标签单独控制",
            "S07": "ColorDots.otpu -> SCATTER.OTP；避免默认彩虹色和黑底，改为白底点阵",
        },
    }
    _write_json(OUTPUT / "manifests" / "figure_registry_manual_v3.json", registry)
    literature = pd.DataFrame(LITERATURE_AUDIT)
    _write_csv(OUTPUT / "literature_audit" / "literature_style_audit.csv", literature)


def write_captions() -> None:
    for figure_id, caption in CAPTIONS.items():
        path = OUTPUT / "captions_CN" / f"{figure_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {figure_id} {FIGURES[figure_id]['name']}\n\n{caption}\n", encoding="utf-8")


def write_manifests(render_records: Mapping[str, Any]) -> None:
    for figure_id in PROTOTYPE_IDS:
        source_paths = (
            list((OUTPUT / "source_data" / "V02").glob("*.csv"))
            if figure_id == "V02" else [OUTPUT / "source_data" / f"{figure_id}_source_data.csv"]
        )
        outputs = [path for path in _output_stem(figure_id).parent.glob(f"{_output_stem(figure_id).name}.*") if path.is_file()]
        manifest = {
            "figure_id": figure_id,
            "name": FIGURES[figure_id]["name"],
            "backend": FIGURES[figure_id]["backend"],
            "template": FIGURES[figure_id]["template"],
            "caption": CAPTIONS[figure_id],
            "source_data": [{"path": str(p.relative_to(OUTPUT)), "sha256": _sha256(p), "bytes": p.stat().st_size} for p in source_paths],
            "outputs": [{"path": str(p.relative_to(OUTPUT)), "sha256": _sha256(p), "bytes": p.stat().st_size} for p in sorted(outputs)],
            "render_record": render_records.get(figure_id),
        }
        _write_json(OUTPUT / "manifests" / f"{figure_id}.json", manifest)


def _trimmed_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    return ImageChops.difference(rgb, background).getbbox()


def build_thumbnail_index() -> Path:
    previews: list[tuple[str, Image.Image]] = []
    for figure_id in PROTOTYPE_IDS:
        png = _output_stem(figure_id).with_suffix(".png")
        if png.is_file():
            image = Image.open(png).convert("RGB")
            image.thumbnail((900, 620), Image.Resampling.LANCZOS)
            previews.append((figure_id, image.copy()))
    canvas = Image.new("RGB", (1100, len(previews) * 700 + 60), "#F4F4F4")
    draw = ImageDraw.Draw(canvas)
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    title_font = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else ImageFont.load_default()
    y = 30
    for figure_id, image in previews:
        draw.text((30, y), f"{figure_id}  {FIGURES[figure_id]['name']}", fill="#222222", font=title_font)
        canvas.paste(image, (170, y + 25))
        y += 700
    target = OUTPUT / "thumbnail_index" / "prototype_contact_sheet.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, dpi=(150, 150))
    return target


def qa_all(start_audit: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    figures: dict[str, Any] = {}
    for figure_id in PROTOTYPE_IDS:
        stem = _output_stem(figure_id)
        item: dict[str, Any] = {}
        required = ("pdf", "png", "tiff")
        for extension in required:
            path = stem.with_suffix(f".{extension}")
            item[extension] = path.is_file() and path.stat().st_size > 0
            if not item[extension]:
                errors.append(f"{figure_id}缺少{extension}")
        svg = stem.with_suffix(".svg")
        item["svg"] = svg.is_file() and svg.stat().st_size > 0
        if not item["svg"]:
            warnings.append(f"{figure_id}未提供通过校验的SVG")
        png = stem.with_suffix(".png")
        if png.is_file():
            with Image.open(png) as image:
                item["pixel_size"] = list(image.size)
                item["dpi"] = list(image.info.get("dpi", (0, 0)))
                item["nonwhite_bbox"] = _trimmed_bbox(image)
                if image.width < 3000:
                    errors.append(f"{figure_id} PNG宽度不足600dpi双栏交付")
        figures[figure_id] = item

    if not (OUTPUT / "origin_projects" / "M02.opju").is_file():
        errors.append("M02缺少可编辑OPJU")
    if not _output_stem("V02").with_suffix(".fig").is_file():
        errors.append("V02缺少可编辑FIG")

    final_hashes = {
        "paper_final": _tree_digest(OLD_PAPER_FINAL)[0],
        "paper_redraw_origin_v2": _tree_digest(OLD_ORIGIN_V2)[0],
    }
    start_hashes = {name: data["sha256"] for name, data in start_audit["old_directories"].items()}
    old_unchanged = final_hashes == start_hashes
    if not old_unchanged:
        errors.append("旧制图目录发生变化")
    report = {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "prototype_gate_state": "completed" if not errors else "qa_failed",
        "figures": figures,
        "old_directories_unchanged": old_unchanged,
        "old_directories_at_end": final_hashes,
    }
    _write_json(OUTPUT / "qa" / "prototype_qa_report.json", report)
    return report


def run_prototype_gate(origin_visible: bool = True) -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    start_audit = audit_inputs(write_snapshot=True)
    write_registry_and_literature()
    write_captions()
    sources = build_source_data()
    records = render_python_prototypes(sources)
    records["M02"] = render_origin_m02(sources["M02"], visible=origin_visible)
    records["V02"] = render_matlab_v02()
    write_manifests(records)
    contact_sheet = build_thumbnail_index()
    report = qa_all(start_audit)
    _write_json(OUTPUT / "manifests" / "prototype_gate_status.json", {
        "state": report["prototype_gate_state"],
        "prototype_ids": list(PROTOTYPE_IDS),
        "contact_sheet": str(contact_sheet),
        "qa_report": str(OUTPUT / "qa" / "prototype_qa_report.json"),
        "remaining_figures_rendered": False,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit-inputs")
    subparsers.add_parser("build-source-data")
    render = subparsers.add_parser("render-prototypes")
    render.add_argument("--origin-hidden", action="store_true")
    subparsers.add_parser("qa")
    args = parser.parse_args()

    if args.command == "audit-inputs":
        print(json.dumps(audit_inputs(write_snapshot=True), ensure_ascii=False, indent=2))
    elif args.command == "build-source-data":
        audit_inputs(write_snapshot=True)
        write_registry_and_literature()
        write_captions()
        sources = build_source_data()
        print(json.dumps({key: (len(value) if isinstance(value, pd.DataFrame) else list(value)) for key, value in sources.items()}, ensure_ascii=False, indent=2))
    elif args.command == "render-prototypes":
        print(json.dumps(run_prototype_gate(origin_visible=not args.origin_hidden), ensure_ascii=False, indent=2))
    elif args.command == "qa":
        start = json.loads((OUTPUT / "qa" / "old_directories_at_start.json").read_text(encoding="utf-8"))
        print(json.dumps(qa_all(start), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

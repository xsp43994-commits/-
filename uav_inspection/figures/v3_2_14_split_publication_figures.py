"""v3.2.14 第二次实验的独立小图、路线局部视窗与 DSM 地形图流水线。

该脚本只读取已冻结的21,648条正式结果和路线，调用经审计的原始绘图函数，
然后把6张正文图和8张补充图拆分成可独立插入论文的小图。不修改模型、任务、
评价器、统计协议或冻结结果。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import proj3d
from PIL import Image, ImageChops, ImageDraw

from uav_inspection.figures import v3_2_14_publication_figures as base


ROOT = WORKSPACE_ROOT
# 清理后只保留这一套正式论文图，避免再次生成并行的旧版目录。
DEFAULT_OUTPUT = base.RUN / "figures" / "paper_final"

# 可调参数集中在此：路线视窗只根据冻结任务几何扩展，不根据算法成绩裁剪。
LOCAL_VIEW_MARGIN = 0.12
LOCAL_VIEW_MARGIN = 0.12
LOCAL_VIEW_MIN_PAD = 5.0
SPLIT_PAD_INCH = 0.025
ROUTE_CELL_METERS = 30
EXPORT_DPI = base.EXPORT_DPI
THUMB_DPI = 170

# V1专用视觉参数：只控制展示图，不影响任何实验、统计或其余图件。
V1_HILLSHADE_AZDEG = 315
V1_HILLSHADE_AZDEG = 315
V1_HILLSHADE_ALTDEG = 42
V1_SCALE_BAR_METERS = 500
V1_VERTICAL_EXAGGERATION = 1.5
V1_VIEW_MARGIN = 0.10
V1_MASTER_WIDTH_MM = 183.0
V1_MASTER_HEIGHT_MM = 145.0
V1_COMPACT_WIDTH_MM = 89.0
V1_COMPACT_HEIGHT_MM = 71.0
V1_CAMERA_CANDIDATES = ((32.0, -58.0), (36.0, -72.0), (30.0, 122.0), (36.0, 138.0))
V1_PLANFORM_LONG_TO_SHORT_RATIO = 1.30
V1_ROUTE_CENTER_TOLERANCE = 0.20
V1_PAD_INCH = 1.2 / 25.4
V1_PANEL_STEM = "figV01_3d_taihang_route"

# 由V1构建器写入，随后并入最终manifest；不承载实验数据，也不影响其它图。
V1_RENDER_METADATA: Dict[str, Any] = {}


@dataclass(frozen=True)
class PanelSpec:
    """一张独立小图的父图、面板、坐标轴与源数据映射。"""

    panel: str
    axes: tuple[int, ...]
    source_key: str
    title: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


PANEL_SPECS: Dict[str, tuple[PanelSpec, ...]] = {
    "fig01_study_design": (
        PanelSpec("a", (0,), "a", "任务场景与安全返航约束"),
        PanelSpec("b", (1,), "b", "冻结地图资产概览"),
        PanelSpec("c", (2,), "c", "正式评价矩阵"),
        PanelSpec("d", (3,), "d", "五条互补证据链"),
    ),
    "fig02_integrated_score": (
        PanelSpec("a", (0, 5), "a", "七个效应维度"),
        PanelSpec("b", (1,), "b", "100分制算术综合得分"),
        PanelSpec("c", (2,), "c", "PPO+Pointer相对A2C+Pointer的层级bootstrap"),
        PanelSpec("d", (3, 6), "d", "归一化下限×权重联合敏感性"),
        PanelSpec("e", (4,), "e", "PPO相对A2C的维度贡献"),
    ),
    "fig03_operational_tradeoffs": (
        PanelSpec("a", (0,), "a", "合成与真实DSM的安全加权覆盖率"),
        PanelSpec("b", (1,), "b", "覆盖、安全和返航的地图级效应"),
        PanelSpec("c", (2, 5, 6, 7), "c", "安全路线的资源与总任务时间"),
        PanelSpec("d", (3,), "d", "在线规划时间ECDF"),
        PanelSpec("e", (4,), "e", "安全加权覆盖率—在线规划时间Pareto视图"),
    ),
    "fig04_training": tuple(
        PanelSpec(panel, (idx,), panel, title)
        for idx, (panel, title) in enumerate(
            (
                ("a", "五种子收敛过程"),
                ("b", "学习曲线AUC"),
                ("c", "阈值样本效率"),
                ("d", "尾段与跨种子稳定性"),
                ("e", "PPO更新诊断"),
            )
        )
    ),
    "fig05_ablation": tuple(
        PanelSpec(panel, (idx,), panel, title)
        for idx, (panel, title) in enumerate(
            (
                ("a", "四个消融的地图级总体效应"),
                ("b", "显式优先级偏置"),
                ("c", "资源塑形与瓶颈类型"),
                ("d", "域随机化与扰动退化"),
                ("e", "返航储备的仿真安全效应"),
            )
        )
    ),
    "fig06_generalization_robustness_routes": (
        PanelSpec("a", (0,), "a", "24张未见合成地图的程序化泛化"),
        PanelSpec("b", (1,), "b", "8张真实DSM的零样本仿真迁移"),
        PanelSpec("c", (2,), "c", "风、功率、DEM与定位误差的退化热力图"),
        PanelSpec("d", (3,), "d", "跨扰动保持率与地图一致性"),
        PanelSpec("e1", (4,), "e", "未见合成任务：PPO+Pointer"),
        PanelSpec("e2", (5,), "e", "未见合成任务：A2C+Pointer"),
        PanelSpec("e3", (6,), "e", "未见合成任务：传统PPO"),
        PanelSpec("e4", (7,), "e", "未见合成任务：MILP"),
        PanelSpec("f1", (8,), "e", "真实DSM任务：PPO+Pointer"),
        PanelSpec("f2", (9,), "e", "真实DSM任务：A2C+Pointer"),
        PanelSpec("f3", (10,), "e", "真实DSM任务：传统PPO"),
        PanelSpec("f4", (11,), "e", "真实DSM任务：MILP"),
    ),
    "figS01_audit": tuple(
        PanelSpec(panel, (idx,), panel, title)
        for idx, (panel, title) in enumerate(
            (("a", "评价家族完整性"), ("b", "算法×评价家族行数"), ("c", "嵌套结构与独立单位"), ("d", "冻结哈希与审计状态"))
        )
    ),
    "figS02_scenarios": (
        PanelSpec("a", (0,), "a", "节点规模（训练范围内）"),
        PanelSpec("b", (1, 4), "b", "认证难度"),
        PanelSpec("c", (2,), "c", "约束类型"),
        PanelSpec("d", (3, 5), "d", "优先级布局"),
    ),
    "figS03_baselines": tuple(
        PanelSpec(panel, (idx,), panel, title)
        for idx, (panel, title) in enumerate(
            (("a", "传统基线任务效果"), ("b", "参考解差距与区间"), ("c", "传统规划器计算代价"), ("d", "MILP求解状态与gap"))
        )
    ),
    "figS04_training_all": (
        PanelSpec("a", (0,), "a", "七个学习模型的共同定义训练指标"),
        PanelSpec("b", (1,), "b", "七个学习模型的返航与安全过程"),
    ),
    "figS05_score_sensitivity": tuple(
        PanelSpec(panel, (idx,), panel, title)
        for idx, (panel, title) in enumerate(
            (("a", "算术聚合敏感性"), ("b", "几何聚合诊断"), ("c", "全权重网格分差分布"), ("d", "D4/D6/D7配对证据"))
        )
    ),
    "figS06_ablation_maps": (
        PanelSpec("a", (0,), "a", "合成地图的消融方向一致性"),
        PanelSpec("b", (1, 4), "b", "真实DSM的消融方向一致性"),
        PanelSpec("c", (2, 5), "c", "终止原因全集"),
        PanelSpec("d", (3, 6), "d", "首次失败约束"),
    ),
    "figS07_robustness_failures": (
        PanelSpec("a", (0, 4), "a", "扰动下的安全率"),
        PanelSpec("b", (1, 5), "b", "扰动下的返航率"),
        PanelSpec("c", (2, 6), "c", "扰动下的违规率"),
        PanelSpec("d", (3, 7), "d", "扰动下的Stranded率"),
    ),
    "figS08_route_atlas": tuple(
        PanelSpec(chr(ord("a") + idx), (idx,), "a-h", f"真实DSM路线图集 {idx + 1}") for idx in range(8)
    ),
    "figV01_3d_route": (PanelSpec("a", (0, 1, 2), "a", "DSM地形路线与航迹高程剖面"),),
    "figV02_outcome_flow": (PanelSpec("a", (0,), "a", "算法→覆盖→终止结果流"),),
}


# V1改为单幅三维场景：一个主坐标轴加一个嵌入式高程色标。
PANEL_SPECS["figV01_3d_route"] = (
    PanelSpec("a", (0, 1), "a", "太行山DSM、公路巡检点与三维安全返航航迹"),
)


def _task_view_bounds(task: Mapping[str, Any], terrain_shape: Sequence[int]) -> tuple[float, float, float, float]:
    """生成任务级共同视窗：同一任务的所有算法完全共享，不使用成绩信息。"""
    points = np.asarray(task["inspection_points_xyz"], dtype=float)[:, :2]
    start = np.asarray(task["start_xy"], dtype=float).reshape(1, 2)
    coords = np.vstack([points, start])
    span = np.maximum(np.ptp(coords, axis=0), 1.0)
    pad = np.maximum(span * LOCAL_VIEW_MARGIN, LOCAL_VIEW_MIN_PAD)
    low = coords.min(axis=0) - pad
    high = coords.max(axis=0) + pad
    width = float(terrain_shape[1] - 1)
    height = float(terrain_shape[0] - 1)
    return max(0.0, low[0]), min(width, high[0]), max(0.0, low[1]), min(height, high[1])


def _sample_terrain(terrain: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xi = np.clip(np.rint(xy[:, 0]).astype(int), 0, terrain.shape[1] - 1)
    yi = np.clip(np.rint(xy[:, 1]).astype(int), 0, terrain.shape[0] - 1)
    return terrain[yi, xi]


def _plot_local_route(
    ax: plt.Axes,
    task: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
    model: str,
    show_title: bool = True,
) -> Dict[str, Any]:
    """在任务走廊视窗中绘制路线，指标移到源数据/图注，不再覆盖地图。"""
    map_id = str(task["map_id"])
    with np.load(base._map_bundle_path(map_id), allow_pickle=False) as data:
        terrain = np.asarray(data["terrain"], dtype=float)
    xmin, xmax, ymin, ymax = _task_view_bounds(task, terrain.shape)
    rotate_corridor = (ymax - ymin) > 1.25 * (xmax - xmin)

    def display_xy(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if not rotate_corridor:
            return values.copy()
        transformed = values.copy()
        transformed[..., 0] = values[..., 1]
        transformed[..., 1] = (terrain.shape[1] - 1) - values[..., 0]
        return transformed

    if rotate_corridor:
        display_terrain = np.rot90(terrain, k=1)
        display_extent = (0, display_terrain.shape[1] - 1, 0, display_terrain.shape[0] - 1)
        view_bounds = (ymin, ymax, (terrain.shape[1] - 1) - xmax, (terrain.shape[1] - 1) - xmin)
    else:
        display_terrain = terrain
        display_extent = (0, terrain.shape[1] - 1, 0, terrain.shape[0] - 1)
        view_bounds = (xmin, xmax, ymin, ymax)
    ax.imshow(
        display_terrain,
        origin="lower",
        cmap="gist_earth",
        alpha=0.94,
        extent=display_extent,
        interpolation="bilinear",
    )
    for segment in base._road_segments(map_id):
        shown = display_xy(segment[:, :2])
        ax.plot(shown[:, 0], shown[:, 1], color="white", lw=1.45, alpha=0.92, zorder=2)
        ax.plot(shown[:, 0], shown[:, 1], color="#505050", lw=0.55, alpha=0.98, zorder=3)

    points = np.asarray(task["inspection_points_xyz"], dtype=float)
    shown_points = display_xy(points[:, :2])
    priorities = np.asarray(task["priorities"], dtype=int)
    for priority, color, size in ((1, "#B5C2CE", 13), (2, "#E3AB56", 20), (3, "#C84C4C", 29)):
        sub = shown_points[priorities == priority]
        if len(sub):
            ax.scatter(sub[:, 0], sub[:, 1], s=size, color=color, edgecolor="white", lw=0.45, zorder=5)
    start = np.asarray(task["start_xy"], dtype=float)
    shown_start = display_xy(start.reshape(1, 2))[0]
    ax.scatter(shown_start[0], shown_start[1], marker="P", s=48, color="#0F4D92", edgecolor="white", lw=0.55, zorder=7)

    row: Dict[str, Any] = {
        "task_id": str(task["id"]), "map_id": map_id, "model": model,
        "route_found": payload is not None, "display_rotation_deg": 90 if rotate_corridor else 0,
    }
    if payload is not None:
        detail = payload.get("detail", payload)
        path = np.asarray(detail.get("path", []), dtype=float)
        if len(path):
            shown_path = display_xy(path[:, :2])
            ax.plot(shown_path[:, 0], shown_path[:, 1], color="white", lw=2.7, alpha=0.88, zorder=5)
            ax.plot(shown_path[:, 0], shown_path[:, 1], color=base.color_for(model), lw=1.45, zorder=6)
        metrics = detail.get("metrics", {}) or {}
        row.update(
            {
                "distance_m": float(detail.get("distance_m", metrics.get("distance_m", math.nan))),
                "energy_wh": float(detail.get("energy_wh", metrics.get("energy_wh", math.nan))),
                "time_s": float(detail.get("time_s", metrics.get("time_s", math.nan))),
                "weighted_coverage": float(metrics.get("weighted_coverage", math.nan)),
                "returned": bool(metrics.get("returned", False)),
                "termination_reason": str(metrics.get("termination_reason", detail.get("termination_reason", "unknown"))),
            }
        )
    else:
        ax.text(0.5, 0.5, "路线缺失", transform=ax.transAxes, ha="center", va="center", color=base.DELTA_DOWN, weight="bold")

    ax.set_xlim(view_bounds[0], view_bounds[1])
    ax.set_ylim(view_bounds[2], view_bounds[3])
    ax.set_aspect("equal", adjustable="box")
    # 比例信息放到数据区外，避免与路线或巡检点重叠。
    rotation_note = "·旋辐90°" if rotate_corridor else ""
    ax.set_xlabel(f"局部走廊{rotation_note}（1格={ROUTE_CELL_METERS} m）", labelpad=2)
    if show_title:
        ax.set_title(base.label_for(model), color=base.color_for(model), fontsize=6.4, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return row


def _contiguous_chunks(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    return [chunk for chunk in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1) if len(chunk) >= 2]


def _figure_v01_rebuilt(bundle: base.DataBundle, output_dir: Path) -> Dict[str, Any]:
    """重构V1：用正射阴影地形图和航迹剖面替代易遮挡的三维坐标盒。"""
    task = base._task_by_id(bundle, base.REAL_EXAMPLE)
    payload = base._route_payload("full", base.REAL_EXAMPLE, 42)
    if payload is None:
        raise RuntimeError("V1展示图的冻结代表路线缺失。")
    map_id = str(task["map_id"])
    with np.load(base._map_bundle_path(map_id), allow_pickle=False) as data:
        terrain = np.asarray(data["terrain"], dtype=float)

    xmin, xmax, ymin, ymax = _task_view_bounds(task, terrain.shape)
    x0, x1 = max(0, int(math.floor(xmin))), min(terrain.shape[1] - 1, int(math.ceil(xmax)))
    y0, y1 = max(0, int(math.floor(ymin))), min(terrain.shape[0] - 1, int(math.ceil(ymax)))
    terrain_crop = terrain[y0 : y1 + 1, x0 : x1 + 1]
    display_terrain = np.rot90(terrain_crop, k=1)
    display_extent = (
        0.0,
        float((y1 - y0) * ROUTE_CELL_METERS),
        0.0,
        float((x1 - x0) * ROUTE_CELL_METERS),
    )

    # 代表任务走廊为南北向。统一逆时针旋转90°以适应双栏横图；北向箭头明确记录方向。
    def display_xy(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        transformed = np.empty_like(values, dtype=float)
        transformed[..., 0] = (values[..., 1] - y0) * ROUTE_CELL_METERS
        transformed[..., 1] = (x1 - values[..., 0]) * ROUTE_CELL_METERS
        return transformed

    fig = plt.figure(figsize=(183 * base.MM, 92 * base.MM), facecolor="white")
    map_ax = fig.add_axes([0.012, 0.35, 0.976, 0.64])
    profile_ax = fig.add_axes([0.055, 0.055, 0.925, 0.205])

    # 使用低饱和度高程着色叠加物理hillshade；避免彩虹色和夸张的三维透视。
    finite = display_terrain[np.isfinite(display_terrain)]
    vmin, vmax = np.nanpercentile(finite, [2, 98])
    elevation_cmap = LinearSegmentedColormap.from_list(
        "v1_hypsometric",
        ["#DCE7DC", "#B9CAA8", "#D4C49E", "#B5A188", "#EEECE7"],
    )
    elevation_norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
    light = LightSource(azdeg=V1_HILLSHADE_AZDEG, altdeg=V1_HILLSHADE_ALTDEG)
    hillshade = light.hillshade(
        display_terrain,
        vert_exag=1.0,
        dx=ROUTE_CELL_METERS,
        dy=ROUTE_CELL_METERS,
        fraction=1.05,
    )
    tint = elevation_cmap(elevation_norm(display_terrain))[..., :3]
    relief = np.clip(tint * (0.68 + 0.48 * hillshade[..., None]), 0.0, 1.0)
    map_ax.imshow(relief, origin="lower", extent=display_extent, interpolation="bilinear", zorder=0)
    contour_levels = np.linspace(float(vmin), float(vmax), V1_CONTOUR_LEVELS)
    map_ax.contour(
        np.linspace(display_extent[0], display_extent[1], display_terrain.shape[1]),
        np.linspace(display_extent[2], display_extent[3], display_terrain.shape[0]),
        display_terrain,
        levels=contour_levels,
        colors="#596257",
        linewidths=0.32,
        alpha=0.34,
        zorder=1,
    )

    source_rows: list[Dict[str, Any]] = []
    for road_index, segment in enumerate(base._road_segments(map_id), start=1):
        mask = (segment[:, 0] >= xmin) & (segment[:, 0] <= xmax) & (segment[:, 1] >= ymin) & (segment[:, 1] <= ymax)
        chunks = _contiguous_chunks(mask)
        for chunk in chunks:
            local = segment[chunk]
            shown_local = display_xy(local[:, :2])
            road_z = _sample_terrain(terrain, local[:, :2])
            map_ax.plot(shown_local[:, 0], shown_local[:, 1], color="white", lw=4.0, alpha=0.96, solid_capstyle="round", zorder=3)
            map_ax.plot(shown_local[:, 0], shown_local[:, 1], color="#3D4140", lw=1.55, alpha=0.98, solid_capstyle="round", zorder=4)
            for order, (xy, shown_xy, z) in enumerate(zip(local[:, :2], shown_local, road_z)):
                source_rows.append({"element": f"road_{road_index}", "sequence": order, "x": xy[0], "y": xy[1], "z": z, "display_x_m": shown_xy[0], "display_y_m": shown_xy[1], "priority": math.nan})

    points = np.asarray(task["inspection_points_xyz"], dtype=float)[:, :2]
    shown_points = display_xy(points)
    priorities = np.asarray(task["priorities"], dtype=int)
    point_z = _sample_terrain(terrain, points)
    detail = payload.get("detail", payload)
    metrics = detail.get("metrics", {}) or {}
    visited = {int(index) for index in metrics.get("visited_order", [])}
    priority_colors = {1: "#91A4B6", 2: "#D89A3D", 3: "#C53E3E"}
    priority_sizes = {1: 18, 2: 27, 3: 39}
    for priority in (1, 2, 3):
        indices = np.flatnonzero(priorities == priority)
        unvisited_indices = [idx for idx in indices if int(idx) not in visited]
        visited_indices = [idx for idx in indices if int(idx) in visited]
        if unvisited_indices:
            map_ax.scatter(
                shown_points[unvisited_indices, 0], shown_points[unvisited_indices, 1],
                s=priority_sizes[priority], c=priority_colors[priority],
                edgecolor="white", linewidth=0.55, zorder=7,
            )
        if visited_indices:
            map_ax.scatter(
                shown_points[visited_indices, 0], shown_points[visited_indices, 1],
                s=priority_sizes[priority] + 20, c=priority_colors[priority],
                edgecolor=base.color_for("full"), linewidth=1.25, zorder=8,
            )
    for order, (xy, shown_xy, z, priority) in enumerate(zip(points, shown_points, point_z, priorities)):
        source_rows.append(
            {
                "element": "inspection_point", "sequence": order,
                "x": xy[0], "y": xy[1], "z": z,
                "display_x_m": shown_xy[0], "display_y_m": shown_xy[1],
                "priority": int(priority), "visited": int(order in visited),
            }
        )

    route = np.asarray(detail["flight_path"], dtype=float)
    shown_route = display_xy(route[:, :2])
    map_ax.plot(shown_route[:, 0], shown_route[:, 1], color="white", lw=5.0, alpha=0.95, solid_capstyle="round", solid_joinstyle="round", zorder=9)
    map_ax.plot(shown_route[:, 0], shown_route[:, 1], color=base.color_for("full"), lw=2.35, solid_capstyle="round", solid_joinstyle="round", zorder=10)
    for order, (xyz, shown_xy) in enumerate(zip(route, shown_route)):
        source_rows.append({"element": "flight_path", "sequence": order, "x": xyz[0], "y": xyz[1], "z": xyz[2], "display_x_m": shown_xy[0], "display_y_m": shown_xy[1], "priority": math.nan})

    start = np.asarray(task["start_xy"], dtype=float)
    shown_start = display_xy(start.reshape(1, 2))[0]
    start_z = float(_sample_terrain(terrain, start.reshape(1, 2))[0])
    map_ax.scatter(shown_start[0], shown_start[1], marker="s", s=64, facecolor="white", edgecolor="#153E75", lw=1.25, zorder=12)
    map_ax.text(shown_start[0], shown_start[1], "H", color="#153E75", fontsize=5.8, weight="bold", ha="center", va="center", zorder=13)
    map_ax.annotate(
        "机场／返航点",
        xy=(shown_start[0], shown_start[1]), xytext=(-7, 9), textcoords="offset points",
        fontsize=5.6, color="#153E75", weight="bold", ha="right", va="bottom",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        zorder=13,
    )
    source_rows.append({"element": "airport", "sequence": 0, "x": start[0], "y": start[1], "z": start_z, "display_x_m": shown_start[0], "display_y_m": shown_start[1], "priority": math.nan})

    segments = detail.get("segments", []) or []
    winds = np.asarray([seg.get("mean_wind_mps", [math.nan, math.nan, math.nan]) for seg in segments], dtype=float)
    if winds.ndim == 2 and winds.shape[1] >= 2 and np.isfinite(winds[:, :2]).any():
        wind = np.nanmean(winds, axis=0)
        norm = float(np.linalg.norm(wind[:2]))
        if norm > 1e-9:
            display_wind = np.array([wind[1], -wind[0]], dtype=float) / norm
            arrow_length = 0.075
            anchor = np.array([0.89, 0.88])
            end = anchor + arrow_length * display_wind
            map_ax.annotate(
                "", xy=end, xytext=anchor, xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": "#5C437C", "lw": 1.35, "mutation_scale": 9},
                zorder=15,
            )
            map_ax.text(0.885, 0.91, f"平均风 {norm:.1f} m/s", transform=map_ax.transAxes, fontsize=5.3, color="#5C437C", ha="center", va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0})
            source_rows.append({"element": "mean_wind", "sequence": 0, "x": wind[0], "y": wind[1], "z": wind[2] if len(wind) > 2 else math.nan, "display_x_m": display_wind[0], "display_y_m": display_wind[1], "priority": math.nan})

    # 比例尺与旋转后的北向，均放在相对稀疏的左上区域。
    map_ax.plot([0.035, 0.035 + V1_SCALE_BAR_METERS / display_extent[1]], [0.93, 0.93], transform=map_ax.transAxes, color="#202322", lw=2.0, solid_capstyle="butt", zorder=16)
    map_ax.text(0.035, 0.945, f"{V1_SCALE_BAR_METERS} m", transform=map_ax.transAxes, fontsize=5.3, color="#202322", ha="left", va="bottom")
    map_ax.annotate("N", xy=(0.30, 0.93), xytext=(0.225, 0.93), xycoords="axes fraction", textcoords="axes fraction", ha="center", va="center", fontsize=5.8, weight="bold", arrowprops={"arrowstyle": "-|>", "color": "#202322", "lw": 1.0, "mutation_scale": 8})

    # 小型横向高程色标，不占用地图外的大白边。
    colorbar_ax = inset_axes(map_ax, width="19%", height="2.8%", loc="lower left", bbox_to_anchor=(0.035, 0.075, 1, 1), bbox_transform=map_ax.transAxes, borderpad=0)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=elevation_norm, cmap=elevation_cmap), cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([round(float(vmin)), round(float(vmax))])
    colorbar.ax.tick_params(labelsize=4.8, length=1.5, pad=1)
    colorbar.outline.set_linewidth(0.35)
    colorbar.ax.set_title("DSM高程 (m)", fontsize=5.2, loc="left", pad=2)

    # 图例压缩为地图内两行；访问状态用蓝色外环表达，避免增加第四套填色。
    legend_handles = [
        Line2D([0], [0], color=base.color_for("full"), lw=2.2, label="PPO+Pointer航迹"),
        Line2D([0], [0], color="#3D4140", lw=1.6, label="山区公路"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor="#153E75", markersize=5.2, label="机场／返航点"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=priority_colors[1], markeredgecolor="white", markersize=4.2, label="低优先级"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=priority_colors[2], markeredgecolor="white", markersize=5.0, label="中优先级"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=priority_colors[3], markeredgecolor="white", markersize=5.8, label="高优先级"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=base.color_for("full"), markeredgewidth=1.2, markersize=5.8, label="已访问外环"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.278),
        ncol=7,
        fontsize=4.85,
        handlelength=1.8,
        columnspacing=0.9,
        labelspacing=0.25,
        frameon=False,
        borderaxespad=0.0,
    )

    map_ax.set_xlim(display_extent[0], display_extent[1])
    map_ax.set_ylim(display_extent[2], display_extent[3])
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.set_axis_off()

    # 下方剖面用累计三维航程，直接显示地表与飞行高度，替代三维坐标盒的遮挡。
    metric_route = route[:, :3].copy()
    metric_route[:, :2] *= ROUTE_CELL_METERS
    step_distance = np.linalg.norm(np.diff(metric_route, axis=0), axis=1)
    cumulative_km = np.r_[0.0, np.cumsum(step_distance)] / 1000.0
    route_ground = _sample_terrain(terrain, route[:, :2])
    profile_ax.fill_between(cumulative_km, route_ground, float(np.nanmin(route_ground)) - 12.0, color="#B9B0A2", alpha=0.72, linewidth=0)
    profile_ax.plot(cumulative_km, route_ground, color="#5A5148", lw=0.9, label="地表高程", zorder=2)
    profile_ax.fill_between(cumulative_km, route_ground, route[:, 2], color=base.color_for("full"), alpha=V1_PROFILE_CLEARANCE_FILL_ALPHA, linewidth=0)
    profile_ax.plot(cumulative_km, route[:, 2], color="white", lw=3.4, zorder=3)
    profile_ax.plot(cumulative_km, route[:, 2], color=base.color_for("full"), lw=1.55, label="飞行高度", zorder=4)
    profile_ax.set_xlim(0.0, max(0.001, float(cumulative_km[-1])))
    profile_ax.set_ylim(float(np.nanmin(route_ground)) - 8.0, float(max(np.nanmax(route[:, 2]), np.nanmax(route_ground))) + 12.0)
    profile_ax.set_xlabel("累计航程 (km)", labelpad=1)
    profile_ax.set_ylabel("高程 (m)", labelpad=2)
    profile_ax.tick_params(axis="both", labelsize=5.2, length=2.2, width=0.45, pad=1.5)
    profile_ax.grid(axis="y", color="#DADADA", lw=0.35, alpha=0.7)
    profile_ax.spines[["top", "right"]].set_visible(False)
    profile_ax.spines[["left", "bottom"]].set_linewidth(0.45)
    profile_ax.legend(loc="upper right", frameon=False, fontsize=5.0, ncol=2, handlelength=1.8, columnspacing=1.0)
    profile_ax.text(0.002, 0.98, "航迹高程剖面", transform=profile_ax.transAxes, fontsize=5.7, weight="bold", ha="left", va="top")

    for order, (distance_km, ground_z, flight_z) in enumerate(zip(cumulative_km, route_ground, route[:, 2])):
        source_rows.append({"element": "elevation_profile", "sequence": order, "x": route[order, 0], "y": route[order, 1], "z": flight_z, "display_x_m": distance_km * 1000.0, "display_y_m": ground_z, "priority": math.nan, "ground_elevation_m": ground_z})

    # 输出稀疏DSM源数据，既可复算底图又避免CSV体量失控。
    terrain_stride = max(1, int(max(display_terrain.shape) / 70))
    for row_index in range(0, display_terrain.shape[0], terrain_stride):
        for col_index in range(0, display_terrain.shape[1], terrain_stride):
            source_rows.append({"element": "dsm_sample", "sequence": row_index * display_terrain.shape[1] + col_index, "x": math.nan, "y": math.nan, "z": display_terrain[row_index, col_index], "display_x_m": col_index * ROUTE_CELL_METERS, "display_y_m": row_index * ROUTE_CELL_METERS, "priority": math.nan})

    frame = pd.DataFrame(source_rows)
    caption = (
        "展示图V1｜真实DSM山区公路固定巡检点规划。上部为沿道路走廊旋转90°后的正射阴影地形图，"
        "北向箭头给出真实方向；公路、机场、巡检点和冻结PPO+Pointer flight_path共享同一坐标变换。"
        "蓝色外环表示已访问巡检点。下部为同一路线的累计航程—高程剖面。该图用于空间解释，不承担统计证明。"
    )
    return base.save_figure(fig, output_dir, "figV01_3d_route", {"a": frame}, caption)


def _v1_scene_inputs(bundle: base.DataBundle) -> Dict[str, Any]:
    """只读装载V1冻结任务，并构造不依赖算法成绩的任务走廊视窗。"""
    task = base._task_by_id(bundle, base.REAL_EXAMPLE)
    payload = base._route_payload("full", base.REAL_EXAMPLE, 42)
    if payload is None:
        raise RuntimeError("V1冻结代表路线缺失。")
    detail = payload.get("detail", payload)
    map_id = str(task["map_id"])
    with np.load(base._map_bundle_path(map_id), allow_pickle=False) as data:
        terrain = np.asarray(data["terrain"], dtype=float)
    roads = base._road_segments(map_id)
    if len(roads) < 2:
        raise RuntimeError("V1真实DSM应包含两个冻结道路上下文。")

    route = np.asarray(detail["flight_path"], dtype=float)
    points = np.asarray(task["inspection_points_xyz"], dtype=float)[:, :2]
    start = np.asarray(task["start_xy"], dtype=float)
    road_00 = np.asarray(roads[0], dtype=float)[:, :2]
    all_xy = np.vstack([route[:, :2], points, start.reshape(1, 2), road_00])
    span = np.maximum(np.ptp(all_xy, axis=0), 1.0)
    pad = span * V1_VIEW_MARGIN
    low = all_xy.min(axis=0) - pad
    high = all_xy.max(axis=0) + pad
    x0 = max(0, int(math.floor(low[0])))
    x1 = min(terrain.shape[1] - 1, int(math.ceil(high[0])))
    y0 = max(0, int(math.floor(low[1])))
    y1 = min(terrain.shape[0] - 1, int(math.ceil(high[1])))
    # 参考高水平三维路径图的完整山地底板：只扩展真实DSM的短边，使平面接近1.3:1，
    # 不拉伸坐标，也不补造地图。路线保留在任务道路所在的一侧，位置关系与原始DSM一致。
    span_x = x1 - x0
    span_y = y1 - y0

    def expand_interval(low_index: int, high_index: int, target_span: int, maximum_index: int) -> tuple[int, int]:
        target_span = min(maximum_index, max(high_index - low_index, target_span))
        center = 0.5 * (low_index + high_index)
        new_low = int(math.floor(center - 0.5 * target_span))
        new_high = new_low + target_span
        if new_low < 0:
            new_high -= new_low
            new_low = 0
        if new_high > maximum_index:
            new_low -= new_high - maximum_index
            new_high = maximum_index
        return max(0, new_low), min(maximum_index, new_high)

    if span_y > V1_PLANFORM_LONG_TO_SHORT_RATIO * span_x:
        target_x_span = int(math.ceil(span_y / V1_PLANFORM_LONG_TO_SHORT_RATIO))
        x0, x1 = expand_interval(x0, x1, target_x_span, terrain.shape[1] - 1)
    elif span_x > V1_PLANFORM_LONG_TO_SHORT_RATIO * span_y:
        target_y_span = int(math.ceil(span_x / V1_PLANFORM_LONG_TO_SHORT_RATIO))
        y0, y1 = expand_interval(y0, y1, target_y_span, terrain.shape[0] - 1)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("V1任务走廊裁剪范围无效。")

    # 将南北向长走廊刚性旋转到横向版面；原始坐标完整保存在source data。
    def display_xy(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        shown = np.empty_like(values, dtype=float)
        shown[..., 0] = (values[..., 1] - y0) * ROUTE_CELL_METERS
        shown[..., 1] = (x1 - values[..., 0]) * ROUTE_CELL_METERS
        return shown

    terrain_crop = terrain[y0 : y1 + 1, x0 : x1 + 1]
    z_reference = float(math.floor(float(np.nanmin(terrain_crop)) / 10.0) * 10.0)

    def display_z(values: np.ndarray | float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return z_reference + (values - z_reference) * V1_VERTICAL_EXAGGERATION

    road_01 = np.asarray(roads[1], dtype=float)[:, :2]
    second_road_distance_m = float(np.min(np.linalg.norm(road_01 - start, axis=1)) * ROUTE_CELL_METERS)
    road_pair_min_m = float(
        min(
            np.min(np.linalg.norm(road_00[i : i + 512, None, :] - road_01[None, :, :], axis=2))
            for i in range(0, len(road_00), 512)
        )
        * ROUTE_CELL_METERS
    )
    return {
        "task": task,
        "payload": payload,
        "detail": detail,
        "terrain": terrain,
        "terrain_crop": terrain_crop,
        "road_00": road_00,
        "road_01": road_01,
        "route": route,
        "points": points,
        "start": start,
        "bounds": (x0, x1, y0, y1),
        "display_xy": display_xy,
        "display_z": display_z,
        "z_reference": z_reference,
        "second_road_distance_m": second_road_distance_m,
        "road_pair_min_m": road_pair_min_m,
    }


def _v1_route_center_delta(ax: plt.Axes, route_xyz: np.ndarray) -> tuple[float, float]:
    """返回投影后航迹包围盒中心相对坐标轴中心的归一化偏差。"""
    projected = proj3d.proj_transform(route_xyz[:, 0], route_xyz[:, 1], route_xyz[:, 2], ax.get_proj())
    pixels = ax.transData.transform(np.column_stack(projected[:2]))
    route_center = np.array(
        [
            0.5 * (float(np.min(pixels[:, 0])) + float(np.max(pixels[:, 0]))),
            0.5 * (float(np.min(pixels[:, 1])) + float(np.max(pixels[:, 1]))),
        ]
    )
    box = ax.get_window_extent()
    axes_center = np.array([0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1)])
    axes_size = np.array([max(box.width, 1.0), max(box.height, 1.0)])
    delta = np.abs(route_center - axes_center) / axes_size
    return float(delta[0]), float(delta[1])


def _build_v1_scene(
    bundle: base.DataBundle,
    *,
    width_mm: float,
    height_mm: float,
    compact: bool,
) -> tuple[plt.Figure, pd.DataFrame, str, Dict[str, Any]]:
    """构建一张单幅、可追溯的三维太行山巡检航迹图。"""
    scene = _v1_scene_inputs(bundle)
    task = scene["task"]
    payload = scene["payload"]
    detail = scene["detail"]
    terrain = scene["terrain"]
    terrain_crop = scene["terrain_crop"]
    road = scene["road_00"]
    route = scene["route"]
    points = scene["points"]
    start = scene["start"]
    display_xy = scene["display_xy"]
    display_z = scene["display_z"]
    x0, x1, y0, y1 = scene["bounds"]

    fig = plt.figure(figsize=(width_mm * base.MM, height_mm * base.MM), facecolor="white")
    axes_position = [0.005, 0.020, 0.900, 0.920] if compact else [0.015, 0.035, 0.97, 0.925]
    ax = fig.add_axes(axes_position, projection="3d", computed_zorder=False)
    ax.set_proj_type("ortho")

    raw_y, raw_x = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    surface_xy = display_xy(np.column_stack([raw_x.ravel(), raw_y.ravel()])).reshape(raw_x.shape + (2,))
    surface_x = surface_xy[..., 0]
    surface_y = surface_xy[..., 1]
    surface_z = display_z(terrain_crop)
    finite = terrain_crop[np.isfinite(terrain_crop)]
    vmin, vmax = [float(value) for value in np.nanpercentile(finite, [2, 98])]
    elevation_cmap = LinearSegmentedColormap.from_list(
        "v1_terrain",
        ["#DDE8DF", "#AFC4A2", "#C9BE94", "#A88F72", "#E9E5DD"],
    )
    elevation_norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    light = LightSource(azdeg=V1_HILLSHADE_AZDEG, altdeg=V1_HILLSHADE_ALTDEG)
    hillshade = light.hillshade(
        terrain_crop,
        vert_exag=1.0,
        dx=ROUTE_CELL_METERS,
        dy=ROUTE_CELL_METERS,
        fraction=1.0,
    )
    tint = elevation_cmap(elevation_norm(terrain_crop))[..., :3]
    relief = np.clip(tint * (0.70 + 0.43 * hillshade[..., None]), 0.0, 1.0)
    ax.plot_surface(
        surface_x,
        surface_y,
        surface_z,
        facecolors=relief,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.98,
        rasterized=True,
        zorder=0,
    )

    source_rows: list[Dict[str, Any]] = []
    common_source = {
        "task_id": str(task["id"]),
        "map_id": str(task["map_id"]),
        "model": "full",
        "seed": 42,
        "task_hash": str(task.get("task_hash", "")),
        "map_hash": str(task.get("map_hash", "")),
        "vertical_exaggeration": V1_VERTICAL_EXAGGERATION,
    }

    def append_source(
        element_type: str,
        element_id: str,
        sequence: int,
        raw_xyz: Sequence[float] | None,
        shown_xyz: Sequence[float] | None,
        **extra: Any,
    ) -> None:
        row: Dict[str, Any] = {
            **common_source,
            "element_type": element_type,
            "element_id": element_id,
            "sequence": int(sequence),
            "raw_x_grid": math.nan,
            "raw_y_grid": math.nan,
            "raw_z_m": math.nan,
            "display_x_m": math.nan,
            "display_y_m": math.nan,
            "display_z_m": math.nan,
            "priority": math.nan,
            "visited": math.nan,
            "note": "",
        }
        if raw_xyz is not None:
            row.update({"raw_x_grid": raw_xyz[0], "raw_y_grid": raw_xyz[1], "raw_z_m": raw_xyz[2]})
        if shown_xyz is not None:
            row.update({"display_x_m": shown_xyz[0], "display_y_m": shown_xyz[1], "display_z_m": shown_xyz[2]})
        row.update(extra)
        source_rows.append(row)

    road_ground = _sample_terrain(terrain, road)
    shown_road = display_xy(road)
    shown_road_z = display_z(road_ground)
    ax.plot(shown_road[:, 0], shown_road[:, 1], shown_road_z, color="white", lw=4.0 if not compact else 2.7, alpha=0.98, solid_capstyle="round", zorder=3)
    ax.plot(shown_road[:, 0], shown_road[:, 1], shown_road_z, color="#3B3F3E", lw=1.55 if not compact else 1.05, alpha=0.98, solid_capstyle="round", zorder=4)
    for order, (xy, z, shown_xy, shown_z) in enumerate(zip(road, road_ground, shown_road, shown_road_z)):
        append_source("road", "road_00", order, (xy[0], xy[1], z), (shown_xy[0], shown_xy[1], shown_z), note="任务相关道路，严格贴合DSM")

    priorities = np.asarray(task["priorities"], dtype=int)
    point_ground = _sample_terrain(terrain, points)
    shown_points = display_xy(points)
    shown_point_z = display_z(point_ground)
    metrics = detail.get("metrics", {}) or {}
    visited_order = [int(index) for index in metrics.get("visited_order", [])]
    visited = set(visited_order)
    priority_colors = {1: "#91A4B6", 2: "#D99A3D", 3: "#C63F3F"}
    priority_sizes = {1: 21 if not compact else 10, 2: 31 if not compact else 15, 3: 43 if not compact else 21}
    for priority in (1, 2, 3):
        indices = np.flatnonzero(priorities == priority)
        ax.scatter(
            shown_points[indices, 0],
            shown_points[indices, 1],
            shown_point_z[indices],
            s=priority_sizes[priority],
            c=priority_colors[priority],
            edgecolor="white",
            linewidth=0.8 if not compact else 0.55,
            depthshade=False,
            zorder=7,
        )
        visited_indices = np.asarray([idx for idx in indices if int(idx) in visited], dtype=int)
        if len(visited_indices):
            ax.scatter(
                shown_points[visited_indices, 0],
                shown_points[visited_indices, 1],
                shown_point_z[visited_indices],
                s=np.asarray([priority_sizes[priority] + (32 if not compact else 14)] * len(visited_indices)),
                facecolors="none",
                edgecolors=base.color_for("full"),
                linewidth=1.5 if not compact else 1.0,
                depthshade=False,
                zorder=8,
            )

    shown_route_xy = display_xy(route[:, :2])
    shown_route_z = display_z(route[:, 2])
    shown_route = np.column_stack([shown_route_xy, shown_route_z])
    for point_index in visited_order:
        matches = np.flatnonzero(np.linalg.norm(route[:, :2] - points[point_index], axis=1) < 1e-5)
        if not len(matches):
            continue
        top = float(np.max(shown_route_z[matches]))
        ax.plot(
            [shown_points[point_index, 0]] * 2,
            [shown_points[point_index, 1]] * 2,
            [shown_point_z[point_index], top],
            color=base.color_for("full"),
            lw=0.65 if not compact else 0.45,
            ls=(0, (2, 2)),
            alpha=0.34,
            zorder=6,
        )

    ax.plot(shown_route[:, 0], shown_route[:, 1], shown_route[:, 2], color="white", lw=4.4 if not compact else 2.8, alpha=0.98, solid_capstyle="round", solid_joinstyle="round", zorder=10)
    ax.plot(shown_route[:, 0], shown_route[:, 1], shown_route[:, 2], color=base.color_for("full"), lw=2.2 if not compact else 1.45, solid_capstyle="round", solid_joinstyle="round", zorder=11)

    horizontal = np.linalg.norm(np.diff(shown_route[:, :2], axis=0), axis=1)
    arrow_candidates = np.flatnonzero(horizontal > 80.0)
    if len(arrow_candidates):
        chosen = np.unique(arrow_candidates[np.linspace(0, len(arrow_candidates) - 1, min(3, len(arrow_candidates))).round().astype(int)])
        for index in chosen:
            vector = shown_route[index + 1] - shown_route[index]
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-9:
                continue
            arrow = vector / norm * min(115.0, 0.24 * norm)
            origin = shown_route[index] + 0.48 * vector
            ax.quiver(*origin, *arrow, color=base.color_for("full"), linewidth=0.9 if not compact else 0.6, arrow_length_ratio=0.28, normalize=False, zorder=12)

    for order, (xyz, shown_xyz) in enumerate(zip(route, shown_route)):
        append_source("flight_path", "ppo_pointer", order, xyz, shown_xyz, note="正式结果原始flight_path；仅显示z使用1.5倍比例")
    for order, (xy, z, shown_xy, shown_z, priority) in enumerate(zip(points, point_ground, shown_points, shown_point_z, priorities)):
        append_source("inspection_point", f"point_{order:02d}", order, (xy[0], xy[1], z), (shown_xy[0], shown_xy[1], shown_z), priority=int(priority), visited=int(order in visited), note="地面高程由冻结DSM回采样")

    start_ground = float(_sample_terrain(terrain, start.reshape(1, 2))[0])
    shown_start_xy = display_xy(start.reshape(1, 2))[0]
    shown_start_z = float(display_z(start_ground))
    airport_top = float(max(shown_route_z[np.linalg.norm(route[:, :2] - start, axis=1) < 1e-5]))
    ax.plot([shown_start_xy[0]] * 2, [shown_start_xy[1]] * 2, [shown_start_z, airport_top], color="#153E75", lw=0.8, alpha=0.55, zorder=8)
    ax.scatter(shown_start_xy[0], shown_start_xy[1], shown_start_z, marker="s", s=82 if not compact else 38, facecolor="white", edgecolor="#153E75", linewidth=1.4 if not compact else 0.90, depthshade=False, zorder=13)
    ax.text(shown_start_xy[0], shown_start_xy[1], shown_start_z, "H", color="#153E75", fontsize=7.0 if not compact else 5.6, weight="bold", ha="center", va="center", zorder=14)
    append_source("airport", "airport_return", 0, (start[0], start[1], start_ground), (shown_start_xy[0], shown_start_xy[1], shown_start_z), note="机场/返航点；竖线连接起飞与返航高度")

    segments = detail.get("segments", []) or []
    winds = np.asarray([segment.get("mean_wind_mps", [math.nan, math.nan, math.nan]) for segment in segments], dtype=float)
    mean_wind = np.nanmean(winds, axis=0) if winds.ndim == 2 and winds.shape[1] >= 3 else np.array([math.nan] * 3)
    wind_speed = float(np.linalg.norm(mean_wind)) if np.isfinite(mean_wind).all() else math.nan

    x_span = float(np.max(surface_x) - np.min(surface_x))
    y_span = float(np.max(surface_y) - np.min(surface_y))
    true_z_min = float(np.nanmin(terrain_crop))
    true_z_max = float(max(np.nanmax(terrain_crop), np.nanmax(route[:, 2])))
    z_min = float(display_z(true_z_min - 8.0))
    z_max = float(display_z(true_z_max + 18.0))
    z_span = z_max - z_min

    # 比例尺与北向箭头放在场景上方，避免与山体发生深度冲突。
    annotation_z = z_max - 0.04 * z_span
    scale_x0 = 0.07 * x_span
    scale_y = 0.06 * y_span
    scale_x1 = scale_x0 + V1_SCALE_BAR_METERS
    ax.plot([scale_x0, scale_x1], [scale_y, scale_y], [annotation_z, annotation_z], color="#202322", lw=2.1 if not compact else 1.4, zorder=15)
    ax.plot([scale_x0, scale_x0], [scale_y, scale_y], [annotation_z - 8, annotation_z + 8], color="#202322", lw=1.0, zorder=15)
    ax.plot([scale_x1, scale_x1], [scale_y, scale_y], [annotation_z - 8, annotation_z + 8], color="#202322", lw=1.0, zorder=15)
    ax.text(0.5 * (scale_x0 + scale_x1), scale_y, annotation_z + 14, "500 m", fontsize=6.4 if not compact else 5.3, ha="center", va="bottom", color="#202322", zorder=16)
    north_origin = np.array([0.79 * x_span, 0.08 * y_span, annotation_z])
    ax.quiver(*north_origin, 280.0, 0.0, 0.0, color="#202322", linewidth=1.2 if not compact else 0.8, arrow_length_ratio=0.28, normalize=False, zorder=15)
    ax.text(north_origin[0] + 310.0, north_origin[1], north_origin[2], "N", fontsize=7.0 if not compact else 5.6, weight="bold", ha="left", va="center", color="#202322", zorder=16)

    if np.isfinite(wind_speed) and wind_speed > 1e-9:
        shown_wind = np.array([mean_wind[1], -mean_wind[0], mean_wind[2] * V1_VERTICAL_EXAGGERATION], dtype=float)
        shown_wind /= max(float(np.linalg.norm(shown_wind)), 1e-9)
        wind_vector = shown_wind * 360.0
        wind_origin = np.array([0.18 * x_span, 0.84 * y_span, annotation_z])
        ax.quiver(*wind_origin, *wind_vector, color="#6D4B8E", linewidth=1.5 if not compact else 1.0, arrow_length_ratio=0.30, normalize=False, zorder=15)
        ax.text(wind_origin[0], wind_origin[1], wind_origin[2] + 18, f"平均风 {wind_speed:.1f} m/s", fontsize=6.2 if not compact else 5.2, color="#6D4B8E", weight="bold", ha="center", va="bottom", zorder=16)
        append_source("mean_wind", "mean_wind", 0, (mean_wind[0], mean_wind[1], mean_wind[2]), (shown_wind[0], shown_wind[1], shown_wind[2]), note="8个正式航段的平均三维风矢量；图中仅归一化方向并统一缩放箭头")

    # 坐标轴严格对应真实DSM底板，取消旧版为居中航迹而产生的大块透明扩展区。
    x_limits = (0.0, float(x_span))
    y_limits = (0.0, float(y_span))
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_span))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000.0:.1f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000.0:.1f}"))
    ax.set_xticks(np.linspace(max(0.0, x_limits[0]), x_limits[1], 5))
    ax.set_yticks(np.linspace(max(0.0, y_limits[0]), y_limits[1], 4))
    available_z_ticks = np.arange(math.ceil(true_z_min / 50.0) * 50.0, math.floor(true_z_max / 50.0) * 50.0 + 1.0, 50.0)
    true_z_ticks = available_z_ticks if len(available_z_ticks) <= 4 else available_z_ticks[np.linspace(0, len(available_z_ticks) - 1, 4).round().astype(int)]
    ax.set_zticks(display_z(true_z_ticks))
    ax.set_zticklabels([f"{value:.0f}" for value in true_z_ticks])
    label_size = 7.0 if not compact else 5.5
    tick_size = 6.0 if not compact else 5.3
    if compact:
        # 单栏检查版采用短标签并向坐标轴内收，防止三维轴标题越过画布边界。
        ax.set_xlabel("北向 (km)", labelpad=-1, fontsize=label_size)
        ax.set_ylabel("西向 (km)", labelpad=-5, fontsize=label_size)
        # 单栏版的高程单位已由紧邻z轴的刻度及DSM色标共同给出，避免重复标题被裁切。
        ax.set_zlabel("")
    else:
        ax.set_xlabel("北向距离 (km)", labelpad=2, fontsize=label_size)
        ax.set_ylabel("西向距离 (km)", labelpad=-1, fontsize=label_size)
        ax.set_zlabel("高程 (m)", labelpad=-3, fontsize=label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size, pad=0.5, length=2.0, width=0.45, colors="#3E4442")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((0.7, 0.7, 0.7, 0.28))
        axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
        axis._axinfo["axisline"]["color"] = (0.25, 0.27, 0.26, 0.62)

    camera_trials: list[Dict[str, Any]] = []
    chosen_camera = V1_CAMERA_CANDIDATES[0]
    for elevation, azimuth in V1_CAMERA_CANDIDATES:
        ax.view_init(elev=elevation, azim=azimuth)
        fig.canvas.draw()
        delta = _v1_route_center_delta(ax, shown_route)
        camera_trials.append({"elev_deg": elevation, "azim_deg": azimuth, "route_center_delta": list(delta)})
        chosen_camera = (elevation, azimuth)
        if max(delta) <= V1_ROUTE_CENTER_TOLERANCE:
            break
    ax.view_init(elev=chosen_camera[0], azim=chosen_camera[1])

    legend_labels = (
        ("航迹", "道路", "机场", "低", "中", "高", "已访问")
        if compact
        else ("PPO+Pointer航迹", "任务道路 road_00", "机场/返航点", "低优先级", "中优先级", "高优先级", "已访问外环")
    )
    legend_handles = [
        Line2D([0], [0], color=base.color_for("full"), lw=2.6, label=legend_labels[0]),
        Line2D([0], [0], color="#3B3F3E", lw=1.8, label=legend_labels[1]),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor="#153E75", markersize=5.7, label=legend_labels[2]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=priority_colors[1], markeredgecolor="white", markersize=4.8, label=legend_labels[3]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=priority_colors[2], markeredgecolor="white", markersize=5.5, label=legend_labels[4]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=priority_colors[3], markeredgecolor="white", markersize=6.2, label=legend_labels[5]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=base.color_for("full"), markeredgewidth=1.4, markersize=6.2, label=legend_labels[6]),
    ]
    legend_anchor = (0.965, 0.915) if compact else (0.955, 0.795)
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=legend_anchor,
        ncol=4,
        fontsize=5.5,
        handlelength=1.8,
        columnspacing=0.85 if not compact else 0.72,
        labelspacing=0.25,
        frameon=False,
    )
    returned = bool(metrics.get("returned", False))
    info_y = 0.970 if compact else 0.795
    fig.text(
        0.018,
        info_y,
        f"24个巡检点  |  访问{len(visited_order)}个  |  {'安全返航' if returned else '未返航'}",
        ha="left",
        va="top",
        fontsize=6.5 if not compact else 5.5,
        weight="bold",
        color="#153E75",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D7DFE7", "linewidth": 0.5, "alpha": 0.90},
    )
    exaggeration_y = 0.835 if compact else 0.725
    fig.text(0.018, exaggeration_y, "垂向夸张 1.5×", ha="left", va="top", fontsize=5.7 if not compact else 5.5, color="#555B59")

    colorbar_position = [0.030, 0.060, 0.250, 0.018] if compact else [0.735, 0.225, 0.210, 0.018]
    colorbar_ax = fig.add_axes(colorbar_position)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=elevation_norm, cmap=elevation_cmap), cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([round(vmin), round(vmax)])
    colorbar.ax.tick_params(labelsize=5.3 if not compact else 5.5, length=1.8, pad=1)
    colorbar.outline.set_linewidth(0.4)
    colorbar.ax.set_title("DSM高程 (m)", fontsize=5.7 if not compact else 5.5, loc="left", pad=2)

    for row_index in range(0, terrain_crop.shape[0], max(1, int(max(terrain_crop.shape) / 80))):
        for col_index in range(0, terrain_crop.shape[1], max(1, int(max(terrain_crop.shape) / 80))):
            raw_xyz = (x0 + col_index, y0 + row_index, terrain_crop[row_index, col_index])
            shown_xy = display_xy(np.asarray([[raw_xyz[0], raw_xyz[1]]], dtype=float))[0]
            shown_xyz = (shown_xy[0], shown_xy[1], float(display_z(raw_xyz[2])))
            append_source("dsm_sample", "terrain", row_index * terrain_crop.shape[1] + col_index, raw_xyz, shown_xyz, note="稀疏source-data样本；正式曲面使用完整裁剪DSM")
    append_source("omitted_context", "road_01", 0, None, None, note=f"独立道路上下文，距机场{scene['second_road_distance_m']:.1f} m，位于本任务视窗外，不参与绘图")

    route_path = Path(str(payload.get("_source_path", "")))
    route_sha256 = _sha256(route_path) if route_path.exists() else ""
    final_delta = _v1_route_center_delta(ax, shown_route)
    metadata = {
        "task_id": str(task["id"]),
        "map_id": str(task["map_id"]),
        "model": "full",
        "seed": 42,
        "task_hash": str(task.get("task_hash", "")),
        "map_hash": str(task.get("map_hash", "")),
        "route_source_sha256": route_sha256,
        "node_count": int(task["node_count"]),
        "visited_count": len(visited_order),
        "returned": returned,
        "crop_bounds_grid": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
        "rigid_display_rotation_deg": 90.0,
        "planform_long_to_short_ratio": max(x_span, y_span) / max(min(x_span, y_span), 1.0),
        "planform_coordinate_scaling": "none",
        "vertical_exaggeration": V1_VERTICAL_EXAGGERATION,
        "camera": {"elev_deg": chosen_camera[0], "azim_deg": chosen_camera[1], "projection": "orthographic"},
        "camera_trials": camera_trials,
        "route_center_delta": list(final_delta),
        "route_center_tolerance": V1_ROUTE_CENTER_TOLERANCE,
        "route_draw_order_overlay": True,
        "coordinate_note": "航迹坐标未改变；白色描边和绘制顺序仅增强可见性。",
        "excluded_road_01": {
            "reason": "road_01属于同一DSM中的独立道路上下文，位于冻结任务走廊外；禁止虚构道路交点。",
            "distance_to_airport_m": scene["second_road_distance_m"],
            "minimum_road_pair_distance_m": scene["road_pair_min_m"],
        },
        "nominal_width_mm": width_mm,
        "nominal_height_mm": height_mm,
        "compact": compact,
    }
    frame = pd.DataFrame(source_rows)
    caption = (
        "展示图V1｜真实太行山DSM中的固定巡检点与PPO+Pointer三维安全返航航迹。"
        f"冻结任务为 `{task['id']}`，使用完整模型训练种子42的正式原始flight_path；24个巡检点中访问{len(visited_order)}个。"
        "红、橙和灰蓝分别表示高、中、低优先级，蓝色外环表示已访问点。公路、机场和巡检点均从同一冻结DSM回采样高程，"
        "蓝色航迹保持真实飞行高度；为提高公里级场景中的地形净空可读性，仅显示坐标采用垂向夸张1.5×，原始高程未修改。"
        f"路线安全返航，航程{float(detail.get('distance_m', metrics.get('distance_m', math.nan))):.1f} m，"
        f"能耗{float(detail.get('energy_wh', metrics.get('energy_wh', math.nan))):.2f} Wh，总任务时间{float(detail.get('time_s', metrics.get('time_s', math.nan))):.1f} s。"
        f"同一DSM的road_01属于独立道路上下文，距本任务机场约{scene['second_road_distance_m'] / 1000.0:.2f} km，位于任务视窗外，故未绘制且未虚构交叉口。"
        "该图用于空间解释，不承担算法优越性的统计证明。"
    )
    return fig, frame, caption, metadata


def _figure_v01_rebuilt_v2(bundle: base.DataBundle, output_dir: Path) -> Dict[str, Any]:
    fig, frame, caption, metadata = _build_v1_scene(
        bundle,
        width_mm=V1_MASTER_WIDTH_MM,
        height_mm=V1_MASTER_HEIGHT_MM,
        compact=False,
    )
    _scale_v1_figure_to_export_width(fig, V1_MASTER_WIDTH_MM)
    V1_RENDER_METADATA.clear()
    V1_RENDER_METADATA.update(metadata)
    return base.save_figure(fig, output_dir, "figV01_3d_route", {"a": frame}, caption)


def _render_v1_compact(bundle: base.DataBundle, output_dir: Path) -> Dict[str, Any]:
    """以同一数据和相机规则生成89 mm可读性复核版。"""
    fig, frame, caption, metadata = _build_v1_scene(
        bundle,
        width_mm=V1_COMPACT_WIDTH_MM,
        height_mm=V1_COMPACT_HEIGHT_MM,
        compact=True,
    )
    bbox = _scale_v1_figure_to_export_width(fig, V1_COMPACT_WIDTH_MM)
    stem = f"{V1_PANEL_STEM}_89mm"
    files = _save_panel_files(fig, bbox, output_dir, stem)
    plt.close(fig)
    source_dir = output_dir / "source_data" / stem
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "source.csv"
    frame.to_csv(source_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    caption_path = output_dir / "captions" / f"{stem}.md"
    _write_text(caption_path, caption + "\n\n本文件为89 mm单栏可读性复核版。\n")
    return {
        "files": files,
        "caption": {"path": str(caption_path.relative_to(output_dir)), "sha256": _sha256(caption_path)},
        "source_data": {"path": str(source_path.relative_to(output_dir)), "rows": int(len(frame)), "sha256": _sha256(source_path)},
        "metadata": metadata,
    }


def _frame_for_panel(stem: str, spec: PanelSpec, panel_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frame = panel_frames.get(spec.source_key, pd.DataFrame()).copy()
    if stem == "fig06_generalization_robustness_routes" and spec.panel[0] in {"e", "f"} and not frame.empty:
        task_id = base.SYNTHETIC_EXAMPLE if spec.panel.startswith("e") else base.REAL_EXAMPLE
        models = ("full", "a2c_pointer", "traditional_ppo", "milp")
        model = models[int(spec.panel[1]) - 1]
        if "task_id" in frame and "model" in frame:
            frame = frame[(frame["task_id"].astype(str) == task_id) & (frame["model"].astype(str) == model)].copy()
    elif stem == "figS08_route_atlas" and not frame.empty:
        idx = ord(spec.panel) - ord("a")
        if idx < len(frame):
            frame = frame.iloc[[idx]].copy()
    return frame


def _panel_bbox_inches(fig: plt.Figure, axes: Sequence[plt.Axes]) -> Bbox:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in axes if ax.get_visible()]
    if not boxes:
        raise RuntimeError("面板没有可见坐标轴，无法导出。")
    # get_tightbbox对3D坐标轴常明显高估空白；用实际渲染像素再做一次内容边界收紧。
    rgba = np.asarray(fig.canvas.buffer_rgba())
    ink = np.any(rgba[:, :, :3] < 252, axis=2) & (rgba[:, :, 3] > 0)
    ys, xs = np.where(ink)
    if len(xs) and len(ys):
        height = rgba.shape[0]
        pixel_box = Bbox.from_extents(float(xs.min()), float(height - 1 - ys.max()), float(xs.max() + 1), float(height - ys.min()))
        return pixel_box.transformed(fig.dpi_scale_trans.inverted())
    return Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())


def _scale_v1_figure_to_export_width(fig: plt.Figure, target_width_mm: float) -> Bbox:
    """等比例校准V1画布，使紧边导出后的物理宽度等于指定版宽。"""
    target_width_inch = target_width_mm * base.MM
    bbox = _panel_bbox_inches(fig, list(fig.axes))
    for _ in range(5):
        export_width_inch = bbox.width + 2.0 * V1_PAD_INCH
        scale = target_width_inch / max(export_width_inch, 1.0e-9)
        if abs(scale - 1.0) <= 2.0e-4:
            break
        current = fig.get_size_inches()
        fig.set_size_inches(current[0] * scale, current[1] * scale, forward=True)
        bbox = _panel_bbox_inches(fig, list(fig.axes))
    return bbox


def _save_panel_files(fig: plt.Figure, bbox: Bbox, output_dir: Path, panel_stem: str) -> Dict[str, Any]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {kind: figure_dir / f"{panel_stem}.{kind}" for kind in ("svg", "pdf", "png", "tiff")}
    is_v1 = panel_stem.startswith(V1_PANEL_STEM)
    pad_inches = V1_PAD_INCH if is_v1 else SPLIT_PAD_INCH
    # Matplotlib对显式Bbox不会可靠应用pad_inches；V1手工扩展1.2 mm，保证实际版宽和安全边距。
    if is_v1:
        export_bbox = Bbox.from_extents(
            bbox.x0 - pad_inches,
            bbox.y0 - pad_inches,
            bbox.x1 + pad_inches,
            bbox.y1 + pad_inches,
        )
        common = {"bbox_inches": export_bbox, "pad_inches": 0.0, "facecolor": "white"}
    else:
        common = {"bbox_inches": bbox, "pad_inches": pad_inches, "facecolor": "white"}
    fig.savefig(paths["svg"], format="svg", **common)
    fig.savefig(paths["pdf"], format="pdf", **common)
    fig.savefig(paths["png"], format="png", dpi=EXPORT_DPI, **common)
    # Windows Pillow在宽幅V1的600 dpi TIFF-LZW编码上会原生崩溃；该图单独使用
    # 未压缩无损TIFF，像素与DPI不变。其余图继续沿用已验证的LZW输出。
    tiff_kwargs = {} if is_v1 else {"pil_kwargs": {"compression": "tiff_lzw"}}
    fig.savefig(paths["tiff"], format="tiff", dpi=EXPORT_DPI, **tiff_kwargs, **common)
    return {
        kind: {"path": str(path.relative_to(output_dir)), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for kind, path in paths.items()
    }


CREATED: Dict[str, Dict[str, Any]] = {}


def _split_save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    panel_frames: Mapping[str, pd.DataFrame],
    caption: str,
    dpi: int = EXPORT_DPI,
) -> Dict[str, Any]:
    """取代原组合图保存函数：每个面板单独裁切并导出四种格式。"""
    specs = PANEL_SPECS[stem]
    # inset_axes的定位器依赖父坐标轴；拆图时会隐藏其他面板，因此先冻结已渲染的最终位置。
    fig.canvas.draw()
    frozen_positions = [ax.get_position().frozen() for ax in fig.axes]
    for ax, position in zip(fig.axes, frozen_positions):
        ax.set_axes_locator(None)
        ax.set_position(position)
    # 组合图最右侧的色条刻度可能超出原画布；把整个面板组左移并保留刻度宽度。
    for spec in specs:
        colorbar_indices = [index for index in spec.axes if fig.axes[index].get_label() == "<colorbar>"]
        if not colorbar_indices:
            continue
        group_positions = [fig.axes[index].get_position().frozen() for index in spec.axes]
        right_edge = max(position.x1 for position in group_positions)
        overflow = max(0.0, right_edge + 0.130 - 0.99)
        if overflow > 0:
            for index, position in zip(spec.axes, group_positions):
                fig.axes[index].set_position(Bbox.from_bounds(position.x0 - overflow, position.y0, position.width, position.height))
    if stem == "figS06_ablation_maps":
        # 零值单元格使用极浅紫而非纯白，保留数据边界并避免被误解为空白区。
        no_white_purple = base.LinearSegmentedColormap.from_list("Purples_no_white", ["#EFE8F4", "#3F007D"])
        if fig.axes[2].images:
            fig.axes[2].images[0].set_cmap(no_white_purple)
    # 路线面板在拆分后都需要自己的算法标题和面板编号。
    if stem == "fig06_generalization_robustness_routes":
        models = ("full", "a2c_pointer", "traditional_ppo", "milp")
        for spec in specs[4:]:
            ax = fig.axes[spec.axes[0]]
            ax.set_title(base.label_for(models[(int(spec.panel[1]) - 1)]), color=base.color_for(models[(int(spec.panel[1]) - 1)]), fontsize=6.4, pad=4)
            if not any(text.get_text() == spec.panel for text in ax.texts):
                base.panel_label(ax, spec.panel, x=-0.07, y=1.08)

    for spec in specs:
        selected_axes = [fig.axes[index] for index in spec.axes]
        visibility = [ax.get_visible() for ax in fig.axes]
        for index, ax in enumerate(fig.axes):
            ax.set_visible(index in spec.axes)
        # 独立小图不保留组合图的a/b/c面板字母，避免与标题重叠。
        for ax in selected_axes:
            for text_artist in ax.texts:
                if text_artist.get_text() in {"a", "b", "c", "d", "e", "f", "g", "h", "e1", "e2", "e3", "e4", "f1", "f2", "f3", "f4"}:
                    text_artist.set_visible(False)
        bbox = _panel_bbox_inches(fig, selected_axes)
        panel_stem = V1_PANEL_STEM if stem == "figV01_3d_route" else f"{stem}_{spec.panel}"
        files = _save_panel_files(fig, bbox, output_dir, panel_stem)
        for ax, visible in zip(fig.axes, visibility):
            ax.set_visible(visible)

        source_frame = _frame_for_panel(stem, spec, panel_frames)
        source_dir = output_dir / "source_data" / panel_stem
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path = source_dir / "source.csv"
        float_format = "%.17g" if panel_stem == V1_PANEL_STEM else "%.10g"
        source_frame.to_csv(source_path, index=False, encoding="utf-8-sig", float_format=float_format)
        caption_path = output_dir / "captions" / f"{panel_stem}.md"
        _write_text(caption_path, f"{spec.title}。\n\n来自父图 `{stem}`。{caption}\n")
        with Image.open(output_dir / files["png"]["path"]) as image:
            width_px, height_px = image.size
        CREATED[panel_stem] = {
            "parent": stem,
            "panel": spec.panel,
            "title": spec.title,
            "files": files,
            "caption": {"path": str(caption_path.relative_to(output_dir)), "sha256": _sha256(caption_path)},
            "source_data": {"path": str(source_path.relative_to(output_dir)), "rows": int(len(source_frame)), "sha256": _sha256(source_path)},
            "pixel_size": [width_px, height_px],
            "dpi": dpi,
        }
        if panel_stem == V1_PANEL_STEM:
            CREATED[panel_stem]["v1_render_metadata"] = dict(V1_RENDER_METADATA)
    plt.close(fig)
    return {"split_panels": [f"{stem}_{spec.panel}" for spec in specs], "source_data": {}}


def _nonwhite_bbox(image: Image.Image, threshold: int = 250) -> tuple[int, int, int, int] | None:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, background).convert("L").point(lambda x: 255 if x > (255 - threshold) else 0)
    return diff.getbbox()


def _run_qa(output_dir: Path, records: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    per_panel: Dict[str, Any] = {}
    failures: list[str] = []
    for stem, record in records.items():
        errors: list[str] = []
        for kind, info in record["files"].items():
            path = output_dir / info["path"]
            if not path.exists() or not path.stat().st_size:
                errors.append(f"{kind}缺失或为空")
            elif _sha256(path) != info["sha256"]:
                errors.append(f"{kind}哈希回读不一致")
        png = output_dir / record["files"]["png"]["path"]
        if png.exists():
            with Image.open(png) as image:
                bbox = _nonwhite_bbox(image)
                if bbox is None:
                    errors.append("PNG疑似空白")
                    margins = None
                else:
                    left, top, right, bottom = bbox
                    margins = [left, top, image.width - right, image.height - bottom]
                    # 600 dpi下60 px约为2.54 mm；这是外边缘最大安全线，非数据区内部留白。
                    if max(margins) > 60:
                        errors.append(f"外部白边过大: {margins}px")
                gray = np.asarray(image.convert("L").resize((192, 192)), dtype=float)
                if float(gray.std()) < 4.0:
                    errors.append("PNG疑似空白或对比度过低")
        svg = output_dir / record["files"]["svg"]["path"]
        if svg.exists():
            text = svg.read_text(encoding="utf-8", errors="ignore")
            if "<text" not in text:
                errors.append("SVG未保留可编辑文字")
            if "ppo_mlp" in text:
                errors.append("检出被排除的旧ppo_mlp")
        per_panel[stem] = {"passed": not errors, "errors": errors, "outer_margins_px": margins if png.exists() else None}
        failures.extend([f"{stem}: {error}" for error in errors])

    report = {"passed": not failures, "panel_count": len(records), "failures": failures, "per_panel": per_panel}
    _write_json(output_dir / "qa_report.json", report)
    lines = ["# 独立小图自动QA", "", f"- 状态：{'**通过**' if report['passed'] else '**未通过**'}", f"- 小图数：{len(records)}", "- 检查：四格式与哈希、非空白、SVG可编辑文字、排除旧模型、外部白边。", ""]
    if failures:
        lines += ["## 待修复", ""] + [f"- {item}" for item in failures]
    else:
        lines += ["所有程序化检查均通过。还需结合分页缩略图进行人工视觉复核。"]
    _write_text(output_dir / "qa_report_CN.md", "\n".join(lines) + "\n")
    return report


def _build_contact_sheets(output_dir: Path, stems: Sequence[str], per_page: int = 12) -> list[Dict[str, Any]]:
    review_dir = output_dir / "review_contact_sheets"
    review_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Dict[str, Any]] = []
    for page_index, start in enumerate(range(0, len(stems), per_page), start=1):
        batch = stems[start : start + per_page]
        cols, rows = 3, 4
        cell_w, cell_h = 700, 555
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        for index, stem in enumerate(batch):
            with Image.open(output_dir / "figures" / f"{stem}.png") as source:
                thumb = source.convert("RGB")
                thumb.thumbnail((cell_w - 30, cell_h - 55), Image.Resampling.LANCZOS)
            col, row = index % cols, index // cols
            x = col * cell_w + (cell_w - thumb.width) // 2
            y = row * cell_h + 38 + (cell_h - 50 - thumb.height) // 2
            sheet.paste(thumb, (x, y))
            draw.text((col * cell_w + 14, row * cell_h + 10), stem, fill="#202020")
        path = review_dir / f"contact_sheet_{page_index:02d}.png"
        sheet.save(path, dpi=(THUMB_DPI, THUMB_DPI))
        pages.append({"path": str(path.relative_to(output_dir)), "sha256": _sha256(path), "panels": list(batch)})
    return pages


def _write_index(output_dir: Path, records: Mapping[str, Mapping[str, Any]], audit: Mapping[str, Any]) -> None:
    if set(records) == {V1_PANEL_STEM}:
        lines = [
            "# V1三维太行山巡检航迹图",
            "",
            "本目录只包含重构后的V1及其89 mm可读性复核版；旧V1未覆盖，其他主图、补充图和展示图均未改动。",
            "",
            f"- 冻结矩阵 SHA-256：`{audit['matrix_sha256']}`",
            f"- 正式结果 SHA-256：`{audit['results_sha256']}`",
            "- [183 mm正式版PNG](figures/figV01_3d_taihang_route.png) ｜ [SVG](figures/figV01_3d_taihang_route.svg) ｜ [PDF](figures/figV01_3d_taihang_route.pdf) ｜ [TIFF](figures/figV01_3d_taihang_route.tiff)",
            "- [89 mm检查版PNG](figures/figV01_3d_taihang_route_89mm.png) ｜ [SVG](figures/figV01_3d_taihang_route_89mm.svg) ｜ [PDF](figures/figV01_3d_taihang_route_89mm.pdf) ｜ [TIFF](figures/figV01_3d_taihang_route_89mm.tiff)",
            "- [中文图注](captions/figV01_3d_taihang_route.md) ｜ [逐元素source data](source_data/figV01_3d_taihang_route/source.csv)",
            "- [V1专项QA](v1_qa_report_CN.md) ｜ [图件manifest](figure_manifest.json) ｜ [灰度与色觉缺陷预览](review/)",
            "",
        ]
        _write_text(output_dir / "README_CN.md", "\n".join(lines))
        return
    grouped: Dict[str, list[str]] = {"main": [], "supplement": [], "showcase": []}
    for stem, record in records.items():
        parent = record["parent"]
        tier = "main" if parent.startswith("fig0") else "supplement" if parent.startswith("figS") else "showcase"
        grouped[tier].append(stem)
    labels = {"main": "正文独立小图", "supplement": "补充独立小图", "showcase": "展示图"}
    lines = [
        "# v3.2.14 独立小图正式输出",
        "",
        "6张正文组合图和8张补充组合图已拆为独立小图；组合画布不再作为论文插图交付。",
        "",
        f"- 冻结矩阵 SHA-256：`{audit['matrix_sha256']}`",
        f"- 正式结果 SHA-256：`{audit['results_sha256']}`",
        f"- 独立小图：{len(records)}张，每张含SVG/PDF/PNG/TIFF、图注和source data。",
        "- 路线图均按冻结任务几何裁剪到局部走廊；同一任务的算法共享同一视窗。",
        "",
    ]
    for tier in ("main", "supplement", "showcase"):
        lines += [f"## {labels[tier]}", ""]
        for stem in grouped[tier]:
            record = records[stem]
            lines.append(f"- [{stem}.png](figures/{stem}.png) — {record['title']} ｜ [图注](captions/{stem}.md)")
        lines.append("")
    lines += ["## 复核入口", "", "- [自动QA](qa_report_CN.md)", "- [图件manifest](figure_manifest.json)", "- [分页缩略图](review_contact_sheets/)"]
    _write_text(output_dir / "README_CN.md", "\n".join(lines) + "\n")


def _run_v1_post_qa(bundle: base.DataBundle, output_dir: Path) -> Dict[str, Any]:
    """对V1执行坐标真实性、导出、白边和辅助视觉模式复核。"""
    failures: list[str] = []
    checks: Dict[str, Any] = {}
    source_path = output_dir / "source_data" / V1_PANEL_STEM / "source.csv"
    if not source_path.exists():
        failures.append("V1 source-data缺失")
        frame = pd.DataFrame()
    else:
        frame = pd.read_csv(source_path, encoding="utf-8-sig")

    task = base._task_by_id(bundle, base.REAL_EXAMPLE)
    payload = base._route_payload("full", base.REAL_EXAMPLE, 42)
    if payload is None:
        failures.append("冻结V1路线载荷缺失")
        route = np.empty((0, 3))
    else:
        route = np.asarray(payload.get("detail", payload)["flight_path"], dtype=float)

    if not frame.empty:
        route_rows = frame[frame["element_type"] == "flight_path"].sort_values("sequence")
        source_route = route_rows[["raw_x_grid", "raw_y_grid", "raw_z_m"]].to_numpy(dtype=float)
        route_exact = source_route.shape == route.shape and bool(np.allclose(source_route, route, atol=1e-8, rtol=0.0))
        checks["flight_path_exact"] = route_exact
        if not route_exact:
            failures.append("source-data中的flight_path与正式路线不一致")

        point_rows = frame[frame["element_type"] == "inspection_point"].sort_values("sequence")
        point_count_ok = len(point_rows) == int(task["node_count"]) == 24
        checks["inspection_point_count"] = int(len(point_rows))
        if not point_count_ok:
            failures.append(f"巡检点数量异常: {len(point_rows)}")

        airport_rows = frame[frame["element_type"] == "airport"]
        airport_ok = len(airport_rows) == 1 and bool(
            np.allclose(
                airport_rows[["raw_x_grid", "raw_y_grid"]].iloc[0].to_numpy(dtype=float),
                np.asarray(task["start_xy"], dtype=float),
                atol=1e-8,
                rtol=0.0,
            )
        )
        checks["airport_exact"] = airport_ok
        if not airport_ok:
            failures.append("机场坐标与冻结start_xy不一致")

        points_xy = point_rows[["raw_x_grid", "raw_y_grid"]].to_numpy(dtype=float)
        with np.load(base._map_bundle_path(str(task["map_id"])), allow_pickle=False) as data:
            terrain = np.asarray(data["terrain"], dtype=float)
        sampled = _sample_terrain(terrain, points_xy)
        ground_ok = bool(np.allclose(sampled, point_rows["raw_z_m"].to_numpy(dtype=float), atol=1e-8, rtol=0.0))
        checks["inspection_points_draped_on_dsm"] = ground_ok
        if not ground_ok:
            failures.append("巡检点高程未严格贴合冻结DSM")

        exaggeration_ok = bool(
            np.allclose(
                frame.loc[frame["raw_z_m"].notna(), "vertical_exaggeration"].to_numpy(dtype=float),
                V1_VERTICAL_EXAGGERATION,
                atol=0.0,
                rtol=0.0,
            )
        )
        checks["vertical_exaggeration_recorded"] = exaggeration_ok
        if not exaggeration_ok:
            failures.append("纵向夸张比例记录异常")

    center_delta = V1_RENDER_METADATA.get("route_center_delta", [math.inf, math.inf])
    center_ok = max(center_delta) <= V1_ROUTE_CENTER_TOLERANCE
    checks["route_center_delta"] = center_delta
    checks["route_center_passed"] = center_ok
    if not center_ok:
        failures.append(f"航迹投影中心偏差超过{100.0 * V1_ROUTE_CENTER_TOLERANCE:.0f}%: {center_delta}")

    planform_ratio = float(V1_RENDER_METADATA.get("planform_long_to_short_ratio", math.inf))
    planform_ok = planform_ratio <= V1_PLANFORM_LONG_TO_SHORT_RATIO + 0.02
    checks["planform_long_to_short_ratio"] = planform_ratio
    checks["planform_ratio_passed"] = planform_ok
    checks["planform_coordinate_scaling"] = V1_RENDER_METADATA.get("planform_coordinate_scaling")
    if not planform_ok:
        failures.append(f"真实DSM底板长短边比例异常: {planform_ratio:.3f}")
    if V1_RENDER_METADATA.get("planform_coordinate_scaling") != "none":
        failures.append("V1平面坐标出现非真实性缩放")

    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    export_checks: Dict[str, Any] = {}
    for stem in (V1_PANEL_STEM, f"{V1_PANEL_STEM}_89mm"):
        png_path = output_dir / "figures" / f"{stem}.png"
        svg_path = output_dir / "figures" / f"{stem}.svg"
        pdf_path = output_dir / "figures" / f"{stem}.pdf"
        if not png_path.exists() or not svg_path.exists() or not pdf_path.exists():
            failures.append(f"{stem}导出不完整")
            continue
        with Image.open(png_path) as image:
            rgb = image.convert("RGB")
            bbox = _nonwhite_bbox(rgb)
            if bbox is None:
                failures.append(f"{stem}疑似空白")
                margins = None
            else:
                margins = [bbox[0], bbox[1], rgb.width - bbox[2], rgb.height - bbox[3]]
                if max(margins) > 60:
                    failures.append(f"{stem}外部白边超过2.54 mm: {margins}")
                if min(margins) < 18:
                    failures.append(f"{stem}安全边距不足0.76 mm: {margins}")
            array = np.asarray(rgb, dtype=np.uint8)
            blue_pixels = int(
                np.sum(
                    (array[..., 2].astype(int) > array[..., 0].astype(int) + 24)
                    & (array[..., 2].astype(int) > array[..., 1].astype(int) + 8)
                    & (array[..., 0] < 110)
                )
            )
            if blue_pixels < 300:
                failures.append(f"{stem}可辨识蓝色航迹像素不足: {blue_pixels}")
            dpi = image.info.get("dpi", (EXPORT_DPI, EXPORT_DPI))
            width_mm = rgb.width / float(dpi[0]) * 25.4
            target_width_mm = V1_MASTER_WIDTH_MM if stem == V1_PANEL_STEM else V1_COMPACT_WIDTH_MM
            width_error_mm = width_mm - target_width_mm
            if abs(width_error_mm) > 0.15:
                failures.append(f"{stem}实际版宽偏差超过0.15 mm: {width_mm:.3f} mm")
            if abs(float(dpi[0]) - EXPORT_DPI) > 0.1 or abs(float(dpi[1]) - EXPORT_DPI) > 0.1:
                failures.append(f"{stem} PNG不是600 dpi: {dpi}")
            export_checks[stem] = {
                "pixel_size": [rgb.width, rgb.height],
                "dpi": [float(dpi[0]), float(dpi[1])],
                "target_width_mm": target_width_mm,
                "rendered_width_mm": width_mm,
                "width_error_mm": width_error_mm,
                "outer_margins_px": margins,
                "blue_route_pixels": blue_pixels,
            }
            if stem == V1_PANEL_STEM:
                rgb.convert("L").save(review_dir / f"{stem}_grayscale.png", dpi=dpi)
                normalized = np.asarray(rgb, dtype=float) / 255.0
                matrix = np.array([[0.625, 0.375, 0.0], [0.700, 0.300, 0.0], [0.0, 0.300, 0.700]])
                simulated = np.clip(normalized @ matrix.T, 0.0, 1.0)
                Image.fromarray(np.uint8(np.round(simulated * 255.0)), mode="RGB").save(
                    review_dir / f"{stem}_deuteranopia.png",
                    dpi=dpi,
                )
        svg_text = svg_path.read_text(encoding="utf-8", errors="ignore")
        svg_text_elements = svg_text.count("<text")
        if svg_text_elements == 0:
            failures.append(f"{stem} SVG文字不可编辑")
        pdf_bytes = pdf_path.read_bytes()
        pdf_font_embedded = b"/FontFile2" in pdf_bytes and b"/Subtype /Type0" in pdf_bytes
        if not pdf_font_embedded:
            failures.append(f"{stem} PDF未检出嵌入的Type0 TrueType字体")
        export_checks[stem]["svg_text_elements"] = svg_text_elements
        export_checks[stem]["pdf_font_embedded"] = pdf_font_embedded
    checks["exports"] = export_checks

    report = {
        "schema": "v3.2.14-v1-3d-qa-v1",
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "review_files": [
            str((review_dir / f"{V1_PANEL_STEM}_grayscale.png").relative_to(output_dir)),
            str((review_dir / f"{V1_PANEL_STEM}_deuteranopia.png").relative_to(output_dir)),
        ],
    }
    _write_json(output_dir / "v1_qa_report.json", report)
    lines = [
        "# V1三维图专项QA",
        "",
        f"- 状态：{'**通过**' if report['passed'] else '**未通过**'}",
        "- 数据真实性：正式flight_path逐点回读、机场、巡检点数量与DSM贴合。",
        "- 空间比例：仅扩展真实DSM短边至约1.30∶1，不进行平面坐标拉伸或虚构补图。",
        "- 视觉导出：航迹居中、四格式、白边、可编辑SVG、灰度与色觉缺陷预览。",
        "",
    ]
    if failures:
        lines += ["## 待修复", ""] + [f"- {item}" for item in failures]
    else:
        lines += ["所有程序化专项检查均通过。", ""]
    _write_text(output_dir / "v1_qa_report_CN.md", "\n".join(lines) + "\n")
    return report


def render_all(output_dir: Path, parents: Sequence[str] | None = None) -> Dict[str, Any]:
    base.apply_style()
    bundle = base.load_bundle()
    audit = base.audit_inputs(bundle)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "input_audit.json", audit)

    selected = list(parents or base.FIGURE_ORDER)
    invalid = sorted(set(selected) - set(base.FIGURE_ORDER))
    if invalid:
        raise ValueError(f"未知父图: {invalid}")

    CREATED.clear()
    old_save = base.save_figure
    old_route = base._plot_route_axis
    old_v01 = base.figure_v01
    try:
        base.save_figure = _split_save_figure
        base._plot_route_axis = _plot_local_route
        base.figure_v01 = _figure_v01_rebuilt_v2
        for index, name in enumerate(selected, start=1):
            print(f"[{index:02d}/{len(selected):02d}] rendering and splitting {name}", flush=True)
            builder = base._builder(name)
            if name == "figS01_audit":
                builder(bundle, output_dir, audit)
            else:
                builder(bundle, output_dir)
    finally:
        base.save_figure = old_save
        base._plot_route_axis = old_route
        base.figure_v01 = old_v01

    stems = list(CREATED)
    if "figV01_3d_route" in selected and V1_PANEL_STEM in CREATED:
        CREATED[V1_PANEL_STEM]["compact_89mm"] = _render_v1_compact(bundle, output_dir)
    contact_sheets = _build_contact_sheets(output_dir, stems)
    qa = _run_qa(output_dir, CREATED)
    v1_qa = _run_v1_post_qa(bundle, output_dir) if V1_PANEL_STEM in CREATED else None
    manifest = {
        "schema": "v3.2.14-publication-split-figures-v3",
        "backend": "Python/Matplotlib",
        "script": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": _sha256(Path(__file__))},
        "base_script": {"path": str(Path(base.__file__).relative_to(ROOT)), "sha256": _sha256(Path(base.__file__))},
        "frozen_input_audit": audit,
        "parent_figures": selected,
        "panel_count": len(CREATED),
        "panels": CREATED,
        "contact_sheets": contact_sheets,
        "qa_passed": qa["passed"] and (v1_qa is None or v1_qa["passed"]),
        "v1_qa": v1_qa,
        "parameters": {
            "local_view_margin": LOCAL_VIEW_MARGIN,
            "local_view_min_pad": LOCAL_VIEW_MIN_PAD,
            "split_pad_inch": SPLIT_PAD_INCH,
            "route_cell_meters": ROUTE_CELL_METERS,
            "export_dpi": EXPORT_DPI,
            "v1_hillshade_azdeg": V1_HILLSHADE_AZDEG,
            "v1_hillshade_altdeg": V1_HILLSHADE_ALTDEG,
            "v1_scale_bar_meters": V1_SCALE_BAR_METERS,
            "v1_vertical_exaggeration": V1_VERTICAL_EXAGGERATION,
            "v1_view_margin": V1_VIEW_MARGIN,
            "v1_planform_long_to_short_ratio": V1_PLANFORM_LONG_TO_SHORT_RATIO,
            "v1_camera_candidates": V1_CAMERA_CANDIDATES,
            "v1_route_center_tolerance": V1_ROUTE_CENTER_TOLERANCE,
            "v1_master_size_mm": [V1_MASTER_WIDTH_MM, V1_MASTER_HEIGHT_MM],
            "v1_compact_size_mm": [V1_COMPACT_WIDTH_MM, V1_COMPACT_HEIGHT_MM],
        },
    }
    _write_json(output_dir / "figure_manifest.json", manifest)
    _write_index(output_dir, CREATED, audit)
    if not qa["passed"]:
        print(f"[warning] automatic QA found {len(qa['failures'])} issues; see qa_report_CN.md", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="生成v3.2.14独立小图与重构DSM路线图。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parents", help="逗号分隔的父图stem；默认全部16组。")
    args = parser.parse_args()
    parents = [item.strip() for item in args.parents.split(",") if item.strip()] if args.parents else None
    manifest = render_all(args.output.resolve(), parents)
    print(json.dumps({"output": str(args.output.resolve()), "panel_count": manifest["panel_count"], "qa_passed": manifest["qa_passed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

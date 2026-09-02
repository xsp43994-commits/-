#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 PPO v2 使用的真实尺度山区双国道巡检场景。

本模块只负责把既有的两条模拟道路放到 GeoTIFF 的真实物理尺度中，并生成
固定、可复现的机场、巡检点、风险等级和空间风场。无人机仍可在点间自由飞行，
道路中心线仅用于定义巡检点位置，不会被当作飞行约束。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:  # pragma: no cover - 由调用入口给出明确依赖错误
    rasterio = None
    Window = None


SCENARIO_SCHEMA_VERSION = 1
DEFAULT_DEM_PATH = WORKSPACE_ROOT / "map_data/AP_15010_FBS_F2760_RT1.dem.tif"
RISK_COMPONENT_NAMES = ("terrain_slope", "overhead_relief", "roughness", "road_curvature")


@dataclass(frozen=True)
class ScenarioConfig:
    """场景关键参数；集中放置，避免道路、风场和风险逻辑出现魔法数字。"""

    expected_crs: str = "EPSG:32651"
    coordinate_scale_m_per_unit: float = 12.5
    intersection_samples: int = 30_000
    arm_length_m: float = 800.0
    segments_per_arm: int = 4
    segment_length_m: float = 200.0
    min_point_spacing_m: float = 150.0
    min_global_euclidean_spacing_m: float = 130.0
    crop_margin_m: float = 300.0
    risk_patch_radius_m: float = 50.0
    risk_percentile_low: float = 5.0
    risk_percentile_high: float = 95.0
    slope_weight: float = 0.35
    overhead_relief_weight: float = 0.25
    roughness_weight: float = 0.20
    curvature_weight: float = 0.20
    high_priority_count: int = 5
    medium_priority_count: int = 6
    low_priority_count: int = 5
    service_time_s: float = 20.0
    wind_spacing_m: float = 250.0
    wind_height_agl_m: float = 18.0
    wind_seed: int = 2026
    wind_min_speed_mps: float = 2.0
    wind_max_speed_mps: float = 6.2
    wind_takeoff_base_limit_mps: float = 6.2
    wind_domain_scale_max: float = 1.20
    wind_vertical_bias_max_mps: float = 1.0
    seed: int = 42

    def validate(self) -> None:
        if self.coordinate_scale_m_per_unit <= 0.0:
            raise ValueError("coordinate_scale_m_per_unit 必须大于0。")
        if self.intersection_samples < 2_000:
            raise ValueError("intersection_samples 至少为2000，才能稳定求道路交点。")
        if self.segments_per_arm != 4:
            raise ValueError("当前任务定义要求每条道路分支固定布置4个巡检点。")
        if not math.isclose(
            self.arm_length_m,
            self.segments_per_arm * self.segment_length_m,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("arm_length_m 必须等于 segments_per_arm × segment_length_m。")
        if (
            self.min_point_spacing_m <= 0.0
            or self.min_global_euclidean_spacing_m <= 0.0
            or self.crop_margin_m <= 0.0
        ):
            raise ValueError("巡检点最小间距和DEM裁剪余量必须大于0。")
        weights = np.array(
            [
                self.slope_weight,
                self.overhead_relief_weight,
                self.roughness_weight,
                self.curvature_weight,
            ],
            dtype=np.float64,
        )
        if np.any(weights < 0.0) or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-9):
            raise ValueError("四项风险权重必须非负且总和为1。")
        expected_points = 4 * self.segments_per_arm
        if self.high_priority_count + self.medium_priority_count + self.low_priority_count != expected_points:
            raise ValueError("高/中/低优先级数量总和必须等于16。")
        if not 0.0 <= self.risk_percentile_low < self.risk_percentile_high <= 100.0:
            raise ValueError("风险归一化分位数范围无效。")
        if self.wind_spacing_m <= 0.0 or self.wind_height_agl_m <= 0.0:
            raise ValueError("风场采样间距和离地高度必须大于0。")
        if not 0.0 <= self.wind_min_speed_mps <= self.wind_max_speed_mps:
            raise ValueError("风速上下限无效。")
        worst_horizontal = self.wind_max_speed_mps * self.wind_domain_scale_max
        worst_vertical = 0.55 * self.wind_domain_scale_max + self.wind_vertical_bias_max_mps
        if math.hypot(worst_horizontal, worst_vertical) > 8.0 + 1e-6:
            raise ValueError("基础风场在最强域随机化后会超过8 m/s起降限制。")


@dataclass
class TrainingScenario:
    """可直接交给 PPO v2 的固定训练场景。"""

    terrain: np.ndarray
    start_pos: np.ndarray
    inspection_points: np.ndarray
    priorities: np.ndarray
    service_times_s: np.ndarray
    road_1: np.ndarray
    road_2: np.ndarray
    point_arm_ids: np.ndarray
    point_segment_ids: np.ndarray
    point_along_arm_distances_m: np.ndarray
    risk_scores: np.ndarray
    risk_components: np.ndarray
    risk_components_raw: np.ndarray
    wind_positions: np.ndarray
    wind_vectors: np.ndarray
    uniform_wind_vector: np.ndarray
    witness_order: np.ndarray
    source_affine: np.ndarray
    local_affine: np.ndarray
    crop_origin_global_pixel: np.ndarray
    airport_global_pixel: np.ndarray
    airport_utm: np.ndarray
    coordinate_scale_m_per_unit: float
    crs: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    scenario_hash: str = ""

    @property
    def points(self) -> np.ndarray:
        return self.inspection_points

    @property
    def wind_data(self) -> Dict[str, np.ndarray]:
        return {
            "positions": self.wind_positions.copy(),
            "vectors": self.wind_vectors.copy(),
            "uniform_vector": self.uniform_wind_vector.copy(),
        }

    def as_training_inputs(self) -> Dict[str, Any]:
        """返回训练器需要的字段，并显式声明12.5米尺度与逐点拍摄时间。"""

        return {
            "start": self.start_pos.copy(),
            "start_pos": self.start_pos.copy(),
            "points": self.inspection_points.copy(),
            "priorities": self.priorities.copy(),
            "terrain": self.terrain.copy(),
            "wind_data": self.wind_data,
            "service_times_s": self.service_times_s.copy(),
            "coordinate_scale_m_per_unit": float(self.coordinate_scale_m_per_unit),
            "cfg": {
                "coordinate_scale_m_per_unit": float(self.coordinate_scale_m_per_unit),
                "service_times_s": self.service_times_s.copy(),
                "point_z_mode": "terrain",
                "terrain_clearance_m": float(self.metadata.get("wind_height_agl_m", 18.0)),
                "return_to_start": True,
            },
        }

    def local_pixel_to_global_pixel(self, xyz: Sequence[Sequence[float]]) -> np.ndarray:
        values = _as_points(xyz)
        result = values.copy()
        result[:, 0] += float(self.crop_origin_global_pixel[0])
        result[:, 1] += float(self.crop_origin_global_pixel[1])
        return _restore_point_shape(xyz, result)

    def global_pixel_to_local_pixel(self, xyz: Sequence[Sequence[float]]) -> np.ndarray:
        values = _as_points(xyz)
        result = values.copy()
        result[:, 0] -= float(self.crop_origin_global_pixel[0])
        result[:, 1] -= float(self.crop_origin_global_pixel[1])
        return _restore_point_shape(xyz, result)

    def local_pixel_to_utm(self, xyz: Sequence[Sequence[float]]) -> np.ndarray:
        values = _as_points(xyz)
        result = values.copy()
        # 模型整数索引表示像元中心；GeoTIFF affine整数位置表示像元左上角。
        result[:, :2] = _affine_forward(values[:, :2] + 0.5, self.local_affine)
        return _restore_point_shape(xyz, result)

    def utm_to_local_pixel(self, xyz: Sequence[Sequence[float]]) -> np.ndarray:
        values = _as_points(xyz)
        result = values.copy()
        result[:, :2] = _affine_inverse(values[:, :2], self.local_affine) - 0.5
        return _restore_point_shape(xyz, result)


def _as_points(values: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError("坐标必须是长度2/3的向量或形状为[N,2/3]的数组。")
    if not np.all(np.isfinite(array)):
        raise ValueError("坐标中包含NaN或Inf。")
    return array


def _restore_point_shape(original: Any, values: np.ndarray) -> np.ndarray:
    return values[0] if np.asarray(original).ndim == 1 else values


def _affine_forward(xy: np.ndarray, affine: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f = np.asarray(affine, dtype=np.float64).reshape(6)
    x = a * xy[:, 0] + b * xy[:, 1] + c
    y = d * xy[:, 0] + e * xy[:, 1] + f
    return np.column_stack([x, y])


def _affine_inverse(xy: np.ndarray, affine: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f = np.asarray(affine, dtype=np.float64).reshape(6)
    determinant = a * e - b * d
    if abs(float(determinant)) <= 1e-12:
        raise ValueError("仿射变换不可逆。")
    # 2×2显式逆避免Windows环境中rasterio/GDAL与BLAS运行库首次加载发生冲突。
    shifted_x = xy[:, 0] - c
    shifted_y = xy[:, 1] - f
    col = (e * shifted_x - b * shifted_y) / determinant
    row = (-d * shifted_x + a * shifted_y) / determinant
    return np.column_stack([col, row])


def enu_wind_to_model(vectors_enu: np.ndarray) -> np.ndarray:
    """把东-北-上风矢量转换为模型使用的东-南-上风矢量。"""

    vectors = np.asarray(vectors_enu, dtype=np.float32)
    if vectors.shape[-1] != 3:
        raise ValueError("风矢量最后一维必须为3。")
    result = vectors.copy()
    result[..., 1] *= -1.0
    return result


def model_wind_to_enu(vectors_model: np.ndarray) -> np.ndarray:
    """东-南-上到东-北-上的逆变换。"""

    return enu_wind_to_model(vectors_model)


def _road_xy(rows: int, cols: int, road_id: int, n_samples: int) -> np.ndarray:
    """复用现有可视化文件的双国道路形公式，仅去除PyVista依赖。"""

    t = np.linspace(0.0, 1.0, int(n_samples), dtype=np.float64)
    if road_id == 1:
        x = (0.08 + 0.84 * t) * (cols - 1)
        y = (
            (0.22 + 0.58 * t) * (rows - 1)
            + 0.10 * (rows - 1) * np.sin(2.0 * np.pi * t + 0.4)
            + 0.04 * (rows - 1) * np.sin(7.0 * np.pi * t + 0.7)
        )
    elif road_id == 2:
        x = (0.12 + 0.76 * t) * (cols - 1)
        y = (
            (0.78 - 0.60 * t) * (rows - 1)
            + 0.06 * (rows - 1) * np.sin(2.6 * np.pi * t + 1.1)
            - 0.03 * (rows - 1) * np.sin(5.0 * np.pi * t)
        )
    else:
        raise ValueError("road_id 只能为1或2。")
    return np.column_stack([x, np.clip(y, 0.0, rows - 1)])


def _find_intersection(road_1_xy: np.ndarray, road_2_xy: np.ndarray) -> Tuple[np.ndarray, int, int, float]:
    """利用道路X单调性进行高密度最近求交，返回两中心线样本中点。"""

    y2_at_x1 = np.interp(road_1_xy[:, 0], road_2_xy[:, 0], road_2_xy[:, 1])
    delta = road_1_xy[:, 1] - y2_at_x1
    sign_changes = np.flatnonzero(delta[:-1] * delta[1:] <= 0.0)
    if sign_changes.size:
        candidate = sign_changes[np.argmin(np.abs(delta[sign_changes]))]
        d0, d1 = float(delta[candidate]), float(delta[candidate + 1])
        fraction = abs(d0) / max(abs(d0) + abs(d1), 1e-12)
        x = float((1.0 - fraction) * road_1_xy[candidate, 0] + fraction * road_1_xy[candidate + 1, 0])
        y1 = float((1.0 - fraction) * road_1_xy[candidate, 1] + fraction * road_1_xy[candidate + 1, 1])
        y2 = float(np.interp(x, road_2_xy[:, 0], road_2_xy[:, 1]))
        point = np.array([x, 0.5 * (y1 + y2)], dtype=np.float64)
    else:
        candidate = int(np.argmin(np.abs(delta)))
        x = float(road_1_xy[candidate, 0])
        point = np.array([x, 0.5 * (road_1_xy[candidate, 1] + y2_at_x1[candidate])])
    idx1 = int(np.argmin(np.linalg.norm(road_1_xy - point, axis=1)))
    idx2 = int(np.argmin(np.linalg.norm(road_2_xy - point, axis=1)))
    sample_gap_px = float(np.linalg.norm(road_1_xy[idx1] - road_2_xy[idx2]))
    return point, idx1, idx2, sample_gap_px


def _bilinear(dem: np.ndarray, xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    rows, cols = dem.shape
    x = np.clip(points[:, 0], 0.0, cols - 1.0)
    y = np.clip(points[:, 1], 0.0, rows - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, cols - 1)
    y1 = np.minimum(y0 + 1, rows - 1)
    tx = x - x0
    ty = y - y0
    return (
        dem[y0, x0] * (1.0 - tx) * (1.0 - ty)
        + dem[y0, x1] * tx * (1.0 - ty)
        + dem[y1, x0] * (1.0 - tx) * ty
        + dem[y1, x1] * tx * ty
    ).astype(np.float64)


def _fill_dem(dem: np.ndarray) -> np.ndarray:
    values = np.asarray(dem, dtype=np.float32)
    valid = np.isfinite(values)
    if not np.any(valid):
        raise ValueError("机场局部DEM没有有效高程。")
    if not np.all(valid):
        values = np.where(valid, values, float(np.nanmedian(values))).astype(np.float32)
    return values


def _road_with_height(road_xy_global: np.ndarray, origin: np.ndarray, terrain: np.ndarray) -> np.ndarray:
    local_xy = road_xy_global - origin.reshape(1, 2)
    inside = (
        (local_xy[:, 0] >= 0.0)
        & (local_xy[:, 0] <= terrain.shape[1] - 1.0)
        & (local_xy[:, 1] >= 0.0)
        & (local_xy[:, 1] <= terrain.shape[0] - 1.0)
    )
    local_xy = local_xy[inside]
    z = _bilinear(terrain, local_xy)
    return np.column_stack([local_xy, z]).astype(np.float32)


def _arm_from_road(
    road_xy: np.ndarray,
    intersection_idx: int,
    intersection_xy: np.ndarray,
    reverse: bool,
    scale: float,
    arm_length_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tail = road_xy[: intersection_idx + 1][::-1] if reverse else road_xy[intersection_idx:]
    if tail.size == 0:
        raise ValueError("道路交点无法分割为四条有效分支。")
    if np.linalg.norm(tail[0] - intersection_xy) > 1e-9:
        tail = np.vstack([intersection_xy, tail])
    else:
        tail = tail.copy()
        tail[0] = intersection_xy
    distances = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(tail, axis=0), axis=1) * scale)]
    )
    keep = distances <= arm_length_m + scale
    return tail[keep], distances[keep], np.flatnonzero(keep)


def _curvature_physical(points_xy: np.ndarray, scale: float) -> np.ndarray:
    xy = np.asarray(points_xy, dtype=np.float64) * scale
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    return (np.abs(dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-12)).astype(np.float64)


def _risk_raw_at_points(
    terrain: np.ndarray,
    points_local_xy: np.ndarray,
    curvature_values: np.ndarray,
    config: ScenarioConfig,
) -> np.ndarray:
    radius = max(1, int(math.ceil(config.risk_patch_radius_m / config.coordinate_scale_m_per_unit)))
    rows, cols = terrain.shape
    result = np.zeros((len(points_local_xy), 4), dtype=np.float64)
    for index, (x, y) in enumerate(points_local_xy):
        xi = int(np.clip(round(float(x)), 0, cols - 1))
        yi = int(np.clip(round(float(y)), 0, rows - 1))
        patch = terrain[
            max(0, yi - radius) : min(rows, yi + radius + 1),
            max(0, xi - radius) : min(cols, xi + radius + 1),
        ].astype(np.float64)
        gy, gx = np.gradient(patch, config.coordinate_scale_m_per_unit)
        slope = float(np.mean(np.hypot(gx, gy)))
        ground = float(_bilinear(terrain, np.array([[x, y]], dtype=np.float64))[0])
        overhead = max(float(np.percentile(patch, 95.0)) - ground, 0.0)
        roughness = float(np.std(patch))
        result[index] = [slope, overhead, roughness, float(curvature_values[index])]
    return result


def _percentile_normalize(values: np.ndarray, low: float, high: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    for column in range(values.shape[1]):
        lower, upper = np.percentile(values[:, column], [low, high])
        if upper > lower + 1e-12:
            result[:, column] = np.clip((values[:, column] - lower) / (upper - lower), 0.0, 1.0)
    return result


def _constrained_risk_selection(
    candidate_records: Sequence[Tuple[int, int, int, np.ndarray, float, float]],
    candidate_xy_global: np.ndarray,
    scores: np.ndarray,
    config: ScenarioConfig,
) -> Sequence[int]:
    """确定性约束回溯：每区间选一点，并优先尝试高风险候选。

    前向检查负责尽早剪掉会使其他区间无候选的分支；获得首个可行解后，再逐组
    尝试提升风险分数。这样不会用简单贪心把后续区间堵死，也无需引入外部求解器。
    """

    candidate_count = len(candidate_records)
    arm_ids = np.array([record[0] for record in candidate_records], dtype=np.int16)
    group_ids = np.array(
        [record[0] * config.segments_per_arm + record[1] for record in candidate_records],
        dtype=np.int16,
    )
    along_distances = np.array([record[5] for record in candidate_records], dtype=np.float64)

    # 候选规模约千点，完整布尔兼容矩阵仅占约1MB，可显著简化回溯中的前向检查。
    deltas = candidate_xy_global[:, None, :] - candidate_xy_global[None, :, :]
    euclidean_m = np.linalg.norm(deltas, axis=2) * config.coordinate_scale_m_per_unit
    compatible = euclidean_m >= config.min_global_euclidean_spacing_m - 1e-6
    same_arm = arm_ids[:, None] == arm_ids[None, :]
    along_ok = (
        np.abs(along_distances[:, None] - along_distances[None, :])
        >= config.min_point_spacing_m - 1e-6
    )
    compatible &= (~same_arm) | along_ok
    compatible[group_ids[:, None] == group_ids[None, :]] = False

    group_count = 4 * config.segments_per_arm
    domains: Dict[int, np.ndarray] = {}
    for group in range(group_count):
        indices = np.flatnonzero(group_ids == group)
        if indices.size == 0:
            raise ValueError(f"巡检点约束组{group}没有候选点。")
        # 风险分数优先；同分时用沿臂距离和原始索引保证跨运行确定性。
        ordered = sorted(
            indices.tolist(),
            key=lambda idx: (-float(scores[idx]), float(along_distances[idx]), int(idx)),
        )
        domains[group] = np.asarray(ordered, dtype=np.int32)

    # 静态冲突压力用于MRV平局：优先处理与其他组兼容比例最低的组。
    conflict_pressure: Dict[int, float] = {}
    for group, domain in domains.items():
        pressure = 0.0
        for other_group, other_domain in domains.items():
            if other_group == group:
                continue
            pressure += 1.0 - float(np.mean(compatible[np.ix_(domain, other_domain)]))
        conflict_pressure[group] = pressure

    visited_nodes = 0

    def propagate(domain_map: Dict[int, np.ndarray]) -> Optional[Dict[int, np.ndarray]]:
        """弧一致性：删除对任一其他组都没有兼容搭档的候选。"""

        reduced = {group: domain.copy() for group, domain in domain_map.items()}
        changed = True
        while changed:
            changed = False
            groups = sorted(reduced)
            for group in groups:
                domain = reduced[group]
                keep = np.ones(domain.size, dtype=bool)
                for other_group in groups:
                    if other_group == group:
                        continue
                    other_domain = reduced[other_group]
                    keep &= np.any(compatible[np.ix_(domain, other_domain)], axis=1)
                    if not np.any(keep):
                        return None
                if int(np.sum(keep)) != domain.size:
                    reduced[group] = domain[keep]
                    changed = True
        return reduced

    def search(
        remaining: Dict[int, np.ndarray], assignments: Dict[int, int]
    ) -> Optional[Dict[int, int]]:
        nonlocal visited_nodes
        visited_nodes += 1
        if not remaining:
            return dict(assignments)
        # MRV + 冲突压力是确定性的fail-first策略，不改变风险候选的尝试顺序。
        group = min(
            remaining,
            key=lambda item: (
                int(remaining[item].size),
                -float(conflict_pressure[item]),
                int(item),
            ),
        )
        other_groups = [item for item in remaining if item != group]
        # 先尝试对其余区间限制最少的候选，得到可行骨架后再做风险坐标上升。
        candidate_order = sorted(
            remaining[group].tolist(),
            key=lambda candidate: (
                -min(
                    int(np.sum(compatible[int(candidate), remaining[other]]))
                    for other in other_groups
                )
                if other_groups
                else 0,
                -sum(
                    int(np.sum(compatible[int(candidate), remaining[other]]))
                    for other in other_groups
                ),
                -float(scores[int(candidate)]),
                int(candidate),
            ),
        )
        for candidate in candidate_order:
            next_domains: Dict[int, np.ndarray] = {}
            feasible = True
            for other_group, other_domain in remaining.items():
                if other_group == group:
                    continue
                filtered = other_domain[compatible[int(candidate), other_domain]]
                if filtered.size == 0:
                    feasible = False
                    break
                next_domains[other_group] = filtered
            if not feasible:
                continue
            propagated = propagate(next_domains)
            if propagated is None:
                continue
            assignments[group] = int(candidate)
            result = search(propagated, assignments)
            if result is not None:
                return result
            del assignments[group]
        return None

    initial_domains = propagate(domains)
    solution = None if initial_domains is None else search(initial_domains, {})
    if solution is None:
        raise ValueError(
            "16个道路区间不存在同时满足同臂150米和全局"
            f"{config.min_global_euclidean_spacing_m:.1f}米约束的巡检点组合。"
        )

    # 坐标上升：逐组替换为风险更高且与当前其余15点兼容的候选，直到稳定。
    improved = True
    while improved:
        improved = False
        for group in range(group_count):
            current = solution[group]
            for candidate in domains[group]:
                if float(scores[candidate]) <= float(scores[current]) + 1e-12:
                    break
                if all(
                    bool(compatible[int(candidate), int(other_candidate)])
                    for other_group, other_candidate in solution.items()
                    if other_group != group
                ):
                    solution[group] = int(candidate)
                    improved = True
                    break

    selected = [solution[group] for group in range(group_count)]
    if len(selected) != group_count or len(set(selected)) != group_count:
        raise RuntimeError("约束回溯返回了缺失或重复的巡检点。")
    return selected


def _select_inspection_points(
    arms: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    terrain: np.ndarray,
    crop_origin: np.ndarray,
    config: ScenarioConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_records = []
    road_curvatures = [_curvature_physical(arm[0], config.coordinate_scale_m_per_unit) for arm in arms]
    for arm_id, ((arm_xy, distances, _), curvature) in enumerate(zip(arms, road_curvatures)):
        for segment_id in range(config.segments_per_arm):
            low = segment_id * config.segment_length_m
            high = (segment_id + 1) * config.segment_length_m
            mask = (distances > low + 1e-6) & (distances <= high + 1e-6)
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                raise ValueError(f"道路分支{arm_id}的第{segment_id + 1}个200米区间没有候选点。")
            for idx in indices:
                candidate_records.append(
                    (arm_id, segment_id, int(idx), arm_xy[idx], curvature[idx], distances[idx])
                )

    candidate_xy_global = np.stack([record[3] for record in candidate_records])
    candidate_xy_local = candidate_xy_global - crop_origin.reshape(1, 2)
    candidate_curvature = np.array([record[4] for record in candidate_records], dtype=np.float64)
    raw = _risk_raw_at_points(terrain, candidate_xy_local, candidate_curvature, config)
    normalized = _percentile_normalize(raw, config.risk_percentile_low, config.risk_percentile_high)
    weights = np.array(
        [config.slope_weight, config.overhead_relief_weight, config.roughness_weight, config.curvature_weight],
        dtype=np.float64,
    )
    scores = normalized @ weights

    selected = list(
        _constrained_risk_selection(
            candidate_records, candidate_xy_global, scores, config
        )
    )
    points_xy_local = candidate_xy_local[selected]
    point_z = _bilinear(terrain, points_xy_local)
    points = np.column_stack([points_xy_local, point_z]).astype(np.float32)
    arm_ids = np.array([candidate_records[idx][0] for idx in selected], dtype=np.int16)
    segment_ids = np.array([candidate_records[idx][1] for idx in selected], dtype=np.int16)
    along_distances = np.array([candidate_records[idx][5] for idx in selected], dtype=np.float32)
    return (
        points,
        arm_ids,
        segment_ids,
        along_distances,
        scores[selected].astype(np.float32),
        normalized[selected].astype(np.float32),
        raw[selected].astype(np.float32),
    )


def _assign_priorities(scores: np.ndarray, config: ScenarioConfig) -> np.ndarray:
    priorities = np.ones(len(scores), dtype=np.int32)
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    priorities[order[: config.high_priority_count]] = 3
    middle_end = config.high_priority_count + config.medium_priority_count
    priorities[order[config.high_priority_count : middle_end]] = 2
    return priorities


def _build_wind_field(terrain: np.ndarray, start_pos: np.ndarray, config: ScenarioConfig):
    spacing = max(1, int(round(config.wind_spacing_m / config.coordinate_scale_m_per_unit)))
    xs = np.unique(np.append(np.arange(0, terrain.shape[1], spacing), terrain.shape[1] - 1)).astype(int)
    ys = np.unique(np.append(np.arange(0, terrain.shape[0], spacing), terrain.shape[0] - 1)).astype(int)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    z = _bilinear(terrain, xy) + config.wind_height_agl_m
    positions = np.column_stack([xy, z]).astype(np.float32)

    gy, gx = np.gradient(terrain.astype(np.float64), config.coordinate_scale_m_per_unit)
    rng = np.random.default_rng(config.wind_seed)
    elevation_low = float(np.percentile(terrain, 5.0))
    elevation_span = max(float(np.percentile(terrain, 95.0)) - elevation_low, 1.0)
    vectors = np.zeros((len(positions), 3), dtype=np.float32)
    for index, (x, y, ground_plus_clearance) in enumerate(positions):
        xi = int(np.clip(round(float(x)), 0, terrain.shape[1] - 1))
        yi = int(np.clip(round(float(y)), 0, terrain.shape[0] - 1))
        slope_x, slope_y = float(gx[yi, xi]), float(gy[yi, xi])
        direction = np.array(
            [1.0 - 0.22 * slope_x, 0.18 - 0.22 * slope_y, 0.0], dtype=np.float64
        )
        direction[:2] /= max(float(np.linalg.norm(direction[:2])), 1e-9)
        relative_height = np.clip(
            (float(ground_plus_clearance) - config.wind_height_agl_m - elevation_low) / elevation_span,
            0.0,
            1.0,
        )
        speed = 3.0 + 1.5 * relative_height + 0.35 * min(math.hypot(slope_x, slope_y), 2.0)
        speed += float(rng.uniform(-0.25, 0.25))
        speed = float(np.clip(speed, config.wind_min_speed_mps, config.wind_max_speed_mps))
        vertical = float(np.clip(0.18 * (slope_x * direction[0] + slope_y * direction[1]), -0.55, 0.55))
        vectors[index] = [direction[0] * speed, direction[1] * speed, vertical]

    # 把机场本身加入支持点；位置严格是地形+18米，不使用可视化的95米抬高量。
    start_wind_position = start_pos.copy().astype(np.float32)
    start_wind_position[2] = float(_bilinear(terrain, start_pos[:2].reshape(1, 2))[0]) + config.wind_height_agl_m
    nearest = int(np.argmin(np.linalg.norm((positions[:, :2] - start_pos[:2]) * config.coordinate_scale_m_per_unit, axis=1)))
    start_vector = vectors[nearest].copy()
    start_norm = float(np.linalg.norm(start_vector))
    if start_norm > config.wind_takeoff_base_limit_mps:
        start_vector *= config.wind_takeoff_base_limit_mps / start_norm
    positions = np.vstack([positions, start_wind_position]).astype(np.float32)
    vectors = np.vstack([vectors, start_vector]).astype(np.float32)
    uniform = np.mean(vectors, axis=0).astype(np.float32)
    return positions, vectors, uniform


def _nearest_neighbor_2opt(start_xy: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    remaining = set(range(len(points_xy)))
    order = []
    current = np.asarray(start_xy, dtype=np.float64)
    while remaining:
        next_idx = min(remaining, key=lambda idx: (float(np.linalg.norm(points_xy[idx] - current)), idx))
        order.append(next_idx)
        remaining.remove(next_idx)
        current = points_xy[next_idx]

    def tour_length(sequence):
        route = np.vstack([start_xy, points_xy[sequence], start_xy])
        return float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))

    best = order
    improved = True
    while improved:
        improved = False
        current_length = tour_length(best)
        for left in range(0, len(best) - 1):
            for right in range(left + 2, len(best) + 1):
                candidate = best[:left] + best[left:right][::-1] + best[right:]
                candidate_length = tour_length(candidate)
                if candidate_length + 1e-9 < current_length:
                    best = candidate
                    current_length = candidate_length
                    improved = True
    return np.asarray(best, dtype=np.int32)


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(digest: "hashlib._Hash", name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())


def compute_scenario_hash(scenario: TrainingScenario) -> str:
    digest = hashlib.sha256()
    for name in (
        "terrain",
        "start_pos",
        "inspection_points",
        "priorities",
        "service_times_s",
        "road_1",
        "road_2",
        "point_arm_ids",
        "point_segment_ids",
        "point_along_arm_distances_m",
        "risk_scores",
        "risk_components",
        "risk_components_raw",
        "wind_positions",
        "wind_vectors",
        "uniform_wind_vector",
        "witness_order",
        "source_affine",
        "local_affine",
        "crop_origin_global_pixel",
        "airport_global_pixel",
        "airport_utm",
    ):
        _array_digest(digest, name, np.asarray(getattr(scenario, name)))
    stable_metadata = {
        key: value
        for key, value in scenario.metadata.items()
        # 场景身份依赖DEM内容摘要和生成数据，不依赖同一文件放在哪个目录。
        if key
        not in {
            "created_at",
            "npz_path",
            "json_path",
            "dem_source",
            "dem_source_absolute",
        }
    }
    digest.update(str(float(scenario.coordinate_scale_m_per_unit)).encode("ascii"))
    digest.update(scenario.crs.encode("utf-8"))
    digest.update(json.dumps(stable_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def build_training_scenario(
    dem_path: Union[str, Path] = DEFAULT_DEM_PATH,
    config: Optional[ScenarioConfig] = None,
) -> TrainingScenario:
    """从原始GeoTIFF一次性构建固定16点场景。"""

    cfg = config or ScenarioConfig()
    cfg.validate()
    if rasterio is None or Window is None:
        raise ImportError("构建真实尺度场景需要 rasterio。")
    source_path = Path(dem_path)
    if not source_path.exists():
        raise FileNotFoundError(f"DEM文件不存在：{source_path}")

    with rasterio.open(source_path) as src:
        rows, cols = src.shape
        crs = str(src.crs)
        source_affine = np.array(
            [src.transform.a, src.transform.b, src.transform.c, src.transform.d, src.transform.e, src.transform.f],
            dtype=np.float64,
        )
        resolution = (abs(float(src.transform.a)), abs(float(src.transform.e)))
        if crs != cfg.expected_crs:
            raise ValueError(f"DEM坐标系应为{cfg.expected_crs}，当前为{crs}。")
        if not np.allclose(resolution, cfg.coordinate_scale_m_per_unit, atol=1e-6):
            raise ValueError(
                f"DEM像元应为{cfg.coordinate_scale_m_per_unit}米，当前为{resolution}。"
            )

        road_1_global_xy = _road_xy(rows, cols, 1, cfg.intersection_samples)
        road_2_global_xy = _road_xy(rows, cols, 2, cfg.intersection_samples)
        airport_global_xy, idx1, idx2, sample_gap_px = _find_intersection(
            road_1_global_xy, road_2_global_xy
        )
        half_width_px = int(
            math.ceil((cfg.arm_length_m + cfg.crop_margin_m) / cfg.coordinate_scale_m_per_unit)
        )
        col0 = max(0, int(math.floor(airport_global_xy[0])) - half_width_px)
        row0 = max(0, int(math.floor(airport_global_xy[1])) - half_width_px)
        col1 = min(cols, int(math.ceil(airport_global_xy[0])) + half_width_px + 1)
        row1 = min(rows, int(math.ceil(airport_global_xy[1])) + half_width_px + 1)
        window = Window(col0, row0, col1 - col0, row1 - row0)
        terrain_band = src.read(1, window=window, masked=True).astype(np.float32)
        terrain = _fill_dem(terrain_band.filled(np.nan))
        local_transform = src.window_transform(window)
        local_affine = np.array(
            [
                local_transform.a,
                local_transform.b,
                local_transform.c,
                local_transform.d,
                local_transform.e,
                local_transform.f,
            ],
            dtype=np.float64,
        )

    crop_origin = np.array([col0, row0], dtype=np.float64)
    airport_local_xy = airport_global_xy - crop_origin
    airport_ground = float(_bilinear(terrain, airport_local_xy.reshape(1, 2))[0])
    # 比地面抬高1毫米，避免float32坐标回插时因舍入被误判为“低于地面”。
    start_pos = np.array(
        [airport_local_xy[0], airport_local_xy[1], airport_ground + 1e-3], dtype=np.float32
    )

    arms = (
        _arm_from_road(road_1_global_xy, idx1, airport_global_xy, True, cfg.coordinate_scale_m_per_unit, cfg.arm_length_m),
        _arm_from_road(road_1_global_xy, idx1, airport_global_xy, False, cfg.coordinate_scale_m_per_unit, cfg.arm_length_m),
        _arm_from_road(road_2_global_xy, idx2, airport_global_xy, True, cfg.coordinate_scale_m_per_unit, cfg.arm_length_m),
        _arm_from_road(road_2_global_xy, idx2, airport_global_xy, False, cfg.coordinate_scale_m_per_unit, cfg.arm_length_m),
    )
    (
        points,
        arm_ids,
        segment_ids,
        along_distances,
        risk_scores,
        risk_components,
        risk_components_raw,
    ) = _select_inspection_points(arms, terrain, crop_origin, cfg)
    priorities = _assign_priorities(risk_scores, cfg)
    service_times = np.full((len(points),), cfg.service_time_s, dtype=np.float32)
    road_1 = _road_with_height(road_1_global_xy, crop_origin, terrain)
    road_2 = _road_with_height(road_2_global_xy, crop_origin, terrain)
    wind_positions, wind_vectors, uniform_wind = _build_wind_field(terrain, start_pos, cfg)
    witness_order = _nearest_neighbor_2opt(start_pos[:2], points[:, :2])
    airport_utm = _affine_forward(airport_local_xy.reshape(1, 2) + 0.5, local_affine)[0]
    pairwise_xy_m = np.linalg.norm(
        (points[:, None, :2] - points[None, :, :2]) * cfg.coordinate_scale_m_per_unit,
        axis=2,
    )
    pairwise_xy_m[pairwise_xy_m <= 1e-9] = np.inf
    global_min_spacing = float(np.min(pairwise_xy_m))
    same_arm_adjacent_spacings = []
    for arm_id in range(4):
        arm_distances = np.sort(along_distances[arm_ids == arm_id].astype(np.float64))
        same_arm_adjacent_spacings.extend(np.diff(arm_distances).tolist())
    same_arm_min_spacing = float(min(same_arm_adjacent_spacings))
    if global_min_spacing < cfg.min_global_euclidean_spacing_m - 1e-5:
        raise RuntimeError("约束搜索输出违反全局欧氏间距，拒绝构建场景。")
    if same_arm_min_spacing < cfg.min_point_spacing_m - 1e-5:
        raise RuntimeError("约束搜索输出违反同臂沿路间距，拒绝构建场景。")

    metadata = {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "dem_source": source_path.as_posix(),
        "dem_source_absolute": str(source_path.resolve()),
        "dem_sha256": _file_sha256(source_path),
        "dem_crop_sha256": hashlib.sha256(np.ascontiguousarray(terrain).tobytes()).hexdigest(),
        "crs": crs,
        "source_shape": [int(rows), int(cols)],
        "crop_shape": [int(terrain.shape[0]), int(terrain.shape[1])],
        "crop_origin_global_pixel": crop_origin.tolist(),
        "source_affine": source_affine.tolist(),
        "local_affine": local_affine.tolist(),
        "pixel_coordinate_convention": (
            "integer model x/y denotes the GeoTIFF pixel center; UTM conversion applies +0.5 "
            "before affine and -0.5 after inverse"
        ),
        "airport_global_pixel": airport_global_xy.tolist(),
        "airport_utm": airport_utm.tolist(),
        "intersection_centerline_sample_gap_m": sample_gap_px * cfg.coordinate_scale_m_per_unit,
        "coordinate_scale_m_per_unit": cfg.coordinate_scale_m_per_unit,
        "road_definition": "v2_3d_visualization.generate_road_line + generate_cross_road_line",
        "road_parameters": {
            "intersection_samples": cfg.intersection_samples,
            "arm_length_m": cfg.arm_length_m,
            "segments_per_arm": cfg.segments_per_arm,
            "segment_length_m": cfg.segment_length_m,
            "min_point_spacing_m": cfg.min_point_spacing_m,
            "min_global_euclidean_spacing_m": cfg.min_global_euclidean_spacing_m,
        },
        "spacing_rule": (
            "same_arm_along_road_spacing_m>=150 and global_euclidean_spacing_m>=130; "
            "the global threshold is a transparent geometric compromise"
        ),
        "same_arm_min_along_road_spacing_m": same_arm_min_spacing,
        "global_min_euclidean_spacing_m": global_min_spacing,
        "spacing_geometry_note": (
            "With the unchanged two-road geometry, four points constrained to the first 0-200 m "
            "arm intervals can achieve at most about 138.70 m global minimum Euclidean spacing; "
            "therefore a global 150 m rule is geometrically infeasible and 130 m is used."
        ),
        "point_selection_method": (
            "deterministic arc-consistency + MRV forward-checking backtracking, followed by "
            "constraint-preserving risk coordinate ascent"
        ),
        "risk_component_names": list(RISK_COMPONENT_NAMES),
        "risk_weights": {
            "terrain_slope": cfg.slope_weight,
            "overhead_relief": cfg.overhead_relief_weight,
            "roughness": cfg.roughness_weight,
            "road_curvature": cfg.curvature_weight,
        },
        "risk_normalization_percentiles": [cfg.risk_percentile_low, cfg.risk_percentile_high],
        "priority_counts": {"high": cfg.high_priority_count, "medium": cfg.medium_priority_count, "low": cfg.low_priority_count},
        "wind_axes": "X=east, Y=south, Z=up",
        "wind_spacing_m": cfg.wind_spacing_m,
        "wind_height_agl_m": cfg.wind_height_agl_m,
        "wind_display_offset_m": 0.0,
        "wind_seed": cfg.wind_seed,
        "scenario_seed": cfg.seed,
        "scenario_config": asdict(cfg),
    }
    scenario = TrainingScenario(
        terrain=terrain.astype(np.float32),
        start_pos=start_pos,
        inspection_points=points,
        priorities=priorities,
        service_times_s=service_times,
        road_1=road_1,
        road_2=road_2,
        point_arm_ids=arm_ids,
        point_segment_ids=segment_ids,
        point_along_arm_distances_m=along_distances,
        risk_scores=risk_scores,
        risk_components=risk_components,
        risk_components_raw=risk_components_raw,
        wind_positions=wind_positions,
        wind_vectors=wind_vectors,
        uniform_wind_vector=uniform_wind,
        witness_order=witness_order,
        source_affine=source_affine,
        local_affine=local_affine,
        crop_origin_global_pixel=crop_origin,
        airport_global_pixel=airport_global_xy.astype(np.float64),
        airport_utm=airport_utm.astype(np.float64),
        coordinate_scale_m_per_unit=cfg.coordinate_scale_m_per_unit,
        crs=crs,
        metadata=metadata,
    )
    scenario.scenario_hash = compute_scenario_hash(scenario)
    return scenario


def _scenario_arrays(scenario: TrainingScenario) -> Dict[str, np.ndarray]:
    names = (
        "terrain",
        "start_pos",
        "inspection_points",
        "priorities",
        "service_times_s",
        "road_1",
        "road_2",
        "point_arm_ids",
        "point_segment_ids",
        "point_along_arm_distances_m",
        "risk_scores",
        "risk_components",
        "risk_components_raw",
        "wind_positions",
        "wind_vectors",
        "uniform_wind_vector",
        "witness_order",
        "source_affine",
        "local_affine",
        "crop_origin_global_pixel",
        "airport_global_pixel",
        "airport_utm",
    )
    return {name: np.asarray(getattr(scenario, name)) for name in names}


def _scenario_paths(path: Union[str, Path]) -> Tuple[Path, Path]:
    target = Path(path)
    if target.suffix.lower() == ".npz":
        return target, target.with_suffix(".json")
    if target.suffix.lower() == ".json":
        return target.with_suffix(".npz"), target
    return target.with_suffix(".npz"), target.with_suffix(".json")


def save_training_scenario(
    scenario: TrainingScenario, path: Union[str, Path]
) -> Tuple[Path, Path]:
    """同时保存机器可读NPZ和可审计JSON，并校验哈希未过期。"""

    npz_path, json_path = _scenario_paths(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    current_hash = compute_scenario_hash(scenario)
    if scenario.scenario_hash and scenario.scenario_hash != current_hash:
        raise ValueError("场景数组已变化但scenario_hash未更新，拒绝保存不一致文件。")
    scenario.scenario_hash = current_hash
    metadata_json = json.dumps(scenario.metadata, ensure_ascii=False, sort_keys=True)
    np.savez_compressed(
        npz_path,
        **_scenario_arrays(scenario),
        coordinate_scale_m_per_unit=np.array([scenario.coordinate_scale_m_per_unit], dtype=np.float64),
        crs=np.array(scenario.crs),
        scenario_hash=np.array(scenario.scenario_hash),
        metadata_json=np.array(metadata_json),
    )
    point_records = []
    for index in range(len(scenario.inspection_points)):
        point_records.append(
            {
                "index": index,
                "local_pixel_xyz": scenario.inspection_points[index].astype(float).tolist(),
                "global_pixel_xyz": np.asarray(
                    scenario.local_pixel_to_global_pixel(scenario.inspection_points[index])
                ).astype(float).tolist(),
                "utm_xyz": np.asarray(scenario.local_pixel_to_utm(scenario.inspection_points[index])).astype(float).tolist(),
                "road_arm": int(scenario.point_arm_ids[index]),
                "road_segment": int(scenario.point_segment_ids[index]),
                "along_arm_distance_m": float(scenario.point_along_arm_distances_m[index]),
                "priority": int(scenario.priorities[index]),
                "risk_score": float(scenario.risk_scores[index]),
                "risk_components_normalized": {
                    name: float(scenario.risk_components[index, component])
                    for component, name in enumerate(RISK_COMPONENT_NAMES)
                },
                "risk_components_raw": {
                    name: float(scenario.risk_components_raw[index, component])
                    for component, name in enumerate(RISK_COMPONENT_NAMES)
                },
                "service_time_s": float(scenario.service_times_s[index]),
            }
        )
    json_payload = {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "scenario_hash": scenario.scenario_hash,
        "npz_file": npz_path.name,
        "metadata": scenario.metadata,
        "start_local_pixel_xyz": scenario.start_pos.astype(float).tolist(),
        "start_global_pixel_xyz": np.asarray(scenario.local_pixel_to_global_pixel(scenario.start_pos)).astype(float).tolist(),
        "start_utm_xyz": np.asarray(scenario.local_pixel_to_utm(scenario.start_pos)).astype(float).tolist(),
        "witness_order": scenario.witness_order.astype(int).tolist(),
        "inspection_points": point_records,
    }
    json_path.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return npz_path, json_path


def load_training_scenario(path: Union[str, Path]) -> TrainingScenario:
    """加载NPZ/JSON同名前缀场景；JSON存在时同时交叉验证哈希。"""

    npz_path, json_path = _scenario_paths(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"场景NPZ不存在：{npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        kwargs = {name: np.asarray(data[name]).copy() for name in _scenario_arrays_placeholder()}
        scenario = TrainingScenario(
            **kwargs,
            coordinate_scale_m_per_unit=float(np.asarray(data["coordinate_scale_m_per_unit"]).reshape(-1)[0]),
            crs=str(data["crs"].item()),
            metadata=metadata,
            scenario_hash=str(data["scenario_hash"].item()),
        )
    actual_hash = compute_scenario_hash(scenario)
    if actual_hash != scenario.scenario_hash:
        raise ValueError("场景NPZ哈希校验失败，文件可能被修改或损坏。")
    if json_path.exists():
        json_hash = str(json.loads(json_path.read_text(encoding="utf-8")).get("scenario_hash", ""))
        if json_hash != scenario.scenario_hash:
            raise ValueError("场景JSON与NPZ的哈希不一致。")
    return scenario


def _scenario_arrays_placeholder() -> Tuple[str, ...]:
    return (
        "terrain",
        "start_pos",
        "inspection_points",
        "priorities",
        "service_times_s",
        "road_1",
        "road_2",
        "point_arm_ids",
        "point_segment_ids",
        "point_along_arm_distances_m",
        "risk_scores",
        "risk_components",
        "risk_components_raw",
        "wind_positions",
        "wind_vectors",
        "uniform_wind_vector",
        "witness_order",
        "source_affine",
        "local_affine",
        "crop_origin_global_pixel",
        "airport_global_pixel",
        "airport_utm",
    )


def preflight_scenario(
    scenario: TrainingScenario,
    ppo_cfg: Optional[Mapping[str, Any]] = None,
    *,
    randomized_scenarios: int = 8,
    minimum_initial_feasible_points: int = 12,
    strict: bool = True,
) -> Dict[str, Any]:
    """用PPO v2同一套动力学证明见证路线和随机化初始可行性。"""

    from uav_inspection.core import final_python_ppo_pointer as ppo

    inputs = scenario.as_training_inputs()
    cfg = dict(ppo_cfg or {})
    cfg.update(inputs["cfg"])
    cfg = ppo.resolve_config(cfg)
    state = ppo.build_episode(
        inputs["start_pos"], inputs["points"], inputs["terrain"], cfg, inputs["wind_data"], randomize=False
    )
    witness_error = None
    try:
        for action in scenario.witness_order.astype(int).tolist():
            state, _, done = ppo.step_env_improved(
                state, action, inputs["points"], inputs["priorities"], inputs["terrain"], cfg, inputs["wind_data"]
            )
            if done:
                raise RuntimeError("见证路线在覆盖全部巡检点前提前终止。")
        state, _, done = ppo.step_env_improved(
            state,
            len(inputs["points"]),
            inputs["points"],
            inputs["priorities"],
            inputs["terrain"],
            cfg,
            inputs["wind_data"],
        )
        if not done or state["termination_reason"] != "returned_full":
            raise RuntimeError("见证路线未在全覆盖后返航。")
    except Exception as exc:  # 将原因写入报告；strict模式随后抛出
        witness_error = f"{type(exc).__name__}: {exc}"

    randomized_counts = []
    randomization_records = []
    for index in range(int(randomized_scenarios)):
        rng = np.random.default_rng(int(cfg["seed"]) + 10_000 + index)
        randomized_state = ppo.build_episode(
            inputs["start_pos"],
            inputs["points"],
            inputs["terrain"],
            cfg,
            inputs["wind_data"],
            rng=rng,
            randomize=True,
        )
        _, masks = ppo._compute_action_context(randomized_state, inputs["priorities"])
        count = int(np.sum(masks["legal"][:-1]))
        randomized_counts.append(count)
        randomization_records.append(copy.deepcopy(randomized_state["episode_randomization"]))

    # 显式验证域随机化立方体的保守角点，避免8次随机抽样碰巧没有覆盖真正最差组合。
    theta = math.radians(float(cfg["wind_rotation_deg"]))
    cosine, sine = math.cos(theta), math.sin(theta)
    worst_vectors = np.asarray(inputs["wind_data"]["vectors"], dtype=np.float32).copy()
    x_component = worst_vectors[:, 0].copy()
    y_component = worst_vectors[:, 1].copy()
    worst_vectors[:, 0] = cosine * x_component - sine * y_component
    worst_vectors[:, 1] = sine * x_component + cosine * y_component
    worst_vectors *= float(cfg["wind_scale_max"])
    worst_vectors[:, 2] += float(cfg["wind_vertical_bias_mps"])
    worst_uniform = np.asarray(inputs["wind_data"]["uniform_vector"], dtype=np.float32).copy()
    uniform_x, uniform_y = float(worst_uniform[0]), float(worst_uniform[1])
    worst_uniform[0] = cosine * uniform_x - sine * uniform_y
    worst_uniform[1] = sine * uniform_x + cosine * uniform_y
    worst_uniform *= float(cfg["wind_scale_max"])
    worst_uniform[2] += float(cfg["wind_vertical_bias_mps"])
    worst_wind_data = {
        "positions": np.asarray(inputs["wind_data"]["positions"], dtype=np.float32),
        "vectors": worst_vectors,
        "uniform_vector": worst_uniform,
    }
    worst_cfg = dict(cfg)
    worst_cfg.update(
        {
            "initial_soc": float(cfg["initial_soc_min"]),
            "max_route_distance": float(cfg["max_route_distance"])
            * float(cfg["distance_budget_scale_min"]),
            "max_mission_time_s": float(cfg["max_mission_time_s"])
            * float(cfg["time_budget_scale_min"]),
        }
    )
    worst_state = ppo.build_episode(
        inputs["start_pos"],
        inputs["points"],
        inputs["terrain"],
        worst_cfg,
        worst_wind_data,
        randomize=False,
    )
    _, worst_masks = ppo._compute_action_context(worst_state, inputs["priorities"])
    worst_corner_feasible_count = int(np.sum(worst_masks["legal"][:-1]))

    nominal_ok = witness_error is None
    constraint_ok = nominal_ok and (
        state["total_energy_consumed"] <= state["energy_budget_wh"] + 1e-6
        and state["total_distance"] <= state["max_route_distance"] + 1e-6
        and state["total_time_s"] <= state["max_mission_time_s"] + 1e-6
    )
    randomized_ok = bool(randomized_counts) and min(randomized_counts) >= int(minimum_initial_feasible_points)
    worst_case_has_patrol_action = worst_corner_feasible_count > 0
    passed = nominal_ok and constraint_ok and randomized_ok and worst_case_has_patrol_action
    report = {
        "passed": bool(passed),
        "scenario_hash": scenario.scenario_hash,
        "witness_order": scenario.witness_order.astype(int).tolist(),
        "witness_error": witness_error,
        "nominal": {
            "returned_full": bool(nominal_ok and state.get("termination_reason") == "returned_full"),
            "visited_count": int(len(state.get("visited", []))),
            "energy_wh": float(state.get("total_energy_consumed", float("nan"))),
            "energy_budget_wh": float(state.get("energy_budget_wh", float("nan"))),
            "distance_m": float(state.get("total_distance", float("nan"))),
            "distance_budget_m": float(state.get("max_route_distance", float("nan"))),
            "time_s": float(state.get("total_time_s", float("nan"))),
            "time_budget_s": float(state.get("max_mission_time_s", float("nan"))),
            "constraint_violations": 0 if constraint_ok else 1,
        },
        "randomized_initial_feasible_counts": randomized_counts,
        "minimum_required_initial_feasible_points": int(minimum_initial_feasible_points),
        "randomization_records": randomization_records,
        "worst_case_has_patrol_action": worst_case_has_patrol_action,
        "worst_corner": {
            "initial_soc": float(worst_cfg["initial_soc"]),
            "distance_budget_scale": float(cfg["distance_budget_scale_min"]),
            "time_budget_scale": float(cfg["time_budget_scale_min"]),
            "wind_scale": float(cfg["wind_scale_max"]),
            "wind_rotation_deg": float(cfg["wind_rotation_deg"]),
            "wind_vertical_bias_mps": float(cfg["wind_vertical_bias_mps"]),
            "initial_feasible_patrol_points": worst_corner_feasible_count,
            "return_action_legal": bool(worst_masks["legal"][-1]),
        },
    }
    if strict and not passed:
        raise RuntimeError("PPO真实尺度场景预检失败：" + json.dumps(report, ensure_ascii=False))
    return report


__all__ = [
    "DEFAULT_DEM_PATH",
    "RISK_COMPONENT_NAMES",
    "SCENARIO_SCHEMA_VERSION",
    "ScenarioConfig",
    "TrainingScenario",
    "build_training_scenario",
    "compute_scenario_hash",
    "enu_wind_to_model",
    "load_training_scenario",
    "model_wind_to_enu",
    "preflight_scenario",
    "save_training_scenario",
]

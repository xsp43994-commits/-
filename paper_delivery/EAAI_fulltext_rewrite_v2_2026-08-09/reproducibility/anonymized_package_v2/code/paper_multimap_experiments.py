#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多地图泛化实验的数据获取、地图注册与程序化地形入口。

真实测试地图的选择只读取DEM和道路输入，不导入或运行任何待比较算法。
训练与正式评价仍由独立入口完成，避免地图筛选阶段发生结果泄漏。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = WORKSPACE_ROOT
DEFAULT_PROTOCOL = (
    ROOT
    / "paper_runs"
    / "protocols"
    / "multimap_generalization_v3_2"
    / "protocol.json"
)
# v3.2 复用经过父协议审计的地图与训练任务，但所有新训练、冻结和评价
# 结果均写入独立目录，避免覆盖 v3.1.17 的历史证据。
DEFAULT_OUTPUT_ROOT = ROOT / "paper_runs" / "multimap_v3_2"
DEFAULT_MAP_ROOT = ROOT / "map_data" / "multimap_v3_1"
DEFAULT_LEGACY_DEM = ROOT / "map_data" / "AP_15010_FBS_F2760_RT1.dem.tif"
PARENT_DIFFICULTY_PROTOCOL = (
    ROOT / "paper_runs" / "protocols" / "difficulty_test_v2_1" / "protocol.json"
)
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
HIGHWAY_TYPES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service",
    "track",
}

# Windows 文件索引、杀毒或监控读取可能在极短时间内占用状态文件。
# 此重试只覆盖原子替换的瞬时共享冲突，不改变训练、场景或评价逻辑。
ATOMIC_REPLACE_MAX_ATTEMPTS = 7
ATOMIC_REPLACE_INITIAL_BACKOFF_SECONDS = 0.05
RETRYABLE_WINDOWS_REPLACE_WINERRORS = {5, 32, 33}


def _canonical_hash(
    payload: Mapping[str, Any], excluded: Sequence[str] = ()
) -> str:
    normalized = {
        key: value for key, value in payload.items() if key not in set(excluded)
    }
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _minimum_valid_initial_soc(protocol: Mapping[str, Any]) -> float:
    """返回统一评价器允许的严格最小初始SOC，并验证协议内部一致性。"""

    safety = protocol["task_generation"]["evaluator_safety_bounds"]
    reserve = float(safety["battery_reserve_ratio"])
    margin = float(safety["initial_soc_strict_margin"])
    minimum = float(safety["minimum_initial_soc"])
    if not (0.0 <= reserve < 1.0 and margin > 0.0):
        raise RuntimeError("SOC安全边界配置无效。")
    if not math.isclose(
        minimum, reserve + margin, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("minimum_initial_soc必须等于返航储备率加严格裕量。")
    for section_name in (
        "mixed_budget_calibration",
        "geometry_budget_compensation",
        "single_constraint_budget_calibration",
    ):
        lower = float(
            protocol["task_generation"][section_name]["parameter_bounds"][
                "initial_soc"
            ][0]
        )
        if lower < minimum - 1e-12:
            raise RuntimeError(
                f"{section_name}的initial_soc下界低于统一安全边界。"
            )
    return minimum


def _atomic_replace(temporary: Path, path: Path) -> None:
    """在 Windows 瞬时文件锁下有限重试原子替换。"""

    for attempt in range(ATOMIC_REPLACE_MAX_ATTEMPTS):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            retryable = getattr(exc, "winerror", None) in (
                RETRYABLE_WINDOWS_REPLACE_WINERRORS
            )
            if not retryable or attempt + 1 >= ATOMIC_REPLACE_MAX_ATTEMPTS:
                raise
            delay = ATOMIC_REPLACE_INITIAL_BACKOFF_SECONDS * (2**attempt)
            time.sleep(delay)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        _atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}必须是JSON对象。")
            rows.append(value)
    return rows


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> Dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = str(protocol.get("protocol_hash", ""))
    actual = _canonical_hash(protocol, excluded=("protocol_hash",))
    if expected != actual:
        raise RuntimeError(
            f"多地图协议哈希不一致：expected={expected}, actual={actual}"
        )
    regions = list(protocol.get("regions", ()))
    design = dict(protocol["real_dem_design"])
    if len(regions) != int(design["count"]):
        raise RuntimeError("真实DEM地域数量与协议不一致。")
    if sum(region["group"] == "china" for region in regions) != int(
        design["china_count"]
    ):
        raise RuntimeError("中国DEM地域数量与协议不一致。")
    if sum(region["group"] == "global" for region in regions) != int(
        design["global_count"]
    ):
        raise RuntimeError("全球DEM地域数量与协议不一致。")
    _minimum_valid_initial_soc(protocol)
    return protocol


def _asset_protocol_hash(protocol: Mapping[str, Any]) -> str:
    """返回地图/任务资产应匹配的协议身份；v3.2 可只读复用 v3.1.17 资产。"""

    return str(protocol.get("asset_parent_protocol_hash", protocol["protocol_hash"]))


def _optional_geo_imports() -> Tuple[Any, ...]:
    try:
        import rasterio
        import requests
        from pyproj import CRS, Transformer
        from rasterio.enums import Resampling
        from rasterio.transform import from_origin
        from rasterio.warp import reproject
        from shapely.geometry import LineString, box
        from shapely.ops import linemerge, transform as transform_geometry, unary_union
    except ImportError as exc:  # pragma: no cover - 由CLI给出环境指引
        raise ImportError(
            "DEM获取需要geo_env中的rasterio、requests、pyproj和shapely；"
            "请使用 D:\\anaconda3\\envs\\geo_env\\python.exe。"
        ) from exc
    return (
        rasterio,
        requests,
        CRS,
        Transformer,
        Resampling,
        from_origin,
        reproject,
        LineString,
        box,
        linemerge,
        transform_geometry,
        unary_union,
    )


def _tile_token(latitude: float, longitude: float) -> str:
    lat_floor = math.floor(float(latitude))
    lon_floor = math.floor(float(longitude))
    lat_text = f"{'N' if lat_floor >= 0 else 'S'}{abs(lat_floor):02d}_00"
    lon_text = f"{'E' if lon_floor >= 0 else 'W'}{abs(lon_floor):03d}_00"
    return f"Copernicus_DSM_COG_10_{lat_text}_{lon_text}_DEM"


def copernicus_tile_url(
    latitude: float, longitude: float, protocol: Mapping[str, Any]
) -> str:
    token = _tile_token(latitude, longitude)
    bucket = str(protocol["dem_source"]["aws_bucket"]).rstrip("/")
    return f"{bucket}/{token}/{token}.tif"


def candidate_centers(
    region: Mapping[str, Any], grid_side: int
) -> List[Dict[str, Any]]:
    west, south, east, north = (
        float(value) for value in region["bbox_wgs84"]
    )
    if not (west < east and south < north and int(grid_side) >= 2):
        raise ValueError(f"地域{region['region_id']}的候选网格无效。")
    # 避开区域边界，减少裁片越界和跨1度DEM瓦片的概率。
    lon_values = np.linspace(west, east, int(grid_side) + 2)[1:-1]
    lat_values = np.linspace(south, north, int(grid_side) + 2)[1:-1]
    records: List[Dict[str, Any]] = []
    index = 0
    for latitude in lat_values:
        for longitude in lon_values:
            records.append(
                {
                    "candidate_id": f"{region['region_id']}__c{index:03d}",
                    "region_id": str(region["region_id"]),
                    "longitude": float(longitude),
                    "latitude": float(latitude),
                }
            )
            index += 1
    return records


def _utm_epsg(longitude: float, latitude: float) -> int:
    zone = min(60, max(1, int(math.floor((longitude + 180.0) / 6.0)) + 1))
    return (32600 if latitude >= 0.0 else 32700) + zone


def _window_transform(
    longitude: float, latitude: float, size_m: float, resolution_m: float
) -> Tuple[Any, int, int, int]:
    (
        _rasterio,
        _requests,
        CRS,
        Transformer,
        _Resampling,
        from_origin,
        _reproject,
        *_rest,
    ) = _optional_geo_imports()
    epsg = _utm_epsg(longitude, latitude)
    transformer = Transformer.from_crs("EPSG:4326", CRS.from_epsg(epsg), always_xy=True)
    center_x, center_y = transformer.transform(longitude, latitude)
    width = int(round(float(size_m) / float(resolution_m)))
    height = width
    transform = from_origin(
        center_x - width * resolution_m / 2.0,
        center_y + height * resolution_m / 2.0,
        resolution_m,
        resolution_m,
    )
    return transform, height, width, epsg


def read_dem_crop(
    url_or_path: str,
    *,
    longitude: float,
    latitude: float,
    size_m: float,
    resolution_m: float,
) -> Dict[str, Any]:
    (
        rasterio,
        _requests,
        CRS,
        _Transformer,
        Resampling,
        _from_origin,
        reproject,
        *_rest,
    ) = _optional_geo_imports()
    dst_transform, height, width, epsg = _window_transform(
        longitude, latitude, size_m, resolution_m
    )
    destination = np.full((height, width), np.nan, dtype=np.float32)
    env_options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    }
    with rasterio.Env(**env_options):
        with rasterio.open(url_or_path) as source:
            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=dst_transform,
                dst_crs=CRS.from_epsg(epsg),
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            source_meta = {
                "source_crs": str(source.crs),
                "source_shape": [int(source.height), int(source.width)],
                "source_nodata": source.nodata,
                "source_dtype": str(source.dtypes[0]),
            }
    return {
        "terrain": destination,
        "transform": [
            float(dst_transform.a),
            float(dst_transform.b),
            float(dst_transform.c),
            float(dst_transform.d),
            float(dst_transform.e),
            float(dst_transform.f),
        ],
        "crs": f"EPSG:{epsg}",
        "source_meta": source_meta,
    }


def terrain_metrics(terrain: np.ndarray, resolution_m: float) -> Dict[str, float]:
    values = np.asarray(terrain, dtype=np.float64)
    finite = np.isfinite(values)
    nodata_fraction = 1.0 - float(np.mean(finite))
    if not np.any(finite):
        return {
            "nodata_fraction": 1.0,
            "relief_m": 0.0,
            "slope_median": 0.0,
            "slope_iqr": 0.0,
            "elevation_median": 0.0,
        }
    filled = values.copy()
    filled[~finite] = float(np.nanmedian(values))
    low, high = np.percentile(filled, [1.0, 99.0])
    gy, gx = np.gradient(filled, float(resolution_m))
    slope = np.hypot(gx, gy)
    slope_q25, slope_median, slope_q75 = np.percentile(
        slope[finite], [25.0, 50.0, 75.0]
    )
    return {
        "nodata_fraction": nodata_fraction,
        "relief_m": float(high - low),
        "slope_median": float(slope_median),
        "slope_iqr": float(slope_q75 - slope_q25),
        "elevation_median": float(np.median(filled[finite])),
    }


def _lon_lat_bbox(
    longitude: float, latitude: float, size_m: float
) -> Tuple[float, float, float, float]:
    half_lat = (float(size_m) / 2.0) / 111_320.0
    half_lon = half_lat / max(math.cos(math.radians(latitude)), 0.1)
    return (
        longitude - half_lon,
        latitude - half_lat,
        longitude + half_lon,
        latitude + half_lat,
    )


def fetch_osm_roads(
    bbox_wgs84: Sequence[float],
    *,
    timeout_s: float = 120.0,
    endpoints: Sequence[str] = OVERPASS_ENDPOINTS,
) -> Dict[str, Any]:
    (
        _rasterio,
        requests,
        _CRS,
        _Transformer,
        _Resampling,
        _from_origin,
        _reproject,
        *_rest,
    ) = _optional_geo_imports()
    west, south, east, north = (float(value) for value in bbox_wgs84)
    query = (
        f'[out:json][timeout:{int(timeout_s)}];'
        f'way["highway"]({south},{west},{north},{east});'
        "out tags geom;"
    )
    errors: List[str] = []
    for endpoint in endpoints:
        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                timeout=float(timeout_s) + 20.0,
                headers={"User-Agent": "PPO-Pointer-multimap-research/3.1"},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise TypeError("Overpass返回值不是JSON对象。")
            return dict(payload)
        except Exception as exc:
            errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
    raise RuntimeError("所有Overpass端点均失败：" + " | ".join(errors))


def _road_lines_utm(
    payload: Mapping[str, Any],
    *,
    epsg: int,
    crop_bounds_utm: Sequence[float],
) -> List[Any]:
    (
        _rasterio,
        _requests,
        _CRS,
        Transformer,
        _Resampling,
        _from_origin,
        _reproject,
        LineString,
        box,
        _linemerge,
        transform_geometry,
        _unary_union,
    ) = _optional_geo_imports()
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    crop = box(*[float(value) for value in crop_bounds_utm])
    lines: List[Any] = []
    for element in payload.get("elements", ()):
        if element.get("type") != "way":
            continue
        highway = str((element.get("tags") or {}).get("highway", ""))
        if highway not in HIGHWAY_TYPES:
            continue
        geometry = element.get("geometry") or ()
        coordinates = [
            (float(point["lon"]), float(point["lat"]))
            for point in geometry
            if "lon" in point and "lat" in point
        ]
        if len(coordinates) < 2:
            continue
        line = transform_geometry(transformer.transform, LineString(coordinates))
        clipped = line.intersection(crop)
        if clipped.is_empty:
            continue
        if clipped.geom_type == "LineString":
            lines.append(clipped)
        elif clipped.geom_type == "MultiLineString":
            lines.extend(list(clipped.geoms))
    return [line for line in lines if float(line.length) >= 30.0]


def road_metrics_and_tracks(lines: Sequence[Any]) -> Dict[str, Any]:
    (
        _rasterio,
        _requests,
        _CRS,
        _Transformer,
        _Resampling,
        _from_origin,
        _reproject,
        _LineString,
        _box,
        linemerge,
        _transform_geometry,
        unary_union,
    ) = _optional_geo_imports()
    if not lines:
        return {
            "road_length_m": 0.0,
            "road_branch_count": 0,
            "tracks": [],
        }
    union = unary_union(list(lines))
    merged = linemerge(union)
    if merged.geom_type == "LineString":
        merged_lines = [merged]
    else:
        merged_lines = list(getattr(merged, "geoms", ()))
    merged_lines.sort(key=lambda item: (-float(item.length), item.wkt))
    tracks = [
        np.asarray(line.coords, dtype=np.float64)
        for line in merged_lines
        if float(line.length) >= 1500.0
    ][:2]
    adjacency: Dict[Tuple[int, int], set] = {}
    for line in lines:
        coordinates = list(line.coords)
        for left, right in zip(coordinates[:-1], coordinates[1:]):
            left_key = (int(round(left[0])), int(round(left[1])))
            right_key = (int(round(right[0])), int(round(right[1])))
            if left_key == right_key:
                continue
            adjacency.setdefault(left_key, set()).add(right_key)
            adjacency.setdefault(right_key, set()).add(left_key)
    # OSM交叉点可能位于way内部，必须按完整折线节点度数统计，不能只数way端点。
    branch_count = sum(len(neighbours) >= 3 for neighbours in adjacency.values())
    return {
        "road_length_m": float(sum(float(line.length) for line in lines)),
        "road_branch_count": int(branch_count),
        "tracks": tracks,
    }


def _crop_bounds_from_transform(
    transform_values: Sequence[float], shape: Sequence[int]
) -> Tuple[float, float, float, float]:
    a, b, c, d, e, f = (float(value) for value in transform_values)
    rows, cols = int(shape[0]), int(shape[1])
    corners = [
        (c, f),
        (c + a * cols + b * rows, f + d * cols + e * rows),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _standardized_scores(
    records: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
) -> Dict[str, float]:
    fields = tuple(weights)
    values = {
        field: np.asarray([float(record[field]) for record in records])
        for field in fields
    }
    output: Dict[str, float] = {}
    for record_index, record in enumerate(records):
        score = 0.0
        for field, weight in weights.items():
            column = values[field]
            spread = float(np.std(column))
            z_value = (
                0.0
                if spread <= 1e-12
                else (float(column[record_index]) - float(np.mean(column))) / spread
            )
            score += float(weight) * z_value
        output[str(record["candidate_id"])] = float(score)
    return output


def evaluate_region_candidates(
    protocol: Mapping[str, Any],
    region: Mapping[str, Any],
    *,
    terrain_only_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    design = dict(protocol["real_dem_design"])
    centers = candidate_centers(region, int(design["candidate_grid_side"]))
    terrain_records: List[Dict[str, Any]] = []
    for candidate in centers:
        url = copernicus_tile_url(
            candidate["latitude"], candidate["longitude"], protocol
        )
        try:
            crop = read_dem_crop(
                url,
                longitude=float(candidate["longitude"]),
                latitude=float(candidate["latitude"]),
                size_m=float(design["crop_size_m"]),
                resolution_m=float(design["target_resolution_m"]),
            )
            metrics = terrain_metrics(
                crop["terrain"], float(design["target_resolution_m"])
            )
            record = {
                **candidate,
                **metrics,
                "dem_url": url,
                "dem_crs": crop["crs"],
                "dem_transform": crop["transform"],
                "dem_shape": list(np.asarray(crop["terrain"]).shape),
                "terrain_eligible": (
                    float(metrics["nodata_fraction"])
                    <= float(design["maximum_nodata_fraction"])
                    and float(metrics["relief_m"])
                    >= float(design["minimum_relief_m"])
                ),
            }
        except Exception as exc:
            record = {
                **candidate,
                "dem_url": url,
                "terrain_eligible": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        terrain_records.append(record)

    eligible = [
        record for record in terrain_records if record.get("terrain_eligible")
    ]
    eligible.sort(
        key=lambda item: (
            -float(item.get("relief_m", 0.0)),
            str(item["candidate_id"]),
        )
    )
    road_probe_count = min(len(eligible), int(terrain_only_limit or 5))
    for record in eligible[:road_probe_count]:
        bbox = _lon_lat_bbox(
            float(record["longitude"]),
            float(record["latitude"]),
            float(design["crop_size_m"]),
        )
        try:
            osm = fetch_osm_roads(bbox)
            bounds = _crop_bounds_from_transform(
                record["dem_transform"], record["dem_shape"]
            )
            epsg = int(str(record["dem_crs"]).split(":")[-1])
            lines = _road_lines_utm(osm, epsg=epsg, crop_bounds_utm=bounds)
            road = road_metrics_and_tracks(lines)
            record.update(
                {
                    "road_length_m": float(road["road_length_m"]),
                    "road_branch_count": int(road["road_branch_count"]),
                    "track_count": len(road["tracks"]),
                    "road_eligible": (
                        float(road["road_length_m"])
                        >= float(design["minimum_usable_road_length_m"])
                        and int(road["road_branch_count"])
                        >= int(design["minimum_road_branch_count"])
                        and len(road["tracks"])
                        >= int(design["road_tracks_per_dem"])
                    ),
                    "_osm_payload": osm,
                    "_tracks": road["tracks"],
                }
            )
        except Exception as exc:
            record.update(
                {
                    "road_eligible": False,
                    "road_error": f"{type(exc).__name__}: {exc}",
                }
            )
    scored = [
        record
        for record in eligible[:road_probe_count]
        if record.get("road_eligible")
    ]
    if scored:
        scores = _standardized_scores(
            scored, design["candidate_score_weights"]
        )
        for record in scored:
            record["selection_score"] = scores[str(record["candidate_id"])]
        scored.sort(
            key=lambda item: (
                -float(item["selection_score"]),
                str(item["candidate_id"]),
            )
        )
    return terrain_records


def _download_file(
    url: str,
    target: Path,
    *,
    timeout_s: float = 300.0,
    maximum_attempts: int = 5,
) -> None:
    (
        _rasterio,
        requests,
        _CRS,
        _Transformer,
        _Resampling,
        _from_origin,
        _reproject,
        *_rest,
    ) = _optional_geo_imports()
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # 使用稳定.part名称保留已下载字节；网络中断后用HTTP Range继续。
    temporary = target.with_name(f".{target.name}.part")
    errors: List[str] = []
    for attempt in range(1, int(maximum_attempts) + 1):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(
                url,
                stream=True,
                timeout=timeout_s,
                headers=headers,
            ) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    # 服务端忽略Range时重新完整写入，不能把两份文件拼接。
                    offset = 0
                    mode = "wb"
                else:
                    mode = "ab" if offset else "wb"
                content_range = str(response.headers.get("Content-Range", ""))
                if "/" in content_range:
                    total_size = int(content_range.rsplit("/", 1)[-1])
                else:
                    total_size = offset + int(
                        response.headers.get("Content-Length", "0") or 0
                    )
                with temporary.open(mode) as stream:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            stream.write(block)
            actual_size = temporary.stat().st_size
            if total_size > 0 and actual_size != total_size:
                raise IOError(
                    f"下载长度不完整：actual={actual_size}, expected={total_size}"
                )
            os.replace(temporary, target)
            return
        except Exception as exc:
            errors.append(f"attempt={attempt}: {type(exc).__name__}: {exc}")
            if attempt < int(maximum_attempts):
                time.sleep(min(2**attempt, 15))
    raise RuntimeError(
        f"下载失败且已保留断点文件{temporary}：" + " | ".join(errors)
    )


def _write_dem_crop(
    path: Path,
    terrain: np.ndarray,
    transform_values: Sequence[float],
    crs: str,
) -> None:
    (
        rasterio,
        _requests,
        _CRS,
        _Transformer,
        _Resampling,
        _from_origin,
        _reproject,
        *_rest,
    ) = _optional_geo_imports()
    from affine import Affine

    array = np.asarray(terrain, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.tif")
    try:
        with rasterio.open(
            temporary,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=Affine(*[float(value) for value in transform_values]),
            nodata=np.nan,
            tiled=True,
            compress="deflate",
            predictor=3,
        ) as destination:
            destination.write(array, 1)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def acquire_real_dem_registry(
    protocol_path: Path,
    map_root: Path,
    output_root: Path,
    *,
    resume_existing: bool = False,
    region_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    design = dict(protocol["real_dem_design"])
    selected_regions = [
        region
        for region in protocol["regions"]
        if not region_ids or str(region["region_id"]) in set(region_ids)
    ]
    registry_path = map_root / "real" / "map_registry.json"
    existing: Dict[str, Any] = {}
    if registry_path.exists():
        existing = json.loads(registry_path.read_text(encoding="utf-8"))
        if not resume_existing:
            raise FileExistsError(
                f"真实地图注册表已存在：{registry_path}；恢复请使用--resume-existing。"
            )
        if existing.get("protocol_hash") != protocol["protocol_hash"]:
            raise RuntimeError("现有真实地图注册表与当前协议不一致。")
    map_records = {
        str(record["map_id"]): dict(record)
        for record in existing.get("maps", ())
    }
    audit_root = output_root / "dem_acquisition"
    audit_root.mkdir(parents=True, exist_ok=True)

    for region in selected_regions:
        map_id = str(region["region_id"])
        if map_id in map_records:
            continue
        candidate_audit_path = audit_root / f"{map_id}_candidates.json"
        candidates: List[Dict[str, Any]]
        if resume_existing and candidate_audit_path.is_file():
            saved = json.loads(candidate_audit_path.read_text(encoding="utf-8"))
            if saved.get("protocol_hash") != protocol["protocol_hash"]:
                raise RuntimeError(f"{map_id}候选审计与当前协议不一致。")
            candidates = [dict(record) for record in saved.get("candidates", ())]
            reusable = [
                record
                for record in candidates
                if record.get("terrain_eligible") and record.get("road_eligible")
            ]
            reusable.sort(
                key=lambda item: (
                    -float(item.get("selection_score", -math.inf)),
                    str(item["candidate_id"]),
                )
            )
            if reusable:
                # 候选身份和输入统计已冻结；恢复时只重新取得所选点的原始道路载荷。
                selected = reusable[0]
                bbox = _lon_lat_bbox(
                    float(selected["longitude"]),
                    float(selected["latitude"]),
                    float(design["crop_size_m"]),
                )
                osm_payload = fetch_osm_roads(bbox)
                bounds = _crop_bounds_from_transform(
                    selected["dem_transform"], selected["dem_shape"]
                )
                epsg = int(str(selected["dem_crs"]).split(":")[-1])
                road = road_metrics_and_tracks(
                    _road_lines_utm(
                        osm_payload,
                        epsg=epsg,
                        crop_bounds_utm=bounds,
                    )
                )
                if (
                    float(road["road_length_m"])
                    < float(design["minimum_usable_road_length_m"])
                    or int(road["road_branch_count"])
                    < int(design["minimum_road_branch_count"])
                    or len(road["tracks"])
                    < int(design["road_tracks_per_dem"])
                ):
                    raise RuntimeError(
                        f"{map_id}恢复时道路输入已不再满足冻结门槛。"
                    )
                selected["road_length_m"] = float(road["road_length_m"])
                selected["road_branch_count"] = int(road["road_branch_count"])
                selected["track_count"] = len(road["tracks"])
                selected["_osm_payload"] = osm_payload
                selected["_tracks"] = road["tracks"]
            else:
                candidates = evaluate_region_candidates(protocol, region)
        else:
            candidates = evaluate_region_candidates(protocol, region)
            serializable_candidates = []
            for record in candidates:
                copy_record = {
                    key: value
                    for key, value in record.items()
                    if not key.startswith("_")
                }
                serializable_candidates.append(copy_record)
            _atomic_json(
                candidate_audit_path,
                {
                    "schema_version": 1,
                    "protocol_hash": protocol["protocol_hash"],
                    "region": region,
                    "candidates": serializable_candidates,
                },
            )
        eligible = [
            record
            for record in candidates
            if record.get("terrain_eligible") and record.get("road_eligible")
        ]
        eligible.sort(
            key=lambda item: (
                -float(item.get("selection_score", -math.inf)),
                str(item["candidate_id"]),
            )
        )
        if not eligible:
            raise RuntimeError(
                f"地域{map_id}没有同时通过地形与道路门槛的候选；"
                f"候选审计已写入{audit_root}。"
            )
        chosen = eligible[0]
        map_dir = map_root / "real" / map_id
        tile_name = Path(str(chosen["dem_url"])).parent.name
        raw_tile = map_root / "raw_tiles" / f"{tile_name}.tif"
        _download_file(str(chosen["dem_url"]), raw_tile)
        crop = read_dem_crop(
            str(raw_tile),
            longitude=float(chosen["longitude"]),
            latitude=float(chosen["latitude"]),
            size_m=float(design["crop_size_m"]),
            resolution_m=float(design["target_resolution_m"]),
        )
        dem_path = map_dir / "dem.tif"
        _write_dem_crop(
            dem_path, crop["terrain"], crop["transform"], crop["crs"]
        )
        osm_payload = dict(chosen["_osm_payload"])
        _atomic_json(map_dir / "osm_raw.json", osm_payload)
        tracks = [np.asarray(track, dtype=np.float64) for track in chosen["_tracks"]]
        road_npz = map_dir / "road_tracks.npz"
        map_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            road_npz,
            track_0=tracks[0],
            track_1=tracks[1],
        )
        map_record = {
            "map_id": map_id,
            "region_name_zh": str(region["name_zh"]),
            "region_group": str(region["group"]),
            "source_product": protocol["dem_source"]["primary_product"],
            "source_release": protocol["dem_source"]["primary_release"],
            "product_kind": protocol["dem_source"]["product_kind"],
            "source_url": str(chosen["dem_url"]),
            "selected_candidate_id": str(chosen["candidate_id"]),
            "center_wgs84": [
                float(chosen["longitude"]),
                float(chosen["latitude"]),
            ],
            "crs": str(crop["crs"]),
            "resolution_m": float(design["target_resolution_m"]),
            "shape": list(np.asarray(crop["terrain"]).shape),
            "raw_tile_file": str(raw_tile.relative_to(map_root)),
            "raw_tile_sha256": _sha256_file(raw_tile),
            "dem_file": str(dem_path.relative_to(map_root)),
            "dem_sha256": _sha256_file(dem_path),
            "road_file": str(road_npz.relative_to(map_root)),
            "road_sha256": _sha256_file(road_npz),
            "osm_raw_file": str((map_dir / "osm_raw.json").relative_to(map_root)),
            "osm_raw_sha256": _sha256_file(map_dir / "osm_raw.json"),
            "terrain_metrics": terrain_metrics(
                crop["terrain"], float(design["target_resolution_m"])
            ),
            "road_metrics": {
                "road_length_m": float(chosen["road_length_m"]),
                "road_branch_count": int(chosen["road_branch_count"]),
                "track_count": int(chosen["track_count"]),
                "track_lengths_m": [
                    float(
                        np.sum(
                            np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1)
                        )
                    )
                    for track in tracks
                ],
            },
            "selection_score": float(chosen["selection_score"]),
            "selection_used_algorithm_results": False,
        }
        map_record["map_record_hash"] = _canonical_hash(
            map_record, excluded=("map_record_hash",)
        )
        map_records[map_id] = map_record
        registry = {
            "schema_version": 1,
            "protocol_hash": protocol["protocol_hash"],
            "source_product": protocol["dem_source"]["primary_product"],
            "source_release": protocol["dem_source"]["primary_release"],
            "selection_complete": len(map_records) == int(design["count"]),
            "maps": [
                map_records[key] for key in sorted(map_records)
            ],
        }
        registry["registry_hash"] = _canonical_hash(
            registry, excluded=("registry_hash",)
        )
        _atomic_json(registry_path, registry)
    return audit_real_dem_registry(
        protocol_path,
        map_root,
        output_path=output_root / "audits" / "audit_real_dem_registry.json",
    )


def audit_real_dem_registry(
    protocol_path: Path,
    map_root: Path,
    *,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    design = dict(protocol["real_dem_design"])
    registry_path = map_root / "real" / "map_registry.json"
    if not registry_path.exists():
        report = {
            "passed": False,
            "reason": "missing_registry",
            "registry_path": str(registry_path.resolve()),
        }
        if output_path is not None:
            _atomic_json(Path(output_path), report)
        return report
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    reasons: List[str] = []
    expected_registry_protocol_hash = (
        protocol["protocol_hash"]
        if split == "synthetic_test"
        else _asset_protocol_hash(protocol)
    )
    if registry.get("protocol_hash") != expected_registry_protocol_hash:
        reasons.append("protocol_hash_mismatch")
    if _canonical_hash(registry, excluded=("registry_hash",)) != registry.get(
        "registry_hash"
    ):
        reasons.append("registry_hash_mismatch")
    maps = list(registry.get("maps", ()))
    if len(maps) != int(design["count"]):
        reasons.append(f"map_count={len(maps)}")
    if sum(record.get("region_group") == "china" for record in maps) != int(
        design["china_count"]
    ):
        reasons.append("china_count_mismatch")
    if sum(record.get("region_group") == "global" for record in maps) != int(
        design["global_count"]
    ):
        reasons.append("global_count_mismatch")
    products = {
        (record.get("source_product"), record.get("source_release"))
        for record in maps
    }
    if len(products) > 1:
        reasons.append("mixed_dem_products")
    seen_ids = set()
    seen_dem_hashes = set()
    for record in maps:
        map_id = str(record.get("map_id", ""))
        if not map_id or map_id in seen_ids:
            reasons.append(f"duplicate_or_empty_map_id={map_id}")
        seen_ids.add(map_id)
        if _canonical_hash(
            record, excluded=("map_record_hash",)
        ) != record.get("map_record_hash"):
            reasons.append(f"map_record_hash_mismatch={map_id}")
        for file_field, hash_field in (
            ("raw_tile_file", "raw_tile_sha256"),
            ("dem_file", "dem_sha256"),
            ("road_file", "road_sha256"),
            ("osm_raw_file", "osm_raw_sha256"),
            *(
                (("map_file", "map_file_sha256"),)
                if registry.get("bundles_sealed")
                else ()
            ),
        ):
            path = map_root / str(record.get(file_field, ""))
            if not path.is_file():
                reasons.append(f"missing_{file_field}={map_id}")
            elif _sha256_file(path) != record.get(hash_field):
                reasons.append(f"{hash_field}_mismatch={map_id}")
        dem_hash = str(record.get("dem_sha256", ""))
        if dem_hash in seen_dem_hashes:
            reasons.append(f"duplicate_dem_hash={map_id}")
        seen_dem_hashes.add(dem_hash)
        metrics = dict(record.get("terrain_metrics") or {})
        roads = dict(record.get("road_metrics") or {})
        if float(metrics.get("nodata_fraction", 1.0)) > float(
            design["maximum_nodata_fraction"]
        ):
            reasons.append(f"nodata_threshold_failed={map_id}")
        if float(metrics.get("relief_m", 0.0)) < float(
            design["minimum_relief_m"]
        ):
            reasons.append(f"relief_threshold_failed={map_id}")
        if int(roads.get("track_count", 0)) != int(
            design["road_tracks_per_dem"]
        ):
            reasons.append(f"road_track_count_failed={map_id}")
        if bool(record.get("selection_used_algorithm_results", True)):
            reasons.append(f"algorithm_result_selection={map_id}")
    report = {
        "schema_version": 1,
        "passed": not reasons,
        "protocol_hash": protocol["protocol_hash"],
        "registry_path": str(registry_path.resolve()),
        "map_count": len(maps),
        "reasons": reasons,
    }
    report["audit_hash"] = _canonical_hash(report, excluded=("audit_hash",))
    if output_path is not None:
        _atomic_json(Path(output_path), report)
    return report


def seal_real_map_bundles(
    protocol_path: Path, map_root: Path
) -> Dict[str, Any]:
    """把GeoTIFF与UTM道路转换为训练器统一读取的只读NPZ地图包。"""

    protocol = load_protocol(protocol_path)
    registry_path = map_root / "real" / "map_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("protocol_hash") != _asset_protocol_hash(protocol):
        raise RuntimeError("真实地图注册表与当前协议不一致。")
    (
        rasterio,
        _requests,
        _CRS,
        _Transformer,
        _Resampling,
        _from_origin,
        _reproject,
        *_rest,
    ) = _optional_geo_imports()
    updated: List[Dict[str, Any]] = []
    for raw_record in registry["maps"]:
        record = dict(raw_record)
        dem_path = map_root / str(record["dem_file"])
        road_path = map_root / str(record["road_file"])
        with rasterio.open(dem_path) as source:
            terrain = source.read(1).astype(np.float32)
            inverse = ~source.transform
            source_transform = [
                float(source.transform.a),
                float(source.transform.b),
                float(source.transform.c),
                float(source.transform.d),
                float(source.transform.e),
                float(source.transform.f),
            ]
        roads_local: List[np.ndarray] = []
        with np.load(road_path, allow_pickle=False) as data:
            for key in sorted(data.files):
                coordinates = np.asarray(data[key], dtype=np.float64)
                local = np.asarray(
                    [inverse * (float(x), float(y)) for x, y in coordinates[:, :2]],
                    dtype=np.float32,
                )
                roads_local.append(local)
        road_points, road_offsets = _pack_roads(roads_local)
        metadata = {
            "map_id": str(record["map_id"]),
            "split": "real_external_test",
            "terrain_family": "copernicus_glo30_real",
            "road_topology": "osm_observed",
            "coordinate_scale_m_per_unit": float(record["resolution_m"]),
            "crs": str(record["crs"]),
            "local_affine": source_transform,
            "source_dem_sha256": str(record["dem_sha256"]),
            "source_road_sha256": str(record["road_sha256"]),
        }
        map_hash = _map_hash(terrain, road_points, road_offsets, metadata)
        bundle_path = (
            map_root / "real" / str(record["map_id"]) / "map_bundle.npz"
        )
        with tempfile.NamedTemporaryFile(
            dir=bundle_path.parent,
            prefix=f".{bundle_path.stem}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            np.savez_compressed(
                temporary_path,
                terrain=terrain,
                road_points=road_points,
                road_offsets=road_offsets,
                metadata_json=np.asarray(
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
                map_hash=np.asarray(map_hash),
            )
            os.replace(temporary_path, bundle_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        record.update(
            {
                "map_file": str(bundle_path.relative_to(map_root)),
                "map_file_sha256": _sha256_file(bundle_path),
                "map_hash": map_hash,
            }
        )
        record["map_record_hash"] = _canonical_hash(
            record, excluded=("map_record_hash",)
        )
        updated.append(record)
    registry["maps"] = sorted(updated, key=lambda item: item["map_id"])
    registry["bundles_sealed"] = True
    registry["registry_hash"] = _canonical_hash(
        registry, excluded=("registry_hash",)
    )
    _atomic_json(registry_path, registry)
    return {
        "passed": True,
        "map_count": len(updated),
        "registry_hash": registry["registry_hash"],
        "protocol_hash": protocol["protocol_hash"],
    }


def _spectral_terrain(
    rng: np.random.Generator,
    shape: Tuple[int, int],
    *,
    hurst: float,
    target_relief_m: float,
) -> np.ndarray:
    rows, cols = shape
    fy = np.fft.fftfreq(rows)[:, None]
    fx = np.fft.fftfreq(cols)[None, :]
    frequency = np.hypot(fx, fy)
    frequency[0, 0] = 1.0
    amplitude = frequency ** (-(float(hurst) + 1.0))
    phase = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    surface = np.fft.ifft2(phase * amplitude).real
    # 叠加定向山脊，避免程序化地图退化为各向同性噪声。
    yy, xx = np.mgrid[-1.0:1.0:complex(rows), -1.0:1.0:complex(cols)]
    angle = float(rng.uniform(0.0, math.pi))
    ridge_axis = xx * math.cos(angle) + yy * math.sin(angle)
    surface += 0.8 * np.exp(-((ridge_axis - rng.uniform(-0.35, 0.35)) / 0.2) ** 2)
    low, high = np.percentile(surface, [1.0, 99.0])
    normalized = np.clip((surface - low) / max(high - low, 1e-9), 0.0, 1.0)
    base = float(rng.uniform(200.0, 1800.0))
    return (base + normalized * float(target_relief_m)).astype(np.float32)


def _legacy_dem_calibration(
    path: Path, *, crop_size_m: float
) -> Dict[str, Any]:
    (
        rasterio,
        _requests,
        _CRS,
        _Transformer,
        _Resampling,
        _from_origin,
        _reproject,
        *_rest,
    ) = _optional_geo_imports()
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"旧DEM校准文件不存在：{source_path}")
    with rasterio.open(source_path) as source:
        resolution = float(
            0.5 * (abs(float(source.transform.a)) + abs(float(source.transform.e)))
        )
        side = min(
            int(round(float(crop_size_m) / resolution)),
            int(source.height),
            int(source.width),
        )
        row_off = (int(source.height) - side) // 2
        col_off = (int(source.width) - side) // 2
        from rasterio.windows import Window

        values = source.read(
            1,
            window=Window(col_off, row_off, side, side),
            masked=True,
        ).astype(np.float32)
        terrain = values.filled(np.nan)
    return {
        "path": str(source_path),
        "sha256": _sha256_file(source_path),
        "resolution_m": resolution,
        "crop_shape": list(terrain.shape),
        "terrain_metrics": terrain_metrics(terrain, resolution),
    }


def _procedural_roads(
    rng: np.random.Generator,
    shape: Tuple[int, int],
    topology: str,
) -> List[np.ndarray]:
    rows, cols = shape
    cx = (cols - 1) / 2.0 + float(rng.uniform(-0.05, 0.05) * cols)
    cy = (rows - 1) / 2.0 + float(rng.uniform(-0.05, 0.05) * rows)
    t = np.linspace(-1.0, 1.0, 400)
    roads: List[np.ndarray] = []
    if topology == "multi_arm":
        for angle in np.linspace(0.0, 2.0 * math.pi, 5, endpoint=False):
            radius = np.linspace(0.0, 0.46 * min(rows, cols), 220)
            curve = 0.08 * radius * np.sin(2.0 * radius / max(radius[-1], 1.0))
            x = cx + radius * math.cos(angle) - curve * math.sin(angle)
            y = cy + radius * math.sin(angle) + curve * math.cos(angle)
            roads.append(np.column_stack([x, y]))
    elif topology == "t_y":
        trunk_x = cx + 0.08 * cols * np.sin(math.pi * t)
        trunk_y = cy + 0.47 * rows * t
        roads.append(np.column_stack([trunk_x, trunk_y]))
        for sign in (-1.0, 1.0):
            branch_t = np.linspace(0.0, 1.0, 220)
            roads.append(
                np.column_stack(
                    [
                        cx + sign * 0.44 * cols * branch_t,
                        cy - 0.05 * rows - 0.25 * rows * branch_t,
                    ]
                )
            )
    elif topology == "curved_trunk_branches":
        roads.append(
            np.column_stack(
                [
                    cx + 0.44 * cols * t,
                    cy + 0.18 * rows * np.sin(math.pi * t),
                ]
            )
        )
        for anchor, sign in ((-0.45, -1.0), (0.0, 1.0), (0.45, -1.0)):
            branch_t = np.linspace(0.0, 1.0, 180)
            anchor_x = cx + 0.44 * cols * anchor
            anchor_y = cy + 0.18 * rows * math.sin(math.pi * anchor)
            roads.append(
                np.column_stack(
                    [
                        anchor_x + 0.2 * cols * sign * branch_t,
                        anchor_y + 0.32 * rows * sign * branch_t,
                    ]
                )
            )
    elif topology == "loop_spur":
        theta = np.linspace(0.0, 2.0 * math.pi, 500)
        roads.append(
            np.column_stack(
                [
                    cx + 0.28 * cols * np.cos(theta),
                    cy + 0.22 * rows * np.sin(theta),
                ]
            )
        )
        for angle in (0.25 * math.pi, 1.25 * math.pi):
            radius = np.linspace(0.0, 0.3 * min(rows, cols), 200)
            anchor_x = cx + 0.28 * cols * math.cos(angle)
            anchor_y = cy + 0.22 * rows * math.sin(angle)
            roads.append(
                np.column_stack(
                    [
                        anchor_x + radius * math.cos(angle),
                        anchor_y + radius * math.sin(angle),
                    ]
                )
            )
    else:
        raise ValueError(f"未知程序化道路拓扑：{topology}")
    clipped: List[np.ndarray] = []
    for road in roads:
        value = np.asarray(road, dtype=np.float32)
        value[:, 0] = np.clip(value[:, 0], 1.0, cols - 2.0)
        value[:, 1] = np.clip(value[:, 1], 1.0, rows - 2.0)
        clipped.append(value)
    return clipped


def _pack_roads(roads: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    points: List[np.ndarray] = []
    for road in roads:
        value = np.asarray(road, dtype=np.float32)
        points.append(value)
        offsets.append(offsets[-1] + len(value))
    return np.vstack(points), np.asarray(offsets, dtype=np.int32)


def _map_hash(
    terrain: np.ndarray,
    road_points: np.ndarray,
    road_offsets: np.ndarray,
    metadata: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("terrain", terrain),
        ("road_points", road_points),
        ("road_offsets", road_offsets),
    ):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def prepare_procedural_maps(
    protocol_path: Path,
    map_root: Path,
    *,
    splits: Sequence[str],
    real_registry_path: Path,
    training_freeze: Optional[Path] = None,
    resume_existing: bool = False,
    legacy_dem_path: Path = DEFAULT_LEGACY_DEM,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    for split in splits:
        if split not in {"training", "validation", "synthetic_test"}:
            raise ValueError(f"未知程序化地图split={split}")
        if split == "synthetic_test":
            if training_freeze is None or not Path(training_freeze).is_file():
                raise RuntimeError("训练协议冻结前不得生成synthetic_test地图。")
    registry = json.loads(real_registry_path.read_text(encoding="utf-8"))
    if not registry.get("selection_complete"):
        raise RuntimeError("8张真实DEM尚未完成封存，不能校准程序化地图。")
    # v3.2 只读复用父协议封存的真实 DSM；新生成的 synthetic_test
    # 则必须绑定当前协议，不能把父协议身份错误地写入新资产。
    if registry.get("protocol_hash") != _asset_protocol_hash(protocol):
        raise RuntimeError("真实DEM注册表与多地图协议不一致。")
    terrain_stats = [
        dict(record["terrain_metrics"]) for record in registry["maps"]
    ]
    legacy_calibration = _legacy_dem_calibration(
        legacy_dem_path,
        crop_size_m=float(protocol["real_dem_design"]["crop_size_m"]),
    )
    terrain_stats.append(dict(legacy_calibration["terrain_metrics"]))
    reliefs = np.asarray(
        [float(item["relief_m"]) for item in terrain_stats], dtype=np.float64
    )
    relief_low, relief_high = np.percentile(reliefs, [10.0, 90.0])
    config = dict(protocol["procedural_terrain"])
    topologies = list(config["topologies"])
    outputs: List[Dict[str, Any]] = []
    for split in splits:
        design = dict(protocol["map_splits"][split])
        split_root = map_root / "procedural" / split
        registry_path = split_root / "map_registry.json"
        if registry_path.exists() and not resume_existing:
            raise FileExistsError(
                f"{split}地图注册表已存在；恢复请使用--resume-existing。"
            )
        records: List[Dict[str, Any]] = []
        for index in range(int(design["map_count"])):
            map_id = f"{split}__map_{index:03d}"
            seed = int(
                hashlib.sha256(
                    f"{design['seed']}|{map_id}".encode("utf-8")
                ).hexdigest()[:16],
                16,
            )
            rng = np.random.default_rng(seed)
            topology = topologies[index % len(topologies)]
            hurst = float(rng.uniform(*config["hurst_range"]))
            target_relief = float(rng.uniform(relief_low, relief_high))
            terrain = _spectral_terrain(
                rng,
                tuple(int(value) for value in config["grid_shape"]),
                hurst=hurst,
                target_relief_m=target_relief,
            )
            roads = _procedural_roads(rng, terrain.shape, topology)
            road_points, road_offsets = _pack_roads(roads)
            metadata = {
                "map_id": map_id,
                "split": split,
                "map_seed": seed,
                "terrain_family": "calibrated_multiscale_spectral_mountain",
                "road_topology": topology,
                "hurst": hurst,
                "target_relief_m": target_relief,
                "coordinate_scale_m_per_unit": float(
                    config["coordinate_scale_m_per_unit"]
                ),
                "calibration": {
                    "real_registry_hash": registry["registry_hash"],
                    "legacy_dem_sha256": legacy_calibration["sha256"],
                    "aggregate_relief_percentile_10_m": float(relief_low),
                    "aggregate_relief_percentile_90_m": float(relief_high),
                    "real_pixel_copy_forbidden": True,
                },
            }
            map_hash = _map_hash(
                terrain, road_points, road_offsets, metadata
            )
            path = split_root / f"{map_id}.npz"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=path.parent,
                    prefix=f".{path.stem}.",
                    suffix=".npz",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                try:
                    np.savez_compressed(
                        temporary_path,
                        terrain=terrain,
                        road_points=road_points,
                        road_offsets=road_offsets,
                        metadata_json=np.asarray(
                            json.dumps(
                                metadata,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        ),
                        map_hash=np.asarray(map_hash),
                    )
                    os.replace(temporary_path, path)
                finally:
                    if temporary_path.exists():
                        temporary_path.unlink()
            record = {
                **metadata,
                "map_file": str(path.relative_to(map_root)),
                "map_file_sha256": _sha256_file(path),
                "map_hash": map_hash,
                "shape": list(terrain.shape),
                "terrain_metrics": terrain_metrics(
                    terrain, float(config["coordinate_scale_m_per_unit"])
                ),
            }
            record["record_hash"] = _canonical_hash(
                record, excluded=("record_hash",)
            )
            records.append(record)
        split_registry = {
            "schema_version": 1,
            "protocol_hash": protocol["protocol_hash"],
            "split": split,
            "map_count": len(records),
            "real_calibration_registry_hash": registry["registry_hash"],
            "legacy_calibration": legacy_calibration,
            "maps": records,
        }
        split_registry["registry_hash"] = _canonical_hash(
            split_registry, excluded=("registry_hash",)
        )
        _atomic_json(registry_path, split_registry)
        outputs.append(
            {
                "split": split,
                "registry": str(registry_path.resolve()),
                "registry_hash": split_registry["registry_hash"],
                "map_count": len(records),
            }
        )
    return {"prepared": outputs, "protocol_hash": protocol["protocol_hash"]}


def audit_procedural_registry(
    protocol_path: Path, map_root: Path, split: str
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    registry_path = map_root / "procedural" / split / "map_registry.json"
    reasons: List[str] = []
    if not registry_path.is_file():
        return {"passed": False, "reasons": ["missing_registry"]}
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("protocol_hash") != _asset_protocol_hash(protocol):
        reasons.append("protocol_hash_mismatch")
    if _canonical_hash(registry, excluded=("registry_hash",)) != registry.get(
        "registry_hash"
    ):
        reasons.append("registry_hash_mismatch")
    expected = int(protocol["map_splits"][split]["map_count"])
    maps = list(registry.get("maps", ()))
    if len(maps) != expected:
        reasons.append(f"map_count={len(maps)}")
    seen_hashes = set()
    geometry_combinations_checked = 0
    for record in maps:
        map_id = str(record.get("map_id", ""))
        path = map_root / str(record.get("map_file", ""))
        if not path.is_file():
            reasons.append(f"missing_map_file={map_id}")
            continue
        if _sha256_file(path) != record.get("map_file_sha256"):
            reasons.append(f"file_hash_mismatch={map_id}")
        if record.get("map_hash") in seen_hashes:
            reasons.append(f"duplicate_map_hash={map_id}")
        seen_hashes.add(record.get("map_hash"))
        if record.get("split") != split:
            reasons.append(f"split_mismatch={map_id}")
        if _canonical_hash(record, excluded=("record_hash",)) != record.get(
            "record_hash"
        ):
            reasons.append(f"record_hash_mismatch={map_id}")
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            actual = _map_hash(
                np.asarray(data["terrain"]),
                np.asarray(data["road_points"]),
                np.asarray(data["road_offsets"]),
                metadata,
            )
            if actual != str(data["map_hash"].item()):
                reasons.append(f"content_hash_mismatch={map_id}")
        # 在生成MILP任务前先做纯道路几何审计，防止稀疏拓扑在运行中途才暴露布点失败。
        try:
            bundle = _load_map_bundle(map_root, record)
        except (KeyError, RuntimeError, ValueError) as exc:
            reasons.append(
                f"geometry_bundle_failed={map_id}:{type(exc).__name__}:{exc}"
            )
            continue
        for node_count in protocol["node_counts"]:
            for difficulty in protocol["difficulty_bands"]:
                try:
                    _effective_task_radius_range(
                        record,
                        bundle,
                        protocol,
                        node_count=int(node_count),
                        difficulty=str(difficulty),
                    )
                    geometry_combinations_checked += 1
                except (KeyError, RuntimeError, ValueError) as exc:
                    reasons.append(
                        "geometry_support_failed="
                        f"{map_id}:{int(node_count)}:{difficulty}:"
                        f"{type(exc).__name__}:{exc}"
                    )
    return {
        "passed": not reasons,
        "protocol_hash": protocol["protocol_hash"],
        "split": split,
        "map_count": len(maps),
        "geometry_combinations_checked": geometry_combinations_checked,
        "reasons": reasons,
        "registry_path": str(registry_path.resolve()),
    }


@dataclass
class FrozenMapProvider:
    """训练器使用的只读地图提供器；按map_id缓存，不把DEM复制进manifest。"""

    map_root: Path
    records: Mapping[str, Mapping[str, Any]]
    provider_hash: str

    def __post_init__(self) -> None:
        self.map_root = Path(self.map_root).resolve()
        self._cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def from_registries(
        cls, map_root: Path, registry_paths: Sequence[Path]
    ) -> "FrozenMapProvider":
        records: Dict[str, Mapping[str, Any]] = {}
        identities: List[Dict[str, str]] = []
        for registry_path in registry_paths:
            payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
            for record in payload.get("maps", ()):
                map_id = str(record["map_id"])
                if map_id in records:
                    raise ValueError(f"地图ID重复：{map_id}")
                records[map_id] = dict(record)
                identities.append(
                    {
                        "map_id": map_id,
                        "map_hash": str(record["map_hash"]),
                        "map_file_sha256": str(record["map_file_sha256"]),
                    }
                )
        provider_hash = hashlib.sha256(
            json.dumps(
                sorted(identities, key=lambda item: item["map_id"]),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return cls(map_root=Path(map_root), records=records, provider_hash=provider_hash)

    def __call__(self, instance: Mapping[str, Any]) -> Dict[str, Any]:
        map_id = str(instance.get("map_id", ""))
        if map_id not in self.records:
            raise KeyError(f"冻结地图提供器不存在map_id={map_id}")
        if str(instance.get("map_hash", "")) != str(
            self.records[map_id]["map_hash"]
        ):
            raise RuntimeError(f"冻结实例的地图身份漂移：{map_id}")
        if map_id not in self._cache:
            record = self.records[map_id]
            path = self.map_root / str(record["map_file"])
            if _sha256_file(path) != record["map_file_sha256"]:
                raise RuntimeError(f"冻结地图文件哈希漂移：{map_id}")
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                terrain = np.asarray(data["terrain"], dtype=np.float32)
                road_points = np.asarray(data["road_points"], dtype=np.float32)
                road_offsets = np.asarray(data["road_offsets"], dtype=np.int32)
                map_hash = str(data["map_hash"].item())
            if map_hash != record["map_hash"]:
                raise RuntimeError(f"冻结地图内容身份漂移：{map_id}")
            self._cache[map_id] = {
                "terrain": terrain,
                "road_points": road_points,
                "road_offsets": road_offsets,
                "metadata": metadata,
            }
        cached = self._cache[map_id]
        terrain = cached["terrain"]
        start_xy = np.asarray(
            instance.get(
                "start_xy",
                [(terrain.shape[1] - 1) / 2.0, (terrain.shape[0] - 1) / 2.0],
            ),
            dtype=np.float32,
        )
        x = float(np.clip(start_xy[0], 0.0, terrain.shape[1] - 1))
        y = float(np.clip(start_xy[1], 0.0, terrain.shape[0] - 1))
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        x1, y1 = min(x0 + 1, terrain.shape[1] - 1), min(
            y0 + 1, terrain.shape[0] - 1
        )
        wx, wy = x - x0, y - y0
        start_z = float(
            (terrain[y0, x0] * (1.0 - wx) + terrain[y0, x1] * wx)
            * (1.0 - wy)
            + (terrain[y1, x0] * (1.0 - wx) + terrain[y1, x1] * wx) * wy
        )
        wind = instance.get("base_wind_data") or {
            "uniform_vector": np.asarray([3.0, 0.4, 0.0], dtype=np.float32)
        }
        return {
            "start_pos": np.asarray([start_xy[0], start_xy[1], start_z + 1e-3]),
            "terrain": terrain,
            "wind_data": wind,
            "cfg_overrides": {
                "coordinate_scale_m_per_unit": float(
                    cached["metadata"]["coordinate_scale_m_per_unit"]
                )
            },
            "map_id": map_id,
            "map_hash": str(self.records[map_id]["map_hash"]),
        }


def _load_map_bundle(
    map_root: Path, record: Mapping[str, Any]
) -> Dict[str, Any]:
    path = Path(map_root) / str(record["map_file"])
    if _sha256_file(path) != str(record["map_file_sha256"]):
        raise RuntimeError(f"地图文件哈希漂移：{record['map_id']}")
    with np.load(path, allow_pickle=False) as data:
        terrain = np.asarray(data["terrain"], dtype=np.float32)
        road_points = np.asarray(data["road_points"], dtype=np.float32)
        road_offsets = np.asarray(data["road_offsets"], dtype=np.int32)
        metadata = json.loads(str(data["metadata_json"].item()))
        map_hash = str(data["map_hash"].item())
    if map_hash != str(record["map_hash"]):
        raise RuntimeError(f"地图内容哈希漂移：{record['map_id']}")
    roads = [
        road_points[int(left) : int(right)].copy()
        for left, right in zip(road_offsets[:-1], road_offsets[1:])
        if int(right) - int(left) >= 2
    ]
    return {
        "terrain": terrain,
        "roads": roads,
        "metadata": metadata,
        "map_hash": map_hash,
    }


def _polyline_resample(
    line: np.ndarray, spacing_units: float
) -> np.ndarray:
    points = np.asarray(line, dtype=np.float64)
    lengths = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))]
    )
    if lengths[-1] <= 1e-9:
        return points[:1, :2]
    samples = np.arange(0.0, lengths[-1] + 1e-9, float(spacing_units))
    if samples[-1] < lengths[-1]:
        samples = np.append(samples, lengths[-1])
    output = np.empty((len(samples), 2), dtype=np.float64)
    output[:, 0] = np.interp(samples, lengths, points[:, 0])
    output[:, 1] = np.interp(samples, lengths, points[:, 1])
    return output


def _candidate_road_points(
    bundle: Mapping[str, Any],
    *,
    coordinate_scale_m_per_unit: float,
    minimum_depot_distance_m: float = 180.0,
    maximum_depot_distance_m: float = 2600.0,
    sampling_spacing_m: float = 90.0,
    minimum_candidate_count: int = 24,
    depot_override_xy: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    terrain = np.asarray(bundle["terrain"])
    center = np.asarray(
        [(terrain.shape[1] - 1) / 2.0, (terrain.shape[0] - 1) / 2.0],
        dtype=np.float64,
    )
    all_vertices = np.vstack([np.asarray(road)[:, :2] for road in bundle["roads"]])
    if depot_override_xy is None:
        depot = all_vertices[
            int(np.argmin(np.linalg.norm(all_vertices - center, axis=1)))
        ]
    else:
        # 真实道路走廊任务可冻结各自的起飞基地，避免误用整幅DSM中心。
        depot = np.asarray(depot_override_xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(depot)):
            raise ValueError("depot_override_xy必须是有限二维坐标")
    samples = np.vstack(
        [
            _polyline_resample(
                road, float(sampling_spacing_m) / coordinate_scale_m_per_unit
            )
            for road in bundle["roads"]
        ]
    )
    distances_m = (
        np.linalg.norm(samples - depot.reshape(1, 2), axis=1)
        * coordinate_scale_m_per_unit
    )
    samples = samples[
        (distances_m >= float(minimum_depot_distance_m))
        & (distances_m <= float(maximum_depot_distance_m))
    ]
    if len(samples) < int(minimum_candidate_count):
        raise RuntimeError(
            f"地图道路在任务半径内不足以放置{int(minimum_candidate_count)}"
            "个候选巡检点。"
        )
    # 按毫米级坐标去重，避免OSM或程序化支路在交点产生重复候选。
    rounded = np.round(samples, decimals=3)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return depot.astype(np.float32), samples[np.sort(unique_indices)].astype(np.float32)


def _farthest_sample(
    candidates: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    coordinate_scale_m_per_unit: float,
    minimum_spacing_m: float = 120.0,
) -> np.ndarray:
    values = np.asarray(candidates, dtype=np.float64)
    if len(values) < int(count):
        raise RuntimeError("候选点数量少于目标节点数。")
    selected = [int(rng.integers(0, len(values)))]
    while len(selected) < int(count):
        remaining = np.asarray(
            [index for index in range(len(values)) if index not in set(selected)],
            dtype=np.int64,
        )
        distances = np.min(
            np.linalg.norm(
                values[remaining, None, :] - values[np.asarray(selected)][None, :, :],
                axis=2,
            ),
            axis=1,
        )
        eligible = (
            distances * float(coordinate_scale_m_per_unit)
            >= float(minimum_spacing_m)
        )
        if not np.any(eligible):
            raise RuntimeError("候选道路无法满足巡检点最小间距。")
        eligible_indices = remaining[eligible]
        eligible_distances = distances[eligible]
        maximum = float(np.max(eligible_distances))
        tied = eligible_indices[
            np.isclose(eligible_distances, maximum, rtol=0.0, atol=1e-9)
        ]
        selected.append(int(tied[int(rng.integers(0, len(tied)))]))
    return values[np.asarray(selected)].astype(np.float32)


def _effective_task_radius_range(
    map_record: Mapping[str, Any],
    bundle: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    node_count: int,
    difficulty: str,
    depot_override_xy: Optional[Sequence[float]] = None,
) -> Tuple[float, Tuple[float, float]]:
    """仅按道路几何确定可布点半径，不读取任何算法或任务结果。"""

    generation = protocol["task_generation"]
    geometry = generation["geometry_feasibility"]
    coordinate_scale = float(
        bundle["metadata"]["coordinate_scale_m_per_unit"]
    )
    base_range = generation["radius_base_ranges_m"][str(difficulty)]
    offset = float(generation["node_radius_offsets_m"][str(node_count)])
    nominal_low = float(base_range[0]) + offset
    nominal_high = float(base_range[1]) + offset
    step = float(geometry["expansion_step_m"])
    maximum = float(geometry["maximum_radius_m"])
    trials = int(geometry["deterministic_trials"])
    minimum_successes = int(geometry["minimum_successes"])
    minimum_feasible: Optional[float] = None
    # 先在预注册名义区间内寻找最小可行半径，仅在名义上界仍失败时向外扩展。
    radius = nominal_low
    while radius <= maximum + 1e-9:
        try:
            _depot, candidates = _candidate_road_points(
                bundle,
                coordinate_scale_m_per_unit=coordinate_scale,
                minimum_depot_distance_m=float(
                    generation["minimum_depot_distance_m"]
                ),
                maximum_depot_distance_m=radius,
                sampling_spacing_m=float(
                    generation["road_sampling_spacing_m"]
                ),
                minimum_candidate_count=int(node_count),
                depot_override_xy=depot_override_xy,
            )
        except RuntimeError:
            radius += step
            continue
        successes = 0
        for trial in range(trials):
            seed = _task_seed(
                0,
                "geometry_feasibility",
                map_record["map_id"],
                int(node_count),
                trial,
            )
            try:
                _farthest_sample(
                    candidates,
                    int(node_count),
                    np.random.default_rng(seed),
                    coordinate_scale_m_per_unit=coordinate_scale,
                    minimum_spacing_m=float(
                        generation["minimum_node_spacing_m"]
                    ),
                )
                successes += 1
            except RuntimeError:
                pass
        if successes >= minimum_successes:
            minimum_feasible = float(radius)
            break
        radius += step
    if minimum_feasible is None:
        raise RuntimeError(
            f"{map_record['map_id']}在{maximum:.0f} m内无法稳定放置"
            f"{int(node_count)}个巡检点。"
        )
    effective_low = max(nominal_low, minimum_feasible)
    effective_high = max(
        nominal_high,
        effective_low + float(geometry["effective_range_width_m"]),
    )
    if effective_high > maximum + 1e-9:
        raise RuntimeError(
            f"{map_record['map_id']}的有效任务半径超过预注册上限。"
        )
    return minimum_feasible, (effective_low, effective_high)


def _task_priorities(
    points_xy: np.ndarray,
    depot_xy: np.ndarray,
    layout: str,
    rng: np.random.Generator,
) -> np.ndarray:
    from uav_inspection.experiments import paper_difficulty_experiments as difficulty

    count = len(points_xy)
    high_count, medium_count, _low_count = difficulty._priority_counts(count)
    priorities = np.ones(count, dtype=np.float32)
    distances = np.linalg.norm(
        np.asarray(points_xy) - np.asarray(depot_xy).reshape(1, 2), axis=1
    )
    if layout == "clustered":
        anchor = int(rng.integers(0, count))
        high_order = np.argsort(
            np.linalg.norm(points_xy - points_xy[anchor], axis=1),
            kind="mergesort",
        )
    elif layout == "dispersed":
        chosen = [int(np.argmax(distances))]
        while len(chosen) < high_count:
            remaining = [
                index for index in range(count) if index not in set(chosen)
            ]
            score = [
                min(
                    float(np.linalg.norm(points_xy[index] - points_xy[other]))
                    for other in chosen
                )
                for index in remaining
            ]
            chosen.append(remaining[int(np.argmax(score))])
        high_order = np.asarray(
            chosen
            + [index for index in np.argsort(-distances) if index not in chosen],
            dtype=np.int64,
        )
    elif layout == "far_high_conflict":
        high_order = np.argsort(-distances, kind="mergesort")
    else:
        raise ValueError(f"未知优先级布局：{layout}")
    high_indices = [int(value) for value in high_order[:high_count]]
    priorities[high_indices] = 3.0
    remaining = [
        index
        for index in np.argsort(distances, kind="mergesort")
        if index not in set(high_indices)
    ]
    priorities[remaining[:medium_count]] = 2.0
    return priorities


def _task_design(map_index: int, task_index: int) -> Dict[str, Any]:
    node_counts = (16, 20, 24)
    difficulties = ("moderate", "hard", "extreme")
    layouts = ("clustered", "dispersed", "far_high_conflict")
    constraints = ("energy", "distance", "time", "mixed")
    return {
        "node_count": node_counts[int(task_index) // 3],
        "difficulty": difficulties[int(task_index) % 3],
        "priority_layout": layouts[(int(task_index) + int(map_index)) % 3],
        "constraint_type": constraints[
            (int(map_index) * 9 + int(task_index)) % 4
        ],
    }


def _task_seed(master_seed: int, *parts: Any) -> int:
    text = "|".join([str(master_seed), *(str(part) for part in parts)])
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
    )


def _task_candidate(
    map_record: Mapping[str, Any],
    bundle: Mapping[str, Any],
    multimap_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    *,
    split: str,
    map_index: int,
    task_index: int,
    attempt: int,
    master_seed: int,
    geometry_radius_range_m: Optional[Tuple[float, float]] = None,
    geometry_minimum_feasible_radius_m: Optional[float] = None,
    seed_namespace: Optional[str] = None,
    road_index: Optional[int] = None,
    depot_override_xy: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    design = _task_design(map_index, task_index)
    seed = _task_seed(
        master_seed,
        split,
        map_record["map_id"],
        seed_namespace or "all_roads",
        task_index,
        attempt,
    )
    rng = np.random.default_rng(seed)
    coordinate_scale = float(
        bundle["metadata"]["coordinate_scale_m_per_unit"]
    )
    # 多地图的物理跨度不同，任务点半径必须在米制空间内按难度冻结。
    # 这里只使用输入地图和预注册难度标签，不读取任何待比较算法结果。
    generation = multimap_protocol["task_generation"]
    base_radius_range = generation["radius_base_ranges_m"][
        design["difficulty"]
    ]
    radius_offset_m = float(
        generation["node_radius_offsets_m"][str(design["node_count"])]
    )
    if geometry_radius_range_m is None:
        (
            geometry_minimum_feasible_radius_m,
            geometry_radius_range_m,
        ) = _effective_task_radius_range(
            map_record,
            bundle,
            multimap_protocol,
            node_count=int(design["node_count"]),
            difficulty=str(design["difficulty"]),
            depot_override_xy=depot_override_xy,
        )
    task_radius_m = float(rng.uniform(*geometry_radius_range_m))
    depot_xy, candidates = _candidate_road_points(
        bundle,
        coordinate_scale_m_per_unit=coordinate_scale,
        minimum_depot_distance_m=float(
            generation["minimum_depot_distance_m"]
        ),
        maximum_depot_distance_m=task_radius_m,
        sampling_spacing_m=float(generation["road_sampling_spacing_m"]),
        minimum_candidate_count=int(design["node_count"]),
        depot_override_xy=depot_override_xy,
    )
    points_xy = _farthest_sample(
        candidates,
        int(design["node_count"]),
        rng,
        coordinate_scale_m_per_unit=coordinate_scale,
        minimum_spacing_m=float(generation["minimum_node_spacing_m"]),
    )
    points = np.column_stack(
        [points_xy, np.zeros((len(points_xy),), dtype=np.float32)]
    )
    priorities = _task_priorities(
        points_xy, depot_xy, str(design["priority_layout"]), rng
    )
    # 原固定地图的绝对预算范围不能直接迁移到8 km多地图。
    # 候选范围故意较宽，最终是否进入数据集只由MILP覆盖区间和瓶颈证书决定。
    budget_ranges = generation["budget_candidate_ranges"]
    soc_range, distance_range, time_range = budget_ranges[
        design["difficulty"]
    ][design["constraint_type"]]
    initial_soc = float(rng.uniform(*soc_range))
    distance_scale = float(rng.uniform(*distance_range))
    time_scale = float(rng.uniform(*time_range))
    if design["constraint_type"] == "mixed":
        # 高节点数需要更大的几何半径才能满足点间距，因此同步放宽预算；
        # 随后的MILP仍须证明至少两类资源共同成为瓶颈。
        mixed_offsets = generation["mixed_node_budget_offsets"]
        soc_offset, distance_offset, time_offset = mixed_offsets[
            str(design["node_count"])
        ]
        initial_soc = min(1.0, initial_soc + soc_offset)
        distance_scale = min(1.2, distance_scale + distance_offset)
        time_scale = min(1.2, time_scale + time_offset)
    # 候选范围可覆盖储备阈值附近，但传入统一评价器前必须保留严格正裕量。
    initial_soc = max(
        initial_soc, _minimum_valid_initial_soc(multimap_protocol)
    )
    compensation = generation["geometry_budget_compensation"]
    nominal_radius_low = float(base_radius_range[0]) + radius_offset_m
    nominal_radius_high = float(base_radius_range[1]) + radius_offset_m
    effective_width = float(
        geometry_radius_range_m[1] - geometry_radius_range_m[0]
    )
    radius_quantile = (
        0.0
        if effective_width <= 1e-12
        else float(
            np.clip(
                (task_radius_m - float(geometry_radius_range_m[0]))
                / effective_width,
                0.0,
                1.0,
            )
        )
    )
    nominal_radius_at_sample = nominal_radius_low + radius_quantile * (
        nominal_radius_high - nominal_radius_low
    )
    compensation_factor = 1.0
    uncompensated_budgets = {
        "initial_soc": initial_soc,
        "distance_budget_scale": distance_scale,
        "time_budget_scale": time_scale,
    }
    compensation_applies = bool(compensation["enabled"]) and not (
        bool(compensation["single_constraint_tasks_only"])
        and design["constraint_type"] == "mixed"
    )
    if compensation_applies:
        compensation_factor = max(
            1.0,
            (task_radius_m / nominal_radius_at_sample)
            ** float(compensation["radius_elasticity"]),
        )
        compensated_values = {}
        for parameter, value in uncompensated_budgets.items():
            intended_parameter = {
                "energy": "initial_soc",
                "distance": "distance_budget_scale",
                "time": "time_budget_scale",
            }.get(str(design["constraint_type"]))
            if (
                bool(compensation["non_intended_resources_only"])
                and parameter == intended_parameter
            ):
                compensated_values[parameter] = float(value)
                continue
            bounds = compensation["parameter_bounds"][parameter]
            compensated_values[parameter] = float(
                np.clip(
                    float(value) * compensation_factor,
                    float(bounds[0]),
                    float(bounds[1]),
                )
            )
        initial_soc = compensated_values["initial_soc"]
        distance_scale = compensated_values["distance_budget_scale"]
        time_scale = compensated_values["time_budget_scale"]
    task_id = f"{split}__{map_record['map_id']}"
    if road_index is not None:
        task_id += f"__road_{int(road_index):02d}"
    task_id += f"__task_{task_index:02d}"
    record = {
        "id": task_id,
        "split": split,
        "map_id": str(map_record["map_id"]),
        "map_hash": str(map_record["map_hash"]),
        "map_file_sha256": str(map_record["map_file_sha256"]),
        "task_index": int(task_index),
        "instance_seed": int(seed % (2**32)),
        "generation_attempt": int(attempt),
        "start_xy": depot_xy.astype(float).tolist(),
        "task_radius_m": task_radius_m,
        "nominal_task_radius_range_m": [
            float(base_radius_range[0]) + radius_offset_m,
            float(base_radius_range[1]) + radius_offset_m,
        ],
        "geometry_minimum_feasible_radius_m": float(
            geometry_minimum_feasible_radius_m
        ),
        "nominal_radius_at_sample_m": nominal_radius_at_sample,
        "geometry_budget_compensation_factor": compensation_factor,
        "uncompensated_budget_values": uncompensated_budgets,
        "effective_task_radius_range_m": [
            float(geometry_radius_range_m[0]),
            float(geometry_radius_range_m[1]),
        ],
        "node_count": int(design["node_count"]),
        "difficulty": str(design["difficulty"]),
        "constraint_type": str(design["constraint_type"]),
        "priority_layout": str(design["priority_layout"]),
        "inspection_points_xyz": points.astype(float).tolist(),
        "priorities": priorities.astype(float).tolist(),
        "service_times_s": [20.0] * int(design["node_count"]),
        "initial_soc": initial_soc,
        "distance_budget_scale": distance_scale,
        "time_budget_scale": time_scale,
        "wind_scale": float(rng.uniform(1.0, 1.2)),
        "wind_rotation_deg": float(rng.uniform(-15.0, 15.0)),
        "wind_vertical_bias_mps": float(rng.uniform(-1.0, 1.0)),
        "power_scale": 1.0,
    }
    if road_index is not None:
        record["road_index"] = int(road_index)
    return record


def _certify_multimap_task(
    record: Mapping[str, Any],
    provider: FrozenMapProvider,
    parent_protocol: Mapping[str, Any],
    *,
    time_limit_s: float,
) -> Tuple[bool, Dict[str, Any], str]:
    from uav_inspection.core import final_python_ppo_pointer as ppo
    from uav_inspection.experiments import paper_difficulty_experiments as difficulty
    from python_classical_algs.common import PlannerBudget, make_problem
    from python_classical_algs.milp import plan_milp_orienteering

    context = provider(record)
    base_cfg = ppo.resolve_config(
        {
            "reward_schema": "multimap_v3_1",
            "coordinate_scale_m_per_unit": context["cfg_overrides"][
                "coordinate_scale_m_per_unit"
            ],
            "point_z_mode": "terrain",
            "terrain_clearance_m": 18.0,
            "service_times_s": record["service_times_s"],
        }
    )
    scenario_cfg, scenario_wind = ppo.apply_frozen_domain_instance(
        base_cfg, context["wind_data"], record
    )
    problem = make_problem(
        context["start_pos"],
        np.asarray(record["inspection_points_xyz"], dtype=np.float32),
        np.asarray(record["priorities"], dtype=np.float32),
        context["terrain"],
        scenario_cfg,
        scenario_wind,
        name=str(record["id"]),
    )
    result = plan_milp_orienteering(
        problem,
        seed=42,
        budget=PlannerBudget(
            max_evaluations=None, time_limit_s=float(time_limit_s)
        ),
        params={
            "objective_mode": "weighted_coverage",
            "mip_rel_gap": float(
                parent_protocol["certification"]["mip_rel_gap"]
            ),
            "presolve": True,
        },
    )
    metadata = dict(result.metadata)
    metrics = dict(result.metrics)
    lower = metadata.get("weighted_coverage_incumbent")
    upper = metadata.get("weighted_coverage_upper_bound")
    gap = metadata.get("mip_gap")
    certificate = {
        "algorithm": "milp_weighted_coverage",
        "solver_status": metadata.get("solver_status"),
        "solver_success": metadata.get("solver_success"),
        "solver_message": metadata.get("solver_message"),
        "status": result.status,
        "mip_gap": gap,
        "weighted_coverage_lower_bound": lower,
        "weighted_coverage_upper_bound": upper,
        "optimality_certified": bool(metadata.get("optimality_certified")),
        "visit_order": list(result.visit_order),
        "visited_count": int(metrics.get("visited_count", len(result.visit_order))),
        "returned": bool(metrics.get("returned", False)),
        "energy_utilization": float(
            metrics.get("energy_utilization", math.inf)
        ),
        "distance_utilization": float(
            metrics.get("distance_utilization", math.inf)
        ),
        "time_utilization": float(metrics.get("time_utilization", math.inf)),
        "runtime_s": float(result.runtime_s),
        "scenario_hash": str(result.scenario_hash),
        "map_id": str(record["map_id"]),
        "map_hash": str(record["map_hash"]),
    }
    certification = parent_protocol["certification"]
    if lower is None or upper is None or gap is None:
        return False, certificate, "missing_solver_bound"
    lower_value, upper_value, gap_value = float(lower), float(upper), float(gap)
    if not all(
        math.isfinite(value) for value in (lower_value, upper_value, gap_value)
    ):
        return False, certificate, "nonfinite_solver_bound"
    if lower_value > upper_value + 1e-7:
        return False, certificate, "inverted_solver_bounds"
    if not certificate["returned"] or certificate["visited_count"] < int(
        certification["minimum_visited_count"]
    ):
        return False, certificate, "no_safe_partial_route"
    band_low, band_high = (
        float(value)
        for value in parent_protocol["difficulty_bands"][
            str(record["difficulty"])
        ]
    )
    tolerance = float(certification["band_tolerance"])
    if lower_value < band_low - tolerance or lower_value > band_high + tolerance:
        return False, certificate, "incumbent_outside_band"
    if upper_value >= float(certification["full_coverage_upper_bound_max"]) - tolerance:
        return False, certificate, "full_coverage_not_excluded"
    same_band = upper_value <= band_high + tolerance
    small_gap = gap_value <= float(certification["mip_rel_gap"]) + 1e-10
    certificate["difficulty_certificate"] = (
        "bounds_same_band" if same_band else "mip_gap"
    )
    if not (same_band or small_gap):
        return False, certificate, "bounds_and_gap_insufficient"
    bottlenecks = difficulty._resource_bottlenecks(
        metrics,
        minimum=float(certification["bottleneck_utilization_min"]),
        max_gap=float(certification["single_bottleneck_max_gap"]),
    )
    certificate["bottleneck_resources"] = list(bottlenecks)
    intended = str(record["constraint_type"])
    if intended == "mixed":
        if len(bottlenecks) < int(certification["mixed_min_active_resources"]):
            return False, certificate, "mixed_bottleneck_not_active"
    elif intended not in bottlenecks:
        return False, certificate, "intended_bottleneck_not_active"
    return True, certificate, "accepted"


def _certify_with_resource_thresholds(
    record: Mapping[str, Any],
    provider: FrozenMapProvider,
    multimap_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
) -> Tuple[bool, Dict[str, Any], str]:
    """用低阈值可行路线和高阈值资源下界组成严格的双边难度证书。"""

    from uav_inspection.core import final_python_ppo_pointer as ppo
    from uav_inspection.experiments import paper_difficulty_experiments as difficulty
    from python_classical_algs.common import MissionEvaluator, make_problem
    from python_classical_algs.milp import solve_resource_threshold_milp

    intended = str(record.get("constraint_type", ""))
    if intended not in {"energy", "distance", "time"}:
        return False, {}, "resource_threshold_ineligible_constraint"
    priorities = np.asarray(record.get("priorities", ()), dtype=np.float64)
    if (
        priorities.ndim != 1
        or priorities.size == 0
        or not np.all(np.isfinite(priorities))
        or not np.allclose(priorities, np.rint(priorities), atol=1e-9)
    ):
        return False, {}, "resource_threshold_requires_integer_priorities"
    total_priority = float(np.sum(priorities))
    if total_priority <= 0.0:
        return False, {}, "resource_threshold_invalid_priority_total"
    band_low, band_high = (
        float(value)
        for value in multimap_protocol["difficulty_bands"][
            str(record["difficulty"])
        ]
    )
    low_required = int(math.ceil(band_low * total_priority - 1e-9))
    high_required = int(math.floor(band_high * total_priority + 1e-9)) + 1
    if not 0 < low_required < high_required <= int(round(total_priority)):
        return False, {}, "resource_threshold_invalid_discrete_bounds"

    context = provider(record)
    base_cfg = ppo.resolve_config(
        {
            "reward_schema": "multimap_v3_1",
            "coordinate_scale_m_per_unit": context["cfg_overrides"][
                "coordinate_scale_m_per_unit"
            ],
            "point_z_mode": "terrain",
            "terrain_clearance_m": 18.0,
            "service_times_s": record["service_times_s"],
        }
    )
    scenario_cfg, scenario_wind = ppo.apply_frozen_domain_instance(
        base_cfg, context["wind_data"], record
    )
    problem = make_problem(
        context["start_pos"],
        np.asarray(record["inspection_points_xyz"], dtype=np.float32),
        priorities.astype(np.float32),
        context["terrain"],
        scenario_cfg,
        scenario_wind,
        name=str(record["id"]),
    )
    fallback = multimap_protocol["task_generation"][
        "resource_threshold_fallback"
    ]
    low_proof = solve_resource_threshold_milp(
        problem,
        resource_name=intended,
        minimum_priority_weight=float(low_required),
        time_limit_s=float(fallback["lower_time_limit_s"]),
    )
    low_order = low_proof.get("visit_order")
    if low_order is None:
        return False, {"low_threshold": low_proof}, "low_threshold_no_route"
    evaluator = MissionEvaluator(problem)
    low_evaluation = evaluator.evaluate_order(low_order)
    if not low_evaluation.returned:
        return (
            False,
            {"low_threshold": low_proof},
            "low_threshold_route_not_safe",
        )
    lower = float(low_evaluation.weighted_coverage)
    tolerance = float(parent_protocol["certification"]["band_tolerance"])
    if not band_low - tolerance <= lower <= band_high + tolerance:
        return (
            False,
            {"low_threshold": low_proof},
            "low_threshold_route_outside_band",
        )

    high_proof = solve_resource_threshold_milp(
        problem,
        resource_name=intended,
        minimum_priority_weight=float(high_required),
        time_limit_s=float(fallback["upper_time_limit_s"]),
    )
    if not bool(high_proof["threshold_impossible_under_actual_budget"]):
        return (
            False,
            {"low_threshold": low_proof, "high_threshold": high_proof},
            "high_threshold_not_excluded",
        )
    upper = float(high_required - 1) / total_priority
    if upper > band_high + tolerance:
        return (
            False,
            {"low_threshold": low_proof, "high_threshold": high_proof},
            "resource_threshold_upper_outside_band",
        )
    metrics = {
        "energy_utilization": (
            float(low_evaluation.energy_wh) / evaluator.energy_budget_wh
        ),
        "distance_utilization": (
            float(low_evaluation.distance_m) / evaluator.distance_budget_m
        ),
        "time_utilization": (
            float(low_evaluation.time_s) / evaluator.time_budget_s
        ),
    }
    bottlenecks = difficulty._resource_bottlenecks(
        metrics,
        minimum=float(
            parent_protocol["certification"]["bottleneck_utilization_min"]
        ),
        max_gap=float(
            parent_protocol["certification"]["single_bottleneck_max_gap"]
        ),
    )
    if intended not in bottlenecks:
        return (
            False,
            {"low_threshold": low_proof, "high_threshold": high_proof},
            "resource_threshold_intended_bottleneck_not_active",
        )
    relative_gap = max(0.0, upper - lower) / max(lower, 1e-12)
    certificate = {
        "algorithm": "milp_weighted_coverage",
        "solver_status": high_proof["solver_status"],
        "solver_success": True,
        "solver_message": high_proof["solver_message"],
        "status": str(low_evaluation.termination_reason),
        "mip_gap": relative_gap,
        "weighted_coverage_lower_bound": lower,
        "weighted_coverage_upper_bound": upper,
        "optimality_certified": math.isclose(
            lower, upper, rel_tol=0.0, abs_tol=1e-12
        ),
        "visit_order": list(low_evaluation.order),
        "visited_count": len(low_evaluation.order),
        "returned": True,
        "energy_utilization": metrics["energy_utilization"],
        "distance_utilization": metrics["distance_utilization"],
        "time_utilization": metrics["time_utilization"],
        "runtime_s": float(low_proof["runtime_s"])
        + float(high_proof["runtime_s"]),
        "scenario_hash": str(problem.scenario_hash),
        "map_id": str(record["map_id"]),
        "map_hash": str(record["map_hash"]),
        "difficulty_certificate": "resource_threshold_bounds_same_band",
        "bottleneck_resources": list(bottlenecks),
        "resource_threshold_proof": {
            "resource_name": intended,
            "low_required_priority": low_required,
            "high_required_priority": high_required,
            "total_priority": total_priority,
            "low_threshold": low_proof,
            "high_threshold": high_proof,
        },
    }
    return True, certificate, "accepted"


def _screening_bounds_intersect_band(
    record: Mapping[str, Any],
    certificate: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
) -> bool:
    """短时求解只负责排除必定不在目标区间的候选。"""

    try:
        lower = float(certificate["weighted_coverage_lower_bound"])
        upper = float(certificate["weighted_coverage_upper_bound"])
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(lower) or not math.isfinite(upper):
        return False
    band_low, band_high = (
        float(value)
        for value in parent_protocol["difficulty_bands"][
            str(record["difficulty"])
        ]
    )
    tolerance = float(parent_protocol["certification"]["band_tolerance"])
    return bool(
        certificate.get("returned")
        and int(certificate.get("visited_count", 0)) >= 1
        and upper >= band_low - tolerance
        and lower <= band_high + tolerance
    )


def _screening_certificate_is_decisive(
    screening_ok: bool, certificate: Mapping[str, Any]
) -> bool:
    """短时证书已满足正式判据或已认证最优时，无需重复运行长时MILP。"""

    return bool(screening_ok) or bool(certificate.get("optimality_certified"))


def _certify_mixed_with_lower_threshold_route(
    record: Mapping[str, Any],
    standard_certificate: Mapping[str, Any],
    provider: FrozenMapProvider,
    multimap_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
) -> Tuple[bool, Dict[str, Any], Dict[str, Any], str]:
    """以安全下阈值路线补足混合约束下界，并保留收紧前的严格上界。"""

    from uav_inspection.core import final_python_ppo_pointer as ppo
    from uav_inspection.experiments import paper_difficulty_experiments as difficulty
    from python_classical_algs.common import MissionEvaluator, make_problem
    from python_classical_algs.milp import solve_resource_threshold_milp

    config = multimap_protocol["task_generation"].get(
        "mixed_threshold_certificate", {}
    )
    if not bool(config.get("enabled", False)) or str(
        record.get("constraint_type", "")
    ) != "mixed":
        return False, dict(record), {}, "mixed_threshold_ineligible"
    priorities = np.asarray(record.get("priorities", ()), dtype=np.float64)
    if (
        priorities.ndim != 1
        or priorities.size == 0
        or not np.all(np.isfinite(priorities))
        or not np.allclose(priorities, np.rint(priorities), atol=1e-9)
    ):
        return (
            False,
            dict(record),
            {},
            "mixed_threshold_requires_integer_priorities",
        )
    total_priority = float(np.sum(priorities))
    band_low, band_high = (
        float(value)
        for value in multimap_protocol["difficulty_bands"][
            str(record["difficulty"])
        ]
    )
    tolerance = float(parent_protocol["certification"]["band_tolerance"])
    standard_upper = float(
        standard_certificate.get("weighted_coverage_upper_bound", math.nan)
    )
    if (
        not math.isfinite(standard_upper)
        or standard_upper > band_high + tolerance
        or standard_upper
        >= float(
            parent_protocol["certification"][
                "full_coverage_upper_bound_max"
            ]
        )
        - tolerance
    ):
        return (
            False,
            dict(record),
            {},
            "mixed_threshold_requires_same_band_upper",
        )
    low_required = int(math.ceil(band_low * total_priority - 1e-9))
    if not 0 < low_required <= int(round(total_priority)):
        return (
            False,
            dict(record),
            {},
            "mixed_threshold_invalid_discrete_lower_bound",
        )
    standard_lower = float(
        standard_certificate.get("weighted_coverage_lower_bound", math.nan)
    )
    if not math.isfinite(standard_lower):
        return (
            False,
            dict(record),
            {},
            "mixed_threshold_missing_standard_lower",
        )
    incumbent_priority = int(round(standard_lower * total_priority))
    shortfall = max(0, low_required - incumbent_priority)
    if shortfall > int(config["maximum_incumbent_priority_shortfall"]):
        return (
            False,
            dict(record),
            {
                "low_required_priority": low_required,
                "standard_incumbent_priority": incumbent_priority,
                "priority_shortfall": shortfall,
            },
            "mixed_threshold_incumbent_too_far_below_band",
        )

    context = provider(record)

    def build_problem(candidate: Mapping[str, Any]) -> Any:
        base_cfg = ppo.resolve_config(
            {
                "reward_schema": "multimap_v3_1",
                "coordinate_scale_m_per_unit": context["cfg_overrides"][
                    "coordinate_scale_m_per_unit"
                ],
                "point_z_mode": "terrain",
                "terrain_clearance_m": 18.0,
                "service_times_s": candidate["service_times_s"],
            }
        )
        scenario_cfg, scenario_wind = ppo.apply_frozen_domain_instance(
            base_cfg, context["wind_data"], candidate
        )
        return make_problem(
            context["start_pos"],
            np.asarray(
                candidate["inspection_points_xyz"], dtype=np.float32
            ),
            priorities.astype(np.float32),
            context["terrain"],
            scenario_cfg,
            scenario_wind,
            name=str(candidate["id"]),
        )

    original_problem = build_problem(record)
    original_evaluator = MissionEvaluator(original_problem)
    attempted_proofs: List[Dict[str, Any]] = []
    resource_order = tuple(
        str(value) for value in config.get("resource_order", ())
    )
    if (
        not resource_order
        or len(set(resource_order)) != len(resource_order)
        or not set(resource_order) <= {"energy", "distance", "time"}
    ):
        raise ValueError("mixed_threshold_certificate resource_order无效。")
    route_sources: List[Tuple[str, Optional[List[int]]]] = []
    standard_order = standard_certificate.get("visit_order")
    if bool(config.get("standard_incumbent_fast_path", False)) and standard_order:
        route_sources.append(
            ("standard_incumbent", list(standard_order))
        )
    route_sources.extend((resource, None) for resource in resource_order)
    for resource_name, provided_order in route_sources:
        if resource_name == "standard_incumbent":
            order = provided_order
        else:
            proof = solve_resource_threshold_milp(
                original_problem,
                resource_name=resource_name,
                minimum_priority_weight=float(low_required),
                time_limit_s=float(config["lower_time_limit_s"]),
            )
            attempted_proofs.append(proof)
            order = proof.get("visit_order")
        if order is None:
            continue
        evaluation = original_evaluator.evaluate_order(order)
        lower = float(evaluation.weighted_coverage)
        if (
            not evaluation.returned
            or not band_low - tolerance <= lower <= band_high + tolerance
        ):
            continue
        metrics = {
            "energy_utilization": float(evaluation.energy_wh)
            / original_evaluator.energy_budget_wh,
            "distance_utilization": float(evaluation.distance_m)
            / original_evaluator.distance_budget_m,
            "time_utilization": float(evaluation.time_s)
            / original_evaluator.time_budget_s,
        }
        route_certificate = {
            "weighted_coverage_lower_bound": lower,
            "weighted_coverage_upper_bound": standard_upper,
            "returned": True,
            "visited_count": len(evaluation.order),
            "energy_utilization": metrics["energy_utilization"],
            "distance_utilization": metrics["distance_utilization"],
            "time_utilization": metrics["time_utilization"],
            "scenario_hash": str(original_problem.scenario_hash),
        }
        bottlenecks = difficulty._resource_bottlenecks(
            metrics,
            minimum=float(
                parent_protocol["certification"][
                    "bottleneck_utilization_min"
                ]
            ),
            max_gap=float(
                parent_protocol["certification"][
                    "single_bottleneck_max_gap"
                ]
            ),
        )
        updated = dict(record)
        monotone_adjustment = None
        if len(bottlenecks) < int(
            parent_protocol["certification"]["mixed_min_active_resources"]
        ):
            updated_candidate = _calibrate_mixed_candidate(
                record,
                route_certificate,
                multimap_protocol,
                parent_protocol,
                iteration=len(
                    record.get("mixed_budget_calibration_trace", ())
                )
                + 1,
            )
            if updated_candidate is None:
                continue
            changed = [
                name
                for name in (
                    "initial_soc",
                    "distance_budget_scale",
                    "time_budget_scale",
                )
                if not math.isclose(
                    float(updated_candidate[name]),
                    float(record[name]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
            if len(changed) != 1:
                continue
            parameter = changed[0]
            before, after = (
                float(record[parameter]),
                float(updated_candidate[parameter]),
            )
            # 只允许收紧预算；这样旧场景的上界对新场景仍是严格上界。
            if after >= before - 1e-12:
                continue
            updated = updated_candidate
            monotone_adjustment = {
                "parameter": parameter,
                "before": before,
                "after": after,
                "rule": "tighten_one_budget_from_in_band_threshold_route",
            }

        final_problem = build_problem(updated)
        final_evaluator = MissionEvaluator(final_problem)
        final_evaluation = final_evaluator.evaluate_order(order)
        if not final_evaluation.returned:
            continue
        lower = float(final_evaluation.weighted_coverage)
        final_metrics = {
            "energy_utilization": float(final_evaluation.energy_wh)
            / final_evaluator.energy_budget_wh,
            "distance_utilization": float(final_evaluation.distance_m)
            / final_evaluator.distance_budget_m,
            "time_utilization": float(final_evaluation.time_s)
            / final_evaluator.time_budget_s,
        }
        bottlenecks = difficulty._resource_bottlenecks(
            final_metrics,
            minimum=float(
                parent_protocol["certification"][
                    "bottleneck_utilization_min"
                ]
            ),
            max_gap=float(
                parent_protocol["certification"][
                    "single_bottleneck_max_gap"
                ]
            ),
        )
        if (
            not band_low - tolerance <= lower <= band_high + tolerance
            or len(bottlenecks)
            < int(
                parent_protocol["certification"][
                    "mixed_min_active_resources"
                ]
            )
        ):
            continue
        relative_gap = max(0.0, standard_upper - lower) / max(
            lower, 1e-12
        )
        certificate = {
            "algorithm": "milp_weighted_coverage",
            "solver_status": standard_certificate.get("solver_status"),
            "solver_success": True,
            "solver_message": (
                "mixed lower-threshold route plus monotone same-band upper"
            ),
            "status": str(final_evaluation.termination_reason),
            "mip_gap": relative_gap,
            "weighted_coverage_lower_bound": lower,
            "weighted_coverage_upper_bound": standard_upper,
            "optimality_certified": math.isclose(
                lower, standard_upper, rel_tol=0.0, abs_tol=1e-12
            ),
            "visit_order": list(final_evaluation.order),
            "visited_count": len(final_evaluation.order),
            "returned": True,
            "energy_utilization": final_metrics["energy_utilization"],
            "distance_utilization": final_metrics[
                "distance_utilization"
            ],
            "time_utilization": final_metrics["time_utilization"],
            "runtime_s": float(
                standard_certificate.get("runtime_s", 0.0)
            )
            + sum(
                float(item.get("runtime_s", 0.0))
                for item in attempted_proofs
            ),
            "scenario_hash": str(final_problem.scenario_hash),
            "map_id": str(record["map_id"]),
            "map_hash": str(record["map_hash"]),
            "difficulty_certificate": (
                "mixed_lower_threshold_plus_monotone_same_band_upper"
            ),
            "bottleneck_resources": list(bottlenecks),
            "mixed_threshold_proof": {
                "resource_order": list(resource_order),
                "route_source": (
                    "standard_incumbent"
                    if resource_name == "standard_incumbent"
                    else "resource_threshold"
                ),
                "successful_resource": resource_name,
                "low_required_priority": low_required,
                "total_priority": total_priority,
                "low_threshold_attempts": attempted_proofs,
                "standard_upper_certificate": {
                    "scenario_hash": str(
                        standard_certificate.get("scenario_hash", "")
                    ),
                    "weighted_coverage_upper_bound": standard_upper,
                    "solver_status": standard_certificate.get(
                        "solver_status"
                    ),
                    "solver_message": standard_certificate.get(
                        "solver_message"
                    ),
                    "visit_order": list(
                        standard_certificate.get("visit_order") or ()
                    ),
                },
                "monotone_adjustment": monotone_adjustment,
            },
        }
        return True, updated, certificate, "accepted"
    return (
        False,
        dict(record),
        {"low_threshold_attempts": attempted_proofs},
        "mixed_threshold_no_acceptable_route",
    )


def _calibrate_mixed_candidate(
    candidate: Mapping[str, Any],
    certificate: Mapping[str, Any],
    multimap_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    *,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    """只用MILP证书最小幅度收紧第二瓶颈预算，不读取待比较算法结果。"""

    calibration = multimap_protocol["task_generation"][
        "mixed_budget_calibration"
    ]
    if not bool(calibration["enabled"]) or str(
        candidate.get("constraint_type", "")
    ) != "mixed":
        return None
    try:
        lower = float(certificate["weighted_coverage_lower_bound"])
        utilities = np.asarray(
            [
                certificate["energy_utilization"],
                certificate["distance_utilization"],
                certificate["time_utilization"],
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not bool(certificate.get("returned"))
        or int(certificate.get("visited_count", 0)) < 1
        or not math.isfinite(lower)
        or not np.all(np.isfinite(utilities))
        or np.any(utilities <= 0.0)
    ):
        return None
    band = parent_protocol["difficulty_bands"][str(candidate["difficulty"])]
    if bool(calibration["requires_incumbent_in_target_band"]) and not (
        float(band[0]) <= lower <= float(band[1])
    ):
        return None
    order = np.argsort(-utilities, kind="mergesort")
    top_index, second_index = int(order[0]), int(order[1])
    maximum_gap = float(
        parent_protocol["certification"]["single_bottleneck_max_gap"]
    )
    current_gap = float(utilities[top_index] - utilities[second_index])
    if current_gap <= maximum_gap + 1e-12:
        return None
    target_gap = maximum_gap - float(calibration["target_gap_margin"])
    if target_gap <= 0.0:
        raise ValueError("mixed_budget_calibration target gap must be positive")
    target_utilization = float(utilities[top_index] - target_gap)
    if target_utilization <= 0.0:
        return None
    parameter_names = (
        "initial_soc",
        "distance_budget_scale",
        "time_budget_scale",
    )
    parameter = parameter_names[second_index]
    before = float(candidate[parameter])
    bounds = calibration["parameter_bounds"][parameter]
    if parameter == "initial_soc":
        bounds = (
            max(
                float(bounds[0]),
                _minimum_valid_initial_soc(multimap_protocol),
            ),
            float(bounds[1]),
        )
    after = float(
        np.clip(
            before * float(utilities[second_index]) / target_utilization,
            float(bounds[0]),
            float(bounds[1]),
        )
    )
    if math.isclose(before, after, rel_tol=0.0, abs_tol=1e-9):
        return None
    updated = dict(candidate)
    updated[parameter] = after
    trace = list(candidate.get("mixed_budget_calibration_trace", ()))
    trace.append(
        {
            "iteration": int(iteration),
            "adjusted_resource": ("energy", "distance", "time")[second_index],
            "adjusted_parameter": parameter,
            "parameter_before": before,
            "parameter_after": after,
            "resource_utilizations_before": utilities.astype(float).tolist(),
            "target_utilization": target_utilization,
            "weighted_coverage_lower_bound_before": lower,
            "certificate_scenario_hash_before": str(
                certificate.get("scenario_hash", "")
            ),
        }
    )
    updated["mixed_budget_calibration_trace"] = trace
    return updated


def _calibrate_single_constraint_candidate(
    candidate: Mapping[str, Any],
    certificate: Mapping[str, Any],
    multimap_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    *,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    """按MILP覆盖方向单调调整唯一目标资源预算，最多执行预注册轮数。"""

    intended = str(candidate.get("constraint_type", ""))
    if intended not in {"energy", "distance", "time"}:
        return None
    calibration = multimap_protocol["task_generation"][
        "single_constraint_budget_calibration"
    ]
    if not bool(calibration["enabled"]):
        return None
    resource_names = ("energy", "distance", "time")
    parameter_names = (
        "initial_soc",
        "distance_budget_scale",
        "time_budget_scale",
    )
    intended_index = resource_names.index(intended)
    try:
        lower = float(certificate["weighted_coverage_lower_bound"])
        upper = float(certificate["weighted_coverage_upper_bound"])
        utilities = np.asarray(
            [
                certificate["energy_utilization"],
                certificate["distance_utilization"],
                certificate["time_utilization"],
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not bool(certificate.get("returned"))
        or int(certificate.get("visited_count", 0)) < 1
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or not np.all(np.isfinite(utilities))
        or np.any(utilities <= 0.0)
    ):
        return None
    band_low, band_high = (
        float(value)
        for value in parent_protocol["difficulty_bands"][
            str(candidate["difficulty"])
        ]
    )
    intended_utilization = float(utilities[intended_index])
    direction: str
    if lower < band_low:
        target_utilization = float(
            calibration["below_band_target_utilization"]
        )
        direction = "loosen"
    elif lower > band_high:
        target_utilization = float(
            calibration["above_band_target_utilization"]
        )
        direction = "tighten"
    else:
        maximum_utilization = float(np.max(utilities))
        maximum_gap = float(
            parent_protocol["certification"]["single_bottleneck_max_gap"]
        )
        if maximum_utilization - intended_utilization <= maximum_gap + 1e-12:
            return None
        target_utilization = maximum_utilization - (
            maximum_gap - float(calibration["target_gap_margin"])
        )
        direction = "tighten"
    if target_utilization <= 0.0:
        return None
    parameter = parameter_names[intended_index]
    before = float(candidate[parameter])
    proposed = before * intended_utilization / target_utilization
    if lower < band_low:
        proposed = max(
            proposed,
            before * float(calibration["minimum_loosen_factor"]),
        )
    elif lower > band_high:
        proposed = min(
            proposed,
            before * float(calibration["maximum_tighten_factor"]),
        )
    if direction == "loosen" and proposed <= before + 1e-9:
        return None
    if direction == "tighten" and proposed >= before - 1e-9:
        return None
    bounds = calibration["parameter_bounds"][parameter]
    if parameter == "initial_soc":
        bounds = (
            max(
                float(bounds[0]),
                _minimum_valid_initial_soc(multimap_protocol),
            ),
            float(bounds[1]),
        )
    after = float(np.clip(proposed, float(bounds[0]), float(bounds[1])))
    if math.isclose(before, after, rel_tol=0.0, abs_tol=1e-9):
        return None
    updated = dict(candidate)
    updated[parameter] = after
    trace = list(candidate.get("single_constraint_budget_calibration_trace", ()))
    trace.append(
        {
            "iteration": int(iteration),
            "intended_resource": intended,
            "adjusted_parameter": parameter,
            "direction": direction,
            "parameter_before": before,
            "parameter_after": after,
            "resource_utilizations_before": utilities.astype(float).tolist(),
            "target_utilization": target_utilization,
            "weighted_coverage_bounds_before": [lower, upper],
            "certificate_scenario_hash_before": str(
                certificate.get("scenario_hash", "")
            ),
        }
    )
    updated["single_constraint_budget_calibration_trace"] = trace
    return updated


def _resource_threshold_fallback_is_triggered(
    attempt: int, fallback: Mapping[str, Any]
) -> bool:
    """按显式节点或固定周期触发昂贵的资源阈值认证。"""

    value = int(attempt)
    if value in {int(item) for item in fallback.get("trigger_attempts", ())}:
        return True
    schedule = dict(fallback.get("periodic_trigger_schedule") or {})
    if not bool(schedule.get("enabled", False)):
        return False
    first = int(schedule["first_attempt"])
    last = int(schedule["last_attempt"])
    interval = int(schedule["interval_attempts"])
    return (
        interval > 0
        and first <= value <= last
        and (value - first) % interval == 0
    )


def _relax_non_intended_resources(
    candidate: Mapping[str, Any],
    multimap_protocol: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """单约束后备路径先释放非目标资源，避免把假性瓶颈归因于目标资源。"""

    intended = str(candidate.get("constraint_type", ""))
    resources = ("energy", "distance", "time")
    parameters = (
        "initial_soc",
        "distance_budget_scale",
        "time_budget_scale",
    )
    if intended not in resources:
        return None
    fallback = multimap_protocol["task_generation"].get(
        "resource_threshold_fallback", {}
    )
    relaxation = dict(
        fallback.get("non_intended_resource_relaxation") or {}
    )
    if not bool(relaxation.get("enabled", False)):
        return None
    if candidate.get("non_intended_resource_relaxation"):
        return None
    parameter_bounds = multimap_protocol["task_generation"][
        "single_constraint_budget_calibration"
    ]["parameter_bounds"]
    before = {parameter: float(candidate[parameter]) for parameter in parameters}
    after = dict(before)
    for resource, parameter in zip(resources, parameters):
        if resource != intended:
            after[parameter] = float(parameter_bounds[parameter][1])
    updated = dict(candidate)
    updated.update(after)
    updated["non_intended_resource_relaxation"] = {
        "rule": "single_constraint_release_non_intended_to_upper_bound_v1",
        "intended_resource": intended,
        "before": before,
        "after": after,
        "released_parameters": [
            parameter
            for resource, parameter in zip(resources, parameters)
            if resource != intended
        ],
    }
    return updated


def _audit_budget_transform_record(
    record: Mapping[str, Any],
    multimap_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
) -> List[str]:
    """重放几何补偿与MILP预算校准，防止任务记录被静默篡改。"""

    task_id = str(record.get("id", ""))
    reasons: List[str] = []
    generation = multimap_protocol["task_generation"]
    parameters = (
        "initial_soc",
        "distance_budget_scale",
        "time_budget_scale",
    )
    resources = ("energy", "distance", "time")
    intended = str(record.get("constraint_type", ""))
    parameter_for_resource = dict(zip(resources, parameters))
    try:
        node_count = int(record["node_count"])
        difficulty = str(record["difficulty"])
        radius = float(record["task_radius_m"])
        effective_low, effective_high = (
            float(value)
            for value in record["effective_task_radius_range_m"]
        )
        base_low, base_high = (
            float(value)
            for value in generation["radius_base_ranges_m"][difficulty]
        )
        offset = float(
            generation["node_radius_offsets_m"][str(node_count)]
        )
        uncompensated = {
            parameter: float(record["uncompensated_budget_values"][parameter])
            for parameter in parameters
        }
    except (KeyError, TypeError, ValueError):
        return [f"budget_transform_fields_invalid={task_id}"]
    if not all(
        math.isfinite(value)
        for value in (
            radius,
            effective_low,
            effective_high,
            base_low,
            base_high,
            offset,
            *uncompensated.values(),
        )
    ):
        return [f"budget_transform_nonfinite={task_id}"]
    minimum_initial_soc = _minimum_valid_initial_soc(multimap_protocol)
    if (
        float(record.get("initial_soc", -math.inf))
        < minimum_initial_soc - 1e-12
        or uncompensated["initial_soc"] < minimum_initial_soc - 1e-12
    ):
        reasons.append(f"initial_soc_below_evaluator_floor={task_id}")
    width = effective_high - effective_low
    quantile = (
        0.0
        if width <= 1e-12
        else float(np.clip((radius - effective_low) / width, 0.0, 1.0))
    )
    expected_nominal = (base_low + offset) + quantile * (
        base_high - base_low
    )
    recorded_nominal = float(record.get("nominal_radius_at_sample_m", math.nan))
    if not math.isclose(
        recorded_nominal, expected_nominal, rel_tol=0.0, abs_tol=1e-9
    ):
        reasons.append(f"nominal_radius_sample_mismatch={task_id}")

    compensation = generation["geometry_budget_compensation"]
    compensation_applies = bool(compensation["enabled"]) and not (
        bool(compensation["single_constraint_tasks_only"])
        and intended == "mixed"
    )
    expected_factor = (
        max(
            1.0,
            (radius / expected_nominal)
            ** float(compensation["radius_elasticity"]),
        )
        if compensation_applies
        else 1.0
    )
    recorded_factor = float(
        record.get("geometry_budget_compensation_factor", math.nan)
    )
    if not math.isclose(
        recorded_factor, expected_factor, rel_tol=0.0, abs_tol=1e-9
    ):
        reasons.append(f"geometry_compensation_factor_mismatch={task_id}")

    current = dict(uncompensated)
    intended_parameter = parameter_for_resource.get(intended)
    if compensation_applies:
        for parameter in parameters:
            if (
                bool(compensation["non_intended_resources_only"])
                and parameter == intended_parameter
            ):
                continue
            low, high = (
                float(value)
                for value in compensation["parameter_bounds"][parameter]
            )
            current[parameter] = float(
                np.clip(current[parameter] * expected_factor, low, high)
            )

    relaxation_trace = dict(
        record.get("non_intended_resource_relaxation") or {}
    )
    fallback = generation.get("resource_threshold_fallback", {})
    relaxation_config = dict(
        fallback.get("non_intended_resource_relaxation") or {}
    )
    if relaxation_trace:
        try:
            before_values = {
                parameter: float(relaxation_trace["before"][parameter])
                for parameter in parameters
            }
            after_values = {
                parameter: float(relaxation_trace["after"][parameter])
                for parameter in parameters
            }
            released = list(relaxation_trace["released_parameters"])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"non_intended_relaxation_invalid={task_id}")
        else:
            expected_released = [
                parameter
                for resource, parameter in zip(resources, parameters)
                if resource != intended
            ]
            bounds = generation["single_constraint_budget_calibration"][
                "parameter_bounds"
            ]
            expected_after = dict(current)
            for parameter in expected_released:
                expected_after[parameter] = float(bounds[parameter][1])
            if (
                intended not in resources
                or not bool(relaxation_config.get("enabled", False))
                or relaxation_trace.get("rule")
                != "single_constraint_release_non_intended_to_upper_bound_v1"
                or relaxation_trace.get("intended_resource") != intended
                or released != expected_released
                or any(
                    not math.isclose(
                        before_values[parameter],
                        current[parameter],
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    for parameter in parameters
                )
                or any(
                    not math.isclose(
                        after_values[parameter],
                        expected_after[parameter],
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    for parameter in parameters
                )
            ):
                reasons.append(
                    f"non_intended_relaxation_replay_failed={task_id}"
                )
            else:
                current = expected_after

    single_trace = list(
        record.get("single_constraint_budget_calibration_trace", ())
    )
    mixed_trace = list(record.get("mixed_budget_calibration_trace", ()))
    single_config = generation["single_constraint_budget_calibration"]
    mixed_config = generation["mixed_budget_calibration"]
    if intended == "mixed":
        if single_trace:
            reasons.append(f"mixed_has_single_calibration_trace={task_id}")
        if len(mixed_trace) > int(mixed_config["maximum_iterations"]):
            reasons.append(f"mixed_calibration_iterations_exceeded={task_id}")
    else:
        if mixed_trace:
            reasons.append(f"single_has_mixed_calibration_trace={task_id}")
        if len(single_trace) > int(single_config["maximum_iterations"]):
            reasons.append(f"single_calibration_iterations_exceeded={task_id}")

    for expected_iteration, item in enumerate(single_trace, start=1):
        try:
            utilities = np.asarray(
                item["resource_utilizations_before"], dtype=float
            )
            lower, upper = (
                float(value)
                for value in item["weighted_coverage_bounds_before"]
            )
            parameter = str(item["adjusted_parameter"])
            before = float(item["parameter_before"])
            after = float(item["parameter_after"])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"single_calibration_trace_invalid={task_id}")
            break
        if (
            int(item.get("iteration", -1)) != expected_iteration
            or intended not in resources
            or item.get("intended_resource") != intended
            or parameter != intended_parameter
            or utilities.shape != (3,)
            or not np.all(np.isfinite(utilities))
            or np.any(utilities <= 0.0)
            or not all(math.isfinite(value) for value in (lower, upper, before, after))
            or not math.isclose(
                before, current.get(parameter, math.nan), rel_tol=0.0, abs_tol=1e-9
            )
        ):
            reasons.append(f"single_calibration_trace_mismatch={task_id}")
            break
        band_low, band_high = (
            float(value)
            for value in multimap_protocol["difficulty_bands"][difficulty]
        )
        intended_index = resources.index(intended)
        intended_utilization = float(utilities[intended_index])
        if lower < band_low:
            target = float(single_config["below_band_target_utilization"])
            proposed = max(
                before * intended_utilization / target,
                before * float(single_config["minimum_loosen_factor"]),
            )
            expected_direction = "loosen"
        elif lower > band_high:
            target = float(single_config["above_band_target_utilization"])
            proposed = min(
                before * intended_utilization / target,
                before * float(single_config["maximum_tighten_factor"]),
            )
            expected_direction = "tighten"
        else:
            max_utilization = float(np.max(utilities))
            max_gap = float(
                parent_protocol["certification"]["single_bottleneck_max_gap"]
            )
            target = max_utilization - (
                max_gap - float(single_config["target_gap_margin"])
            )
            proposed = before * intended_utilization / target
            expected_direction = "tighten"
        low, high = (
            float(value)
            for value in single_config["parameter_bounds"][parameter]
        )
        expected_after = float(np.clip(proposed, low, high))
        if (
            item.get("direction") != expected_direction
            or not math.isclose(
                float(item.get("target_utilization", math.nan)),
                target,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                after, expected_after, rel_tol=0.0, abs_tol=1e-9
            )
            or not str(item.get("certificate_scenario_hash_before", ""))
        ):
            reasons.append(f"single_calibration_replay_failed={task_id}")
            break
        current[parameter] = after

    for expected_iteration, item in enumerate(mixed_trace, start=1):
        try:
            utilities = np.asarray(
                item["resource_utilizations_before"], dtype=float
            )
            lower = float(item["weighted_coverage_lower_bound_before"])
            parameter = str(item["adjusted_parameter"])
            before = float(item["parameter_before"])
            after = float(item["parameter_after"])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"mixed_calibration_trace_invalid={task_id}")
            break
        order = (
            np.argsort(-utilities, kind="mergesort")
            if utilities.shape == (3,)
            else np.asarray([], dtype=int)
        )
        if (
            int(item.get("iteration", -1)) != expected_iteration
            or utilities.shape != (3,)
            or not np.all(np.isfinite(utilities))
            or np.any(utilities <= 0.0)
            or len(order) != 3
            or not all(math.isfinite(value) for value in (lower, before, after))
        ):
            reasons.append(f"mixed_calibration_trace_mismatch={task_id}")
            break
        top_index, second_index = int(order[0]), int(order[1])
        expected_parameter = parameters[second_index]
        max_gap = float(
            parent_protocol["certification"]["single_bottleneck_max_gap"]
        )
        target = float(utilities[top_index]) - (
            max_gap - float(mixed_config["target_gap_margin"])
        )
        low, high = (
            float(value)
            for value in mixed_config["parameter_bounds"][expected_parameter]
        )
        expected_after = float(
            np.clip(
                before * float(utilities[second_index]) / target,
                low,
                high,
            )
        )
        band_low, band_high = (
            float(value)
            for value in multimap_protocol["difficulty_bands"][difficulty]
        )
        if (
            parameter != expected_parameter
            or item.get("adjusted_resource") != resources[second_index]
            or not band_low <= lower <= band_high
            or not math.isclose(
                before,
                current.get(expected_parameter, math.nan),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(item.get("target_utilization", math.nan)),
                target,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                after, expected_after, rel_tol=0.0, abs_tol=1e-9
            )
            or not str(item.get("certificate_scenario_hash_before", ""))
        ):
            reasons.append(f"mixed_calibration_replay_failed={task_id}")
            break
        current[expected_parameter] = after

    if str(multimap_protocol.get("protocol_version", "")) in {
        "multimap_generalization_v3_2_13",
        "multimap_generalization_v3_2_14",
    }:
        bounds = generation["single_constraint_budget_calibration"][
            "parameter_bounds"
        ]
        for item in record.get("certificate_search_trace", ()):
            stage = str(item.get("stage", ""))
            if stage == "global_scale":
                factor = float(item["factor_from_geometry_anchor"])
                for parameter in parameters:
                    low, high = (
                        float(value) for value in bounds[parameter]
                    )
                    current[parameter] = float(
                        np.clip(current[parameter] * factor, low, high)
                    )
            elif stage == "activate_required_bottleneck":
                parameter = str(item["parameter"])
                factor = float(item["factor_from_stage_anchor"])
                low, high = (
                    float(value) for value in bounds[parameter]
                )
                current[parameter] = float(
                    np.clip(current[parameter] * factor, low, high)
                )
            elif stage == "single_constraint_release_nuisance_resources":
                intended_parameter = str(item["intended_parameter"])
                for parameter in parameters:
                    if parameter != intended_parameter:
                        current[parameter] = float(bounds[parameter][1])
            elif stage == "single_constraint_absolute_search":
                parameter = str(item["parameter"])
                fraction = float(item["registered_range_fraction"])
                low, high = (
                    float(value) for value in bounds[parameter]
                )
                current[parameter] = low + fraction * (high - low)
            elif stage == "v3_2_13_transported_resource_threshold":
                fraction = float(item["registered_range_fraction"])
                current["initial_soc"] = float(
                    bounds["initial_soc"][1]
                )
                current["time_budget_scale"] = float(
                    bounds["time_budget_scale"][1]
                )
                low, high = (
                    float(value)
                    for value in bounds["distance_budget_scale"]
                )
                current["distance_budget_scale"] = (
                    low + fraction * (high - low)
                )
            else:
                reasons.append(
                    f"unknown_v3_2_13_certificate_search_stage="
                    f"{task_id}:{stage}"
                )

    for parameter in parameters:
        try:
            final_value = float(record[parameter])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"final_budget_invalid={task_id}:{parameter}")
            continue
        if not math.isclose(
            final_value, current[parameter], rel_tol=0.0, abs_tol=1e-9
        ):
            reasons.append(f"final_budget_replay_failed={task_id}:{parameter}")
    return reasons


def prepare_task_manifest(
    protocol_path: Path,
    map_root: Path,
    output_root: Path,
    *,
    split: str,
    map_registry_path: Path,
    resume_existing: bool = False,
    map_limit: Optional[int] = None,
    map_index_start: Optional[int] = None,
    map_index_stop: Optional[int] = None,
    shard_name: Optional[str] = None,
    task_limit_per_map: Optional[int] = None,
    certification_time_limit_s: float = 60.0,
    screening_time_limit_s: float = 10.0,
    max_attempts_per_task: int = 2000,
    training_freeze: Optional[Path] = None,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    parent_protocol = json.loads(
        PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
    )
    parent_actual = _canonical_hash(
        parent_protocol, excluded=("protocol_hash",)
    )
    if parent_actual != parent_protocol.get("protocol_hash"):
        raise RuntimeError("父困难协议哈希不一致。")
    if split not in {"training", "validation", "synthetic_test"}:
        raise ValueError(f"不支持任务split={split}")
    if split == "synthetic_test":
        if training_freeze is None or not Path(training_freeze).is_file():
            raise RuntimeError("训练协议冻结前不得生成synthetic_test任务。")
        freeze = json.loads(Path(training_freeze).read_text(encoding="utf-8"))
        if (
            freeze.get("protocol_hash") != protocol["protocol_hash"]
            or freeze.get("state") not in {"frozen_for_pilot", "formal_training_complete"}
        ):
            raise RuntimeError("synthetic_test任务必须使用当前协议的有效训练冻结。")
    map_registry = json.loads(
        Path(map_registry_path).read_text(encoding="utf-8")
    )
    expected_registry_protocol_hash = (
        protocol["protocol_hash"]
        if split == "synthetic_test"
        else _asset_protocol_hash(protocol)
    )
    if map_registry.get("protocol_hash") != expected_registry_protocol_hash:
        raise RuntimeError("地图注册表与多地图协议不一致。")
    all_maps = list(map_registry["maps"])
    indexed_maps = list(enumerate(all_maps))
    if map_index_start is not None or map_index_stop is not None:
        start = int(map_index_start or 0)
        stop = int(
            map_index_stop
            if map_index_stop is not None
            else len(indexed_maps)
        )
        if not 0 <= start < stop <= len(indexed_maps):
            raise ValueError("地图分片范围必须满足0 <= start < stop <= map_count。")
        indexed_maps = indexed_maps[start:stop]
    if map_limit is not None:
        indexed_maps = indexed_maps[: int(map_limit)]
    maps = [record for _index, record in indexed_maps]
    task_count = min(9, int(task_limit_per_map or 9))
    sharded = shard_name is not None
    smoke = (
        not sharded
        and (map_limit is not None or task_limit_per_map is not None)
    )
    if sharded:
        if not str(shard_name).replace("_", "").replace("-", "").isalnum():
            raise ValueError("shard_name只能包含字母、数字、下划线或连字符。")
        run_dir = (
            output_root / "manifests" / f"{split}_shards" / str(shard_name)
        )
    elif smoke:
        run_dir = output_root / "smoke" / split
    else:
        run_dir = output_root / "manifests" / split
    records_path = run_dir / "records.jsonl"
    existing = _read_jsonl(records_path)
    if existing and not resume_existing:
        raise FileExistsError(
            f"任务结果已存在：{records_path}；恢复请使用--resume-existing。"
        )
    accepted_by_id = {str(row["id"]): dict(row) for row in existing}
    checkpoint_path = run_dir / "generation_checkpoint.json"
    resume_checkpoint: Dict[str, Any] = {}
    if resume_existing and checkpoint_path.is_file():
        resume_checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
        if resume_checkpoint.get("protocol_hash") != protocol["protocol_hash"]:
            raise RuntimeError("恢复检查点与当前多地图协议哈希不一致。")
    resume_task_id = str(resume_checkpoint.get("current_task_id", ""))
    resume_attempt = int(resume_checkpoint.get("current_attempt", 0) or 0)
    provider = FrozenMapProvider.from_registries(
        map_root, [Path(map_registry_path)]
    )
    rejection_counts: Dict[str, int] = {
        str(key): int(value)
        for key, value in dict(
            resume_checkpoint.get("rejection_counts", {})
        ).items()
    }
    design = dict(protocol["map_splits"][split])
    # 分片仍使用原注册表下标，保证任务设计、种子和单进程结果身份一致。
    for map_index, map_record in indexed_maps:
        bundle = _load_map_bundle(map_root, map_record)
        for task_index in range(task_count):
            task_id = f"{split}__{map_record['map_id']}__task_{task_index:02d}"
            if task_id in accepted_by_id:
                continue
            task_design = _task_design(map_index, task_index)
            (
                geometry_minimum_radius,
                geometry_radius_range,
            ) = _effective_task_radius_range(
                map_record,
                bundle,
                protocol,
                node_count=int(task_design["node_count"]),
                difficulty=str(task_design["difficulty"]),
            )
            accepted: Optional[Dict[str, Any]] = None
            start_attempt = (
                resume_attempt if task_id == resume_task_id else 0
            )
            for attempt in range(
                int(start_attempt), int(max_attempts_per_task)
            ):
                if attempt > 0 and attempt % 10 == 0:
                    _atomic_json(
                        run_dir / "generation_checkpoint.json",
                        {
                            "state": "running",
                            "protocol_hash": protocol["protocol_hash"],
                            "completed": len(accepted_by_id),
                            "expected": len(maps) * task_count,
                            "latest_task_id": (
                                sorted(accepted_by_id)[-1]
                                if accepted_by_id
                                else None
                            ),
                            "current_task_id": task_id,
                            "current_attempt": attempt,
                            "rejection_counts": rejection_counts,
                        },
                    )
                try:
                    candidate = _task_candidate(
                        map_record,
                        bundle,
                        protocol,
                        parent_protocol,
                        split=split,
                        map_index=map_index,
                        task_index=task_index,
                        attempt=attempt,
                        master_seed=int(design["seed"]),
                        geometry_radius_range_m=geometry_radius_range,
                        geometry_minimum_feasible_radius_m=(
                            geometry_minimum_radius
                        ),
                    )
                except RuntimeError as exc:
                    reason = f"candidate_geometry_infeasible:{type(exc).__name__}"
                    rejection_counts[reason] = (
                        rejection_counts.get(reason, 0) + 1
                    )
                    continue
                raw_candidate = dict(candidate)
                calibration_config = protocol["task_generation"].get(
                    "mixed_budget_calibration", {}
                )
                single_calibration_config = protocol["task_generation"].get(
                    "single_constraint_budget_calibration", {}
                )
                maximum_calibrations = (
                    int(calibration_config.get("maximum_iterations", 0))
                    if str(candidate.get("constraint_type", "")) == "mixed"
                    else int(
                        single_calibration_config.get("maximum_iterations", 0)
                    )
                )
                for calibration_iteration in range(maximum_calibrations + 1):
                    screening_ok, screening_certificate, screening_reason = (
                        _certify_multimap_task(
                            candidate,
                            provider,
                            parent_protocol,
                            time_limit_s=min(
                                float(screening_time_limit_s),
                                float(certification_time_limit_s),
                            ),
                        )
                    )
                    if screening_ok or calibration_iteration >= maximum_calibrations:
                        break
                    if str(candidate.get("constraint_type", "")) == "mixed":
                        calibrated = _calibrate_mixed_candidate(
                            candidate,
                            screening_certificate,
                            protocol,
                            parent_protocol,
                            iteration=calibration_iteration + 1,
                        )
                    else:
                        calibrated = _calibrate_single_constraint_candidate(
                            candidate,
                            screening_certificate,
                            protocol,
                            parent_protocol,
                            iteration=calibration_iteration + 1,
                        )
                    if calibrated is None:
                        break
                    candidate = calibrated
                    calibration_key = (
                        "mixed_calibration_applied"
                        if str(candidate.get("constraint_type", "")) == "mixed"
                        else "single_constraint_calibration_applied"
                    )
                    rejection_counts[calibration_key] = (
                        rejection_counts.get(calibration_key, 0) + 1
                    )
                screening_decisive = _screening_certificate_is_decisive(
                    screening_ok, screening_certificate
                )
                fallback_config = protocol["task_generation"].get(
                    "resource_threshold_fallback", {}
                )
                fallback_selected = False
                if (
                    bool(fallback_config.get("enabled", False))
                    and _resource_threshold_fallback_is_triggered(
                        attempt, fallback_config
                    )
                    and str(candidate.get("constraint_type", ""))
                    in set(
                        fallback_config.get(
                            "eligible_constraint_types", ()
                        )
                    )
                    and not screening_decisive
                ):
                    rejection_counts["resource_threshold_fallback_applied"] = (
                        rejection_counts.get(
                            "resource_threshold_fallback_applied", 0
                        )
                        + 1
                    )
                    ok, certificate, reason = (
                        _certify_with_resource_thresholds(
                            candidate,
                            provider,
                            protocol,
                            parent_protocol,
                        )
                    )
                    if ok:
                        fallback_selected = True
                        certification_source = "resource_threshold_fallback"
                        certification_limit_used = float(
                            fallback_config["lower_time_limit_s"]
                        ) + float(
                            fallback_config["upper_time_limit_s"]
                        )
                    else:
                        fallback_key = f"resource_threshold:{reason}"
                        rejection_counts[fallback_key] = (
                            rejection_counts.get(fallback_key, 0) + 1
                        )
                        relaxation_config = dict(
                            fallback_config.get(
                                "non_intended_resource_relaxation"
                            )
                            or {}
                        )
                        if (
                            reason == "low_threshold_no_route"
                            and bool(
                                relaxation_config.get("enabled", False)
                            )
                        ):
                            relaxed_candidate = (
                                _relax_non_intended_resources(
                                    raw_candidate, protocol
                                )
                            )
                            if relaxed_candidate is not None:
                                rejection_counts[
                                    "non_intended_relaxation_applied"
                                ] = (
                                    rejection_counts.get(
                                        "non_intended_relaxation_applied",
                                        0,
                                    )
                                    + 1
                                )
                                relaxed_screening_ok = False
                                relaxed_screening_certificate = {}
                                relaxed_screening_reason = (
                                    "relaxation_not_screened"
                                )
                                for relaxation_iteration in range(
                                    maximum_calibrations + 1
                                ):
                                    (
                                        relaxed_screening_ok,
                                        relaxed_screening_certificate,
                                        relaxed_screening_reason,
                                    ) = _certify_multimap_task(
                                        relaxed_candidate,
                                        provider,
                                        parent_protocol,
                                        time_limit_s=min(
                                            float(screening_time_limit_s),
                                            float(
                                                certification_time_limit_s
                                            ),
                                        ),
                                    )
                                    if (
                                        relaxed_screening_ok
                                        or relaxation_iteration
                                        >= maximum_calibrations
                                    ):
                                        break
                                    calibrated = (
                                        _calibrate_single_constraint_candidate(
                                            relaxed_candidate,
                                            relaxed_screening_certificate,
                                            protocol,
                                            parent_protocol,
                                            iteration=(
                                                relaxation_iteration + 1
                                            ),
                                        )
                                    )
                                    if calibrated is None:
                                        break
                                    relaxed_candidate = calibrated
                                (
                                    relaxed_ok,
                                    relaxed_certificate,
                                    relaxed_reason,
                                ) = _certify_with_resource_thresholds(
                                    relaxed_candidate,
                                    provider,
                                    protocol,
                                    parent_protocol,
                                )
                                if relaxed_ok:
                                    candidate = relaxed_candidate
                                    ok = relaxed_ok
                                    certificate = relaxed_certificate
                                    reason = relaxed_reason
                                    screening_ok = relaxed_screening_ok
                                    screening_certificate = (
                                        relaxed_screening_certificate
                                    )
                                    screening_reason = (
                                        relaxed_screening_reason
                                    )
                                    fallback_selected = True
                                    certification_source = (
                                        "resource_threshold_fallback"
                                    )
                                    certification_limit_used = float(
                                        fallback_config[
                                            "lower_time_limit_s"
                                        ]
                                    ) + float(
                                        fallback_config[
                                            "upper_time_limit_s"
                                        ]
                                    )
                                    rejection_counts[
                                        "non_intended_relaxation_accepted"
                                    ] = (
                                        rejection_counts.get(
                                            "non_intended_relaxation_accepted",
                                            0,
                                        )
                                        + 1
                                    )
                                else:
                                    relaxation_key = (
                                        "non_intended_relaxation:"
                                        f"{relaxed_reason}"
                                    )
                                    rejection_counts[relaxation_key] = (
                                        rejection_counts.get(
                                            relaxation_key, 0
                                        )
                                        + 1
                                    )
                should_finalize = screening_decisive or _screening_bounds_intersect_band(
                    candidate,
                    screening_certificate,
                    parent_protocol,
                )
                if not fallback_selected and not should_finalize:
                    key = f"screening:{screening_reason}"
                    rejection_counts[key] = rejection_counts.get(key, 0) + 1
                    continue
                if fallback_selected:
                    pass
                elif screening_decisive:
                    ok, certificate, reason = (
                        screening_ok,
                        screening_certificate,
                        screening_reason,
                    )
                    certification_source = "screening_sufficient"
                    certification_limit_used = float(screening_time_limit_s)
                elif float(screening_time_limit_s) < float(
                    certification_time_limit_s
                ):
                    ok, certificate, reason = _certify_multimap_task(
                        candidate,
                        provider,
                        parent_protocol,
                        time_limit_s=certification_time_limit_s,
                    )
                    certification_source = "extended_final"
                    certification_limit_used = float(certification_time_limit_s)
                else:
                    ok, certificate, reason = (
                        screening_ok,
                        screening_certificate,
                        screening_reason,
                    )
                    certification_source = "screening_equals_final"
                    certification_limit_used = float(screening_time_limit_s)
                mixed_threshold_config = protocol["task_generation"].get(
                    "mixed_threshold_certificate", {}
                )
                if (
                    not ok
                    and bool(mixed_threshold_config.get("enabled", False))
                    and str(candidate.get("constraint_type", "")) == "mixed"
                    and reason
                    in set(
                        mixed_threshold_config.get(
                            "eligible_standard_reasons", ()
                        )
                    )
                ):
                    (
                        mixed_ok,
                        mixed_candidate,
                        mixed_certificate,
                        mixed_reason,
                    ) = _certify_mixed_with_lower_threshold_route(
                        candidate,
                        certificate,
                        provider,
                        protocol,
                        parent_protocol,
                    )
                    if mixed_ok:
                        candidate = mixed_candidate
                        ok = True
                        certificate = mixed_certificate
                        reason = mixed_reason
                        certification_source = (
                            "mixed_threshold_certificate"
                        )
                        attempted_count = len(
                            certificate["mixed_threshold_proof"][
                                "low_threshold_attempts"
                            ]
                        )
                        certification_limit_used = float(
                            certification_time_limit_s
                        ) + attempted_count * float(
                            mixed_threshold_config["lower_time_limit_s"]
                        )
                    else:
                        mixed_key = f"mixed_threshold:{mixed_reason}"
                        rejection_counts[mixed_key] = (
                            rejection_counts.get(mixed_key, 0) + 1
                        )
                certificate["screening"] = {
                    "time_limit_s": float(screening_time_limit_s),
                    "reason": screening_reason,
                    "weighted_coverage_lower_bound": screening_certificate.get(
                        "weighted_coverage_lower_bound"
                    ),
                    "weighted_coverage_upper_bound": screening_certificate.get(
                        "weighted_coverage_upper_bound"
                    ),
                    "mip_gap": screening_certificate.get("mip_gap"),
                }
                certificate["certification_source"] = certification_source
                certificate["certification_time_limit_s_used"] = (
                    certification_limit_used
                )
                key = "accepted" if ok else f"final:{reason}"
                rejection_counts[key] = rejection_counts.get(key, 0) + 1
                if ok:
                    candidate["certificate"] = certificate
                    candidate["task_hash"] = _canonical_hash(
                        candidate, excluded=("task_hash",)
                    )
                    accepted = candidate
                    break
            if accepted is None:
                _atomic_json(
                    run_dir / "generation_checkpoint.json",
                    {
                        "state": "failed",
                        "protocol_hash": protocol["protocol_hash"],
                        "map_id": map_record["map_id"],
                        "task_index": task_index,
                        "rejection_counts": rejection_counts,
                    },
                )
                raise RuntimeError(
                    f"{task_id}在{max_attempts_per_task}次内未通过MILP认证。"
                )
            accepted_by_id[task_id] = accepted
            if task_id == resume_task_id:
                resume_task_id = ""
                resume_attempt = 0
            ordered = [
                accepted_by_id[key] for key in sorted(accepted_by_id)
            ]
            _atomic_text(records_path, _jsonl_text(ordered))
            _atomic_json(
                run_dir / "generation_checkpoint.json",
                {
                    "state": "running",
                    "protocol_hash": protocol["protocol_hash"],
                    "completed": len(ordered),
                    "expected": len(maps) * task_count,
                    "latest_task_id": task_id,
                    "rejection_counts": rejection_counts,
                },
            )
    records = [accepted_by_id[key] for key in sorted(accepted_by_id)]
    records_sha256 = hashlib.sha256(
        _jsonl_text(records).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "parent_difficulty_protocol_hash": parent_protocol["protocol_hash"],
        "split": split,
        "smoke": smoke,
        "sharded": sharded,
        "shard_name": str(shard_name) if sharded else None,
        "map_index_start": (
            int(indexed_maps[0][0]) if indexed_maps else None
        ),
        "map_index_stop": (
            int(indexed_maps[-1][0]) + 1 if indexed_maps else None
        ),
        "map_registry_path": str(Path(map_registry_path).resolve()),
        "map_registry_hash": map_registry["registry_hash"],
        "map_provider_hash": provider.provider_hash,
        "map_count": len(maps),
        "tasks_per_map": task_count,
        "scenario_count": len(records),
        "records_sha256": records_sha256,
        "selection_used_algorithm_results": False,
    }
    manifest["manifest_hash"] = _canonical_hash(
        manifest, excluded=("manifest_hash",)
    )
    _atomic_json(run_dir / "manifest.json", manifest)
    audit = audit_task_manifest(
        protocol_path,
        map_root,
        run_dir / "manifest.json",
        expected_map_count=len(maps),
        expected_tasks_per_map=task_count,
    )
    _atomic_json(run_dir / "environment_audit.json", audit)
    _atomic_json(
        run_dir / "generation_checkpoint.json",
        {
            "state": "completed" if audit["passed"] else "audit_failed",
            "protocol_hash": protocol["protocol_hash"],
            "completed": len(records),
            "expected": len(maps) * task_count,
            "rejection_counts": rejection_counts,
            "audit_passed": audit["passed"],
        },
    )
    return {"manifest": manifest, "audit": audit}


def merge_task_shards(
    protocol_path: Path,
    map_root: Path,
    output_root: Path,
    *,
    split: str,
    map_registry_path: Path,
    base_records_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """合并独立地图分片；重复任务必须逐字段同一，缺失任务拒绝封存。"""

    protocol = load_protocol(protocol_path)
    if split not in {"training", "validation"}:
        raise ValueError("只允许合并training或validation任务分片。")
    map_registry = json.loads(
        Path(map_registry_path).read_text(encoding="utf-8")
    )
    if map_registry.get("protocol_hash") != protocol["protocol_hash"]:
        raise RuntimeError("待合并地图注册表与协议哈希不一致。")
    shard_root = output_root / "manifests" / f"{split}_shards"
    shard_manifest_paths = sorted(shard_root.glob("*/manifest.json"))
    if not shard_manifest_paths:
        raise RuntimeError(f"没有找到{split}任务分片manifest。")

    rows_by_id: Dict[str, Dict[str, Any]] = {}
    sources_by_id: Dict[str, str] = {}
    duplicate_count = 0

    def add_rows(rows: Sequence[Mapping[str, Any]], source: str) -> None:
        nonlocal duplicate_count
        for raw in rows:
            row = dict(raw)
            task_id = str(row.get("id", ""))
            expected_hash = _canonical_hash(row, excluded=("task_hash",))
            if not task_id or row.get("task_hash") != expected_hash:
                raise RuntimeError(f"分片任务哈希无效：{source}:{task_id}")
            existing = rows_by_id.get(task_id)
            if existing is None:
                rows_by_id[task_id] = row
                sources_by_id[task_id] = source
                continue
            if existing != row:
                raise RuntimeError(
                    "重复任务内容不一致："
                    f"{task_id} sources={sources_by_id[task_id]},{source}"
                )
            duplicate_count += 1

    if base_records_path is not None and Path(base_records_path).is_file():
        add_rows(_read_jsonl(Path(base_records_path)), "serial_base")

    shard_manifest_hashes: Dict[str, str] = {}
    for manifest_path in shard_manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("protocol_hash") != protocol["protocol_hash"]
            or manifest.get("split") != split
            or not bool(manifest.get("sharded", False))
            or manifest.get("map_registry_hash")
            != map_registry.get("registry_hash")
            or manifest.get("manifest_hash")
            != _canonical_hash(manifest, excluded=("manifest_hash",))
        ):
            raise RuntimeError(f"分片manifest身份无效：{manifest_path}")
        records_path = manifest_path.parent / "records.jsonl"
        rows = _read_jsonl(records_path)
        if hashlib.sha256(
            _jsonl_text(rows).encode("utf-8")
        ).hexdigest() != manifest.get("records_sha256"):
            raise RuntimeError(f"分片records哈希无效：{records_path}")
        audit = audit_task_manifest(
            protocol_path,
            map_root,
            manifest_path,
            expected_map_count=int(manifest["map_count"]),
            expected_tasks_per_map=int(manifest["tasks_per_map"]),
        )
        if not audit["passed"]:
            raise RuntimeError(f"分片审计失败：{manifest_path}")
        source = str(manifest.get("shard_name") or manifest_path.parent.name)
        add_rows(rows, source)
        shard_manifest_hashes[source] = str(manifest["manifest_hash"])

    expected_ids = {
        f"{split}__{record['map_id']}__task_{task_index:02d}"
        for record in map_registry["maps"]
        for task_index in range(9)
    }
    actual_ids = set(rows_by_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise RuntimeError(
            "任务分片不完整："
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    records = [rows_by_id[key] for key in sorted(rows_by_id)]
    run_dir = output_root / "manifests" / split
    records_path = run_dir / "records.jsonl"
    _atomic_text(records_path, _jsonl_text(records))
    records_sha256 = hashlib.sha256(
        _jsonl_text(records).encode("utf-8")
    ).hexdigest()
    provider = FrozenMapProvider.from_registries(
        map_root, [Path(map_registry_path)]
    )
    manifest = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "parent_difficulty_protocol_hash": json.loads(
            PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
        )["protocol_hash"],
        "split": split,
        "smoke": False,
        "sharded": False,
        "merged_from_shards": True,
        "map_registry_path": str(Path(map_registry_path).resolve()),
        "map_registry_hash": map_registry["registry_hash"],
        "map_provider_hash": provider.provider_hash,
        "map_count": len(map_registry["maps"]),
        "tasks_per_map": 9,
        "scenario_count": len(records),
        "records_sha256": records_sha256,
        "selection_used_algorithm_results": False,
        "parallel_worker_count": int(
            protocol["task_generation"]["parallel_certification"][
                "worker_count"
            ]
        ),
        "shard_manifest_hashes": shard_manifest_hashes,
    }
    manifest["manifest_hash"] = _canonical_hash(
        manifest, excluded=("manifest_hash",)
    )
    manifest_path = run_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    audit = audit_task_manifest(
        protocol_path,
        map_root,
        manifest_path,
        expected_map_count=len(map_registry["maps"]),
        expected_tasks_per_map=9,
    )
    _atomic_json(run_dir / "environment_audit.json", audit)
    checkpoint = {
        "state": "completed" if audit["passed"] else "audit_failed",
        "protocol_hash": protocol["protocol_hash"],
        "completed": len(records),
        "expected": len(expected_ids),
        "audit_passed": audit["passed"],
        "merged_from_shards": True,
    }
    _atomic_json(run_dir / "generation_checkpoint.json", checkpoint)
    report = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "split": split,
        "passed": bool(audit["passed"]),
        "scenario_count": len(records),
        "duplicate_identical_count": duplicate_count,
        "records_sha256": records_sha256,
        "shard_manifest_hashes": shard_manifest_hashes,
    }
    report["merge_hash"] = _canonical_hash(
        report, excluded=("merge_hash",)
    )
    _atomic_json(run_dir / "parallel_merge_report.json", report)
    return {"manifest": manifest, "audit": audit, "merge_report": report}


def audit_task_manifest(
    protocol_path: Path,
    map_root: Path,
    manifest_path: Path,
    *,
    expected_map_count: Optional[int] = None,
    expected_tasks_per_map: Optional[int] = None,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    parent_protocol = json.loads(
        PARENT_DIFFICULTY_PROTOCOL.read_text(encoding="utf-8")
    )
    root = Path(manifest_path).parent
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = _read_jsonl(root / "records.jsonl")
    reasons: List[str] = []
    if _canonical_hash(
        parent_protocol, excluded=("protocol_hash",)
    ) != parent_protocol.get("protocol_hash"):
        reasons.append("parent_protocol_hash_invalid")
    if manifest.get("parent_difficulty_protocol_hash") != parent_protocol.get(
        "protocol_hash"
    ):
        reasons.append("parent_protocol_identity_mismatch")
    map_registry_path = Path(str(manifest.get("map_registry_path", "")))
    map_registry: Dict[str, Any] = {}
    registered_maps: Dict[str, Dict[str, Any]] = {}
    map_order: Dict[str, int] = {}
    if not map_registry_path.is_file():
        reasons.append("missing_map_registry")
    else:
        map_registry = json.loads(
            map_registry_path.read_text(encoding="utf-8")
        )
        if (
            str(manifest.get("split", "")) == "synthetic_test"
            and str(protocol.get("protocol_version", ""))
            in {
                "multimap_generalization_v3_2_12",
                "multimap_generalization_v3_2_13",
                "multimap_generalization_v3_2_14",
            }
        ):
            # v3.2.12 only changes post-training certificate search.  The
            # already sealed v3.2 unseen procedural pixels are reused after a
            # separate byte/hash and input-semantics audit.
            sealed_protocol = load_protocol(
                ROOT
                / "paper_runs/protocols/multimap_generalization_v3_2/protocol.json"
            )
            expected_registry_protocol_hash = sealed_protocol["protocol_hash"]
        else:
            expected_registry_protocol_hash = (
                protocol["protocol_hash"]
                if str(manifest.get("split", "")) == "synthetic_test"
                else _asset_protocol_hash(protocol)
            )
        if map_registry.get("protocol_hash") != expected_registry_protocol_hash:
            reasons.append("map_registry_protocol_hash_mismatch")
        if map_registry.get("registry_hash") != manifest.get(
            "map_registry_hash"
        ):
            reasons.append("map_registry_identity_mismatch")
        try:
            provider = FrozenMapProvider.from_registries(
                map_root, [map_registry_path]
            )
            if provider.provider_hash != manifest.get("map_provider_hash"):
                reasons.append("map_provider_hash_mismatch")
        except (KeyError, RuntimeError, ValueError) as exc:
            reasons.append(f"map_provider_invalid={type(exc).__name__}")
        registered_maps = {
            str(item["map_id"]): dict(item)
            for item in map_registry.get("maps", ())
        }
        map_order = {
            str(item["map_id"]): index
            for index, item in enumerate(map_registry.get("maps", ()))
        }
    expected_manifest_protocol_hash = (
        protocol["protocol_hash"]
        if str(manifest.get("split", "")) == "synthetic_test"
        else _asset_protocol_hash(protocol)
    )
    if manifest.get("protocol_hash") != expected_manifest_protocol_hash:
        reasons.append("protocol_hash_mismatch")
    if _canonical_hash(
        manifest, excluded=("manifest_hash",)
    ) != manifest.get("manifest_hash"):
        reasons.append("manifest_hash_mismatch")
    if hashlib.sha256(_jsonl_text(records).encode("utf-8")).hexdigest() != str(
        manifest.get("records_sha256")
    ):
        reasons.append("records_hash_mismatch")
    if len(records) != int(manifest.get("scenario_count", -1)):
        reasons.append("scenario_count_mismatch")
    if expected_map_count is not None and int(
        manifest.get("map_count", -1)
    ) != int(expected_map_count):
        reasons.append("map_count_mismatch")
    if expected_tasks_per_map is not None and int(
        manifest.get("tasks_per_map", -1)
    ) != int(expected_tasks_per_map):
        reasons.append("tasks_per_map_mismatch")
    ids = [str(record.get("id", "")) for record in records]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        reasons.append("duplicate_or_empty_task_id")
    per_map: Dict[str, List[Dict[str, Any]]] = {}
    seen_task_hashes = set()
    seen_scenario_hashes = set()
    generation = protocol["task_generation"]
    geometry_cache: Dict[
        Tuple[str, int, str], Tuple[float, Tuple[float, float]]
    ] = {}
    for record in records:
        map_id = str(record.get("map_id", ""))
        per_map.setdefault(map_id, []).append(record)
        reasons.extend(
            _audit_budget_transform_record(record, protocol, parent_protocol)
        )
        registered = registered_maps.get(map_id)
        if registered is None:
            reasons.append(f"unregistered_map={record.get('id')}")
        else:
            for field in ("map_hash", "map_file_sha256"):
                if str(record.get(field, "")) != str(registered.get(field, "")):
                    reasons.append(
                        f"{field}_mismatch={record.get('id')}"
                    )
        if _canonical_hash(record, excluded=("task_hash",)) != record.get(
            "task_hash"
        ):
            reasons.append(f"task_hash_mismatch={record.get('id')}")
        if record.get("task_hash") in seen_task_hashes:
            reasons.append(f"duplicate_task_hash={record.get('id')}")
        seen_task_hashes.add(record.get("task_hash"))
        if int(record.get("node_count", -1)) not in {16, 20, 24}:
            reasons.append(f"invalid_node_count={record.get('id')}")
            continue
        node_count = int(record["node_count"])
        task_index = int(record.get("task_index", -1))
        if map_id in map_order and 0 <= task_index < 9:
            expected_design = _task_design(map_order[map_id], task_index)
            for field in (
                "node_count",
                "difficulty",
                "constraint_type",
                "priority_layout",
            ):
                if record.get(field) != expected_design[field]:
                    reasons.append(
                        f"task_design_mismatch={record.get('id')}:{field}"
                    )
        else:
            reasons.append(f"invalid_task_index={record.get('id')}")
        points = np.asarray(record.get("inspection_points_xyz", ()), dtype=float)
        priorities = np.asarray(record.get("priorities", ()), dtype=float)
        service_times = np.asarray(record.get("service_times_s", ()), dtype=float)
        start_xy = np.asarray(record.get("start_xy", ()), dtype=float)
        if (
            points.shape != (node_count, 3)
            or priorities.shape != (node_count,)
            or service_times.shape != (node_count,)
            or start_xy.shape != (2,)
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(priorities))
            or not np.all(np.isfinite(service_times))
            or not np.all(np.isfinite(start_xy))
        ):
            reasons.append(f"invalid_task_arrays={record.get('id')}")
        else:
            rounded = np.round(points[:, :2], decimals=6)
            if len(np.unique(rounded, axis=0)) != node_count:
                reasons.append(f"duplicate_points={record.get('id')}")
            coordinate_scale = float(
                protocol["procedural_terrain"]["coordinate_scale_m_per_unit"]
            )
            radius = float(record.get("task_radius_m", math.nan))
            distances_m = (
                np.linalg.norm(points[:, :2] - start_xy.reshape(1, 2), axis=1)
                * coordinate_scale
            )
            difficulty = str(record.get("difficulty", ""))
            base_range = generation["radius_base_ranges_m"].get(difficulty)
            offset = generation["node_radius_offsets_m"].get(str(node_count))
            if base_range is None or offset is None or not math.isfinite(radius):
                reasons.append(f"invalid_task_radius={record.get('id')}")
            else:
                nominal = [
                    float(base_range[0]) + float(offset),
                    float(base_range[1]) + float(offset),
                ]
                if not np.allclose(
                    np.asarray(record.get("nominal_task_radius_range_m", ())),
                    np.asarray(nominal),
                    rtol=0.0,
                    atol=1e-9,
                ):
                    reasons.append(
                        f"nominal_radius_range_mismatch={record.get('id')}"
                    )
                cache_key = (map_id, node_count, difficulty)
                if registered is not None and cache_key not in geometry_cache:
                    bundle = _load_map_bundle(map_root, registered)
                    geometry_cache[cache_key] = _effective_task_radius_range(
                        registered,
                        bundle,
                        protocol,
                        node_count=node_count,
                        difficulty=difficulty,
                    )
                if cache_key in geometry_cache:
                    expected_minimum, expected_range = geometry_cache[cache_key]
                else:
                    expected_minimum = math.nan
                    expected_range = (math.nan, math.nan)
                recorded_range = np.asarray(
                    record.get("effective_task_radius_range_m", ()),
                    dtype=float,
                )
                if (
                    not math.isclose(
                        float(
                            record.get(
                                "geometry_minimum_feasible_radius_m", math.nan
                            )
                        ),
                        expected_minimum,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    or recorded_range.shape != (2,)
                    or not np.allclose(
                        recorded_range,
                        np.asarray(expected_range),
                        rtol=0.0,
                        atol=1e-9,
                    )
                ):
                    reasons.append(
                        f"geometry_radius_identity_mismatch={record.get('id')}"
                    )
                expected_low, expected_high = expected_range
                if not expected_low <= radius <= expected_high:
                    reasons.append(f"task_radius_out_of_range={record.get('id')}")
                if np.any(
                    distances_m
                    < float(generation["minimum_depot_distance_m"]) - 1e-6
                ) or np.any(distances_m > radius + 1e-6):
                    reasons.append(f"point_radius_failed={record.get('id')}")
            if node_count > 1:
                pairwise = np.linalg.norm(
                    points[:, None, :2] - points[None, :, :2], axis=2
                )
                pairwise += np.eye(node_count) * 1e12
                if (
                    float(np.min(pairwise)) * coordinate_scale
                    < float(generation["minimum_node_spacing_m"]) - 1e-6
                ):
                    reasons.append(f"point_spacing_failed={record.get('id')}")
        certificate = dict(record.get("certificate") or {})
        lower = float(certificate.get("weighted_coverage_lower_bound", math.nan))
        upper = float(certificate.get("weighted_coverage_upper_bound", math.nan))
        gap = float(certificate.get("mip_gap", math.nan))
        difficulty = str(record.get("difficulty", ""))
        band = protocol["difficulty_bands"].get(difficulty)
        if not (
            math.isfinite(lower)
            and math.isfinite(upper)
            and math.isfinite(gap)
            and 0.0 <= lower <= upper + 1e-7
            and upper < 0.85
            and certificate.get("returned")
            and int(certificate.get("visited_count", 0)) >= 1
        ):
            reasons.append(f"invalid_certificate={record.get('id')}")
        elif band is None or not (
            float(band[0]) - 1e-9
            <= lower
            <= float(band[1]) + 1e-9
        ):
            reasons.append(f"difficulty_band_failed={record.get('id')}")
        elif not (
            gap <= float(protocol["certification"]["mip_rel_gap"]) + 1e-10
            or upper <= float(band[1]) + 1e-9
        ):
            reasons.append(f"certificate_strength_failed={record.get('id')}")
        screening = dict(certificate.get("screening") or {})
        if not math.isclose(
            float(screening.get("time_limit_s", math.nan)),
            float(protocol["certification"]["candidate_screening_time_limit_s"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            reasons.append(f"screening_limit_mismatch={record.get('id')}")
        source = str(certificate.get("certification_source", ""))
        limit_used = float(
            certificate.get("certification_time_limit_s_used", math.nan)
        )
        if source == "screening_sufficient":
            expected_limit = float(
                protocol["certification"]["candidate_screening_time_limit_s"]
            )
        elif source == "resource_threshold_fallback":
            fallback = protocol["task_generation"][
                "resource_threshold_fallback"
            ]
            expected_limit = float(fallback["lower_time_limit_s"]) + float(
                fallback["upper_time_limit_s"]
            )
        elif source == "v3_2_13_transported_resource_threshold":
            fallback = protocol["task_generation"][
                "resource_threshold_fallback"
            ]
            expected_limit = float(fallback["lower_time_limit_s"]) + float(
                fallback["upper_time_limit_s"]
            )
        elif source in {
            "v3_2_13_constructive_mixed_threshold",
            "v3_2_14_constructive_mixed_threshold",
        }:
            proof = dict(
                certificate.get("constructive_mixed_threshold_proof") or {}
            )
            expected_limit = float(
                proof.get("registered_solver_time_limit_s", math.nan)
            )
        elif source == "mixed_threshold_certificate":
            mixed_config = protocol["task_generation"][
                "mixed_threshold_certificate"
            ]
            mixed_proof = dict(
                certificate.get("mixed_threshold_proof") or {}
            )
            attempted_count = len(
                mixed_proof.get("low_threshold_attempts") or ()
            )
            expected_limit = float(
                protocol["certification"]["time_limit_s"]
            ) + attempted_count * float(
                mixed_config["lower_time_limit_s"]
            )
        else:
            expected_limit = float(protocol["certification"]["time_limit_s"])
        if source not in {
            "screening_sufficient",
            "extended_final",
            "screening_equals_final",
            "resource_threshold_fallback",
            "mixed_threshold_certificate",
            "v3_2_12_parametric_fixed_budget_final",
            "v3_2_13_transported_resource_threshold",
            "v3_2_13_constructive_mixed_threshold",
            "v3_2_14_constructive_mixed_threshold",
        } or not math.isclose(
            limit_used,
            expected_limit,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            reasons.append(f"certification_limit_mismatch={record.get('id')}")
        if str(certificate.get("map_hash", "")) != str(record.get("map_hash", "")):
            reasons.append(f"certificate_map_hash_mismatch={record.get('id')}")
        scenario_hash = str(certificate.get("scenario_hash", ""))
        if not scenario_hash or scenario_hash in seen_scenario_hashes:
            reasons.append(f"duplicate_or_empty_scenario_hash={record.get('id')}")
        seen_scenario_hashes.add(scenario_hash)
        bottlenecks = set(certificate.get("bottleneck_resources", ()))
        intended = str(record.get("constraint_type", ""))
        if source == "resource_threshold_fallback":
            proof = dict(certificate.get("resource_threshold_proof") or {})
            low_proof = dict(proof.get("low_threshold") or {})
            high_proof = dict(proof.get("high_threshold") or {})
            total_priority = float(np.sum(priorities))
            band_low, band_high = (
                float(value)
                for value in protocol["difficulty_bands"][difficulty]
            )
            expected_low_weight = int(
                math.ceil(band_low * total_priority - 1e-9)
            )
            expected_high_weight = (
                int(math.floor(band_high * total_priority + 1e-9)) + 1
            )
            if (
                intended not in {"energy", "distance", "time"}
                or proof.get("resource_name") != intended
                or int(proof.get("low_required_priority", -1))
                != expected_low_weight
                or int(proof.get("high_required_priority", -1))
                != expected_high_weight
                or not math.isclose(
                    float(proof.get("total_priority", math.nan)),
                    total_priority,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not bool(
                    high_proof.get(
                        "threshold_impossible_under_actual_budget", False
                    )
                )
                or list(low_proof.get("visit_order") or ())
                != list(certificate.get("visit_order") or ())
                or not math.isclose(
                    upper,
                    float(expected_high_weight - 1) / total_priority,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                reasons.append(
                    f"resource_threshold_proof_invalid={record.get('id')}"
                )
        if source == "v3_2_13_transported_resource_threshold":
            proof = dict(
                certificate.get(
                    "transported_resource_threshold_proof"
                )
                or {}
            )
            low_proof = dict(proof.get("low_threshold") or {})
            source_high = dict(
                proof.get("source_high_threshold") or {}
            )
            transported_high = dict(
                proof.get("transported_high_threshold") or {}
            )
            total_priority = float(np.sum(priorities))
            band_low, band_high = (
                float(value)
                for value in protocol["difficulty_bands"][difficulty]
            )
            expected_low_weight = int(
                math.ceil(band_low * total_priority - 1e-9)
            )
            expected_high_weight = (
                int(math.floor(band_high * total_priority + 1e-9)) + 1
            )
            source_path = Path(str(proof.get("source_proof_path", "")))
            proof_invalid = bool(
                intended not in {"energy", "distance", "time"}
                or proof.get("resource_name") != intended
                or int(proof.get("low_required_priority", -1))
                != expected_low_weight
                or int(proof.get("high_required_priority", -1))
                != expected_high_weight
                or not math.isclose(
                    float(proof.get("total_priority", math.nan)),
                    total_priority,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or list(low_proof.get("visit_order") or ())
                != list(certificate.get("visit_order") or ())
                or not bool(
                    transported_high.get(
                        "threshold_impossible_under_actual_budget", False
                    )
                )
                or not math.isclose(
                    float(
                        transported_high.get(
                            "resource_dual_bound", math.nan
                        )
                    ),
                    float(
                        source_high.get("resource_dual_bound", math.nan)
                    ),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or float(
                    transported_high.get(
                        "resource_dual_bound", -math.inf
                    )
                )
                <= float(
                    transported_high.get(
                        "actual_resource_budget", math.inf
                    )
                )
                + 1e-7
                or str(proof.get("source_invariant_hash", ""))
                != str(proof.get("target_invariant_hash", ""))
                or not source_path.is_file()
                or (
                    source_path.is_file()
                    and _sha256_file(source_path)
                    != str(proof.get("source_proof_sha256", ""))
                )
                or not math.isclose(
                    upper,
                    float(expected_high_weight - 1) / total_priority,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
            if proof_invalid:
                reasons.append(
                    "transported_resource_threshold_proof_invalid="
                    f"{record.get('id')}"
                )
        if source == "v3_2_14_constructive_mixed_threshold":
            proof = dict(
                certificate.get(
                    "constructive_mixed_threshold_proof"
                )
                or {}
            )
            low_proof = dict(proof.get("low_threshold_proof") or {})
            high_proof = dict(proof.get("high_threshold_proof") or {})
            total_priority = float(np.sum(priorities))
            band_low, band_high = (
                float(value)
                for value in protocol["difficulty_bands"][difficulty]
            )
            expected_low_weight = int(
                math.ceil(band_low * total_priority - 1e-9)
            )
            expected_high_weight = (
                int(math.floor(band_high * total_priority + 1e-9)) + 1
            )
            source_paths = (
                (
                    "low_threshold_proof_path",
                    "low_threshold_proof_sha256",
                ),
                ("calibration_path", "calibration_sha256"),
                ("final_cut_proof_path", "final_cut_proof_sha256"),
            )
            files_invalid = False
            for path_key, hash_key in source_paths:
                source_path = Path(str(proof.get(path_key, "")))
                if (
                    not source_path.is_file()
                    or _sha256_file(source_path)
                    != str(proof.get(hash_key, ""))
                ):
                    files_invalid = True
            proof_invalid = bool(
                intended != "mixed"
                or int(proof.get("low_required_priority", -1))
                != expected_low_weight
                or int(proof.get("high_required_priority", -1))
                != expected_high_weight
                or not math.isclose(
                    float(proof.get("total_priority", math.nan)),
                    total_priority,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or list(low_proof.get("visit_order") or ())
                != list(certificate.get("visit_order") or ())
                or int(high_proof.get("solver_status", -1)) != 2
                or not bool(high_proof.get("threshold_infeasible"))
                or high_proof.get("connected_route") is not None
                or int(proof.get("final_cut_count", -1))
                != int(high_proof.get("subtour_cut_count", -2))
                or not math.isclose(
                    upper,
                    float(expected_high_weight - 1) / total_priority,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or files_invalid
            )
            if proof_invalid:
                reasons.append(
                    "constructive_mixed_threshold_proof_invalid="
                    f"{record.get('id')}"
                )
        if source == "mixed_threshold_certificate":
            proof = dict(certificate.get("mixed_threshold_proof") or {})
            low_attempts = list(proof.get("low_threshold_attempts") or ())
            resource_order = list(proof.get("resource_order") or ())
            registered_order = list(
                protocol["task_generation"][
                    "mixed_threshold_certificate"
                ]["resource_order"]
            )
            successful_resource = str(
                proof.get("successful_resource", "")
            )
            route_source = str(proof.get("route_source", ""))
            total_priority = float(np.sum(priorities))
            band_low = float(
                protocol["difficulty_bands"][difficulty][0]
            )
            expected_low_weight = int(
                math.ceil(band_low * total_priority - 1e-9)
            )
            successful_index = (
                resource_order.index(successful_resource)
                if successful_resource in resource_order
                else -1
            )
            standard_upper = dict(
                proof.get("standard_upper_certificate") or {}
            )
            adjustment = proof.get("monotone_adjustment")
            if route_source == "standard_incumbent":
                route_structure_invalid = bool(
                    successful_resource != "standard_incumbent"
                    or low_attempts
                    or list(
                        standard_upper.get("visit_order") or ()
                    )
                    != list(certificate.get("visit_order") or ())
                )
            else:
                route_structure_invalid = bool(
                    route_source != "resource_threshold"
                    or successful_index < 0
                    or len(low_attempts) != successful_index + 1
                    or any(
                        str(item.get("resource_name", ""))
                        != resource_order[index]
                        or not math.isclose(
                            float(
                                item.get(
                                    "minimum_priority_weight", math.nan
                                )
                            ),
                            float(expected_low_weight),
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        )
                        for index, item in enumerate(low_attempts)
                    )
                    or list(low_attempts[-1].get("visit_order") or ())
                    != list(certificate.get("visit_order") or ())
                )
            proof_invalid = bool(
                intended != "mixed"
                or resource_order != registered_order
                or route_structure_invalid
                or int(proof.get("low_required_priority", -1))
                != expected_low_weight
                or not math.isclose(
                    float(proof.get("total_priority", math.nan)),
                    total_priority,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(
                        standard_upper.get(
                            "weighted_coverage_upper_bound", math.nan
                        )
                    ),
                    upper,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
            if adjustment is not None:
                adjustment = dict(adjustment)
                parameter = str(adjustment.get("parameter", ""))
                before = float(adjustment.get("before", math.nan))
                after = float(adjustment.get("after", math.nan))
                proof_invalid = proof_invalid or bool(
                    parameter
                    not in {
                        "initial_soc",
                        "distance_budget_scale",
                        "time_budget_scale",
                    }
                    or not math.isfinite(before)
                    or not math.isfinite(after)
                    or after >= before - 1e-12
                    or not math.isclose(
                        float(record.get(parameter, math.nan)),
                        after,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
            if proof_invalid:
                reasons.append(
                    f"mixed_threshold_proof_invalid={record.get('id')}"
                )
        if intended == "mixed":
            if len(bottlenecks) < 2:
                reasons.append(f"mixed_bottleneck_failed={record.get('id')}")
        elif intended not in bottlenecks:
            reasons.append(f"intended_bottleneck_failed={record.get('id')}")
    for map_id, rows in per_map.items():
        if expected_tasks_per_map is not None and len(rows) != int(
            expected_tasks_per_map
        ):
            reasons.append(f"tasks_per_map_failed={map_id}")
        if len(rows) == 9:
            task_indices = sorted(int(row["task_index"]) for row in rows)
            if task_indices != list(range(9)):
                reasons.append(f"task_index_balance_failed={map_id}")
            counts = {
                node_count: sum(
                    int(row["node_count"]) == node_count for row in rows
                )
                for node_count in (16, 20, 24)
            }
            if counts != {16: 3, 20: 3, 24: 3}:
                reasons.append(f"node_balance_failed={map_id}")
    return {
        "schema_version": 1,
        "passed": not reasons,
        "protocol_hash": protocol["protocol_hash"],
        "manifest_hash": manifest.get("manifest_hash"),
        "scenario_count": len(records),
        "map_count": len(per_map),
        "reasons": reasons,
    }


def audit_task_splits(
    protocol_path: Path,
    map_root: Path,
    training_manifest_path: Path,
    validation_manifest_path: Path,
) -> Dict[str, Any]:
    """证明训练与validation在地图、任务和内容层面完全隔离。"""

    protocol = load_protocol(protocol_path)
    audits = {}
    manifests = {}
    records_by_split = {}
    for split, path in (
        ("training", Path(training_manifest_path)),
        ("validation", Path(validation_manifest_path)),
    ):
        expected_maps = int(protocol["map_splits"][split]["map_count"])
        expected_tasks = int(protocol["map_splits"][split]["tasks_per_map"])
        audits[split] = audit_task_manifest(
            protocol_path,
            map_root,
            path,
            expected_map_count=expected_maps,
            expected_tasks_per_map=expected_tasks,
        )
        manifests[split] = json.loads(path.read_text(encoding="utf-8"))
        records_by_split[split] = _read_jsonl(path.parent / "records.jsonl")

    reasons: List[str] = []
    for split, audit in audits.items():
        if not audit["passed"]:
            reasons.append(f"{split}_audit_failed")
    training_records = records_by_split["training"]
    validation_records = records_by_split["validation"]
    for identity_field in ("id", "task_hash"):
        training_values = {
            str(row.get(identity_field, "")) for row in training_records
        }
        validation_values = {
            str(row.get(identity_field, "")) for row in validation_records
        }
        if training_values & validation_values:
            reasons.append(f"split_overlap={identity_field}")
    training_maps = {str(row["map_hash"]) for row in training_records}
    validation_maps = {str(row["map_hash"]) for row in validation_records}
    if training_maps & validation_maps:
        reasons.append("split_overlap=map_hash")

    def task_content_hash(row: Mapping[str, Any]) -> str:
        return _canonical_hash(
            {
                "points": row["inspection_points_xyz"],
                "priorities": row["priorities"],
                "domain": {
                    field: row[field]
                    for field in (
                        "initial_soc",
                        "distance_budget_scale",
                        "time_budget_scale",
                        "wind_scale",
                        "wind_rotation_deg",
                        "wind_vertical_bias_mps",
                    )
                },
            }
        )

    training_content = {task_content_hash(row) for row in training_records}
    validation_content = {task_content_hash(row) for row in validation_records}
    if training_content & validation_content:
        reasons.append("split_overlap=task_content")
    report = {
        "schema_version": 1,
        "passed": not reasons,
        "protocol_hash": protocol["protocol_hash"],
        "training_manifest_hash": manifests["training"].get("manifest_hash"),
        "validation_manifest_hash": manifests["validation"].get("manifest_hash"),
        "training_scenario_count": len(training_records),
        "validation_scenario_count": len(validation_records),
        "training_map_count": len(training_maps),
        "validation_map_count": len(validation_maps),
        "reasons": reasons,
    }
    report["audit_hash"] = _canonical_hash(
        report, excluded=("audit_hash",)
    )
    return report


def seal_environment(
    protocol_path: Path,
    map_root: Path,
    output_root: Path,
    training_manifest_path: Path,
    validation_manifest_path: Path,
) -> Dict[str, Any]:
    """在试训前封存地图、任务、代码和隔离证据。"""

    protocol = load_protocol(protocol_path)
    real_audit = audit_real_dem_registry(
        protocol_path,
        map_root,
        output_path=output_root / "audits" / "audit_real_dem_registry.json",
    )
    training_map_audit = audit_procedural_registry(
        protocol_path, map_root, "training"
    )
    validation_map_audit = audit_procedural_registry(
        protocol_path, map_root, "validation"
    )
    split_audit = audit_task_splits(
        protocol_path,
        map_root,
        training_manifest_path,
        validation_manifest_path,
    )
    _atomic_json(
        output_root / "audits" / "audit_procedural_training.json",
        training_map_audit,
    )
    _atomic_json(
        output_root / "audits" / "audit_procedural_validation.json",
        validation_map_audit,
    )
    _atomic_json(
        output_root / "audits" / "audit_task_splits.json",
        split_audit,
    )
    component_audits = {
        "real_dem": real_audit,
        "training_maps": training_map_audit,
        "validation_maps": validation_map_audit,
        "task_splits": split_audit,
    }
    failed = [
        name for name, report in component_audits.items() if not report["passed"]
    ]
    synthetic_registry = (
        map_root / "procedural" / "synthetic_test" / "map_registry.json"
    )
    if synthetic_registry.exists():
        failed.append("synthetic_test_generated_before_training_freeze")
    if failed:
        raise RuntimeError("环境封存失败：" + ", ".join(failed))
    code_files = [
        ROOT / "uav_inspection/core/final_python_ppo_pointer.py",
        ROOT / "uav_inspection/experiments/paper_multimap_experiments.py",
        ROOT / "uav_inspection/experiments/paper_v3_2_experiments.py",
        ROOT / "uav_inspection/experiments/paper_difficulty_experiments.py",
        ROOT / "python_classical_algs" / "common.py",
        ROOT / "python_classical_algs" / "milp.py",
    ]
    code_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha256_file(path)
        for path in code_files
    }
    training_manifest = json.loads(
        Path(training_manifest_path).read_text(encoding="utf-8")
    )
    validation_manifest = json.loads(
        Path(validation_manifest_path).read_text(encoding="utf-8")
    )
    freeze = {
        "schema_version": 1,
        "state": "frozen_for_pilot",
        "protocol_hash": protocol["protocol_hash"],
        "training_manifest_hash": training_manifest["manifest_hash"],
        "validation_manifest_hash": validation_manifest["manifest_hash"],
        "training_scenario_count": training_manifest["scenario_count"],
        "validation_scenario_count": validation_manifest["scenario_count"],
        "real_dem_registry_hash": json.loads(
            (map_root / "real" / "map_registry.json").read_text(encoding="utf-8")
        )["registry_hash"],
        "training_map_registry_hash": training_manifest["map_registry_hash"],
        "validation_map_registry_hash": validation_manifest["map_registry_hash"],
        "split_audit_hash": split_audit["audit_hash"],
        "code_sha256": code_hashes,
        "real_external_test_access_during_training": "forbidden",
        "synthetic_test_generation_before_training_freeze": "forbidden",
        "selection_used_algorithm_results": False,
    }
    freeze["freeze_hash"] = _canonical_hash(
        freeze, excluded=("freeze_hash",)
    )
    destination = output_root / "environment_freeze.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != freeze:
            raise RuntimeError("已有环境封存文件与当前身份不一致。")
    else:
        _atomic_json(destination, freeze)
    return freeze


def _load_frozen_task_set(
    protocol_path: Path,
    map_root: Path,
    manifest_path: Path,
    *,
    split: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    protocol = load_protocol(protocol_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = protocol["map_splits"][split]
    audit = audit_task_manifest(
        protocol_path,
        map_root,
        manifest_path,
        expected_map_count=int(expected["map_count"]),
        expected_tasks_per_map=int(expected["tasks_per_map"]),
    )
    if not audit["passed"]:
        raise RuntimeError(f"{split}任务审计未通过：{audit['reasons']}")
    if bool(manifest.get("smoke")):
        raise RuntimeError("smoke任务不得进入试训或正式训练。")
    return manifest, _read_jsonl(Path(manifest_path).parent / "records.jsonl")


def _multimap_training_cfg(
    protocol: Mapping[str, Any],
    provider: FrozenMapProvider,
    context: Mapping[str, Any],
    *,
    variant: str,
    seed: int,
    episodes: int,
    monitor_episodes: Sequence[int],
    run_dir: Path,
    training_manifest_hash: str,
    validation_manifest_hash: str,
    stage: str,
) -> Dict[str, Any]:
    from uav_inspection.core import final_python_ppo_pointer as ppo

    cfg = {
        "experiment_variant": variant,
        "reward_schema": str(protocol["objective"]["reward_schema"]),
        "seed": int(seed),
        "max_episodes": int(episodes),
        "checkpoint_dir": str(run_dir.resolve()),
        "monitor_episodes": [int(value) for value in monitor_episodes],
        "persist_monitor_checkpoints": stage == "pilot",
        "point_z_mode": "terrain",
        "terrain_clearance_m": 18.0,
        "coordinate_scale_m_per_unit": float(
            context["cfg_overrides"]["coordinate_scale_m_per_unit"]
        ),
        "scenario_provider_hash": provider.provider_hash,
        "multimap_protocol_hash": str(protocol["protocol_hash"]),
        "multimap_training_manifest_hash": str(training_manifest_hash),
        "multimap_validation_manifest_hash": str(validation_manifest_hash),
        "experiment_stage": stage,
    }
    return ppo.resolve_config(cfg)


def _multimap_health_callback(
    run_dir: Path,
    *,
    output_root: Path,
    protocol: Mapping[str, Any],
    stage: str,
    variant: str,
    seed: int,
) -> Any:
    metrics_path = run_dir / "training_metrics.jsonl"
    by_episode = {
        int(float(row["episodes_seen"])): dict(row)
        for row in _read_jsonl(metrics_path)
    }
    stage_config = protocol[
        "pilot_training" if stage == "pilot" else "formal_training"
    ]
    monitor_set = {int(value) for value in stage_config["monitor_episodes"]}

    def callback(raw: Mapping[str, Any]) -> None:
        row = copy.deepcopy(dict(raw))
        episode = int(float(row["episodes_seen"]))
        by_episode[episode] = row
        _atomic_text(
            metrics_path,
            _jsonl_text([by_episode[key] for key in sorted(by_episode)]),
        )
        validation = dict(row.get("validation") or {})
        alerts: List[str] = []
        finite_values = [
            float(value)
            for value in (
                row.get("mean_return"),
                row.get("mean_coverage"),
                row.get("mean_weighted_coverage"),
                row.get("return_rate"),
            )
            if value is not None
        ]
        if not all(math.isfinite(value) for value in finite_values):
            alerts.append("nonfinite_training_metric")
        # v3.2 的600回合工程试训只验证实现链路，不得把未预注册的
        # 表现阈值偷渡成停训或调参依据；正式阶段才沿用既有健康阈值。
        score_floor_keys = (
            "floor_safe_rate",
            "zero_visit_warning_rate",
            "floor_median_oracle_attainment",
        )
        enforce_score_floors = all(
            key in protocol["pilot_training"] for key in score_floor_keys
        ) and not bool(
            stage == "pilot"
            and protocol["pilot_training"].get("engineering_only", False)
        )
        if validation and enforce_score_floors:
            if float(validation.get("return_rate", 0.0)) < float(
                protocol["pilot_training"]["floor_safe_rate"]
            ):
                alerts.append("validation_safe_rate_below_floor")
            if float(validation.get("zero_visit_rate", 0.0)) > float(
                protocol["pilot_training"]["zero_visit_warning_rate"]
            ):
                alerts.append("validation_zero_visit_above_limit")
            attainment = validation.get("median_oracle_attainment_lower")
            if attainment is not None and float(attainment) < float(
                protocol["pilot_training"]["floor_median_oracle_attainment"]
            ):
                alerts.append("validation_oracle_attainment_below_floor")
        if episode in monitor_set:
            report = {
                "schema_version": 1,
                "protocol_hash": protocol["protocol_hash"],
                "stage": stage,
                "variant": variant,
                "training_seed": int(seed),
                "episode": episode,
                "training_node_count": row.get("training_node_count"),
                "training_metrics": {
                    key: row.get(key)
                    for key in (
                        "mean_return",
                        "mean_coverage",
                        "mean_weighted_coverage",
                        "return_rate",
                        "mean_energy_utilization",
                        "mean_distance_utilization",
                        "mean_time_utilization",
                        "termination_reason_counts",
                    )
                },
                "validation": validation,
                "alerts": alerts,
                "single_model_alerts_do_not_authorize_protocol_changes": True,
            }
            report["health_hash"] = _canonical_hash(
                report, excluded=("health_hash",)
            )
            _atomic_json(
                run_dir / "health" / f"episode_{episode:04d}.json", report
            )
        _atomic_json(
            run_dir / "status.json",
            {
                "state": "running",
                "variant": variant,
                "training_seed": int(seed),
                "completed": episode,
                "total": int(stage_config["episodes"]),
                "latest_checkpoint": row.get("latest_checkpoint"),
                "monitor_checkpoint": row.get("monitor_checkpoint"),
                "alerts": alerts,
            },
        )
        if stage == "formal" and episode in monitor_set:
            aggregate = _refresh_formal_group_health(
                output_root, protocol, episode
            )
            if aggregate is not None and aggregate["collective_stop_required"]:
                raise RuntimeError(
                    "五种子核心模型在预注册监控点共同触发停止条件；"
                    "已保存latest检查点和聚合健康报告。"
                )

    return callback


def _refresh_formal_group_health(
    output_root: Path,
    protocol: Mapping[str, Any],
    episode: int,
) -> Optional[Dict[str, Any]]:
    # v3.2 的工程试训只覆盖新增传统 PPO；正式健康监测也只等待新训练
    # 变体，父协议的30个冻结模型在训练后审计中以只读方式核验。
    core_variants = [
        str(value)
        for value in protocol["formal_training"].get(
            "new_training_variants", protocol["pilot_training"]["variants"]
        )
    ]
    seeds = [int(value) for value in protocol["formal_training"]["seeds"]]
    total_episodes = int(protocol["formal_training"]["episodes"])
    reports: List[Dict[str, Any]] = []
    for variant in core_variants:
        for seed in seeds:
            path = (
                output_root
                / "formal_training"
                / f"formal_{variant}_seed{seed}_{total_episodes}ep"
                / "health"
                / f"episode_{int(episode):04d}.json"
            )
            if not path.is_file():
                return None
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    by_variant: Dict[str, Dict[str, Any]] = {}
    for variant in core_variants:
        selected = [
            report for report in reports if report["variant"] == variant
        ]
        validations = [dict(report.get("validation") or {}) for report in selected]
        by_variant[variant] = {
            "seed_count": len(selected),
            "seeds": sorted(int(report["training_seed"]) for report in selected),
            "median_safe_rate": float(
                statistics.median(
                    float(value.get("return_rate", 0.0))
                    for value in validations
                )
            ),
            "median_zero_visit_rate": float(
                statistics.median(
                    float(value.get("zero_visit_rate", 1.0))
                    for value in validations
                )
            ),
            "median_oracle_attainment": float(
                statistics.median(
                    float(
                        value.get("median_oracle_attainment_lower")
                        if value.get("median_oracle_attainment_lower") is not None
                        else 0.0
                    )
                    for value in validations
                )
            ),
            "median_safe_weighted_coverage": float(
                statistics.median(
                    float(value.get("safe_weighted_coverage", 0.0))
                    for value in validations
                )
            ),
        }
    engineering_only = bool(protocol["pilot_training"].get("engineering_only", False))
    if engineering_only:
        # 工程试训不以相对成绩决定是否保留传统基线，避免按表现调参。
        all_safety_collapse = False
        all_attainment_collapse = False
        all_zero_visit_failure = False
    else:
        all_safety_collapse = all(
            item["median_safe_rate"]
            < float(protocol["pilot_training"]["floor_safe_rate"])
            for item in by_variant.values()
        )
        all_attainment_collapse = all(
            item["median_oracle_attainment"]
            < float(protocol["pilot_training"]["floor_median_oracle_attainment"])
            for item in by_variant.values()
        )
        all_zero_visit_failure = all(
            item["median_zero_visit_rate"]
            > float(protocol["pilot_training"]["zero_visit_warning_rate"])
            for item in by_variant.values()
        )
    report = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "episode": int(episode),
        "core_model_count": len(reports),
        "all_five_seeds_present_per_variant": True,
        "by_variant": by_variant,
        "all_core_safety_collapse": all_safety_collapse,
        "all_core_attainment_collapse": all_attainment_collapse,
        "all_core_zero_visit_failure": all_zero_visit_failure,
        "engineering_only": engineering_only,
        "collective_stop_required": bool(
            all_safety_collapse
            or all_attainment_collapse
            or all_zero_visit_failure
        ),
        "single_variant_or_seed_lag_does_not_authorize_changes": True,
    }
    report["health_hash"] = _canonical_hash(
        report, excluded=("health_hash",)
    )
    _atomic_json(
        output_root
        / "formal_training"
        / "group_health"
        / f"episode_{int(episode):04d}.json",
        report,
    )
    return report


def run_multimap_training_grid(
    protocol_path: Path,
    map_root: Path,
    output_root: Path,
    training_manifest_path: Path,
    validation_manifest_path: Path,
    *,
    stage: str,
    device: str = "cuda",
    resume_existing: bool = False,
    variants: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    from uav_inspection.core import final_python_ppo_pointer as ppo

    protocol = load_protocol(protocol_path)
    freeze_path = output_root / "environment_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("环境尚未封存，不得启动试训或正式训练。")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("protocol_hash") != protocol["protocol_hash"]:
        raise RuntimeError("环境封存与当前协议不一致。")
    for relative, expected_hash in dict(freeze.get("code_sha256") or {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256_file(path) != str(expected_hash):
            raise RuntimeError(f"环境封存后的代码哈希漂移：{relative}")
    training_manifest, training_records = _load_frozen_task_set(
        protocol_path,
        map_root,
        training_manifest_path,
        split="training",
    )
    validation_manifest, validation_records = _load_frozen_task_set(
        protocol_path,
        map_root,
        validation_manifest_path,
        split="validation",
    )
    split_audit = audit_task_splits(
        protocol_path,
        map_root,
        training_manifest_path,
        validation_manifest_path,
    )
    if not split_audit["passed"]:
        raise RuntimeError("训练与validation隔离审计未通过。")
    if split_audit["audit_hash"] != freeze.get("split_audit_hash"):
        raise RuntimeError("训练与validation隔离证据在封存后发生漂移。")

    if stage == "pilot":
        stage_config = protocol["pilot_training"]
        expected_variants = [str(value) for value in stage_config["variants"]]
        expected_seeds = [int(stage_config["seed"])]
        run_root = output_root / "pilot"
    elif stage == "formal":
        stage_config = protocol["formal_training"]
        decision_path = output_root / "pilot" / "pilot_decision.json"
        if not decision_path.is_file():
            raise RuntimeError("试训尚未评估，不得启动正式训练。")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("decision") not in {
            "pilot_passed",
            "pilot_passed_pointer_lag",
            "pilot_passed_engineering",
        }:
            raise RuntimeError("试训未通过，不得启动新增传统PPO的正式训练。")
        expected_variants = [
            str(value)
            for value in stage_config.get(
                "new_training_variants", stage_config["variants"]
            )
        ]
        expected_seeds = [int(value) for value in stage_config["seeds"]]
        run_root = output_root / "formal_training"
    else:
        raise ValueError("stage只能是pilot或formal。")
    selected_variants = [str(value) for value in (variants or expected_variants)]
    selected_seeds = [int(value) for value in (seeds or expected_seeds)]
    if not set(selected_variants).issubset(expected_variants):
        raise ValueError("请求了当前阶段未注册的学习变体。")
    if not set(selected_seeds).issubset(expected_seeds):
        raise ValueError("请求了当前阶段未注册的训练种子。")
    if set(selected_variants) != set(expected_variants) or set(
        selected_seeds
    ) != set(expected_seeds):
        raise ValueError(
            "多地图试训和正式训练必须按协议整批运行，不允许只训练有利子集。"
        )
    planned = [
        (variant, seed)
        for variant in expected_variants
        if variant in selected_variants
        for seed in expected_seeds
        if seed in selected_seeds
    ]
    episodes = int(stage_config["episodes"])
    if dry_run:
        return {
            "action": "multimap_training_grid",
            "dry_run": True,
            "stage": stage,
            "planned_models": len(planned),
            "episodes_per_model": episodes,
            "planned_episodes": len(planned) * episodes,
            "protocol_hash": protocol["protocol_hash"],
        }
    registry_paths = [
        Path(training_manifest["map_registry_path"]),
        Path(validation_manifest["map_registry_path"]),
    ]
    provider = FrozenMapProvider.from_registries(map_root, registry_paths)
    first_record = training_records[0]
    context = provider(first_record)
    default_points = np.asarray(
        first_record["inspection_points_xyz"], dtype=np.float32
    )
    default_priorities = np.asarray(
        first_record["priorities"], dtype=np.float32
    )
    completed: List[str] = []
    for variant, seed in planned:
        run_dir = run_root / f"{stage}_{variant}_seed{seed}_{episodes}ep"
        latest = run_dir / "latest.pt"
        if run_dir.exists() and not resume_existing:
            raise FileExistsError(
                f"训练目录已存在：{run_dir}；恢复请使用--resume-existing。"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = _multimap_training_cfg(
            protocol,
            provider,
            context,
            variant=variant,
            seed=seed,
            episodes=episodes,
            monitor_episodes=stage_config["monitor_episodes"],
            run_dir=run_dir,
            training_manifest_hash=str(training_manifest["manifest_hash"]),
            validation_manifest_hash=str(validation_manifest["manifest_hash"]),
            stage=stage,
        )
        run_config = {
            "schema_version": 1,
            "stage": stage,
            "variant": variant,
            "training_seed": seed,
            "episodes": episodes,
            "protocol_hash": protocol["protocol_hash"],
            "environment_freeze_hash": freeze["freeze_hash"],
            "training_manifest_hash": training_manifest["manifest_hash"],
            "validation_manifest_hash": validation_manifest["manifest_hash"],
            "scenario_provider_hash": provider.provider_hash,
            "training_config": cfg,
            "paper_eligible": stage == "formal",
        }
        _atomic_json(run_dir / "run_config.json", run_config)
        _atomic_json(
            run_dir / "status.json",
            {
                "state": "starting",
                "variant": variant,
                "training_seed": seed,
                "completed": 0,
                "total": episodes,
            },
        )
        try:
            model, _returns = ppo.train_policy_improved(
                context["start_pos"],
                default_points,
                default_priorities,
                context["terrain"],
                cfg,
                context["wind_data"],
                resume_from=(
                    latest if resume_existing and latest.exists() else None
                ),
                metrics_callback=_multimap_health_callback(
                    run_dir,
                    output_root=output_root,
                    protocol=protocol,
                    stage=stage,
                    variant=variant,
                    seed=seed,
                ),
                target_device=device,
                validation_instances=validation_records,
                training_instances=training_records,
                scenario_provider=provider,
            )
            summary = dict(getattr(model, "training_summary", {}) or {})
            _atomic_json(run_dir / "training_summary.json", summary)
            _atomic_json(
                run_dir / "status.json",
                {
                    "state": "completed",
                    "variant": variant,
                    "training_seed": seed,
                    "completed": episodes,
                    "total": episodes,
                    "selection_kind": summary.get("selection_kind"),
                },
            )
            completed.append(f"{variant}__seed{seed}")
        except Exception as exc:
            metrics = _read_jsonl(run_dir / "training_metrics.jsonl")
            _atomic_json(
                run_dir / "status.json",
                {
                    "state": "failed",
                    "variant": variant,
                    "training_seed": seed,
                    "completed": (
                        int(float(metrics[-1]["episodes_seen"])) if metrics else 0
                    ),
                    "total": episodes,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
    return {
        "stage": stage,
        "completed_models": completed,
        "protocol_hash": protocol["protocol_hash"],
    }


def _evaluate_multimap_checkpoint(
    checkpoint: Path,
    records: Sequence[Mapping[str, Any]],
    provider: FrozenMapProvider,
    *,
    device: str,
    algorithm: str,
    episode: int,
) -> List[Dict[str, Any]]:
    from uav_inspection.core import final_python_ppo_pointer as ppo

    model, payload = ppo.load_checkpoint(checkpoint, map_location=device)
    if str(payload.get("checkpoint_kind")) != "monitor":
        raise RuntimeError(f"试训评估只接受monitor检查点：{checkpoint}")
    cfg = ppo.resolve_config(dict(payload["cfg"]))
    rows: List[Dict[str, Any]] = []
    for record in records:
        context = provider(record)
        provider_cfg = ppo.resolve_config(
            {
                **cfg,
                **dict(context["cfg_overrides"]),
            }
        )
        scenario_cfg, wind = ppo.apply_frozen_domain_instance(
            provider_cfg, context["wind_data"], record
        )
        detail = ppo.plan_with_policy_improved(
            model,
            context["start_pos"],
            np.asarray(record["inspection_points_xyz"], dtype=np.float32),
            np.asarray(record["priorities"], dtype=np.float32),
            context["terrain"],
            scenario_cfg,
            wind,
            return_details=True,
            decode_mode="deterministic",
        )
        metrics = dict(detail["metrics"])
        safe = bool(metrics.get("returned")) and not any(
            bool(metrics.get(field, False))
            for field in (
                "energy_violation",
                "distance_violation",
                "time_violation",
                "dynamics_violation",
            )
        )
        achieved = (
            float(metrics.get("weighted_coverage", 0.0)) if safe else 0.0
        )
        upper = float(
            record["certificate"]["weighted_coverage_upper_bound"]
        )
        rows.append(
            {
                "scenario_id": record["id"],
                "node_count": int(len(record["priorities"])),
                "algorithm": algorithm,
                "episode": int(episode),
                "safe": safe,
                "safe_weighted_coverage": achieved,
                "visited_count": int(metrics.get("visited_count", 0)),
                "partial_return": safe
                and str(metrics.get("termination_reason")) == "returned_partial",
                "oracle_upper": upper,
                "oracle_attainment_lower": min(
                    1.0, achieved / max(upper, 1e-12)
                ),
                "within_one_percent_of_oracle": achieved
                >= upper - 0.01 - 1e-12,
            }
        )
    return rows


def _evaluate_pilot_greedies(
    records: Sequence[Mapping[str, Any]],
    provider: FrozenMapProvider,
) -> List[Dict[str, Any]]:
    from uav_inspection.core import final_python_ppo_pointer as ppo
    from python_classical_algs import run_planner
    from python_classical_algs.common import make_problem

    rows: List[Dict[str, Any]] = []
    for record in records:
        context = provider(record)
        cfg = ppo.resolve_config(
            {
                "reward_schema": "multimap_v3_1",
                "point_z_mode": "terrain",
                "terrain_clearance_m": 18.0,
                **dict(context["cfg_overrides"]),
            }
        )
        scenario_cfg, wind = ppo.apply_frozen_domain_instance(
            cfg, context["wind_data"], record
        )
        problem = make_problem(
            context["start_pos"],
            np.asarray(record["inspection_points_xyz"], dtype=np.float32),
            np.asarray(record["priorities"], dtype=np.float32),
            context["terrain"],
            scenario_cfg,
            wind,
            name=str(record["id"]),
        )
        upper = float(
            record["certificate"]["weighted_coverage_upper_bound"]
        )
        for algorithm in (
            "nearest_feasible",
            "priority_resource_greedy",
        ):
            result = run_planner(algorithm, problem, seed=42)
            metrics = dict(result.metrics)
            safe = bool(metrics.get("returned")) and not any(
                bool(metrics.get(field, False))
                for field in (
                    "energy_violation",
                    "distance_violation",
                    "time_violation",
                    "dynamics_violation",
                )
            )
            achieved = (
                float(metrics.get("weighted_coverage", 0.0)) if safe else 0.0
            )
            rows.append(
                {
                    "scenario_id": record["id"],
                    "node_count": int(len(record["priorities"])),
                    "algorithm": algorithm,
                    "episode": 0,
                    "safe": safe,
                    "safe_weighted_coverage": achieved,
                    "visited_count": int(metrics.get("visited_count", 0)),
                    "partial_return": safe
                    and str(metrics.get("termination_reason"))
                    == "returned_partial",
                    "oracle_upper": upper,
                    "oracle_attainment_lower": min(
                        1.0, achieved / max(upper, 1e-12)
                    ),
                    "within_one_percent_of_oracle": achieved
                    >= upper - 0.01 - 1e-12,
                }
            )
    return rows


def _pilot_row_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    algorithm: str,
    episode: int,
) -> Dict[str, Any]:
    selected = [
        row
        for row in rows
        if str(row["algorithm"]) == algorithm
        and int(row["episode"]) == int(episode)
    ]
    if not selected:
        raise RuntimeError(f"缺少试训评估行：{algorithm}@{episode}")
    return {
        "algorithm": algorithm,
        "episode": int(episode),
        "scenario_count": len(selected),
        "safe_rate": float(np.mean([bool(row["safe"]) for row in selected])),
        "zero_visit_rate": float(
            np.mean([int(row["visited_count"]) == 0 for row in selected])
        ),
        "partial_return_rate": float(
            np.mean([bool(row["partial_return"]) for row in selected])
        ),
        "median_oracle_attainment": float(
            statistics.median(
                float(row["oracle_attainment_lower"]) for row in selected
            )
        ),
        "near_oracle_scene_share": float(
            np.mean(
                [
                    bool(row["within_one_percent_of_oracle"])
                    for row in selected
                ]
            )
        ),
        "mean_safe_weighted_coverage": float(
            np.mean(
                [float(row["safe_weighted_coverage"]) for row in selected]
            )
        ),
    }


def assess_multimap_pilot(
    protocol_path: Path,
    map_root: Path,
    output_root: Path,
    validation_manifest_path: Path,
    *,
    device: str = "cuda",
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    validation_manifest, validation_records = _load_frozen_task_set(
        protocol_path,
        map_root,
        validation_manifest_path,
        split="validation",
    )
    # v3.2 训练/验证资产只读复用父协议目录；工程试训输出目录不复制
    # 清单，以免误把派生结果伪装成新的训练资产。
    training_manifest_path = (
        Path(validation_manifest_path).parent.parent / "training" / "manifest.json"
    )
    if not training_manifest_path.is_file():
        training_manifest_path = (
            output_root / "manifests" / "training" / "manifest.json"
        )
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8")
    )
    provider = FrozenMapProvider.from_registries(
        map_root,
        [
            Path(training_manifest["map_registry_path"]),
            Path(validation_manifest["map_registry_path"]),
        ],
    )
    pilot = protocol["pilot_training"]
    rows: List[Dict[str, Any]] = []
    episodes = [int(value) for value in pilot["monitor_episodes"]]
    final_episode = int(pilot["episodes"])
    for variant in pilot["variants"]:
        run_dir = (
            output_root
            / "pilot"
            / f"pilot_{variant}_seed{pilot['seed']}_{final_episode}ep"
        )
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("state") != "completed":
            raise RuntimeError(f"试训模型尚未完成：{run_dir}")
        for episode in episodes:
            checkpoint = run_dir / f"monitor_ep{episode:04d}.pt"
            if not checkpoint.is_file():
                raise RuntimeError(f"缺少试训monitor检查点：{checkpoint}")
            rows.extend(
                _evaluate_multimap_checkpoint(
                    checkpoint,
                    validation_records,
                    provider,
                    device=device,
                    algorithm=str(variant),
                    episode=episode,
                )
            )
    rows.extend(_evaluate_pilot_greedies(validation_records, provider))
    summaries = [
        _pilot_row_summary(rows, algorithm=str(variant), episode=episode)
        for variant in pilot["variants"]
        for episode in episodes
    ]
    summaries.extend(
        _pilot_row_summary(rows, algorithm=algorithm, episode=0)
        for algorithm in (
            "nearest_feasible",
            "priority_resource_greedy",
        )
    )
    _atomic_text(
        output_root / "pilot" / "pilot_validation_rows.jsonl",
        _jsonl_text(rows),
    )
    _atomic_json(
        output_root / "pilot" / "pilot_summary.json",
        {"schema_version": 1, "summaries": summaries},
    )
    final = {
        str(row["algorithm"]): row
        for row in summaries
        if int(row["episode"]) == final_episode
    }
    greedy = {
        str(row["algorithm"]): row
        for row in summaries
        if int(row["episode"]) == 0
    }
    core_names = [str(value) for value in pilot["variants"]]
    if bool(pilot.get("engineering_only", False)):
        required_counts = {int(value) for value in pilot["required_node_counts"]}
        observed_counts = {
            int(row["node_count"])
            for row in rows
            if str(row.get("algorithm")) in set(core_names)
        }
        final_row = final.get("traditional_ppo")
        finite_metrics = all(
            math.isfinite(float(row.get("safe_weighted_coverage", math.nan)))
            and math.isfinite(float(row.get("oracle_attainment_lower", math.nan)))
            for row in rows
            if str(row.get("algorithm")) in set(core_names)
        )
        safe_return = bool(final_row and float(final_row["safe_rate"]) > 0.0)
        checks = {
            "finite_metrics": finite_metrics,
            "required_node_counts": observed_counts == required_counts,
            "at_least_one_safe_return": safe_return,
        }
        decision = (
            "pilot_passed_engineering" if all(checks.values()) else "pilot_failed_engineering"
        )
        report = {
            "schema_version": 1,
            "protocol_hash": protocol["protocol_hash"],
            "validation_manifest_hash": validation_manifest["manifest_hash"],
            "decision": decision,
            "paper_eligible": False,
            "formal_training_allowed": decision == "pilot_passed_engineering",
            "engineering_only": True,
            "engineering_checks": checks,
            "observed_node_counts": sorted(observed_counts),
            "final_core": final,
            "greedy_reference": greedy,
            "score_based_tuning_forbidden": True,
        }
        report["decision_hash"] = _canonical_hash(
            report, excluded=("decision_hash",)
        )
        _atomic_json(output_root / "pilot" / "pilot_decision.json", report)
        return report
    all_near_oracle = all(
        final[name]["near_oracle_scene_share"]
        >= float(pilot["ceiling_scene_share"])
        for name in core_names
    ) and all(
        row["near_oracle_scene_share"] >= float(pilot["ceiling_scene_share"])
        for row in greedy.values()
    )
    all_attainment_too_low = all(
        final[name]["median_oracle_attainment"]
        < float(pilot["floor_median_oracle_attainment"])
        for name in core_names
    )
    all_safety_too_low = all(
        final[name]["safe_rate"] < float(pilot["floor_safe_rate"])
        for name in core_names
    )
    zero_visit_alert = any(
        final[name]["zero_visit_rate"]
        > float(pilot["zero_visit_warning_rate"])
        for name in core_names
    )
    pointer_lag = final["full"]["mean_safe_weighted_coverage"] < max(
        final["ppo_mlp"]["mean_safe_weighted_coverage"],
        final["a2c_pointer"]["mean_safe_weighted_coverage"],
    )
    if all_near_oracle:
        decision = "revise_task_too_easy"
    elif all_attainment_too_low or all_safety_too_low:
        decision = "revise_task_too_hard"
    elif zero_visit_alert:
        decision = "diagnose_zero_visit"
    elif pointer_lag:
        decision = "pilot_passed_pointer_lag"
    else:
        decision = "pilot_passed"
    report = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "validation_manifest_hash": validation_manifest["manifest_hash"],
        "decision": decision,
        "paper_eligible": False,
        "formal_training_allowed": decision
        in {"pilot_passed", "pilot_passed_pointer_lag"},
        "all_core_and_greedies_near_oracle": all_near_oracle,
        "all_core_attainment_too_low": all_attainment_too_low,
        "all_core_safety_too_low": all_safety_too_low,
        "zero_visit_alert": zero_visit_alert,
        "ppo_pointer_lag": pointer_lag,
        "final_core": final,
        "greedy_reference": greedy,
        "pointer_lag_does_not_authorize_task_changes": True,
    }
    report["decision_hash"] = _canonical_hash(
        report, excluded=("decision_hash",)
    )
    _atomic_json(output_root / "pilot" / "pilot_decision.json", report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPO+Pointer多地图泛化实验的DEM获取、地图注册与审计。"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show-protocol")

    acquire = subparsers.add_parser("acquire-real")
    acquire.add_argument("--resume-existing", action="store_true")
    acquire.add_argument("--region", action="append", dest="regions")

    subparsers.add_parser("audit-real")
    subparsers.add_parser("seal-real-bundles")

    procedural = subparsers.add_parser("prepare-procedural")
    procedural.add_argument(
        "--split",
        action="append",
        required=True,
        choices=("training", "validation", "synthetic_test"),
    )
    procedural.add_argument("--real-registry", type=Path, required=True)
    procedural.add_argument("--training-freeze", type=Path)
    procedural.add_argument("--legacy-dem", type=Path, default=DEFAULT_LEGACY_DEM)
    procedural.add_argument("--resume-existing", action="store_true")

    audit = subparsers.add_parser("audit-procedural")
    audit.add_argument(
        "--split",
        required=True,
        choices=("training", "validation", "synthetic_test"),
    )

    tasks = subparsers.add_parser("prepare-tasks")
    tasks.add_argument(
        "--split",
        required=True,
        choices=("training", "validation", "synthetic_test"),
    )
    tasks.add_argument("--map-registry", type=Path, required=True)
    tasks.add_argument("--training-freeze", type=Path)
    tasks.add_argument("--resume-existing", action="store_true")
    tasks.add_argument("--map-limit", type=int)
    tasks.add_argument("--map-index-start", type=int)
    tasks.add_argument("--map-index-stop", type=int)
    tasks.add_argument("--shard-name")
    tasks.add_argument("--task-limit-per-map", type=int)
    tasks.add_argument("--certification-time-limit-s", type=float, default=60.0)
    tasks.add_argument("--screening-time-limit-s", type=float, default=10.0)
    tasks.add_argument("--max-attempts-per-task", type=int, default=2000)

    merge_tasks = subparsers.add_parser("merge-task-shards")
    merge_tasks.add_argument(
        "--split", required=True, choices=("training", "validation")
    )
    merge_tasks.add_argument("--map-registry", type=Path, required=True)
    merge_tasks.add_argument("--base-records", type=Path)

    task_audit = subparsers.add_parser("audit-tasks")
    task_audit.add_argument("--manifest", type=Path, required=True)

    split_audit = subparsers.add_parser("audit-splits")
    split_audit.add_argument("--training-manifest", type=Path, required=True)
    split_audit.add_argument("--validation-manifest", type=Path, required=True)

    seal = subparsers.add_parser("seal-environment")
    seal.add_argument("--training-manifest", type=Path, required=True)
    seal.add_argument("--validation-manifest", type=Path, required=True)

    train_grid = subparsers.add_parser("train-grid")
    train_grid.add_argument("--stage", choices=("pilot", "formal"), required=True)
    train_grid.add_argument("--training-manifest", type=Path, required=True)
    train_grid.add_argument("--validation-manifest", type=Path, required=True)
    train_grid.add_argument("--device", default="cuda")
    train_grid.add_argument("--variant", action="append", dest="variants")
    train_grid.add_argument("--seed", action="append", type=int, dest="seeds")
    train_grid.add_argument("--resume-existing", action="store_true")
    train_grid.add_argument("--dry-run", action="store_true")

    assess = subparsers.add_parser("assess-pilot")
    assess.add_argument("--validation-manifest", type=Path, required=True)
    assess.add_argument("--device", default="cuda")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    protocol_path = args.protocol.resolve()
    map_root = args.map_root.resolve()
    output_root = args.output_root.resolve()
    if args.command == "show-protocol":
        protocol = load_protocol(protocol_path)
        result = {
            "protocol_name": protocol["protocol_name"],
            "protocol_hash": protocol["protocol_hash"],
            "real_dem_count": protocol["real_dem_design"]["count"],
            "map_splits": protocol["map_splits"],
        }
    elif args.command == "acquire-real":
        result = acquire_real_dem_registry(
            protocol_path,
            map_root,
            output_root,
            resume_existing=bool(args.resume_existing),
            region_ids=args.regions,
        )
    elif args.command == "audit-real":
        result = audit_real_dem_registry(
            protocol_path,
            map_root,
            output_path=output_root / "audits" / "audit_real_dem_registry.json",
        )
    elif args.command == "seal-real-bundles":
        result = seal_real_map_bundles(protocol_path, map_root)
    elif args.command == "prepare-procedural":
        result = prepare_procedural_maps(
            protocol_path,
            map_root,
            splits=args.split,
            real_registry_path=args.real_registry.resolve(),
            training_freeze=(
                args.training_freeze.resolve()
                if args.training_freeze is not None
                else None
            ),
            resume_existing=bool(args.resume_existing),
            legacy_dem_path=args.legacy_dem.resolve(),
        )
    elif args.command == "audit-procedural":
        result = audit_procedural_registry(
            protocol_path, map_root, args.split
        )
    elif args.command == "prepare-tasks":
        result = prepare_task_manifest(
            protocol_path,
            map_root,
            output_root,
            split=args.split,
            map_registry_path=args.map_registry.resolve(),
            resume_existing=bool(args.resume_existing),
            map_limit=args.map_limit,
            map_index_start=args.map_index_start,
            map_index_stop=args.map_index_stop,
            shard_name=args.shard_name,
            task_limit_per_map=args.task_limit_per_map,
            certification_time_limit_s=float(args.certification_time_limit_s),
            screening_time_limit_s=float(args.screening_time_limit_s),
            max_attempts_per_task=int(args.max_attempts_per_task),
            training_freeze=(
                args.training_freeze.resolve()
                if args.training_freeze is not None
                else None
            ),
        )
    elif args.command == "merge-task-shards":
        result = merge_task_shards(
            protocol_path,
            map_root,
            output_root,
            split=args.split,
            map_registry_path=args.map_registry.resolve(),
            base_records_path=(
                args.base_records.resolve()
                if args.base_records is not None
                else None
            ),
        )
    elif args.command == "audit-tasks":
        result = audit_task_manifest(
            protocol_path,
            map_root,
            args.manifest.resolve(),
        )
    elif args.command == "audit-splits":
        result = audit_task_splits(
            protocol_path,
            map_root,
            args.training_manifest.resolve(),
            args.validation_manifest.resolve(),
        )
    elif args.command == "seal-environment":
        result = seal_environment(
            protocol_path,
            map_root,
            output_root,
            args.training_manifest.resolve(),
            args.validation_manifest.resolve(),
        )
    elif args.command == "train-grid":
        result = run_multimap_training_grid(
            protocol_path,
            map_root,
            output_root,
            args.training_manifest.resolve(),
            args.validation_manifest.resolve(),
            stage=args.stage,
            device=args.device,
            resume_existing=bool(args.resume_existing),
            variants=args.variants,
            seeds=args.seeds,
            dry_run=bool(args.dry_run),
        )
    elif args.command == "assess-pilot":
        result = assess_multimap_pilot(
            protocol_path,
            map_root,
            output_root,
            args.validation_manifest.resolve(),
            device=args.device,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())

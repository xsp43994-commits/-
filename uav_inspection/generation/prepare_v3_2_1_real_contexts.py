#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare post-v3.2 real-road-corridor assets without consulting algorithms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from uav_inspection.paths import WORKSPACE_ROOT
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from uav_inspection.experiments import paper_multimap_experiments as multimap


ROOT = WORKSPACE_ROOT
DEFAULT_PROTOCOL = ROOT / "paper_runs/protocols/multimap_generalization_v3_2_4/protocol.json"
DEFAULT_MAP_ROOT = ROOT / "map_data/multimap_v3_1"
DEFAULT_OUTPUT = ROOT / "paper_runs/multimap_v3_2_4"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any], *, excluded: Sequence[str] = ()) -> str:
    payload = {key: item for key, item in value.items() if key not in set(excluded)}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _pack_roads(roads: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    parts: List[np.ndarray] = []
    for road in roads:
        values = np.asarray(road, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 2:
            continue
        parts.append(values)
        offsets.append(offsets[-1] + len(values))
    if not parts:
        raise RuntimeError("道路走廊中没有可用折线")
    return np.vstack(parts), np.asarray(offsets, dtype=np.int32)


def _clip_local_coordinates(points: np.ndarray, shape: Tuple[int, int], epsilon_px: float) -> np.ndarray:
    """Project boundary points strictly inside the raster's valid pixel domain."""
    if not 0.0 < float(epsilon_px) < 0.5:
        raise ValueError("boundary epsilon must be in (0, 0.5) pixels")
    values = np.asarray(points, dtype=np.float32).copy()
    height, width = int(shape[0]), int(shape[1])
    lower = np.asarray([float(epsilon_px), float(epsilon_px)], dtype=np.float32)
    upper = np.asarray([float(width) - 1.0 - float(epsilon_px), float(height) - 1.0 - float(epsilon_px)], dtype=np.float32)
    if np.any(upper <= lower):
        raise ValueError("raster is too small for coordinate boundary clipping")
    return np.clip(values, lower, upper)


def _corridor_roads(
    record: Mapping[str, Any], map_root: Path, protocol: Mapping[str, Any]
) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """Use all clipped OSM segments as one local road corridor; no model output enters here."""
    (
        rasterio,
        _requests,
        _CRS,
        _Transformer,
        _Resampling,
        _from_origin,
        _reproject,
        *_rest,
    ) = multimap._optional_geo_imports()
    dem_path = map_root / str(record["dem_file"])
    raw_path = map_root / str(record["osm_raw_file"])
    with rasterio.open(dem_path) as source:
        inverse = ~source.transform
        bounds = (source.bounds.left, source.bounds.bottom, source.bounds.right, source.bounds.top)
        shape = (int(source.height), int(source.width))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    epsg = int(str(record["crs"]).split(":")[-1])
    lines = multimap._road_lines_utm(raw, epsg=epsg, crop_bounds_utm=bounds)
    epsilon_px = float(protocol["real_corridor_contexts"]["boundary_epsilon_px"])
    roads = [
        _clip_local_coordinates(
            np.asarray([inverse * (float(x), float(y)) for x, y in line.coords], dtype=np.float32),
            shape,
            epsilon_px,
        )
        for line in lines
        if len(line.coords) >= 2
    ]
    if not roads:
        raise RuntimeError(f"{record['map_id']}没有可用道路走廊")
    return roads, shape


def _rank_base_candidates(
    record: Mapping[str, Any], roads: Sequence[np.ndarray], protocol: Mapping[str, Any]
) -> List[Tuple[np.ndarray, int, int]]:
    """Return a deterministic, geometry-only capacity-ranked candidate deck."""
    generation = protocol["task_generation"]
    scale = float(record["resolution_m"])
    spacing = float(generation["road_sampling_spacing_m"]) / scale
    samples = np.vstack([multimap._polyline_resample(road, spacing) for road in roads])
    rounded = np.round(samples, decimals=3)
    _, unique = np.unique(rounded, axis=0, return_index=True)
    samples = samples[np.sort(unique)]
    stride = int(protocol["real_corridor_contexts"]["base_candidate_stride"])
    ranked: List[Tuple[np.ndarray, int, int]] = []
    for index in range(0, len(samples), stride):
        base = np.asarray(samples[index], dtype=np.float32)
        distances = np.linalg.norm(samples - base.reshape(1, 2), axis=1) * scale
        capacity = int(np.sum(
            (distances >= float(generation["minimum_depot_distance_m"]))
            & (distances <= float(generation["geometry_feasibility"]["maximum_radius_m"]))
        ))
        ranked.append((base, capacity, index))
    ranked.sort(key=lambda item: (-int(item[1]), int(item[2])))
    return ranked[: int(protocol["real_corridor_contexts"]["base_preflight_candidate_count"])]


def _feasible_bases(record: Mapping[str, Any], roads: Sequence[np.ndarray], shape: Tuple[int, int], protocol: Mapping[str, Any]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    generation = protocol["task_generation"]
    scale = float(record["resolution_m"])
    spacing = float(generation["road_sampling_spacing_m"]) / scale
    samples = np.vstack([multimap._polyline_resample(road, spacing) for road in roads])
    rounded = np.round(samples, decimals=3)
    _, unique = np.unique(rounded, axis=0, return_index=True)
    samples = samples[np.sort(unique)]
    stride = int(protocol["real_corridor_contexts"]["base_candidate_stride"])
    bundle = {
        "terrain": np.empty(shape, dtype=np.float32),
        "roads": list(roads),
        "metadata": {"coordinate_scale_m_per_unit": scale},
    }
    results: List[Tuple[np.ndarray, Dict[str, Any]]] = []
    for base, _ranked_capacity, index in _rank_base_candidates(record, roads, protocol):
        radii: Dict[str, Any] = {}
        try:
            for node_count in (16, 20, 24):
                minimum, interval = multimap._effective_task_radius_range(
                    record,
                    bundle,
                    protocol,
                    node_count=node_count,
                    difficulty="moderate",
                    depot_override_xy=base,
                )
                radii[str(node_count)] = {"minimum_m": minimum, "interval_m": list(interval)}
        except RuntimeError:
            continue
        # 每个基地必须能承载最大规模；16/20 已在同一循环中显式验证。
        distances = np.linalg.norm(samples - base.reshape(1, 2), axis=1) * scale
        capacity = int(np.sum((distances >= float(generation["minimum_depot_distance_m"])) & (distances <= float(generation["geometry_feasibility"]["maximum_radius_m"]))))
        results.append((base, {"sample_index": index, "capacity": capacity, "radii": radii}))
    return results


def _select_two_bases(candidates: Sequence[Tuple[np.ndarray, Mapping[str, Any]]]) -> List[Tuple[np.ndarray, Mapping[str, Any]]]:
    if len(candidates) < 2:
        raise RuntimeError("道路走廊不足两个可认证起飞基地")
    # 先选覆盖容量最大的基地，再选与其空间分离最大的基地，避免两个上下文退化成重复任务。
    first = max(candidates, key=lambda item: (int(item[1]["capacity"]), -int(item[1]["sample_index"])))
    second = max(
        (item for item in candidates if int(item[1]["sample_index"]) != int(first[1]["sample_index"])),
        key=lambda item: (
            float(np.linalg.norm(item[0] - first[0])),
            int(item[1]["capacity"]),
            -int(item[1]["sample_index"]),
        ),
    )
    return [first, second]


def prepare_contexts(protocol_path: Path, map_root: Path, output_root: Path) -> Dict[str, Any]:
    protocol = multimap.load_protocol(protocol_path)
    if protocol.get("protocol_version") not in {
        "multimap_generalization_v3_2_1",
        "multimap_generalization_v3_2_2",
        "multimap_generalization_v3_2_3",
        "multimap_generalization_v3_2_4",
    }:
        raise RuntimeError("道路走廊资产只能用于v3.2.1或v3.2.2协议")
    registry = json.loads((map_root / "real" / "map_registry.json").read_text(encoding="utf-8"))
    asset_root = output_root / "real_corridor_assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for record in registry.get("maps", ()):
        roads, shape = _corridor_roads(record, map_root, protocol)
        candidates = _feasible_bases(record, roads, shape, protocol)
        selected = _select_two_bases(candidates)
        road_points, road_offsets = _pack_roads(roads)
        path = asset_root / f"{record['map_id']}.npz"
        np.savez_compressed(
            path,
            road_points=road_points,
            road_offsets=road_offsets,
            base_positions=np.asarray([base for base, _meta in selected], dtype=np.float32),
        )
        row = {
            "map_id": str(record["map_id"]),
            "map_hash": str(record["map_hash"]),
            "source_osm_sha256": _sha256_file(map_root / str(record["osm_raw_file"])),
            "asset_path": str(path.resolve()),
            "asset_sha256": _sha256_file(path),
            "road_segment_count": len(roads),
            "contexts": [
                {"road_index": index, "start_xy": base.astype(float).tolist(), **dict(meta)}
                for index, (base, meta) in enumerate(selected)
            ],
        }
        row["context_hash"] = _canonical_hash(row, excluded=("context_hash",))
        rows.append(row)
    manifest = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "map_registry_sha256": _sha256_file(map_root / "real" / "map_registry.json"),
        "context_definition": protocol["real_corridor_contexts"]["definition"],
        "boundary_epsilon_px": float(protocol["real_corridor_contexts"]["boundary_epsilon_px"]),
        "coordinate_domain": "[epsilon, width-1-epsilon] x [epsilon, height-1-epsilon]",
        "maps": sorted(rows, key=lambda item: item["map_id"]),
    }
    manifest["manifest_hash"] = _canonical_hash(manifest, excluded=("manifest_hash",))
    manifest_path = asset_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare v3.2.1 real road-corridor assets")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare_contexts(args.protocol, args.map_root, args.output_root), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

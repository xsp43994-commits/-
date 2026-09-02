"""v3.2.14独立小图流水线的结构、视窗与已生成产物回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uav_inspection.figures import v3_2_14_publication_figures as base
from uav_inspection.figures import v3_2_14_split_publication_figures as split


def test_all_parent_figures_are_split_into_72_independent_panels() -> None:
    assert set(split.PANEL_SPECS) == set(base.FIGURE_ORDER)
    assert sum(len(specs) for specs in split.PANEL_SPECS.values()) == 72


def test_s06_colorbars_are_attached_to_the_correct_panels() -> None:
    specs = {spec.panel: spec.axes for spec in split.PANEL_SPECS["figS06_ablation_maps"]}
    assert specs == {"a": (0,), "b": (1, 4), "c": (2, 5), "d": (3, 6)}


def test_v01_exports_single_3d_scene_and_embedded_elevation_scale() -> None:
    """V1必须是单幅三维场景，并保留主坐标轴与嵌入式高程色标。"""
    spec = split.PANEL_SPECS["figV01_3d_route"][0]
    assert spec.axes == (0, 1)
    assert split.V1_PANEL_STEM == "figV01_3d_taihang_route"
    assert split.V1_VERTICAL_EXAGGERATION == 1.5
    assert split.V1_MASTER_WIDTH_MM == 183.0


def test_real_example_uses_an_input_only_local_corridor_view() -> None:
    bundle = base.load_bundle()
    task = base._task_by_id(bundle, base.REAL_EXAMPLE)
    with np.load(base._map_bundle_path(str(task["map_id"])), allow_pickle=False) as payload:
        terrain = np.asarray(payload["terrain"])
    xmin, xmax, ymin, ymax = split._task_view_bounds(task, terrain.shape)
    points = np.asarray(task["inspection_points_xyz"], dtype=float)[:, :2]
    start = np.asarray(task["start_xy"], dtype=float)
    assert xmin <= min(points[:, 0].min(), start[0])
    assert xmax >= max(points[:, 0].max(), start[0])
    assert ymin <= min(points[:, 1].min(), start[1])
    assert ymax >= max(points[:, 1].max(), start[1])
    assert 0 <= xmin < xmax <= terrain.shape[1] - 1
    assert 0 <= ymin < ymax <= terrain.shape[0] - 1
    assert (ymax - ymin) > 1.25 * (xmax - xmin)


def test_current_complete_output_passes_qa_when_present() -> None:
    output = split.DEFAULT_OUTPUT
    manifest_path = output / "figure_manifest.json"
    qa_path = output / "qa_report.json"
    if not manifest_path.exists() or not qa_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    assert manifest["panel_count"] == 72
    assert len(manifest["panels"]) == 72
    assert qa["panel_count"] == 72
    assert qa["passed"] is True
    assert "figV01_3d_taihang_route" in manifest["panels"]
    assert "figV01_3d_route_a" not in manifest["panels"]
    for record in manifest["panels"].values():
        assert set(record["files"]) == {"svg", "pdf", "png", "tiff"}
        for file_record in record["files"].values():
            assert (output / Path(file_record["path"])).is_file()

"""v3.2.14 第二轮制图交付的只读回归测试。"""

import pandas as pd

from uav_inspection.figures import v3_2_14_origin_redraw as redraw


def test_registry_has_frozen_26_independent_figures() -> None:
    tiers = [meta["tier"] for meta in redraw.FIGURES.values()]
    backends = [meta["backend"] for meta in redraw.FIGURES.values()]
    assert len(redraw.FIGURES) == 26
    assert tiers.count("main") == 12
    assert tiers.count("supplementary") == 12
    assert tiers.count("showcase") == 2
    assert backends.count("origin") == 24
    assert backends.count("python") == 2


def test_frozen_inputs_and_old_figures_are_unchanged() -> None:
    audit = redraw.audit_inputs()
    assert audit["passed"] is True
    assert audit["old_paper_final"]["sha256"] == redraw.OLD_TREE_SHA256_AT_START
    assert audit["analysis_state"] == "ready_for_plotting"


def test_all_source_data_are_nonempty_and_exclude_old_ppo_mlp() -> None:
    for figure_id in redraw.FIGURES:
        path = redraw.OUTPUT / "source_data" / f"{figure_id}_source_data.csv"
        assert path.is_file(), figure_id
        frame = pd.read_csv(path, encoding="utf-8-sig")
        assert not frame.empty, figure_id
        if "model" in frame:
            assert "ppo_mlp" not in set(frame["model"].astype(str)), figure_id


def test_delivery_contains_four_formats_and_24_origin_projects() -> None:
    for figure_id in redraw.FIGURES:
        stem = redraw._output_stem(figure_id)
        for suffix in (".pdf", ".svg", ".tiff", ".png"):
            target = stem.with_suffix(suffix)
            assert target.is_file() and target.stat().st_size > 0, target

    projects = list((redraw.OUTPUT / "origin_projects").glob("*.opju"))
    assert len(projects) == 24
    assert all(project.stat().st_size >= 10_000 for project in projects)


def test_fixed_route_examples_and_seed_contract() -> None:
    assert redraw.SYNTHETIC_EXAMPLE == "synthetic_test__synthetic_test__map_003__task_08"
    assert redraw.REAL_EXAMPLE == "real_test__cn_taihang__road_00__task_08"
    for figure_id in ("S11", "S12"):
        source = pd.read_csv(
            redraw.OUTPUT / "source_data" / f"{figure_id}_source_data.csv",
            encoding="utf-8-sig",
        )
        route_rows = source[source["record_type"].isin(["route", "route_missing"])]
        assert set(route_rows["model"]) == {"full", "a2c_pointer", "traditional_ppo", "milp"}
        assert set(route_rows["evaluation_seed"].dropna().astype(int)) == {redraw.ROUTE_EVALUATION_SEED}


def test_final_qa_report_passes() -> None:
    report = redraw.qa_all()
    assert report["passed"] is True
    assert report["figure_count"] == 26
    assert report["opju_count"] == 24
    assert report["old_unchanged"] is True

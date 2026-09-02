"""只读打开十个v7 Origin工程并导出原生PDF，用于可编辑性与后端审计。"""

from pathlib import Path
import json
import win32com.client


ROOT = Path(r"C:\Users\xsp\Desktop\DRL代码\paper_runs\multimap_v3_2_14\figures\paper_redraw_multibackend_v7_drones_style_fullsuite")
FIGURES = ("M02", "M03", "M04", "M05", "M07", "M08", "M10", "S02", "S03", "S07")


def main() -> None:
    output = (ROOT / "qa" / "origin_native_exports").resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    app = win32com.client.Dispatch("Origin.ApplicationSI")
    app.Visible = 1
    try:
        for figure_id in FIGURES:
            project = (ROOT / "editable" / "origin" / f"{figure_id}.opju").resolve()
            loaded = bool(app.Load(str(project)))
            graph = "M02" if figure_id == "M02" else "Graph1"
            app.Execute(f"win -a {graph};")
            exported = bool(app.Execute(
                f'expGraph type:=pdf path:="{str(output).replace(chr(92), "/")}" '
                f'filename:="{figure_id}_origin_native" overwrite:=replace tr.Margin:=1.5 tr1.Unit:=2;'
            ))
            pdf = output / f"{figure_id}_origin_native.pdf"
            records.append({"figure_id": figure_id, "project_loaded": loaded,
                            "native_pdf_exported": exported and pdf.is_file() and pdf.stat().st_size > 0,
                            "pdf": str(pdf), "project": str(project)})
            app.Execute("doc -s; doc -n;")
    finally:
        app.Exit()
    passed = all(item["project_loaded"] and item["native_pdf_exported"] for item in records)
    (ROOT / "qa" / "origin_native_export_audit.json").write_text(
        json.dumps({"passed": passed, "origin_version": "OriginPro 2021", "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit("Origin native export audit failed")


if __name__ == "__main__":
    main()

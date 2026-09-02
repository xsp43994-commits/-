# V4 build and validation scripts

All commands are run from the workspace root. The scripts create only versioned v4 outputs and do not call the research training/evaluation code.

## Build order

1. `python -X utf8 -B paper_delivery/EAAI_format_translation_v4_2026-08-30/scripts/prepare_figures_v4.py`
2. Run `render_V02_zh_v4.m` in MATLAB R2024b or later.
3. Run `build_docx_package_v4.py` with the bundled document Python runtime.
4. `powershell -ExecutionPolicy Bypass -File paper_delivery/EAAI_format_translation_v4_2026-08-30/scripts/render_docx_with_word_v4.ps1`
5. Run `render_and_contact_v4.py` with the bundled PDF/document Python runtime.
6. Run `build_format_and_glossary_workbooks_v4.mjs` with the bundled Node runtime and `@oai/artifact-tool` available.
7. Run `run_final_audits_v4.py` with the bundled PDF/document Python runtime.

## Important tunable locations

- Page size, margins, fonts, line spacing, columns and figure widths: `build_docx_package_v4.py`, constants and style helpers near the top of the file.
- Main/supplement figure mappings and captions: `build_docx_package_v4.py`, `FIGURE_MAP_*` and supplement lists.
- Chinese F01/M05 typography, export size and DPI: `prepare_figures_v4.py`, figure-specific build functions.
- Chinese V02 typography and export formats: `render_V02_zh_v4.m`, export block near the end.
- Contact-sheet DPI and pages per sheet: `render_and_contact_v4.py`, top-level constants.

Do not change figure data paths to non-frozen outputs. Do not use the two-column proof as the submission manuscript.

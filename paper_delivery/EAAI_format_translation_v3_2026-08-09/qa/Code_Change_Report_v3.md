# V3 build-script change report

- Added a versioned DOCX builder for English single-column, English two-column reading proof and Chinese single-column packages.
- Added frozen-data figure preparation plus a MATLAB V02 renderer; no research-model or experiment interface was changed.
- Added Word COM PDF rendering, Poppler page/contact-sheet rendering, spreadsheet generation and final audit scripts.
- Main document settings are centralized near the top of `build_docx_package_v3.py`; figure sizes/captions are in its figure maps.
- Figure export dimensions and typography are centralized in `prepare_figures_v3.py`; MATLAB V02 export settings are in `render_V02_zh_v3.m`.
- Validation: 11 DOCX, 11 PDFs, 2 XLSX, new figure format sets and all v3 audits completed successfully.

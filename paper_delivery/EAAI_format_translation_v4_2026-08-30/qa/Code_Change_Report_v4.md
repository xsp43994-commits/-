# V4 training-curve correction change report

- Added the read-only `training_curve_correction_v6` analysis and an independent recomputation audit for 35 formal training traces.
- Corrected M06 and S06 data provenance, recomputed D6/D7, all dependent seven-dimension scores and 37,410 sensitivity rows without retraining or changing 21,648 frozen route evaluations.
- Regenerated M06, M07, S06, S07 and S08 in Chinese and English with bilingual captions, Source Data, manifests and four export formats.
- Synchronized the corrected conclusions into the English/Chinese manuscript, supplementary material, cover letter and evidence map, then rebuilt 11 DOCX files.
- Key definitions are centralized in `uav_inspection/analysis/training_curve_correction_v6.py`; corrected-figure styling is centralized in `uav_inspection/figures/v3_2_14_training_corrected_v4.py`.
- Validation: 8 focused tests passed, the independent v6 audit passed, all 11 documents were rendered and visually inspected, and the final package audit passed.

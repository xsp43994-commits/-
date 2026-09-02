# Anonymous reproducibility package v2

This package accompanies the EAAI full-text rewrite. It contains the frozen v3.2.14 protocol, task/evaluation matrix, 21,648 formal result rows, audit state, pre-plot statistics, figure Source Data, implementation snapshots and file hashes. It does not retrain models or recompute route evaluations.

## Evidence identity

- Confirmatory endpoint: map-level `safe_weighted_coverage`.
- Independent inferential unit: map.
- Formal result rows: 21,648.
- Paper-eligible conventional PPO: fixed-slot `FlatMLPActorCritic` (`traditional_ppo`).
- The excluded historical attention-containing prototype is not included.

## Suggested checks

1. Verify `manifest_sha256_v2.json`.
2. Confirm that `results/final_results.jsonl` has 21,648 lines.
3. Inspect `results/final_audit_status.json` before using any result.
4. Run analysis only with the protocol and map-level aggregation rules supplied here.

Example from the workspace checkout:

```powershell
python -X utf8 -B paper_cli.py audit-workspace
python -X utf8 -B paper_cli.py show-paths
```

## Copernicus data

Raw Copernicus DEM/DSM assets are intentionally not redistributed in this anonymous package. The simulation used Copernicus DEM GLO-30 terrain inputs. Reconstruct the geographic task set from the official product record (DOI: 10.5270/ESA-c5d3d65), the public region/task identifiers in the evaluation matrix, and the task-building code. Before public release, the authors must confirm that the region identifiers and attribution wording are suitable for their repository.

## Scope boundary

The geographic results are zero-shot DSM simulation transfer. They are not real-flight validation, deployment certification, or evidence of extrapolation beyond the trained node counts.

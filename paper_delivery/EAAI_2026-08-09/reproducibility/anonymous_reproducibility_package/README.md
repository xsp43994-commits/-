# Anonymous reproducibility package

This package is a manuscript-facing extract of the frozen v3.2.14 evidence. It does not rerun training or evaluation.

## Integrity gates

- final_results.jsonl remains in the frozen workspace and contains 21,648 records.
- final_audit_status.json reports passed=true and ppo_mlp_absent=true.
- Map is the independent statistical unit.
- Post-hoc operational scores do not replace safe_weighted_coverage.

## Copernicus reconstruction

Use the official Copernicus DEM GLO-30 registry: https://registry.opendata.aws/copernicus-dem/. Raw assets are omitted unless redistribution is confirmed. Region identifiers and task manifests are retained in the frozen evaluation package.

## Rebuilding manuscript artifacts

Run the scripts in manuscript_build_scripts in numeric workflow order: evidence snapshot, literature register, workbooks, figures, documents, then QA. Training and evaluation commands are intentionally excluded from manuscript rebuilding.

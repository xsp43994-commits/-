# Terminology and abbreviations v2

| Term | Definition | Usage control |
|---|---|---|
| PPO-Pointer | PPO with Pointer policy | Use for full; do not call coverage champion |
| A2C-Pointer | A2C comparator with Pointer policy | Final coverage is close to PPO-Pointer |
| Flat-MLP PPO | traditional_ppo / FlatMLPActorCritic / flat_mlp_24 | Never use ppo_mlp |
| SWC | safe weighted coverage | Confirmatory endpoint; unsafe/violating routes score zero |
| DSM | digital surface model | Copernicus DEM GLO-30; simulation terrain input |
| unseen synthetic maps | 24 held-out procedural maps | Not unseen node-count generalisation |
| zero-shot DSM simulation transfer | Evaluation on 8 DSM maps without DSM-specific training | Not real flight |
| known domain shift | policy observes the shifted condition used for execution | Wind or power coefficient shift |
| hidden mismatch | planning observation/model differs from execution truth | Keep separate from known shift |
| return-aware multi-resource feasibility mask | composite return reserve mechanism | No independent submask claim |
| map-level pairing | task/seed aggregation within each map | n=24 synthetic or n=8 DSM |
| post-hoc 100-point score | multiobjective sensitivity summary | Supplement only; not abstract evidence |

# English figure captions v2

## Main manuscript

**Figure 1. Return-aware PPO–Pointer planning and frozen evaluation workflow.** The diagram describes the implemented sequence from mountain-road task representation to composite return-feasibility screening and the predefined evidence outputs; it is a non-generative schematic and does not add experimental evidence.

**Figure 2. Coverage and priority-stratum evidence.** (a) Map-level safe priority-weighted coverage on 24 unseen synthetic maps and eight DSM maps; points are independent-map aggregates and vertical markers are medians. (b) Full-model minus ablation differences across priority strata; these stratified values are descriptive and do not replace the confirmatory map-level endpoint.

**Figure 3. Safe-return and resource outcomes.** (a) PPO–Pointer minus comparator percentage-point effects for safe completion and depot return under known shifts and hidden model/perception mismatch; intervals are map-level 95% bootstrap intervals. (b) Median energy, range and mission-time utilization among safe routes; the dashed line denotes the budget limit.

**Figure 4. Online planning time and quality–time trade-off.** (a) Empirical cumulative distributions of per-task online planning time over the frozen evaluation jobs. (b) Map-level D1 versus 95th-percentile planning time on synthetic and DSM tasks. Timing values are specific to the frozen software and hardware protocol and are not cross-platform latency guarantees.

**Figure 5. Corrected validation trajectories, training stability and sample efficiency.** (a) Safe weighted coverage on the same fixed 108-task external validation set at 26 checkpoints over 3,000 episodes. Thin curves are five seeds, heavy curves are medians and bands are interquartile ranges. (b) Corrected post-hoc D6 stability and D7 normalized validation-area-under-the-curve scores, with direction-aligned component and budget-sensitivity values. These training dimensions supplement rather than redefine the confirmatory final-route endpoint.

**Figure 6. Cross-map performance and zero-shot DSM simulation transfer.** (a) Map-level D1 estimates and 95% bootstrap intervals on unseen procedural maps and eight Copernicus DSM maps. (b) Terrain, road network, inspection points and representative routes for one fixed DSM task. The route panel is illustrative and non-inferential; DSM evaluation remains simulation, not flight validation.

**Figure 7. Two-layer robustness evaluation.** D1 retention is reported separately for shifts revealed to the planner and hidden model/perception mismatches. The dashed reference denotes retention of 1.0; maps, not tasks or routes, are the inferential units.

**Figure 8. Four component ablations.** Points are full-model minus ablation mean differences in map-level safe weighted coverage; horizontal intervals are 95% map-bootstrap intervals. Asterisked significance, where reported in the Source Data, follows Holm adjustment within the prespecified ablation family.

## Supplementary material

**Figure S1.** Regret performance profile for all frozen algorithms. Each curve is an empirical cumulative distribution over the stated task set; the panel is descriptive and does not change the map-level inferential unit.

**Figure S2.** Coverage–time trade-off on synthetic and DSM tasks using D1 and the protocol-specific 95th-percentile planning time.

**Figure S3.** Traditional-planner oracle-regret intervals versus planning time. Solver status and certification share must be read together with these values.

**Figure S4.** Scenario-stratified mean D1 for the three principal learning models. Scenario cells are descriptive strata, not independent replication units.

**Figure S5.** Robustness and failure-mode rates under known shifts and hidden mismatch. Values are frozen task aggregates; safe and return rates are not flight-certification evidence.

**Figure S6.** Training trajectories for the seven paper-eligible learning variants. Curves summarize five seeds per model over 3,000 episodes.

**Figure S7.** Seven normalized dimensions and the post-hoc 100-point composite. This score is exploratory, weight-dependent and is not the confirmatory endpoint.

**Figure S8.** Joint sensitivity of the PPO–Pointer first-place share to the combined D6+D7 weight and operational floor after correcting both training dimensions. Each cell summarizes the frozen weight grid; it is a post-hoc ranking-sensitivity diagnostic.

**Figure S9.** Representative routes on one fixed synthetic task. The panel is illustrative and not used for statistical inference.

**Figure S10.** Representative routes on one fixed DSM task with elevation and road context. This is a zero-shot simulation-transfer illustration, not field validation.

# Supplementary material

## S1. Frozen protocol identity and audit trail

The supplementary package refers only to protocol v4.2.14. The formal result file contains exactly 21,648 unique rows and the route directories contain 21,648 corresponding route records. The frozen protocol SHA-256 is `c0ac70fb8fac32bd60afe602b24d8534d9329e531ea381d9a73bf4237c9fbc58`; the final result SHA-256 is `4b620c21566c2e33c875f6bea2017b741b02a7d30d70aa50add60a6d06214a2c`. The excluded historical attention-containing prototype is absent from the paper-eligible model set. The formal set contains 35 learning models (seven variants × five seeds) and 105,000 paper-eligible training episodes.

### Table S1. Formal evaluation families

| Family | Result rows | Independent maps | Role |
|---|---:|---:|---|
| Unseen synthetic learning models | 7,560 | 24 | Main and ablation evaluation |
| Unseen synthetic main baselines | 3,888 | 24 | Main traditional comparison |
| Unseen synthetic supplementary baselines | 504 | stratified subset | A*, particle swarm and Pareto dynamic programming |
| Digital-surface-model learning models | 5,040 | 8 | Geographic simulation transfer |
| Digital-surface-model representative baselines | 1,152 | 8 | Traditional comparison |
| Known domain shift | 1,008 | 8 | Planner observes shifted truth |
| Hidden model/perception mismatch | 2,496 | 8 | Planner and execution truth differ |
| **Total** | **21,648** | — | Frozen audited evidence |

## S2. Learning models and ablations

### Table S2. Model definitions

| Model label | Policy architecture | Update | Removed component |
|---|---|---|---|
| PPO–Pointer | Priority encoder + Pointer actor | PPO | None |
| A2C–Pointer | Priority encoder + Pointer actor | A2C | PPO repeated clipped update |
| Flat-MLP PPO | Fixed 24-node slots + depot slot | PPO | Pointer, attention and node encoder |
| No priority bias | Pointer | PPO | Explicit additive priority bias only |
| No domain randomization | Pointer | PPO | Parameter randomization; multi-map training retained |
| No resource shaping | Pointer | PPO | Secondary incremental resource term |
| No return reserve | Pointer | PPO | Policy-visible projected return reserve |

All models used seeds 42–46 and 3,000 episodes per seed. The complete PPO configuration used a 128-dimensional candidate representation, four attention heads, priority-bias coefficient 0.5, learning rate 10^−4, clipping ratio 0.2, discount 0.99, generalized-advantage parameter 0.95, five epochs per update, minibatches of at most 128 transitions, value coefficient 0.5, gradient norm cap 1.0 and entropy coefficient annealed from 0.02 to 0.002. These values are implementation identities, not post-hoc tuned manuscript parameters.

## S3. Observation and feasibility definitions

Each candidate token had 15 features: relative x, y and z position; normalized priority; visited and depot flags; outgoing distance, energy and time fractions; projected return distance, energy and time fractions; and three normalized outgoing-wind components. The vehicle vector had 14 features: position relative to the depot, previous direction, remaining energy, distance and time fractions, ordinary and priority-weighted coverage, and the local three-component wind vector.

The legal set was the conjunction of visit, energy, distance, time and dynamics masks. For an inspection candidate, resource checks included the outgoing segment, service requirement and projected return segment. The no-return-reserve ablation relaxed only the policy-visible reserve. The simulator retained the full execution check and marked a newly exposed but actually infeasible proposal as stranded without executing the segment. The experiment therefore measures a planning-safety mechanism in simulation and not physical hazard exposure.

## S4. Map/task factorial design

Procedural training, validation and unseen-test splits contained 72/12/24 maps and 648/108/216 tasks. Each map supported factorial variation over 16, 20 and 24 nodes; moderate, hard and extreme difficulty; energy, distance, time and mixed constraints; and multiple priority layouts. Because all three node counts entered training, size-stratified results are within the trained range.

Eight Copernicus DEM GLO-30 digital-surface-model assets defined the geographic simulation set. Each map used two deterministic road/launch contexts and 144 total tasks. The digital surface model is a terrain input to the simulator. No flight, sensing or inspection-image data were collected.

## S5. Map-level primary results

### Table S3. Safe weighted coverage by map

| Family | Algorithm | n maps | Mean | Median | Q1 | Q3 |
|---|---|---:|---:|---:|---:|---:|
| Synthetic | PPO–Pointer | 24 | 0.4858 | 0.4829 | 0.4683 | 0.5042 |
| Synthetic | A2C–Pointer | 24 | 0.4850 | 0.4846 | 0.4700 | 0.4991 |
| Synthetic | Flat-MLP PPO | 24 | 0.2686 | 0.2713 | 0.2536 | 0.2835 |
| Synthetic | ACO | 24 | 0.5525 | 0.5543 | 0.5370 | 0.5663 |
| Synthetic | SA | 24 | 0.5436 | 0.5458 | 0.5295 | 0.5601 |
| Synthetic | MILP | 24 | 0.5711 | 0.5726 | 0.5560 | 0.5852 |
| DSM | PPO–Pointer | 8 | 0.5024 | 0.5006 | 0.4960 | 0.5086 |
| DSM | A2C–Pointer | 8 | 0.5025 | 0.4975 | 0.4920 | 0.5100 |
| DSM | Flat-MLP PPO | 8 | 0.2658 | 0.2706 | 0.2561 | 0.2730 |
| DSM | ACO | 8 | 0.5599 | 0.5603 | 0.5482 | 0.5649 |
| DSM | MILP | 8 | 0.5829 | 0.5850 | 0.5783 | 0.5888 |

### Table S4. Selected paired comparisons against PPO–Pointer

| Family | Comparator | Mean difference | 95% bootstrap interval | Holm p | Interpretation |
|---|---|---:|---|---:|---|
| Synthetic main | Flat-MLP PPO | 0.2172 | 0.1958 to 0.2393 | 9.54 × 10^−7 | PPO–Pointer higher |
| Synthetic main | A2C–Pointer | 0.0008 | −0.0057 to 0.0069 | 0.845 | No detected difference; not equivalence |
| Synthetic main | ACO | −0.0667 | −0.0814 to −0.0534 | 9.54 × 10^−7 | ACO higher |
| Synthetic main | SA | −0.0578 | −0.0729 to −0.0432 | 9.54 × 10^−7 | SA higher |
| Synthetic main | MILP | −0.0853 | −0.0999 to −0.0722 | 9.54 × 10^−7 | MILP higher |
| DSM main | Flat-MLP PPO | 0.2366 | 0.2123 to 0.2618 | 0.0469 | PPO–Pointer higher |
| DSM main | A2C–Pointer | −0.00001 | −0.0090 to 0.0090 | 0.742 | No detected difference; not equivalence |
| DSM main | ACO | −0.0575 | −0.0706 to −0.0456 | 0.0469 | ACO higher |
| DSM main | MILP | −0.0805 | −0.0956 to −0.0658 | 0.0469 | MILP higher |

## S6. Ablation tests

### Table S5. Complete-model minus ablation differences

| Family | Ablation | Mean difference | 95% bootstrap interval | Holm p |
|---|---|---:|---|---:|
| Synthetic | No priority bias | 0.0010 | −0.0066 to 0.0087 | 1.000 |
| Synthetic | No domain randomization | 0.0012 | −0.0071 to 0.0095 | 1.000 |
| Synthetic | No resource shaping | 0.0018 | −0.0050 to 0.0083 | 0.506 |
| Synthetic | No return reserve | 0.3725 | 0.3434 to 0.4018 | 4.77 × 10^−7 |
| DSM | No priority bias | −0.0006 | −0.0097 to 0.0085 | 1.000 |
| DSM | No domain randomization | 0.0009 | −0.0088 to 0.0121 | 1.000 |
| DSM | No resource shaping | 0.0001 | −0.0083 to 0.0082 | 1.000 |
| DSM | No return reserve | 0.3904 | 0.3563 to 0.4245 | 0.0313 |
| Hidden mismatch | No return reserve | 0.3174 | 0.2460 to 0.4042 | 0.0391 |

The first three ablations show limited average evidence for the primary endpoint. Their non-significant differences are not equivalence tests. The last rows apply only to the composite return-aware mechanism.

## S7. Training-aware post-hoc analysis

The corrected analysis used only the formal fixed 108-task external-validation traces. D6 combined 60% cross-seed consistency and 40% within-seed temporal consistency over the final 20% of the 3,000-episode budget. The complete PPO–Pointer and A2C–Pointer scores were 0.9978 and 0.9971; the paired-seed mean difference was 0.00069 and its 10,000-replicate interval crossed zero (−0.00060 to 0.00173). D7 was normalized validation safe-weighted-coverage area under the curve over the common interaction window 80–17,702. The corresponding scores were 0.4872 and 0.4781; the difference was 0.00913 with a bootstrap interval of 0.00491–0.01331. Both paired five-seed Holm-adjusted p values were 0.125. Tail-window (10%, 20%, 30%) and interaction-budget (50%, 75%, 100%) checks were reported as sensitivity analyses. These post-hoc dimensions therefore support near-equal stability and a modest area-under-the-curve difference, not separately confirmed superiority at α = 0.05.

After correcting D6 and D7 while retaining the frozen default weights and 0.60 operational floor, the 100-point arithmetic summary assigned 58.975 to PPO–Pointer, 58.566 to A2C–Pointer and 54.874 to Flat-MLP PPO. A map-outer hierarchical bootstrap estimated a PPO-minus-A2C difference of 0.393 points (95% interval −0.075 to 1.099; 10,000 resamples; probability of a positive difference 0.912). The deterministic summary difference was 0.409 points. The score was designed after results were available, combines heterogeneous dimensions and changes under weighting and normalization choices. It is a diagnostic summary only and does not replace safe weighted coverage.

## S8. Traditional-planner status and computation

Mean synthetic planning times were 1.168 s for PPO–Pointer, 3.832 s for simulated annealing, 20.807 s for mixed-integer programming and 76.701 s for ant-colony optimization. Times refer to the frozen hardware, software and implementation. Mixed-integer records retain termination status, incumbent objective, dual bound and gap. A time-limited feasible incumbent is not described as proven optimal.

## S9. Supplementary figure register

- Figure S1: performance profile across the full algorithm set.
- Figure S2: quality–online-time Pareto display.
- Figure S3: oracle regret and computational cost.
- Figure S4: complete five-seed training curves.
- Figure S5: post-hoc score and weight sensitivity.
- Figure S6: ablation effects by map.
- Figure S7: robustness failures and termination patterns.
- Figure S8: representative route atlas across digital-surface-model maps.
- Figure V1: three-dimensional route illustration; non-inferential.
- Figure V2: outcome-flow illustration; non-inferential.

## S10. Reproducibility and redistribution notes

The reproducibility package provides the frozen protocol, analysis protocol, task/evaluation matrix, final results, statistics, source data, environment description, model metadata, file hashes and rerun commands. Openly redistributable references are included only where licence status permits. Publisher or school-access papers are supplied as DOI/publisher links rather than copied PDFs. Copernicus raw assets are handled separately: the package records the product, region identifiers, attribution and reconstruction procedure and does not assume that local raw files can be redistributed without checking the current access terms.

## S11. Author verification items

[AUTHOR INPUT REQUIRED: permanent repository DOI or URL.]

[AUTHOR INPUT REQUIRED: confirm Copernicus region identifiers that may be published.]

[AUTHOR INPUT REQUIRED: verify the generative-AI disclosure and all author-side declarations.]

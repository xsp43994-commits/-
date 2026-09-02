# Full-text reading note 10

## Identity

- PDF: `10.pdf`
- DOI: `10.1016/j.engappai.2024.109339`
- Title: *A reliable traversability learning method based on human-demonstrated risk cost mapping for mobile robots over uneven terrain*
- Pages: 10
- SHA-256: `b466b4ce7823983b091cd5ff903848f261091761a2f3bdabe1983451583bcf0d`
- Article type: Research paper
- Comparability: medium-high for terrain-risk mapping, fixed-dimensional feature design, safe-detour trade-offs, cross-scenario tests, physical platform reporting and comparator cost/latency; low for IRL and human demonstrations as method analogues.

## Functional architecture

- The Introduction frames manually specified terrain rewards as the bottleneck, motivates learning human risk preferences and maps three contributions to feature mapping, MCE IRL and cost-aware Hybrid A*.
- Related work is divided into demonstrations, terrain feature extraction and uneven-terrain navigation.
- Notation and abbreviations tables precede the formal two-stage problem: learn a reward/cost map, then plan a least-cost trajectory.
- Method proceeds through geometry-based feature extraction, full MCE IRL derivation, cost-map construction and a modified Hybrid A* cost.
- Experiments first isolate the new cost function, then report hardware, metrics, three named scenarios, demonstration-count sensitivity, convergence and comparisons with two IRL baselines.
- Conclusion summarises the pipeline and states material/sensor and out-of-distribution limitations.

## Reporting and evidence style

- The paper makes a central trade-off explicit: safer paths have lower cumulative slope/roughness but longer length and greater planning time.
- Generalisation scenarios are defined separately from the training scenario and do not contain new human demonstrations.
- The number of demonstrations is examined before fixing 20 demonstrations for the main experiments.
- Five metrics cover success, planning time, length, terrain risk and demonstration similarity; applicability of the similarity metric is explicitly restricted to the training scenario.
- Baseline comparisons report mean and best outcomes, including metrics on which the proposed method is worse.
- Planning time is interpreted against the robot’s operational requirement rather than declared `real time` without a threshold.
- Hardware and software stacks are listed, while data are only available on request and an experimental video is supplementary.

## Journal-level candidate conventions

- Define training and generalisation scenarios and identify which learning steps are disabled in held-out scenarios.
- Explain metric applicability and do not fill in meaningless values for scenarios where a metric has no valid reference.
- Express a safety detour as an explicit risk–distance–time trade-off.
- Test sensitivity to the amount of supervision before fixing the main protocol.
- Interpret online time relative to an operational deadline.
- Preserve comparator wins in the same table as proposed-method wins.

These remain candidate conventions until cross-paper frequency is assessed.

## Non-copy boundary

Do not transfer MCE IRL, human demonstrations, terrain-feature bins, reward inversion, Hybrid A* modifications, scenario designs, demonstration counts, hardware, metrics, numerical results or references. In the current work, learning is PPO-based and no reward or cost map is inferred from human behaviour. This source can inform the language of safety–detour trade-offs, but not justify a claim that return-aware masking learns human risk preferences.

## Full-text status

Complete: all 10 pages, including notation, MCE IRL and cost-map equations, sensitivity analysis, outdoor robot protocol, all result tables, metric limitations, conclusions, supplementary link, declarations and references, were reviewed for scientific function and writing structure.

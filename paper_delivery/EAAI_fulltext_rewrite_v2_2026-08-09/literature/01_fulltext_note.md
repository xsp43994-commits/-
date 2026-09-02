# Full-text reading note 01

## Identity

- PDF: `1.pdf`
- DOI: `10.1016/j.engappai.2026.115219`
- Pages: 16
- SHA-256: `c41438e408773c9ede8c1eec90a185da6705bc7557c6860c8e567afe22026fbd`
- Article type: Research paper
- Comparability: high for UAV inspection, endurance-aware action pruning, attention-based routing, exact/heuristic/DRL comparison, and engineering trade-off narration; low for periodic 28-day scheduling details that are absent from the present study.

## Functional architecture

- Introduction narrows from heterogeneous power-inspection frequency requirements to the fixed-nest operational mode, then identifies combinatorial scale, topology, and myopic long-horizon decision making as three challenges.
- Contributions are stated as three mechanism-to-problem mappings: dynamic state pruning, topology-aware attention, and a composite reward/flight-efficiency indicator.
- Literature review is mechanism-organised and ends with an explicit synthesis of why decoupled, MARL, and HRL approaches do not enforce the paper's hard periodic feasibility requirements.
- Problem formulation precedes the learning method. The paper first defines graph sets, operational assumptions, and a MILP, then maps the problem to an MDP.
- Method presentation follows state/action model -> feasibility pruning -> state transitions -> attention actor/critic -> reward -> training algorithm.
- Results follow parameter disclosure -> training behaviour -> exact-solver small-scale verification -> large-scale multi-metric comparison -> ablation -> reward sensitivity.
- The conclusion restates the mechanism and quantitative outcomes, but it also makes broader feasibility/generalisation statements that should not be imitated without corresponding evidence.

## Reporting and evidence style

- Method granularity is high: operational assumptions, equations, notation table, pseudocode, architecture, hyperparameters, hardware, and appendices are provided.
- Results introduce the question and metric before the figure/table, then give quantitative comparisons and a short engineering interpretation.
- Multi-objective evidence is narrated as a quality-efficiency-computation trade-off rather than a single-metric champion claim.
- Solver time limits and incumbent status are reported, although some later language is stronger than the evidence warrants.
- Stochastic DRL evaluation is described using ten independent runs; distributions, averages, and sensitivity tests are placed close to the claims they support.
- Figures progress from scenario and formulation to architecture, training, metric comparisons, distributions, Pareto-style trade-offs, ablation, and supplementary route/coverage visualisation.
- Data availability, funding, CRediT, competing-interest, and appendices follow the conclusion.

## Journal-level candidate conventions

- Separate problem formulation and method sections.
- Explicit contribution list near the end of the Introduction.
- Detailed MDP and constraint definitions before network architecture.
- Parameter/hardware disclosure at the start of the experiment section.
- Results ordered from learning behaviour and controlled verification to scale comparison, ablation, and sensitivity.
- Quantitative evidence and figure/table references remain close to the associated claim.
- Engineering value is framed through competing performance dimensions rather than only reward or path length.

These remain candidate conventions until cross-paper frequency is assessed.

## Non-copy boundary

Do not transfer the paper's periodic frequencies, 28-day horizon, fixed-nest sortie assumptions, linear energy approximation, MILP, DSP/TAA equations, reward terms, FE definition, parameters, baselines, results, limitations, or reference combination. Do not reuse its distinctive phrase `myopic optimization behavior` as the organising label for the present paper. The present return-aware mask and frozen multi-resource evidence must be described from the project's own implementation and data.

## Full-text status

Complete: all 16 pages, including appendices, figures/tables as extracted in context, declarations, and reference list, were reviewed for scientific function and writing structure.

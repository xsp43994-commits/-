# Full-text reading note 06

## Identity

- PDF: `6.pdf`
- DOI: `10.1016/j.engappai.2025.110392`
- Title: *Reinforcement learning based multi-perspective motion planning of manned electric vertical take-off and landing vehicle in urban environment with wind fields*
- Pages: 17
- SHA-256: `0f389da7d22ae41352d6228e006dd9853ff7bd03510d55fd83f34e37362fd5b5`
- Article type: Research paper
- Comparability: high for PPO motion planning, multi-resource objectives, wind-field simulation, baselines, reward ablation and engineering trade-offs; lower for passenger/noise models and continuous eVTOL control.

## Functional architecture

- The Introduction names three stakeholder perspectives—aircraft, passenger and urban environment—before defining the multi-objective gap and mapping five contributions to those perspectives.
- Related work uses an algorithm table and an objective-coverage table to show which prior studies cover energy, time, safety, comfort and noise.
- Methodology begins with the POMDP and decomposes action, observation and reward. Domain models for energy and noise are reported before the RL algorithm and CFD source.
- The experiment section states the training strategies, environment, named wind cases and computational cost before results.
- Results move from PPO implementation comparison to energy-reward comparisons across algorithms, passenger metrics, noise, joint-objective training and component-removal ablation.
- Discussion is separate from Results and contains five limitations: training complexity, exploration restrictions from strict penalties, single-wind-field training, model-free sample cost and real-world deployment constraints.

## Reporting and evidence style

- Multi-objective scope is made auditable through comparison tables rather than broad novelty claims alone.
- Modelling simplifications are disclosed where introduced, including omitted takeoff/landing phases, simplified acoustic propagation and empirical parameter scaling.
- The paper reports computational throughput and training time separately from the anticipated inference cost.
- Cross-algorithm comparisons retain cases in which the proposed reward is not best on individual metrics.
- Ablation names each removed reward component and reports task completion, failure modes and objective metrics together; this prevents a component from appearing beneficial only because unsafe failures are ignored.
- The discussion acknowledges that scalarisation requires careful weight tuning and may slow learning or restrict exploration.
- Some percentage language is potentially confusing when reductions exceed 100%; this is a caution, not a convention to imitate.

## Journal-level candidate conventions

- Use objective-coverage and algorithm-coverage tables when a multi-objective gap would otherwise be difficult to verify.
- Disclose the source, adaptation and limitations of external physical models used to calculate rewards or metrics.
- Report training resource cost and deployment inference cost as separate quantities.
- Evaluate ablations using both target metrics and termination/failure counts.
- Discuss the operational cost of additional objectives and constraint penalties.
- Avoid declaring a global winner when individual algorithms or metrics show exceptions.

These remain candidate conventions until cross-paper frequency is assessed.

## Non-copy boundary

Do not transfer the eVTOL scenario, OSM Atlanta/CFD cases, passenger and acoustic objectives, power/noise equations, modified PPO layers, reward shaping, weights, wind cases, objective tables, numerical results or references. The current project’s post-hoc composite score cannot be elevated into a training objective or primary outcome by analogy. Its four ablations test different mechanisms and must not be described as independent return-submask evidence.

## Full-text status

Complete: all 17 pages, including comparison tables, physical-model assumptions, full reward formulation, CFD data source, computational analysis, all result and ablation tables, discussion, limitations, declarations and references, were reviewed for scientific function and writing structure.

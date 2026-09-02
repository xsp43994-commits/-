# Full-text reading note 04

## Identity

- PDF: `4.pdf`
- DOI: `10.1016/j.engappai.2025.112090`
- Title: *Evaluating reinforcement learning-based neural controllers for quadcopter navigation in windy conditions*
- Pages: 18
- SHA-256: `f61ac26e7ffe6c07b7fb00d7a7a510a5727c6fa46b5173878fc49703ab8a3ba4`
- Article type: Research paper
- Comparability: high for PPO evaluation, domain randomisation, waypoint constraints, training-condition comparison, robustness reporting, simulation boundaries, and negative findings; lower for low-level continuous control and SHAP-specific interpretation.

## Functional architecture

- The Introduction frames wind-disturbed waypoint navigation as an unresolved engineering problem, then narrows the gap to training design and state representation rather than proposing a wholly new controller.
- Related work is organised around three functional themes: UAV DRL control, wind-robust control and explainability. The contribution list follows the evaluated design factors.
- Preliminaries introduce the MDP, PPO and SHAP before the task-specific formulation.
- The method sequence is problem and reward definition, alternative training configurations, alternative state representations, simulator construction, baselines, protocol and metrics.
- Results progress from classical-baseline comparison to training-condition variants, state-representation variants, then policy explanation. Aggregated plots are paired with trajectory-specific appendix plots.
- Conclusions first summarise supported design findings, then discuss interpretability and end with concrete simulation-to-real, simulator-speed, visual-navigation and degraded-sensing limitations.

## Reporting and evidence style

- Training and evaluation conditions are separated. Four named trajectory geometries, no-wind and wind evaluations, three training seeds and repeated rollouts are specified before results.
- Metrics are defined operationally: success requires satisfying waypoint-distance constraints, while steps quantify efficiency. Where success ties, fewer steps determine the preferred setup.
- Tables retain configurations that add complexity but do not improve performance. Frame stacking and multi-goal training are reported as marginal or detrimental rather than omitted.
- The narrative repeatedly exposes trade-offs: wind randomisation improves disturbed-condition robustness but can reduce no-wind efficiency; stricter constraints improve both precision and transfer without extra interactions in these experiments.
- The paper distinguishes aggregate conclusions from trajectory-specific behaviour and places disaggregated evidence in an appendix.
- Explainability results are treated as feature-attribution observations and tentative behavioural interpretations, not as causal proof of the controller mechanism.
- The future-work paragraph explicitly states that physical outdoor testing remains unperformed and identifies slow simulation as a training limitation.

## Journal-level candidate conventions

- A systematic evaluation study may make training design and state representation—not only algorithm novelty—the contribution, provided the evaluation space is explicit and engineering-relevant.
- Define success criteria and tie-breaking rules before ranking configurations.
- Preserve unsuccessful or neutral design variants to support a balanced conclusion.
- Present robustness and nominal-efficiency results together so improvements are not detached from their costs.
- Keep aggregate main-text evidence and disaggregated supplementary evidence linked.
- State the simulator and validation boundary directly when discussing real-world deployment.

These remain candidate conventions until cross-paper frequency is assessed.

## Non-copy boundary

Do not transfer the AirSim wind model, four trajectory shapes, control frequency, PPO hyperparameters, waypoint tolerances, SHAP workflow, state variables, domain-randomisation ranges, numerical results, physical-system suggestions or reference combination. Do not reuse claims that explicit wind observation or stricter waypoint constraints improve generalisation; the current project has different task dynamics and evidence. The present DSM experiments must remain `zero-shot simulation transfer`, not sim-to-real or physical validation.

## Full-text status

Complete: all 18 pages, including the full formulation, simulator and evaluation protocol, tables, SHAP analysis, trajectory-specific appendix, declarations, data-availability statement and references, were reviewed for scientific function and writing structure.

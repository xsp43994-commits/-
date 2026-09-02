# Full-text reading note 09

## Identity

- PDF: `9.pdf`
- DOI: `10.1016/j.engappai.2024.108506`
- Title: *Reinforcement learning based robot navigation using illegal actions for autonomous docking of surface vehicles in unknown environments*
- Pages: 20
- SHA-256: `ea355c02d59c0eb6a7004b176568eaa761077ab0bc6572b97f844775ab6ff220`
- Article type: Research paper
- Comparability: high for A2C/PPO navigation, action masking, safety-aware planning, sample efficiency, unknown-condition testing, conventional baselines, latency and tiered validation; the maritime and physical-validation claims are task-specific.

## Functional architecture

- The Introduction develops the engineering context and safety-critical docking gap, then lists a perception-to-action RL framework, an action-mask mechanism and multi-tier validation as contributions.
- Related work distinguishes predefined-path tracking, high-level decision making, simulation-only methods and real-world deployment losses.
- Method begins with A2C/PPO fundamentals, then defines navigation/perception states, high-level actions, sparse and dense rewards and a training-only illegal-action mask.
- Results explicitly announce three validation tiers: simulation, a controlled physical tank and a real harbour with a different vessel.
- Simulation results move from mask-versus-total-freedom training, through episode behaviour, environmental robustness and unseen structures, to A* quality–time comparison.
- Physical results disclose localization changes, thresholds, initial-pose ranges, episode counts, environmental conditions and failed cases before the final conclusion.

## Reporting and evidence style

- The action-mask contribution is isolated with 10 training instances per condition, mean and standard deviation, three mask-duration variants and event-frequency analysis.
- Training-only masking is explicitly distinguished from unrestricted testing, preventing confusion between exploration guidance and operational action feasibility.
- Success is not used alone: position/heading errors, collision risk, duration, processing time and action/velocity traces explain behaviour.
- Robustness tests use named wind/wave scales, seven severity scenarios and 100 episodes per scenario; unseen structure tests cover five layouts.
- The A* comparison reports both its advantage (shorter path/duration) and disadvantage (higher collision risk and failure under disturbances), plus resolution-dependent timing.
- Controlled physical tests report 27/28 successes and explain the single failure; real-harbour tests are separately presented without inflating their ten successful trials into a broader reliability rate.
- Mechanistic explanations of learned intermediate-zone behaviour are interpretive and should not be copied as causal proof.

## Journal-level candidate conventions

- Isolate a masking or feasibility mechanism with direct variants and event-level diagnostics.
- State whether an action mask applies during training, inference or both.
- Pair success/coverage with resource, safety and latency measures.
- Organise validation tiers from inexpensive simulation to increasingly realistic settings, keeping each tier’s claims separate.
- Report comparator advantages and sensitivity to its configuration.
- Include failure episodes and threshold changes in physical validation.

These remain candidate conventions until cross-paper frequency is assessed.

## Non-copy boundary

Do not transfer the docking task, perception sectors, eleven high-level actions, illegal-action temporal windows, shaped reward, dynamic boundary, curriculum, wind/wave scales, dock layouts, A* settings, ASV hardware, physical results or references. The current project’s return-aware mask is a feasibility mechanism with a different semantics and must not be called an `illegal-action` method. Its submasks were not independently ablated, so this paper cannot justify assigning separate causal effects to them. No physical validation vocabulary may be imported into the DSM simulation results.

## Full-text status

Complete: all 20 pages, including full A2C/PPO and mask equations, state/action/reward design, hyperparameters, simulation repetitions, behaviour traces, robustness and conventional comparisons, controlled and harbour tests, failure reporting, declarations and references, were reviewed for scientific function and writing structure.

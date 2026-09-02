# Full-text reading note 12

## Identity

- Title: *Multi-UAV trajectory optimizer: A sustainable system for wireless data harvesting with deep reinforcement learning*
- Journal: *Engineering Applications of Artificial Intelligence*
- DOI: `10.1016/j.engappai.2023.105891`
- Publication year: 2023
- PDF length: 11 pages
- SHA-256: `34abae9c06434fad1da6f38523ce283db8ee9933fce88ff316b4d09a4430b081`
- Article type: EAAI research article.
- Comparability: medium–high for multi-UAV resource constraints, charging/safety control, parameter randomization, transfer evaluation, Monte Carlo reporting, scaling experiments, and competing-objective trade-offs; low for the wireless-harvesting objective, Dec-POMDP formulation, MARL, and DDQN implementation.

## Functional architecture

1. The Introduction motivates persistent wireless-data harvesting, reviews UAV routing and reinforcement-learning approaches, identifies limits in centralized control and resource awareness, states the proposed multi-agent system, and previews the paper structure.
2. The system model separates the environment, UAV motion and battery/charging processes, and the wireless communication model before presenting the learning algorithm.
3. The decision problem is formalized as a Dec-POMDP with explicit state, observation, action, reward, and safety-controller definitions.
4. The method section explains decentralized DDQN training, transfer learning, and computational complexity.
5. The experiments first specify simulation settings and baselines, then report transfer effects, trajectory visualizations, parameter sensitivity, objective trade-offs, and larger-map tests.
6. The Conclusion summarizes empirical findings and lists prospective extensions rather than claiming deployment readiness.

## Reporting and evidence style

- Resource and safety constraints are defined before reward construction. Boundary and collision violations are handled by a separate safety controller that cancels the unsafe action and keeps the UAV hovering; this enforcement mechanism is not presented as an emergent learned guarantee.
- Training uses a two-stage transfer procedure to balance data collection and charging behavior. The paper contrasts transfer learning, a non-transfer DDQN variant, and a DQN baseline rather than treating one training curve as sufficient evidence.
- Scenario parameters are randomized over explicit ranges and summarized in a parameter table, which makes the simulation distribution visible to the reader.
- Training results include 99% confidence intervals. Main evaluations use 1000 Monte Carlo repetitions, while parameter-effect studies use 300 repetitions per setting.
- Trajectory figures are used to explain behavior, whereas quantitative plots and tables support performance and sensitivity claims.
- Increasing the number of UAVs improves one objective but may degrade the communication-fulfilment objective; the paper reports this conflict instead of declaring a universal winner.
- Larger 100 × 100 environments involve changed settings and additional training/evaluation. They therefore constitute new experiments, not evidence of automatic out-of-distribution scale generalization.
- The parameter “sweet spot” is inferred from post-hoc sensitivity results. That narrative is useful for engineering interpretation but should not be imitated as a confirmatory primary claim without a predeclared selection rule.
- The ablation evidence is limited: comparisons with/without transfer and against DQN do not independently isolate every reward or safety component.
- The data-availability statement identifies the data as confidential, showing that EAAI articles may state a concrete access constraint rather than omit data availability altogether.

## Journal-level candidate conventions

- Define physical/resource states and hard feasibility enforcement before introducing the learned objective.
- Distinguish rule-based safety intervention from the learned policy and calibrate the safety claim accordingly.
- Report scenario-generation ranges and important hyperparameters in compact tables.
- Use repeated Monte Carlo evaluation and visible uncertainty for stochastic learning results.
- Present conflicting engineering objectives and operating trade-offs instead of relying on a single winner label.
- Treat enlarged or altered environments as separate experiments unless the training and test distributions support a genuine generalization claim.
- Keep post-hoc parameter recommendations subordinate to predeclared primary evidence.

These are candidate conventions only. Their final strength is determined by the cross-paper frequency and comparability audit across all 12 exemplars.

## Non-copy boundary

- Do not transfer the wireless-channel model, Dec-POMDP definition, DDQN/MARL architecture, charging thresholds, reward equations, grid sizes, scenario ranges, numerical “sweet spot,” or reference combination.
- The paper’s boundary/collision controller is not evidence for the present project’s return-aware feasibility mask; the mechanisms and evaluated claims are different.
- Do not use its enlarged-map experiment to support training-range-external scale generalization in the present study.
- Do not elevate the present project’s post-hoc 100-point composite score into a primary or abstract-level claim by analogy with its parameter studies.
- Learn only the scientific functions: constraint-first reporting, explicit intervention boundaries, repeated stochastic evaluation, trade-off disclosure, and cautious interpretation.

## Full-text status

- Status: complete.
- Coverage: all 11 PDF pages, including equations, figures, tables, simulation settings, parameter studies, conclusion, declarations, and references, were reviewed from the acquired full text.
- No judgment in this note is based solely on the abstract or metadata.

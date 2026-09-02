# Full-text reading note 05

## Identity

- PDF: `5.pdf`
- DOI: `10.1016/j.engappai.2025.111219`
- Title: *Human-in-the-loop reinforcement learning for dynamic soaring: A trajectory planning and control integrated system*
- Pages: 19
- SHA-256: `18a42d336cce3ea12d0e5b1515252425bc11692aedfdb9ed65d72d3e18921ab5`
- Article type: Research paper
- Comparability: medium for integrated planning/control, expert-informed reward design, model fidelity, disturbance tests and honest deployment limits; low for dynamic-soaring physics, SAC and manual HITL iteration.

## Functional architecture

- The Introduction is subdivided into motivation, challenge, and novelty/contributions. It contrasts optimal-control simplifications with the need to include dynamics, actuators and tracking error.
- Related work is grouped by optimal control, reinforcement learning and human-in-the-loop methods; the gap is the absence of an integrated high-fidelity planning-and-control treatment for the target manoeuvre.
- A conventional optimal-control formulation is presented first and later used as expert knowledge and a comparison reference.
- Environment construction is a standalone section that exposes aerodynamic, controller and wind-model fidelity as well as remaining approximations.
- The proposed method is described as an iterative human–DRL–environment workflow, followed by the stochastic-policy basis, SAC details, phase-specific reward, evaluation criteria and network parameters.
- Results progress from the optimal-control reference, HITL versus non-HITL comparison, the trained controller’s trajectory and state traces, changed wind profiles, random-wind and initialization-bias tests, then a dedicated limitations/discussion section.

## Reporting and evidence style

- A nomenclature table precedes dense physics and control equations.
- The paper separates the reference optimiser’s idealised inputs from the learned system’s actuator/controller execution, making the comparison boundary visible.
- Figures combine three-dimensional trajectories, orthogonal projections, time histories, actions and energy states so that the claimed completion mechanism can be checked from several views.
- Robustness is not reported as a single aggregate score: random wind, position bias and attitude bias are tested and their asymmetric failure sensitivity is retained.
- Numerical differences adverse to the method are discussed, including longer cycle time, larger trajectory footprint, positional sensitivity and training cost.
- The limitations section states that mathematical convergence is unproved, transferability across arbitrary wind fields is limited, only one cycle is studied, and no real-world flight is performed.
- References are broad but highly problem-specific; their combination is unsuitable as a default citation set for the present work.

## Journal-level candidate conventions

- Introduce a conventional or idealised reference method before the learned method when it supplies domain knowledge or establishes a meaningful ceiling.
- Describe simulator fidelity together with its known approximations rather than using `high fidelity` as an unqualified label.
- Use state/action traces and constraints to support a trajectory claim, not only a rendered route.
- Decompose robustness by disturbance source and retain direction-dependent weaknesses.
- Give limitations their own subsection when deployment implications are substantial.
- Distinguish feasibility evidence from completed physical validation.

These remain candidate conventions until cross-paper frequency is assessed.

## Non-copy boundary

Do not transfer the dynamic-soaring/Rayleigh-cycle formulation, GPOPS-II/SNOPT setup, SAC equations, manual trajectory screening, four-phase reward coefficients, aerodynamic model, wind thresholds, robustness perturbations, energy findings, network sizes or reference combination. The current project has no human-in-the-loop intervention and no lower-level flight controller, so this architecture and its deployment language must not be imitated. In particular, simulation robustness must not be converted into onboard-deployment readiness.

## Full-text status

Complete: all 19 pages, including nomenclature, optimal-control baseline, environment assumptions, HITL/SAC formulation, reward coefficients, all trajectory/state figures, robustness tests, limitations, declarations and references, were reviewed for scientific function and writing structure.

# Full-text reading note 08

## Identity

- PDF: `8.pdf`
- DOI: `10.1016/j.engappai.2024.108926`
- Title: *Layered learning in a quadrotor drone: Simultaneous controlling and path planning using optimal fuzzy fractional order proportional integral derivative and proximal policy optimization*
- Pages: 19
- SHA-256: `d713a4adf45bee91082ea5b07df47f68d44d33a94342f11b0393b5d020149721`
- Article type: Research paper
- Comparability: medium for PPO path tracking, layered control, disturbance tests and hardware reporting; low for FOPID/fuzzy/PSO/GA design and the experimental claims’ direct relevance to fixed-point mission planning.

## Functional architecture

- The Introduction surveys quadrotor control, optimisation and DRL, then previews a layered architecture joining attitude control and PPO-based path tracking.
- A conceptual layered-learning section comes before the dynamics and assigns motor, sensing, attitude, navigation and high-level tasks to separate processing layers.
- The problem section is dominated by the nonlinear quadrotor model, PID/fuzzy/FOPID controllers, discretisation, GA/PSO optimisation, Taguchi parameter selection and time-domain objectives.
- PPO path tracking is introduced only after the lower-level controller design, with state/action, network, reward, early termination and manoeuvre constraints.
- Simulation results cover controller step responses and PPO training/network variants; experimental results cover the test stand, sensing/filtering, attitude tests, disturbances and a qualitative loop-flight demonstration.
- The Conclusion uses explicit `Findings`, `Research Limitations` and `Recommendations for Future Research` labels.

## Reporting and evidence style

- Controller comparisons use rise time, settling time, overshoot and steady-state error rather than a single summary measure.
- Hardware components, sensors and actuator specifications are tabulated, and simulation outputs are separated from test-stand outputs.
- PPO details include state/action definitions, network sizes, weighted rewards, early termination and the disturbance range.
- Network-depth experiments report episode count, reward and initial value within a fixed training-time budget, although statistical replication and uncertainty are not presented.
- The flight evidence is largely qualitative and should not be treated as a model for quantitative validation standards.
- The paper explicitly acknowledges simulation dependence, laboratory/environment mismatch and local-optimum/convergence limits of the metaheuristics.

## Journal-level candidate conventions

- In a layered system, present lower-level control validation before attributing high-level planning performance.
- Use multiple physically interpretable control-response metrics rather than a generic reward alone.
- Tabulate experimental hardware and sensors when physical validation is claimed.
- Separate test-stand evidence from unconstrained flight evidence.
- Label findings, limitations and future work clearly when the method spans several subsystems.

These remain candidate conventions until cross-paper frequency is assessed. This paper’s qualitative claims and limited uncertainty reporting must not be promoted to S-High merely because it is an EAAI article.

## Non-copy boundary

Do not transfer the layered autopilot architecture, quadrotor dynamics, fuzzy rules, FOPID/GA/PSO/Taguchi setup, PPO network, manoeuvre reward, early-termination rules, sensor filtering, test-bench results, numerical controller metrics or references. The current project is an online combinatorial mission planner, not a low-level attitude controller; no flight or test-stand validation may be implied.

## Full-text status

Complete: all 19 pages, including the layered architecture, dynamics and controller derivations, optimisation settings, PPO formulation, simulation tables, test-bench hardware, disturbance and flight sections, limitations, declarations and references, were reviewed for scientific function and writing structure.

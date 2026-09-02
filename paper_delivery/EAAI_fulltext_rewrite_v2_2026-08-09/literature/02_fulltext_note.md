# Full-text reading note 02

## Identity

- PDF: `2.pdf`
- DOI: `10.1016/j.engappai.2026.113779`
- Pages: 17
- SHA-256: `2f612ac4dbd9505bc137375e9792031fec60e21839548e55200266de5a2e1d0b`
- Article type: Research paper
- Comparability: medium for PPO engineering navigation, attention, robustness, ablation, unseen scenarios, computational cost, and physical-validation boundary; low for continuous local collision avoidance and sensor-control details.

## Functional architecture

- Introduction moves from the engineering safety need to a taxonomy of global versus local navigation and then three technical families; a compact comparison table exposes limitations before the contribution list.
- The problem is formulated as a POMDP and linked to executable low-level control before the proposed model is introduced.
- Method presentation follows overall system -> observation/action/reward -> replay significance -> local attention -> PPO/asynchronous learning.
- Experiments disclose software, hardware, architecture, hyperparameters, training curriculum, and noise injection before evaluation.
- Evaluation expands in layers: component configurations, parameter sensitivity, comparison on unseen scenarios, density stress test, multi-agent scaling, and real-robot analysis.
- Limitations are reported both locally at the observed failure mode and again in the conclusion.

## Reporting and evidence style

- The paper uses explicit baseline/configuration tables to define what each ablation changes.
- Experimental paragraphs generally state the scenario, sample count, metric definitions, observations, quantitative differences, and a mechanism interpretation in that order.
- Simulation-to-real transfer is not asserted solely from simulation; hardware, sensors, software, test area, task count, and physical evaluation metrics are reported.
- Metric denominators and success-case conditioning are stated, which is relevant to avoiding ambiguous rate and time claims.
- Results use increasingly difficult tests and separate training scenarios from unseen inference scenarios.
- The computational-cost section decomposes runtime by component rather than reporting only an aggregate latency.
- The manuscript sometimes uses causal and superlative language more strongly than the experimental design supports; this is not a journal rule to imitate.

## Journal-level candidate conventions

- Use a related-work comparison table when it genuinely clarifies method families and limitations.
- Define configurations/ablations in a compact table before interpreting their results.
- Present robustness as explicit scenario shifts with sample counts and unchanged metrics.
- Place limitations next to failure evidence and summarise them again at the end.
- For engineering deployment claims, report platform, sensors, runtime environment, task protocol, and quantitative physical results.
- Keep training configuration and inference evaluation structurally separate.

These remain candidate conventions until cross-paper frequency is assessed.

## Non-copy boundary

Do not transfer the ASMAC, SMRR, LAM, MLP reward, asynchronous worker, PID, collision-distance, sensor, curriculum, scenario, parameter, metric, runtime, or physical-experiment details. Do not reuse the paper's distinctive `shortsighted, intrusive, and unnatural motion policies` framing. The current manuscript has no real-robot experiment and must not borrow this article's physical-validation rhetoric.

## Full-text status

Complete: all 17 pages, including figures/tables, model details, sensitivity tests, real-world evaluation, declarations, biographies, and references, were reviewed for scientific function and writing structure.

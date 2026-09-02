# Full-text reading note 03

## Identity

- PDF: `3.pdf`
- DOI: `10.1016/j.engappai.2025.113518`
- Pages: 20
- SHA-256: `e2d5a54a936fb464252a578bfb4424c137587538d2a9efbdeca5756234d61f6c`
- Article type: Research paper
- Comparability: medium-high for UAV path planning, multi-objective reward design, uncertainty experiments, training comparison, robustness, scalability, ablation, and statistical reporting; low for its sensor filtering and swarm-specific hybrid architecture.

## Functional architecture

- Introduction moves from dynamic UAV-swarm navigation to a taxonomy/comparison table, synthesises three deficiencies, maps each proposed module to one deficiency, and lists three contributions.
- The paper devotes separate major sections to a preprocessing/filtering model, a situation-aware optimizer, the underlying DRL algorithm, and the integrated framework.
- Notation tables precede equation-dense sections, reducing ambiguity in long method descriptions.
- The integrated method is expressed as a four-phase data flow and pseudocode before reward details, uncertainty models, and convergence arguments.
- The experiment section begins with an explicit roadmap and a consolidated parameter table, then reports reward-weight selection, filter comparison, training comparison, algorithm comparison, significance tests, ablation, scalability, extreme conditions, and an experimental summary.
- Conclusion separates theoretical contributions, practical outcomes, limitations, and future work.

## Reporting and evidence style

- Experiment purpose, scenario, parameters, number of Monte Carlo runs, metrics, and early-termination rule are stated before numerical results.
- Mean and standard deviation are reported for repeated tests; statistical significance and effect size are discussed together rather than using p-values alone.
- A non-significant swarm path-length result is explicitly retained and discussed, though the phrase `meaningful trend` should not be adopted automatically.
- The paper distinguishes training time from execution/completion time and acknowledges the proposed method's longer training cost.
- Robustness is structured by named uncertainty sources and predefined combined adverse scenarios at two spatial scales.
- Scaling is reported with both task completion time and communication overhead.
- Ablations define exactly what is removed, the test scenario, number of trials, and four metrics before attribution.
- Results conclude with a synthesis table of algorithm characteristics and suitable application conditions.

## Journal-level candidate conventions

- Use notation tables for dense mathematical formulations.
- Precede a long experiment section with a short roadmap and consolidated parameter table.
- Report adverse conditions as explicitly defined scenarios, not an undifferentiated `noise` test.
- Combine statistical significance with effect size and preserve important non-significant outcomes.
- Include costs and limitations that work against the proposed method's preferred narrative.
- Close Results with an evidence synthesis before the Conclusion when many experiments are reported.
- Separate method contribution, practical evidence, limitations, and future work in the final discussion/conclusion sequence.

These remain candidate conventions until cross-paper frequency is assessed.

## Non-copy boundary

Do not transfer AHF, SAPO, CNN-LSTM attention, DDPG, reward weights, Gaussian danger fields, Dubins/wind/sensor parameters, convergence claims, uncertainty scenarios, swarm sizes, results, or references. Do not copy its claim that all modules are `beneficial and indispensable`; the present ablations show limited or conditional contributions for several components. Do not reproduce its causal explanations of robustness or generalisation without direct evidence.

## Full-text status

Complete: all 20 pages, including equations, notation tables, statistical tests, ablation, scalability, extreme-condition tests, limitations, declarations, and references, were reviewed for scientific function and writing structure.

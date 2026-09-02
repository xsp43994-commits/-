# Journal Style Profile v2 — Engineering Applications of Artificial Intelligence

## Scope and evidence rule

This profile is rebuilt from complete reading of all 12 acquired EAAI research-article PDFs. It replaces the first-round profile. It describes recurring scientific functions, not reusable wording.

An item is labelled **S-High** only when it occurs in at least five papers and in at least 70% of the papers for which the pattern is genuinely comparable. Patterns below that gate are **S-Medium** or **O/Low** and are not treated as journal requirements. Official submission rules remain a separate B-class authority and override stylistic frequency.

## S-High conventions to apply

| ID | Stable function | Frequency | Application to the v2 manuscript |
|---|---|---:|---|
| S01 | Engineering constraint → precise gap → explicit contributions | 11/12 (91.7%) | End the Introduction with evidence-bounded AI, feasibility and engineering contributions. |
| S02 | Define task, state/action/resource constraints and evaluation semantics before algorithm detail | 12/12 (100%) | Put the mountain-road task and return feasibility before the PPO–Pointer architecture. |
| S03 | Disclose environment, parameters, baselines, metrics and compute setting before comparisons | 12/12 (100%) | Keep the frozen protocol and map-level unit explicit before Results. |
| S04 | Keep figures/tables close to their claims and give each visual a distinct evidentiary role | 12/12 (100%) | Use the eight predefined evidence units rather than decorative figures. |
| S05 | Report several engineering dimensions and retain comparator wins/adverse costs | 10/12 (83.3%) | State that ACO, SA and MILP can obtain higher coverage while reporting time/status and operating trade-offs. |
| S06 | Separate simulation, held-out, hardware and deployment claim boundaries | 11/12 (91.7%) | Call DSM evidence zero-shot **simulation** transfer and explicitly state that no flight validation was conducted. |
| S07 | Separate training from held-out evaluation and specify what changes | 11/12 (91.7%) | Distinguish trained sizes, unseen synthetic maps, DSM transfer, known shifts and hidden mismatch. |
| S08 | Define named perturbations and limit robustness claims to tested ranges | 9/11 applicable (81.8%) | Report the two robustness layers separately; do not collapse them into generic robustness. |
| S09 | Give limitations and deployment boundaries a dedicated, identifiable discussion | 11/12 (91.7%) | Use a dedicated Discussion and limitations section with claim-specific constraints. |
| S10 | Physical claims require platform, instrumentation, protocol, repetitions/failures and separation from simulation | 5/5 applicable (100%) | Boundary only: because the present study has no physical experiment, no physical or certified-safety claim is allowed. |

## Lower-strength patterns

- **S-Medium M01 (8/12, 66.7%)**: replicated learning/evaluation uncertainty. It will still be used because the frozen project contains repeated seeds and map-level statistics, but it is not presented as a universal EAAI rule.
- **S-Medium M02 (7/12, 58.3%)**: explicit ablations before mechanism claims. The paper will use the four frozen ablations, while refusing to attribute independent effects to untested internal submasks.
- **O/Low M03 (4/8 applicable)**: notation tables are optional and used only if they materially improve the dense formulation.
- **O/Low M04 (3/12)**: related-work coverage tables are optional, not a journal convention.
- **O/Low M05 (4/8 applicable)**: latency decomposition is desirable only where the frozen Source Data support it.
- **O/Low M06 (1/12)**: moving Related work after Discussion is paper-specific and will not be emulated.

## Section-function blueprint

### Introduction

1. Establish the fixed-point mountain-road inspection problem and its multi-resource/return constraints.
2. Review the closest routing, pointer/attention, on-policy RL, safety/constraint and transfer evidence by mechanism.
3. Identify the gap: online combinatorial selection must preserve return feasibility and remain useful under engineering time/resource trade-offs.
4. State contributions with an evidence ceiling: formulation, return-aware PPO–Pointer method, and frozen multi-map evaluation. Do not claim universal coverage superiority.

### Related work and research gap

- Organise by scientific function rather than chronology.
- Every external proposition must map to a fully verified D-class source.
- Do not inherit a sample paper’s citation cluster.
- End each subsection by stating what remains unresolved for the present task.

### Problem formulation and method

- Define fixed points, priority, depot, energy/time/distance resources, termination and feasible action set before neural architecture.
- Separate hard return-aware feasibility enforcement from reward shaping and learned preference.
- Describe the Pointer decoder and PPO objective at the granularity necessary for reproduction, while using only frozen implementation facts.

### Experimental protocol

- Name the frozen protocol/version and data identity.
- State maps as independent units; tasks, routes and seeds are nested observations.
- Separate confirmatory metrics from the post-hoc 100-point composite summary.
- Define synthetic held-out maps, DSM zero-shot simulation transfer, known perturbations, hidden mismatch and four ablations before results.

### Results

Use the sequence:

1. Coverage and priority outcomes;
2. Return safety and resource costs;
3. Online planning time and quality–time trade-off;
4. Training stability and sample efficiency;
5. Unseen synthetic maps and DSM zero-shot simulation transfer;
6. Known shifts and hidden mismatch;
7. Four frozen ablations.

Each result paragraph follows **Observation → quantitative evidence → statistics → minimal interpretation**. Comparator wins and non-significant findings stay in the narrative.

### Discussion and limitations

Label the epistemic level of each sentence:

- **Observation**: directly measured in the frozen evidence.
- **Interpretation**: a bounded engineering explanation consistent with that evidence.
- **Mechanistic possibility**: plausible but not isolated causally.
- **Speculation**: future-facing and not used to support the conclusion.

Explicitly rule out equivalence claims from non-significance, training-range-external scale generalisation, real-flight validation, real-world safety certification and independent effects for unablated return submasks.

## Style controls

- Use restrained, testable verbs: *achieved, reduced, remained, was associated with, was observed under*.
- Reserve *outperformed* for a named metric, comparator, dataset and statistical context.
- Do not use *proved*, *guaranteed safe*, *generalised to arbitrary scales*, *real-world validated* or *state of the art* unless the required evidence exists; it does not in the frozen package.
- Put numerical claims next to their figure/table and Source Data locator.
- Define all abbreviations at first use and keep model names stable: PPO–Pointer, A2C–Pointer and traditional Flat-MLP PPO.
- Use `traditional_ppo` only as the implementation identifier; do not reintroduce excluded `ppo_mlp`.

## Non-copy control

No sentence, equation arrangement, dataset value, parameter, distinctive phrase, contribution wording, figure composition or reference combination is to be copied from the 12 exemplars. Only the recurring scientific function and evidence order may influence the manuscript. The sentence-level similarity audit will compare the completed v2 manuscript against all 12 full texts.

# Full-text reading note 11

## Identity

- PDF: `11.pdf`
- DOI: `10.1016/j.engappai.2023.106703`
- Title: *Subtask-masked curriculum learning for reinforcement learning with application to UAV maneuver decision-making*
- Pages: 14
- SHA-256: `c8e304f95c1905d1438ee35b8cbc9446d0dc24ef0288e244f1a57cc45c30108f`
- Article type: Research paper
- Comparability: medium for action/state masking concepts, sample-efficiency evaluation, ablation, multiple-seed learning curves and transfer limitations; low for curriculum learning, TD3 and military manoeuvre content.

## Functional architecture

- The Introduction defines sparse-reward, concurrent-subtask limitations, contrasts hierarchical and curriculum learning, then introduces task and transfer masks.
- Background separately formalises RL/TD3, curriculum learning and the task MDP before the new method.
- Method defines concepts first, then task generation/sequencing, transfer condition, policy distillation/reuse and the integrated algorithm with pseudocode.
- Empirical analysis states platform, curriculum, state spaces, metrics, hyperparameters, hardware, training time and ten random seeds.
- Results begin with four ablations tied to three explicit questions, then compare standard RL baselines, analyse final policies and compare with an expert policy.
- Discussion is divided into results analysis and limitations; related work appears after Discussion to contrast adjacent paradigms once the method is fully defined.

## Reporting and evidence style

- Ablations distinguish removal of policy distillation/reuse from replacing adaptive weights with decay; learning curves show mean and standard deviation across ten seeds.
- Final performance is evaluated over 1000 episodes, but the paper selects the best-performing training seed for this table; this selection must not be copied without disclosure.
- Negative transfer is not hidden: one ablation performs worse than learning from scratch and is used to qualify the value of curriculum learning.
- Mechanistic accounts such as `policy reuse avoids cold start` are explicitly presented as interpretations or inferences from curves, not direct causal proofs beyond the ablations.
- The expert-policy comparison uses both aggregate metrics and state/angle traces, while avoiding a claim that similar trajectories prove identical internal reasoning.
- Limitations cover method assumptions, deterministic-policy restriction, scenario randomness/scalability and imperfect observations.

## Journal-level candidate conventions

- Turn ablation goals into explicit questions and map each variant to one question.
- Show mean and variability of learning curves across seeds.
- Retain negative-transfer cases that weaken the broad method narrative.
- Disclose any best-seed selection for downstream evaluation.
- Separate observed curve behaviour from inferred mechanism.
- Compare learned and expert behaviour using both performance and trajectory/state evidence.

These remain candidate conventions until cross-paper frequency is assessed.

## Non-copy boundary

Do not transfer the missile scenario, curriculum/task masks, transfer mask, TD3 transfer equations, expert policy, rewards, hyperparameters, seed-selection protocol, numerical results or references. Its `subtask mask` removes curriculum subtasks, whereas the current return-aware feasibility mask blocks infeasible route actions; the terms and mechanisms must not be conflated. This paper cannot justify claiming independent effects for the current project’s internal return-resource submasks.

## Full-text status

Complete: all 14 pages, including definitions and pseudocode, curriculum and transfer derivations, ten-seed ablations, baseline and expert comparisons, negative-transfer discussion, limitations, declarations and references, were reviewed for scientific function and writing structure.

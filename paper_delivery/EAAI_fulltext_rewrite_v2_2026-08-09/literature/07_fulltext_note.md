# Full-text reading note 07

## Identity

- PDF: `7.pdf`
- DOI: `10.1016/j.engappai.2024.109870`
- Title: *Path planning via reinforcement learning with closed-loop motion control and field tests*
- Pages: 13
- SHA-256: `a6371468248b67346d1c2334992820e2490b39907528a23ac0a6174816a990d5`
- Article type: Research paper
- Comparability: medium-high for hierarchical RL planning, feasibility feedback, real-time evidence, domain randomisation, algorithm comparison and engineering validation; task domain and field-test claims are not transferable.

## Functional architecture

- The Introduction balances dynamic feasibility and real-time computation, locates the method in a layered planning/control architecture and defines a standards-based use case.
- Related work spans search, optimisation, RL and MPC, then contributions identify the TD3 polynomial planner, closed-loop MPC evaluation and proving-ground comparison.
- Method reports the one-step RL environment, network/hyperparameters, state/action spaces, whole-manoeuvre reward and polynomial path construction.
- Two dedicated sections describe the vehicle/instrumentation and MPC formulation before any field result.
- Results first compare agent/path-generator combinations, then explain domain randomisation, compare with an adaptive MPC planner in simulation, and finally report controlled field tests against an earlier planner and human drivers.
- Conclusion explicitly gives two feasibility-check mechanisms and identifies the manoeuvre-specific path template as a drawback.

## Reporting and evidence style

- The planner is trained with closed-loop controller performance in the reward, so path feasibility is tied to the execution layer rather than assessed only geometrically.
- Algorithm and representation effects are crossed in the comparison; a slower PPO training path that eventually reaches similar reward is retained.
- Domain-randomisation ranges are tabulated, and the paper distinguishes variation used to prevent overfitting from the specific standards-based field configuration.
- Planning time, path generation time and control-loop turnaround are reported separately.
- Field-test protocol reports hardware, sensor rate, number and type of runs, human-driver population and supervising drivers.
- Simulation errors and physical runs are not conflated; each supports a different part of the claim.
- Conclusion acknowledges impossible scenarios and states how an infeasible output should be escalated to a higher planning layer.

## Journal-level candidate conventions

- When claiming real-time applicability, report planning, generation and control-loop latency separately on target hardware.
- Explain how feasibility is checked and what the system does when no feasible solution exists.
- Cross algorithm choice with representation choice when both can cause performance changes.
- State domain-randomisation ranges and the exact held-out or field-test condition.
- Keep simulation, hardware-in-loop and field evidence explicitly separated.
- Report the structural drawback introduced by a hierarchical solution, not only its safety benefit.

These remain candidate conventions until cross-paper frequency is assessed.

## Non-copy boundary

Do not transfer the ISO 3888-2 scenario, TD3/MPC hierarchy, polynomial/clothoid generator, tire and actuator models, reward thresholds, vehicle hardware, human-driver comparison, timing values, domain-randomisation ranges or field-test claims. The current work has no physical vehicle or UAV tests; DSM evidence cannot be presented using this paper’s sim-to-real or proving-ground vocabulary. The present `return-aware` feasibility mechanism also cannot be equated with a critic-Q feasibility estimate.

## Full-text status

Complete: all 13 pages, including the complete RL and MPC formulations, instrumentation, training comparison, domain-randomisation table, simulation and field protocols, timing and error figures, limitations, declarations and references, were reviewed for scientific function and writing structure.

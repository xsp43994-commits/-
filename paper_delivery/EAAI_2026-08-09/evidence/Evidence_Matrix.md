# Evidence Matrix

| Claim ID | Claim | Evidence | Type | Max strength | Boundary |
|---|---|---|---|---|---|
| C01 | The method enforces return-aware resource feasibility before action selection. | F08; Fig. 1; implementation | method fact | show | No formal safety certification |
| C02 | Pointer attention supports variable-length candidate selection within the trained node-count range. | F05; Fig. 1 | method fact | supports | No claim beyond 16/20/24 nodes |
| C03 | PPO+Pointer strongly improves safe weighted coverage over Flat-MLP PPO. | F10; Fig. 2; pairwise statistics | result | shows | Simulation domains only |
| C04 | PPO+Pointer and A2C+Pointer attain statistically similar confirmatory safe coverage. | F11; Fig. 2 | result | indicates | Non-significance is not proof of equivalence |
| C05 | PPO+Pointer has a stronger combined training profile than A2C+Pointer. | Fig. 5; D6/D7 frozen scores | result | indicates | Mechanism not causally isolated |
| C06 | Return reserve is the dominant independently tested safety-related component. | F13; Fig. 8 | result | supports | Other submasks were not separately ablated |
| C07 | Classical solvers expose a quality–time trade-off rather than universal PPO dominance. | F12; Fig. 4 | result | shows | Hardware-specific time values |
| C08 | The policy transfers zero-shot to independent geographic DSM simulations. | F04; F14; Fig. 6 | result | shows | Not physical-domain transfer |
| C09 | Hidden mismatch degrades performance but preserves a PPO advantage over Flat-MLP PPO. | Fig. 7; robustness tables | result | indicates | Finite perturbation catalogue |
| C10 | The study provides reproducible simulation evidence for engineering planning decisions. | F01–F15; package manifest | synthesis | supports | Requires independent external and flight validation |

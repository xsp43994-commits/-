# Return-aware reinforcement learning with a Pointer policy for resource-constrained mountain-road inspection planning

## Abstract

Mountain-road inspection by unmanned aerial vehicles requires a planner to select valuable fixed inspection points while retaining sufficient energy, distance, time and dynamic feasibility for safe return. This study couples proximal policy optimization with a Pointer policy and a return-aware multi-resource feasibility mask. The artificial-intelligence contribution is feasible sequence construction over variable candidate sets; the engineering application is online planning for priority-labelled points along mountain-road corridors. The protocol used 72 training, 12 validation and 24 unseen procedural maps, followed by zero-shot simulation transfer to eight Copernicus digital-surface-model maps. Seven learning variants and established traditional comparators produced 21,648 frozen route evaluations. The confirmatory endpoint was map-level safe weighted coverage. On unseen synthetic maps, proximal-policy-optimization–Pointer and advantage-actor–critic–Pointer achieved means of 0.486 and 0.485, whereas fixed-slot multilayer-perceptron proximal policy optimization achieved 0.269. Corresponding digital-surface-model means were 0.502, 0.502 and 0.266. Ant-colony, simulated-annealing and mixed-integer planners attained higher coverage in some settings but required longer protocol-specific planning times. The Pointer learners did not differ significantly in final coverage. Corrected post-hoc analysis on a common 108-task validation set found nearly identical late-training stability and a modest validation-area-under-the-curve advantage for proximal policy optimization. Removing return reserve caused the clearest degradation across synthetic, digital-surface-model and hidden-mismatch tests. Return-aware Pointer reinforcement learning therefore provides fast, safe and balanced online planning under the simulator, not universal optimality or flight certification.

**Keywords:** unmanned aerial vehicle; inspection planning; reinforcement learning; Pointer network; safe return; constrained routing

## Glossary

**A2C–Pointer:** advantage-actor–critic learner using the same Pointer policy representation as the principal model.  
**DSM:** digital surface model used as terrain input for zero-shot simulation transfer.  
**Flat-MLP PPO:** fixed-slot multilayer-perceptron proximal policy optimization comparator without attention or a Pointer mechanism.  
**PPO–Pointer:** proximal policy optimization with a Pointer policy over the current candidate set.  
**Return-aware feasibility mask:** composite legal-action mask that accounts for visit status, energy, distance, time, dynamics and projected return.  
**Safe weighted coverage (SWC):** priority-weighted coverage set to zero when a route fails to return safely or violates a hard constraint.

## 1. Introduction

Unmanned aerial vehicles can reduce personnel exposure and access cost in the inspection of geographically dispersed infrastructure, but their route plans must reconcile inspection value with endurance and path feasibility. In fixed-point infrastructure inspection, visiting more locations is not sufficient if the selected route consumes the reserve needed to return, exceeds a distance or mission-time budget, or traverses terrain and wind conditions that violate vehicle limits. Recent inspection planners therefore treat coverage, route cost and resource feasibility as coupled engineering objectives rather than interchangeable reporting metrics (Zhao et al., 2024; Sun et al., 2026). Reviews of reinforcement learning for autonomous aerial navigation similarly show that task definitions, environmental variation and validation platforms materially determine what an apparent performance improvement establishes (AlMahamid and Grolinger, 2022).

The present task differs from continuous road coverage and point-to-point collision avoidance. Two mountain-road corridors meet near a launch site, and a finite set of fixed inspection points is assigned high, medium or low priority. At every decision, the vehicle selects one unvisited point or returns to the launch site. A useful policy must therefore construct a variable-length sequence while tracking the changing set of feasible candidates. Pointer Networks are well suited to this structure because their outputs identify positions in an input sequence rather than classes from a fixed output dictionary (Vinyals et al., 2015). Attention-based policies have consequently been applied to travelling-salesperson, vehicle-routing, orienteering and prize-collecting problems (Vaswani et al., 2017; Kool et al., 2019). Their use in a mountain-road inspection problem nevertheless requires task-specific safety semantics, resource accounting and evidence that separates the contribution of sequence-aware architecture from that of the policy-gradient optimizer.

Safety in reinforcement learning may be introduced through constrained objectives, modified exploration or direct restriction of actions, rather than being left entirely to reward penalties (García and Fernández, 2015; Achiam et al., 2017). Action masking can improve learning and prevent invalid navigation decisions, but its timing and meaning must be explicit: a mask that excludes already visited points is not equivalent to one that certifies a resource-feasible return (Pereira and Pinto, 2024; Hou et al., 2023). Here, each candidate action is screened using its outgoing flight segment, inspection service requirement and projected return segment. The resulting mechanism is evaluated only as a composite return-aware multi-resource feasibility mask; the study does not claim that its energy, distance, time or dynamics components have been independently isolated.

Robustness is another source of overstatement in simulation studies. Domain randomization exposes a policy to varied simulated conditions during training (Tobin et al., 2017), but simulation variation alone does not establish real-world deployment readiness (Dulac-Arnold et al., 2021). We therefore distinguish conditions that are known to the planner from hidden discrepancies between the planning model or perception and execution truth. We also distinguish held-out procedural maps from geographic simulation transfer based on a digital surface model. The latter uses Copernicus DEM GLO-30 terrain assets, but it remains simulation rather than flight validation.

This work addresses three linked gaps. First, fixed-point mountain-road inspection needs an online combinatorial policy that embeds return feasibility rather than repairing an unsafe terminal route after selection. Second, evidence for a Pointer policy should include a genuinely fixed-slot multilayer-perceptron proximal policy optimization comparator, not an attention-containing variant relabelled as conventional proximal policy optimization. Third, final route quality must be interpreted together with training behaviour, computation time, robustness and ablation evidence, without converting a post-hoc aggregate score into the confirmatory endpoint.

The contributions are as follows. (1) We formulate priority-weighted fixed-point mountain-road inspection as a sequential decision problem with explicit energy, distance, time, terrain, wind, dynamics and safe-return constraints. (2) We implement a proximal policy optimization–Pointer actor–critic with priority-biased set encoding and a return-aware composite feasibility mask. (3) We freeze a comparison against an advantage-actor–critic–Pointer model, a fixed-slot multilayer-perceptron proximal policy optimization model, four mechanism ablations and established traditional planners. (4) We evaluate 21,648 routes on unseen procedural maps, Copernicus digital-surface-model simulations, known shifts and hidden model/perception mismatches using map-level inference. The central hypothesis is deliberately conditional: the proposed model should provide an effective online engineering balance, not necessarily the highest coverage among all offline or iterative planners.

## 2. Related work and research gap

### 2.1 Aerial inspection and resource-aware route construction

Infrastructure-inspection routing generally couples a value or coverage objective with flight-path efficiency. Zhao et al. (2024) optimized automated structural inspection paths under coverage and travel considerations, while Sun et al. (2026) studied periodic unmanned-aerial-vehicle task allocation and path planning with endurance-aware feasibility pruning. These studies demonstrate that inspection routes are structured combinatorial objects, but their allocation settings, solver assumptions and validation protocols do not directly resolve online priority-labelled selection under mountain terrain. Other EAAI applications expose related trade-offs: terrain-risk planning may improve safety at a distance or time cost (Zhang et al., 2024), and multi-vehicle resource systems must mediate charging, communication and task objectives (Seong et al., 2023). Such evidence motivates multi-metric evaluation but does not imply a universal winner across incompatible objectives.

### 2.2 Reinforcement learning, sequence models and safe action restriction

Proximal policy optimization uses a clipped probability-ratio surrogate to permit repeated minibatch updates while limiting large policy changes (Schulman et al., 2017). Generalized advantage estimation controls a bias–variance trade-off in policy-gradient advantage estimates (Schulman et al., 2016). The comparison model in this study belongs to the advantage actor–critic lineage associated with asynchronous actor–critic methods (Mnih et al., 2016), but uses the same Pointer architecture and task interface so that optimizer-related training behaviour can be compared without changing the sequence representation.

Pointer Networks construct outputs by attending to input positions (Vinyals et al., 2015). Scaled dot-product and multi-head attention provide a mechanism for modelling interactions among candidates (Vaswani et al., 2017), and attention policies have learned constructive heuristics for classical routing problems (Kool et al., 2019). However, an attention policy alone does not guarantee physical feasibility. Wind-randomized quadcopter studies have shown the importance of separating training variation, nominal evaluation and perturbation testing (Andres et al., 2025). Likewise, action-masking and curriculum studies show that training benefits can be conditional and may coexist with negative transfer or setting-specific failures (Pereira and Pinto, 2024; Hou et al., 2023). The missing link is an inspection-specific action certificate that reserves the resources required for return and is tested against an architecture-matched conventional policy.

### 2.3 Position of the present study

The proposed method is not a new metaphor-based optimizer. It is an engineering application of established proximal policy optimization, attention and Pointer concepts to a constrained fixed-point inspection problem. The artificial-intelligence contribution is the integration of sequence-aware policy representation, priority-biased candidate encoding and return-aware feasible-action construction within a common actor–critic interface. The engineering contribution is a reproducible evaluation protocol that preserves terrain, wind, energy, distance, time and return semantics across learning and traditional planners. This positioning also determines the claim boundary: traditional optimizers may produce higher-coverage routes when given substantially more online computation, while a learned policy may be preferable when fast repeated replanning and predictable feasibility are more important.

## 3. Mountain-road inspection problem formulation

### 3.1 Task and decision process

Let V = {0, 1, …, N} denote the depot 0 and N fixed inspection points. Each point i has a position p_i, priority w_i ≥ 0 and service time τ_i. The tested node counts were N ∈ {16, 20, 24}. A route π = (0, i_1, …, i_K, 0) contains no repeated inspection point and may terminate before all points are visited. Positions were sampled along two road corridors whose intersection was near the depot. Priority layouts included regular patterns and conflict cases in which high-priority points were farther from the depot.

The state comprised candidate-specific features and a global vehicle vector. Each candidate, including the depot token, used 15 features: relative three-dimensional position; normalized priority; visited and depot indicators; normalized outgoing distance, energy and time; normalized return distance, energy and time; and the normalized three-component mean wind vector on the outgoing segment. The 14-dimensional vehicle vector encoded the current position relative to the depot, previous direction, remaining energy/distance/time fractions, visited-point and priority-weighted coverage fractions, and the current normalized wind vector. An action selected one candidate point or the depot.

### 3.2 Resources, dynamics and confirmatory objective

For a candidate i, the simulator evaluated an outgoing segment from the current location, inspection service at i and a return segment from i to the depot. Segment feasibility accounted for terrain clearance, wind and vehicle dynamic limits. Energy, distance and time were accumulated along the executed route. The energy calculation used a transparent engineering proxy with distinct hover, cruise, climb and descent power parameters; rotary-wing propulsion energy is known to vary with flight regime, so this representation is not claimed to be a high-fidelity aircraft power model (Zeng et al., 2019).

Let C_w(π) be the priority-weighted fraction of inspected points. The confirmatory endpoint was safe weighted coverage,

[EQUATION 1: SWC(π) = C_w(π) if the route returned safely without a hard-constraint violation; SWC(π) = 0 otherwise.]

This definition prevents an unsafe high-coverage route from appearing beneficial. Ordinary coverage, priority-stratified coverage, return and violation rates, resource use, total mission time and planning time were auxiliary outcomes. Energy, distance and mission-time summaries were computed only for safe routes and were interpreted jointly with the safe-route rate.

### 3.3 Return-aware multi-resource feasibility

For every unvisited candidate i, the planner estimated resources for the outgoing segment, service and projected return. Five Boolean components represented visit status, energy, distance, time and dynamics. Their conjunction formed the legal-action mask. The return token was always evaluated using the actual return segment. If no legal action other than return remained, the policy returned rather than selecting an inspection point that would strand the vehicle.

The no-return-reserve ablation relaxed what the policy could propose, but the simulator still refused to execute a candidate that violated the full return requirement. Such an attempt terminated the simulated episode as stranded. This design measures the protective contribution of the composite planning mask without executing a knowingly unsafe flight segment. Because the reserve components were not ablated separately, subsequent results apply to the composite mechanism only.

## 4. Return-aware proximal policy optimization–Pointer method

### 4.1 Priority-aware candidate encoder

The principal network projected each 15-dimensional candidate vector into a 128-dimensional representation and applied four-head self-attention. A normalized priority term was added to the attention score with coefficient 0.5 before the softmax. Residual connections, layer normalization and a two-layer feed-forward block produced the encoded candidate set. Removing the explicit priority bias defined one ablation; all other model and task elements were retained.

The encoder preserves a shared candidate representation across the trained node counts. It does not establish extrapolation beyond 24 points because all three evaluated sizes occurred in training. The term “multi-scale” therefore refers only to performance within the trained range.

### 4.2 Pointer actor and value function

The actor projected the global vehicle state to a query, attended to encoded candidates through a masked multi-head glimpse and combined the glimpse with the vehicle query using a learned gate. A final Pointer score was computed for each candidate and set to negative infinity when the corresponding legal-action entry was false. The resulting categorical distribution therefore assigned probability only to currently feasible actions. The critic concatenated the actor query with masked means of the unvisited and currently legal candidate representations, then predicted the state value.

This construction follows the input-position selection principle of Pointer Networks (Vinyals et al., 2015) and the attention operations described by Vaswani et al. (2017), while the candidate, priority and return semantics are specific to the inspection task. It is a constructive online policy: one action is produced at each decision rather than an entire route being optimized offline and repaired afterward.

### 4.3 Proximal policy optimization

Trajectories were optimized with proximal policy optimization using the clipped surrogate of Schulman et al. (2017). Generalized advantage estimates used γ = 0.99 and λ = 0.95 (Schulman et al., 2016). The clipping ratio was 0.2, learning rate 10^−4, maximum of five epochs per update and minibatch size 128. Value loss had coefficient 0.5; gradients were clipped at norm 1.0. The entropy coefficient decreased from 0.02 to 0.002. A target approximate Kullback–Leibler divergence of 0.02 provided an early-stop diagnostic for repeated epochs. Each formal model was trained for 3,000 episodes with seeds 42–46.

Domain randomization varied initial state of charge, distance and time budgets, and wind scaling within the frozen training ranges. Resource shaping imposed a secondary penalty proportional to mean incremental use of energy, distance and time. Its scale remained subordinate to the priority-coverage objective. The no-domain-randomization and no-resource-shaping models removed these components individually. Raw rewards were not compared across variants because the shaping definitions differ.

### 4.4 Controlled learning comparators

The advantage-actor–critic–Pointer model shared the Pointer representation, observation, masks, task distribution and five training seeds, but used a single-pass advantage actor–critic update rather than repeated clipped proximal updates. This comparator isolates optimizer-related training behaviour more closely than a simultaneous architecture change, although it does not constitute a causal proof of any one update component.

The conventional proximal-policy-optimization comparator used a fixed-slot multilayer perceptron. It contained no Pointer mechanism, attention layer or node encoder. Twenty-four candidate slots plus one depot slot were flattened; 16- and 20-point instances used zero-feature padding and illegal empty slots. The actor mapped 389 inputs through two 256-unit hidden layers to 25 logits. The critic additionally received valid-slot and legal-action masks, producing a 439-dimensional input followed by two 256-unit layers. Reward, constraints, randomization, return reserve, training maps, proximal-policy-optimization settings and checkpoint rule were otherwise shared. A historically excluded attention-containing prototype did not enter any analysis.

## 5. Experimental and statistical protocol

### 5.1 Maps, tasks and frozen evaluation matrix

The procedural split contained 72 training maps with 648 tasks, 12 validation maps with 108 tasks and 24 unseen test maps with 216 tasks. Tasks crossed node count (16, 20, 24), difficulty (moderate, hard, extreme), resource constraint (energy, distance, time, mixed) and priority layout. Geographic simulation transfer used eight Copernicus DEM GLO-30 digital surface models, each with two deterministic road/launch contexts and 144 total tasks. Copernicus DEM GLO-30 is a global 30 m digital surface model with attribution and access conditions specified by the data provider (Copernicus Space Component Data Access, 2021). Contexts and tasks were nested within maps and were not treated as independent geographic samples.

Seven learning variants—complete proximal-policy-optimization–Pointer, fixed-slot multilayer-perceptron proximal policy optimization, advantage-actor–critic–Pointer and four ablations—were trained with five seeds. Formal evaluation contained 7,560 synthetic learning routes, 3,888 synthetic main-baseline routes, 504 supplementary-baseline routes, 5,040 digital-surface-model learning routes, 1,152 digital-surface-model baseline routes, 1,008 known-shift routes and 2,496 hidden-mismatch routes, totalling 21,648. Protocol, route files and analysis inputs were frozen before manuscript drafting.

[FIGURE 1 — Non-generative method and frozen evaluation workflow]

### 5.2 Traditional planners

Main synthetic comparisons included nearest-feasible and priority-resource greedy construction, ant-colony optimization, a genetic algorithm, simulated annealing and mixed-integer linear programming. The traditional families were established algorithms rather than newly proposed metaphors: Ant System uses stochastic constructive search with pheromone-mediated positive feedback (Dorigo et al., 1996), and simulated annealing transfers an annealing acceptance principle to combinatorial optimization (Kirkpatrick et al., 1983). A stratified 72-task supplementary subset additionally included A* graph search (Hart et al., 1968), particle-swarm optimization (Kennedy and Eberhart, 1995) and exact Pareto dynamic programming.

The digital-surface-model comparison retained nearest feasible, priority-resource greedy, ant-colony optimization and mixed-integer programming. Mixed-integer problems were solved through SciPy’s interface to HiGHS (Virtanen et al., 2020; SciPy community, 2026). Solver status, bounds and mixed-integer-programming gap were retained; a time-limited incumbent was not relabelled as a proven optimum.

### 5.3 Known shifts and hidden mismatches

Known-domain-shift tests changed wind conditions or increased power-model coefficients by 10%, and the policy received observations consistent with the changed execution condition. Hidden-mismatch tests instead planned with nominal or perturbed observations and evaluated the selected route against a different frozen truth. They covered hidden wind error, actual power at 1.1 times the planning model, digital-elevation error and localization error. All algorithms assigned to one task shared the same perturbation realization. The two robustness families were analysed separately because they answer different engineering questions.

### 5.4 Statistical analysis

The map was the independent unit. Planning seeds and training seeds were aggregated within tasks, and tasks were then aggregated within maps. The six predeclared families were synthetic main algorithms, synthetic ablations, digital-surface-model main algorithms, digital-surface-model ablations, known shifts and hidden mismatches. A Friedman test assessed each map-blocked family, followed by paired two-sided Wilcoxon signed-rank comparisons against the complete model (Demšar, 2006). Holm’s sequential procedure controlled multiplicity within each pairwise family (Holm, 1979).

Paired effects are reported as mean differences with 10,000-replicate map-outer bootstrap intervals, Hodges–Lehmann shifts and rank-biserial effects where applicable. Bootstrap resampling approximates sampling uncertainty by repeated sampling with replacement (Efron, 1979), and the Hodges–Lehmann estimator supplies a rank-based paired location shift (Hodges and Lehmann, 1963). The eight digital-surface-model maps—not the 144 nested tasks—therefore define the independent sample size for geographic simulation inference. Post-hoc multiobjective scores and their normalization/weight sensitivity were restricted to supplementary diagnosis.

## 6. Results

### 6.1 Safe weighted coverage and priority outcomes

Across 24 unseen procedural maps, mean safe weighted coverage was 0.486 for proximal-policy-optimization–Pointer, 0.485 for advantage-actor–critic–Pointer and 0.269 for fixed-slot multilayer-perceptron proximal policy optimization (Fig. 2). The map-paired complete-model difference from the fixed-slot comparator was 0.217 (95% bootstrap interval 0.196–0.239; Holm-adjusted p = 9.54 × 10^−7). The complete and advantage-actor–critic Pointer models differed by only 0.0008 (−0.0057 to 0.0069; adjusted p = 0.845). This non-significant contrast does not demonstrate formal equivalence, but it rules out a coverage-based claim that the proximal optimizer clearly dominated the architecture-matched comparator.

Traditional planners set a higher coverage ceiling in the same protocol. Ant-colony optimization, simulated annealing and mixed-integer programming achieved synthetic map means of 0.553, 0.544 and 0.571. Their paired advantages over proximal-policy-optimization–Pointer were 0.067, 0.058 and 0.085 in absolute safe weighted coverage, respectively; all adjusted p values were 9.54 × 10^−7. Priority-stratified results showed that the fixed-slot model lost much of the high- and medium-priority coverage captured by the Pointer learners. The result supports sequence-aware representation over the conventional fixed-slot policy, but the broader architectural differences prevent attribution to attention alone.

On eight digital-surface-model maps, the complete and advantage-actor–critic Pointer means were both 0.502, compared with 0.266 for fixed-slot proximal policy optimization. The complete-versus-fixed-slot difference was 0.237 (0.212–0.262; adjusted p = 0.0469). The complete-versus-advantage-actor–critic difference was −0.00001 (−0.0090 to 0.0090; adjusted p = 0.742). Ant-colony optimization and mixed-integer programming attained 0.560 and 0.583, exceeding the complete learned policy by 0.057 and 0.080 (both adjusted p = 0.0469). Thus, the geographic simulation results reproduced both parts of the synthetic finding: strong separation from fixed-slot learning and no final-coverage separation from the architecture-matched actor–critic, alongside higher coverage from slower traditional planners.

[FIGURE 2 — M01 and M03: coverage and priority outcomes]

### 6.2 Safety, return and resource costs

Under nominal synthetic and digital-surface-model evaluations, the three core learning models completed the frozen routes without hard-constraint violations, giving mean safe and return rates of 1.0. This nominal safety result reflects the joint effect of the legal-action construction and execution checks; it is not evidence of real-world safety certification. Energy, distance and mission-time summaries were therefore computed over safe routes. The Pointer models spent resources to obtain higher priority-weighted coverage than the fixed-slot model, so lower absolute resource use by the latter cannot be read as a superior route when it largely results from visiting fewer valuable points.

The composite return mechanism was the principal safety result. Removing the reserve reduced synthetic map-level safe weighted coverage by 0.372 (0.343–0.402; adjusted p = 4.77 × 10^−7) and digital-surface-model coverage by 0.390 (0.356–0.424; adjusted p = 0.0313). The simulator intercepted the dangerous proposal rather than executing an infeasible segment, so this ablation measures prevented unsafe planning actions in simulation. It does not isolate energy, distance, time or dynamics submasks and does not imply that a deployed aircraft would remain safe under unmodelled failures.

[FIGURE 3 — M02 and M04: safe return and resource costs]

### 6.3 Online planning time and the quality–time trade-off

Mean synthetic planning time was 1.168 s for proximal-policy-optimization–Pointer. Simulated annealing required 3.832 s, mixed-integer programming 20.807 s and ant-colony optimization 76.701 s under the frozen hardware and implementation. The learned policy therefore occupied a different point on the quality–time frontier: it sacrificed some coverage relative to iterative and optimization-based planners but produced decisions much faster. Fixed-slot proximal policy optimization was faster still, but at substantially lower coverage.

These timings are protocol-specific and should not be generalized to other hardware or implementations. Mixed-integer programming also requires its status and gap to be considered, because a returned incumbent at the time limit is operationally useful but not necessarily certified optimal. The practical finding is conditional: when repeated online planning is valued, proximal-policy-optimization–Pointer offers a favourable compromise; when computation time is secondary and the current model is trusted, ant-colony or mixed-integer planning can yield higher coverage.

[FIGURE 4 — M05 and S02: online planning time and quality–time trade-off]

### 6.4 Training stability and sample efficiency

All core learning models were trained on the same task distribution for 3,000 episodes and five seeds. Figure 5a uses safe weighted coverage from the same fixed 108-task external validation set at 26 checkpoints; it does not use batch reward or final test results. At episode 3,000, the five-seed validation means were 0.497 for the complete model, 0.499 for advantage-actor–critic–Pointer and 0.271 for fixed-slot proximal policy optimization.

The corrected post-hoc D6 stability score combined cross-seed and within-seed consistency over the final 20% of the training budget. It was 0.9978 for the complete model and 0.9971 for advantage-actor–critic–Pointer. Their paired-seed difference was 0.00069; the 10,000-replicate interval crossed zero (−0.00060 to 0.00173). D7 was redefined as normalized validation safe-weighted-coverage area under the curve over the common interaction window 80–17,702. It was 0.4872 for the complete model and 0.4781 for advantage-actor–critic–Pointer, a difference of 0.00913 with a bootstrap interval of 0.00491–0.01331. The paired five-seed Holm-adjusted p value was 0.125 for both dimensions. Tail-window and interaction-budget sensitivity analyses retained the qualitative distinction between near-equal stability and a modest area-under-the-curve advantage.

These results do not support a large stability advantage. They instead show nearly equal late-training stability and a modest sample-efficiency difference under the corrected area-under-the-curve definition. The direction is consistent with clipped repeated-minibatch updating, but the experiment did not independently manipulate each proximal-policy-optimization component; no causal optimizer claim is made.

[FIGURE 5 — M06 and M07: training trajectories, stability and sample efficiency]

### 6.5 Unseen maps and digital-surface-model simulation transfer

The 24 procedural test maps were not used for training or checkpoint selection, supporting cross-map generalization within the frozen task distribution. Node counts of 16, 20 and 24 were all present during training, so the size-stratified results describe trained-range multi-scale performance, not extrapolation to a larger problem. The similarity of proximal-policy-optimization–Pointer means between procedural maps (0.486) and digital-surface-model maps (0.502) indicates that the policy retained useful behaviour when terrain geometry changed to eight Copernicus-derived regions.

This result is best described as zero-shot geographic digital-surface-model simulation transfer: there was no digital-surface-model-specific policy training, but the route evaluator and vehicle model remained simulated. The study neither flew the routes nor tested site-specific sensing, regulation, communication or weather operations. Representative route renderings illustrate the fixed task geometry and are not independent inferential evidence.

[FIGURE 6 — M08 and V01/V02: cross-map and DSM simulation transfer]

### 6.6 Known shifts and hidden model/perception mismatch

In known shifts, the complete model and advantage-actor–critic–Pointer had mean safe weighted coverage of 0.479 and 0.480, with a paired complete-model difference of −0.0008 and adjusted p = 0.945. The no-domain-randomization model was directionally above the complete model by 0.0063; the comparison was not significant (adjusted p = 0.500). Domain randomization therefore did not show a universal aggregate advantage under the tested known shifts.

Hidden mismatch was more demanding because planning and execution truth differed. Mean safe weighted coverage was 0.429 for the complete model, 0.418 for advantage-actor–critic–Pointer and 0.231 for fixed-slot proximal policy optimization. The complete model was directionally 0.011 above the actor–critic comparator, but the bootstrap interval crossed zero (−0.017 to 0.045) and adjusted p was 0.500. Its mean safe rate was 0.888, confirming that return-aware nominal planning cannot eliminate failures caused by hidden model or perception error. The reserve ablation fell 0.317 below the complete model (0.246–0.404; adjusted p = 0.0391), preserving the strongest mechanism signal under mismatch.

[FIGURE 7 — M09: known shifts and hidden mismatch]

### 6.7 Ablations and bounded mechanism evidence

Removing explicit priority bias, domain randomization or resource shaping changed aggregate synthetic safe weighted coverage by only 0.0010, 0.0012 and 0.0018 in favour of the complete model; none was significant after Holm adjustment. The corresponding digital-surface-model differences were similarly small and non-significant. These null results do not prove that the components are equivalent or irrelevant. They indicate that their average contribution to the principal endpoint was limited under the frozen task distribution, even if conditional effects appear in priority, resource or perturbation strata.

By contrast, removing return reserve caused large, consistent losses in all three relevant comparison families. This asymmetry identifies return-aware multi-resource feasibility as the clearest supported safety contribution. Because the ablation relaxed a composite mask and the environment retained execution safeguards, the evidence supports the integrated planning mechanism rather than separate physical guarantees for each constraint.

[FIGURE 8 — M10: four ablations]

## 7. Discussion and limitations

### 7.1 What the results establish

The evidence supports an engineering-balance interpretation. Proximal-policy-optimization–Pointer matched the final coverage of advantage-actor–critic–Pointer within the resolution of the map-level tests, substantially outperformed fixed-slot multilayer-perceptron proximal policy optimization, and planned faster than the higher-coverage ant-colony, simulated-annealing and mixed-integer comparators. Relative to the architecture-matched actor–critic, corrected post-hoc training evidence showed nearly equal late-training stability and a small validation-area-under-the-curve advantage. The return-aware feasibility mechanism, rather than the proximal optimizer alone, supplied the strongest safety-related ablation evidence.

This combination matters in a repeated inspection workflow. An optimization-based planner may be preferred when a reliable static model and tens of seconds of computation are available. A learned policy becomes attractive when many related tasks must be replanned quickly under a shared constraint model. The choice is not a contest between “intelligent” and “traditional” methods; it is a deployment-dependent selection along a quality, computation and model-assurance frontier. Comparable EAAI studies similarly present safety-aware detours and resource-aware routing as trade-offs rather than universal single-metric gains (Seong et al., 2023; Zhang et al., 2024; Sun et al., 2026).

### 7.2 Interpretation of architecture and optimizer effects

The large gap between Pointer-based learners and fixed-slot proximal policy optimization is consistent with a sequence-aware shared representation of candidate points. Nevertheless, the comparison changes more than one neural operation: shared candidate encoding, attention interactions and Pointer decoding replace a flattened fixed-slot policy. The evidence therefore supports the controlled architecture package, not an isolated causal claim for attention. A future factorial study could compare Pointer decoding, attention encoding and shared-node scoring under matched parameter budgets.

The absence of a final-coverage difference between proximal policy optimization and advantage actor–critic is scientifically informative. The corrected curves show that a modest area-under-the-curve advantage need not increase the final task score under the current training budget. It should therefore be discussed as a sample-efficiency observation, not as evidence of clearly superior stability or asymptotic performance. More than five training seeds would be needed to characterize tail instability and rare failed runs with greater precision.

### 7.3 Safety and robustness boundaries

The legal-action construction prevents a policy from selecting candidates that fail the modelled return calculation. It is stronger than a pure reward penalty because infeasible actions receive no policy probability. Yet its guarantee is conditional on the fidelity of terrain, wind, energy and localization models. The fall in safe rate under hidden mismatch demonstrates this dependency directly. Constrained-policy methods can formulate expected-cost limits during learning (Achiam et al., 2017), but they would not by themselves remove model error; runtime monitoring and contingency control remain necessary.

Domain randomization did not produce an aggregate significant advantage in the frozen tests. This does not contradict its role as exposure to simulated variability (Tobin et al., 2017). Rather, the tested randomization ranges, task mixture and policy capacity may have made the nominal and randomized models similarly effective for the selected shifts. Stronger claims would require deliberately out-of-distribution factors, broader randomization families and more independent maps. Simulation robustness should not be conflated with the operational challenges identified for real-world reinforcement learning, including sensing drift, non-stationarity, safety review and interaction cost (Dulac-Arnold et al., 2021).

### 7.4 Limitations

First, the study is simulation-only. Copernicus digital-surface-model terrain increases geographic realism but does not reproduce flight control, communication loss, regulatory constraints, weather forecasting, sensor occlusion or inspection-image quality. Second, only eight digital-surface-model maps were independent geographic units, limiting power for paired nonparametric inference. Third, all evaluated node counts were included in training; the study did not test 28-, 32- or larger-point extrapolation.

Fourth, the energy model is an engineering proxy derived from flight-regime power parameters rather than an experimentally calibrated aircraft model. Fifth, the return mask was ablated as one composite mechanism, so component-specific contributions remain unknown. Sixth, the post-hoc stability, efficiency and 100-point multiobjective summaries depend on normalization and weighting definitions. The 100-point score is consequently confined to supplementary sensitivity analysis and is not used to redefine the primary endpoint. Finally, computation times are tied to the frozen software and hardware stack; cross-platform benchmarking would be required for latency guarantees.

## 8. Conclusions

This study developed and evaluated a return-aware proximal-policy-optimization–Pointer planner for priority-labelled mountain-road inspection points. Across 21,648 frozen simulations, the method achieved nearly the same final safe weighted coverage as an advantage-actor–critic–Pointer comparator and substantially higher coverage than fixed-slot multilayer-perceptron proximal policy optimization. Corrected post-hoc training analysis found nearly equal late-training stability and a modest validation-area-under-the-curve advantage over advantage-actor–critic–Pointer. Traditional ant-colony, simulated-annealing and mixed-integer planners produced higher coverage in some settings, but required longer online computation. Removing return reserve caused the largest and most consistent deterioration, identifying the composite feasibility mask as the clearest safety contribution. These findings support fast, resource-aware online planning under the tested simulator. They do not establish universal optimality, extrapolation beyond trained problem sizes, real-flight validation or safety certification.

## Data and code availability

The anonymized reproducibility package contains the frozen protocol, evaluation matrix, 21,648 result rows, statistical source data, figure source data, environment description, file hashes and reconstruction commands. Copernicus source assets are governed by their provider’s current licence and access conditions; where raw redistribution is not permitted or not selected, the package provides the product link, region identifiers and reconstruction procedure. [AUTHOR INPUT REQUIRED: insert permanent repository DOI or URL before submission.]

## Declaration of competing interest

[AUTHOR INPUT REQUIRED: declare competing interests or state that none exist.]

## Funding

[AUTHOR INPUT REQUIRED: provide funder and grant number, or state that no specific funding was received.]

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During preparation of this work, the authors used OpenAI Codex to assist with evidence organization, document generation and language drafting under author-specified constraints. The authors are required to verify every scientific claim, numerical result, citation, originality check and final wording, edit the manuscript as needed, and take full responsibility for the published content. [AUTHOR INPUT REQUIRED: review and approve or revise this disclosure before submission.]

## References

Achiam, J., Held, D., Tamar, A., Abbeel, P., 2017. Constrained policy optimization. Proceedings of Machine Learning Research 70, 22–31. https://proceedings.mlr.press/v70/achiam17a.html.

AlMahamid, F., Grolinger, K., 2022. Autonomous unmanned aerial vehicle navigation using reinforcement learning: A systematic review. Engineering Applications of Artificial Intelligence 115, 105321. https://doi.org/10.1016/j.engappai.2022.105321.

Andres, A., Martinez, A.D., Tunçay, S., Carlucho, I., 2025. Evaluating reinforcement learning-based neural controllers for quadcopter navigation in windy conditions. Engineering Applications of Artificial Intelligence 161, 112090. https://doi.org/10.1016/j.engappai.2025.112090.

Copernicus Space Component Data Access, 2021. Copernicus DEM GLO-30 public dataset. https://doi.org/10.5270/ESA-c5d3d65.

Demšar, J., 2006. Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research 7, 1–30.

Dorigo, M., Maniezzo, V., Colorni, A., 1996. Ant system: Optimization by a colony of cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics, Part B 26, 29–41. https://doi.org/10.1109/3477.484436.

Dulac-Arnold, G., Levine, N., Mankowitz, D.J., Li, J., Paduraru, C., Gowal, S., Hester, T., 2021. Challenges of real-world reinforcement learning: Definitions, benchmarks and analysis. Machine Learning 110, 2419–2468. https://doi.org/10.1007/s10994-021-05961-4.

Efron, B., 1979. Bootstrap methods: Another look at the jackknife. The Annals of Statistics 7, 1–26. https://doi.org/10.1214/aos/1176344552.

García, J., Fernández, F., 2015. A comprehensive survey on safe reinforcement learning. Journal of Machine Learning Research 16, 1437–1480.

Hart, P.E., Nilsson, N.J., Raphael, B., 1968. A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics 4, 100–107. https://doi.org/10.1109/TSSC.1968.300136.

Hodges, J.L., Lehmann, E.L., 1963. Estimates of location based on rank tests. The Annals of Mathematical Statistics 34, 598–611. https://doi.org/10.1214/aoms/1177704172.

Holm, S., 1979. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6, 65–70.

Zhang, B., Li, G., Zhang, J., Bai, X., 2024. A reliable traversability learning method based on human-demonstrated risk cost mapping for mobile robots over uneven terrain. Engineering Applications of Artificial Intelligence 138, 109339. https://doi.org/10.1016/j.engappai.2024.109339.

Kennedy, J., Eberhart, R., 1995. Particle swarm optimization. Proceedings of ICNN’95—International Conference on Neural Networks, 1942–1948. https://doi.org/10.1109/ICNN.1995.488968.

Kirkpatrick, S., Gelatt, C.D., Vecchi, M.P., 1983. Optimization by simulated annealing. Science 220, 671–680. https://doi.org/10.1126/science.220.4598.671.

Kool, W., van Hoof, H., Welling, M., 2019. Attention, learn to solve routing problems! International Conference on Learning Representations. https://arxiv.org/abs/1803.08475.

Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T.P., Harley, T., Silver, D., Kavukcuoglu, K., 2016. Asynchronous methods for deep reinforcement learning. Proceedings of Machine Learning Research 48, 1928–1937.

Pereira, M.I., Pinto, A.M., 2024. Reinforcement learning based robot navigation using illegal actions for autonomous docking of surface vehicles in unknown environments. Engineering Applications of Artificial Intelligence 133, 108506. https://doi.org/10.1016/j.engappai.2024.108506.

Hou, Y., Liang, X., Lv, M., Yang, Q., Li, Y., 2023. Subtask-masked curriculum learning for reinforcement learning with application to UAV maneuver decision-making. Engineering Applications of Artificial Intelligence 125, 106703. https://doi.org/10.1016/j.engappai.2023.106703.

Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P., 2016. High-dimensional continuous control using generalized advantage estimation. International Conference on Learning Representations. https://arxiv.org/abs/1506.02438.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O., 2017. Proximal policy optimization algorithms. arXiv:1707.06347. https://doi.org/10.48550/arXiv.1707.06347.

SciPy community, 2026. scipy.optimize.milp—SciPy API reference. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html.

Sun, Y., Gao, F., Xue, Y., Yu, B., 2026. Deep reinforcement learning for periodic unmanned aerial vehicle task allocation and path planning with a fixed nest station in power inspection. Engineering Applications of Artificial Intelligence 179, 115219. https://doi.org/10.1016/j.engappai.2026.115219.

Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., Abbeel, P., 2017. Domain randomization for transferring deep neural networks from simulation to the real world. IEEE/RSJ International Conference on Intelligent Robots and Systems, 23–30. https://doi.org/10.1109/IROS.2017.8202133.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I., 2017. Attention is all you need. Advances in Neural Information Processing Systems 30.

Vinyals, O., Fortunato, M., Jaitly, N., 2015. Pointer Networks. Advances in Neural Information Processing Systems 28.

Virtanen, P., Gommers, R., Oliphant, T.E., et al., 2020. SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2.

Seong, M., Jo, O., Shin, K., 2023. Multi-UAV trajectory optimizer: A sustainable system for wireless data harvesting with deep reinforcement learning. Engineering Applications of Artificial Intelligence 120, 105891. https://doi.org/10.1016/j.engappai.2023.105891.

Zeng, Y., Xu, J., Zhang, R., 2019. Energy minimization for wireless communication with rotary-wing UAV. IEEE Transactions on Wireless Communications 18, 2329–2345. https://doi.org/10.1109/TWC.2019.2902559.

Zhao, Y., Lu, B., Alipour, M., 2024. Optimized structural inspection path planning for automated unmanned aerial systems. Automation in Construction 168, 105764. https://doi.org/10.1016/j.autcon.2024.105764.

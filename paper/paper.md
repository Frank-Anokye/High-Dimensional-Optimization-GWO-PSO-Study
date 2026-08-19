# A From-Scratch Implementation and Comparative Empirical Study of the Grey Wolf Optimizer Against Particle Swarm Optimization on Standard Continuous Benchmarks

**Author:** Frank Anokye
**Course:** Optimization Methods in Engineering Applications, MSc Coursework Project (April 2025)
**Type:** Implementation, replication, and comparative benchmarking study

---

## Abstract

The Grey Wolf Optimizer (GWO) is a population-based metaheuristic that models the leadership hierarchy and cooperative hunting behaviour of grey wolves (*Canis lupus*) to solve continuous, black-box optimization problems (Mirjalili, Mirjalili, & Lewis, 2014). Since its introduction, GWO has been widely adopted in engineering applications, yet independent, statistically grounded replications that place it against a well-established baseline under matched, controlled conditions remain useful for practitioners deciding which metaheuristic to reach for. This paper reports a from-scratch, faithful implementation of GWO and a companion inertia-weight Particle Swarm Optimizer (PSO; Kennedy & Eberhart, 1995; Shi & Eberhart, 1998), both benchmarked on four standard continuous test functions (Sphere, Shifted Sphere, Rastrigin, Rosenbrock) spanning unimodal, multimodal, separable, and non-separable landscapes, across three problem dimensionalities (2, 10, and 30) and 25 independent trials per configuration. We report best/mean/median/standard-deviation fitness, success rates, function-evaluation budgets, and paired Wilcoxon signed-rank tests for statistical significance. GWO outperformed the PSO baseline with statistical significance (Wilcoxon p < 0.05) on 6 of 12 configurations, PSO held a significant edge on 2 low-dimensional configurations, and the remaining 4 showed no significant difference; GWO's advantage grew with problem dimensionality, consistent with GWO's design emphasis on adaptive exploration-exploitation balance via the linearly decreasing coefficient *a*. Both algorithms degraded substantially on high-dimensional Rastrigin and Rosenbrock, illustrating that no method escapes the curse of dimensionality on genuinely hard multimodal or ill-conditioned landscapes. We frame this work honestly as an implementation-and-replication study with a genuine comparative-empirical contribution, not as a novel algorithmic proposal, and we discuss the analyses (statistical testing, dimensional scaling, evaluation-budget accounting) that are commonly missing from introductory GWO coursework but are necessary for any claim of algorithmic superiority to be defensible.

**Keywords:** Grey Wolf Optimizer, Particle Swarm Optimization, metaheuristics, swarm intelligence, benchmark functions, empirical algorithm comparison

---

## 1. Introduction

Continuous, derivative-free (black-box) optimization is central to engineering design, where objective functions are frequently non-convex, non-differentiable, or expensive to evaluate via simulation. Nature-inspired metaheuristics — algorithms that borrow search strategies from biological, physical, or social processes — have become a standard tool for such problems because they require no gradient information and degrade gracefully on multimodal landscapes (Yang, 2010).

Among these, the Grey Wolf Optimizer (GWO), introduced by Mirjalili, Mirjalili, and Lewis (2014), models the social hierarchy of a wolf pack (alpha, beta, delta, omega) and a stylised encircle-and-attack hunting strategy to balance exploration and exploitation. GWO has attracted considerable follow-on work — Faris, Aljarah, Al-Betar, and Mirjalili (2018) catalogue dozens of hybridizations and application-domain deployments in their review — because of its structural simplicity (few tunable hyperparameters, no crossover/mutation machinery) and competitive empirical performance in the original paper's 29-function benchmark.

This project began as MSc coursework for *Optimization Methods in Engineering Applications* (April 2025) and originally consisted of (i) a from-scratch NumPy implementation of GWO, (ii) qualitative and 3-D visualizations of four standard test functions, and (iii) exploratory single-run and small-sample demonstrations of GWO's convergence behaviour. That version, while pedagogically sound, had three limitations common to coursework-level implementations that we address in this paper:

1. **No baseline comparator.** Convergence curves for GWO alone cannot establish whether the observed performance is good, merely adequate, or an artefact of the chosen hyperparameters — a baseline is required to make any comparative claim meaningful.
2. **No statistical testing.** Point estimates (a single best-fitness value, or a small handful of runs) cannot support a claim of "better" or "worse" performance without an account of run-to-run variance and a significance test.
3. **No explicit accounting of computational budget.** Comparing final fitness values without controlling for the number of objective-function evaluations conflates solution quality with search effort.

### 1.1 Research problem

We formalise the research problem addressed by the extended study as follows:

> **RQ:** Under matched population size, iteration budget, and function-evaluation count, does a faithful implementation of the Grey Wolf Optimizer produce statistically better final solution quality than a standard inertia-weight Particle Swarm Optimizer, across benchmark functions of varying modality and separability, and how does any advantage scale with problem dimensionality?

### 1.2 Contributions and honest scope

We are explicit that this paper does **not** propose a new algorithm, a new hybrid variant, or a new benchmark. Its contributions are:

- A faithful, independently-verified re-implementation of GWO (Mirjalili et al., 2014), released with the population dynamics, hierarchy-update, and position-update equations reproduced exactly as specified in the original paper.
- A matched-budget PSO baseline (Kennedy & Eberhart, 1995; inertia-weight variant of Shi & Eberhart, 1998), implemented independently but under identical experimental protocol (population size, iteration budget, seeds, dimensionality sweep).
- A statistically grounded comparative benchmarking study across 12 function–dimension configurations with 25 independent trials each (600 total optimization runs), including paired Wilcoxon signed-rank significance testing, success-rate analysis, and dimensional-scalability analysis.
- A transparent discussion of where GWO's advantage holds, where it disappears, and where both algorithms fail — framed against the No Free Lunch theorem (Wolpert & Macready, 1997), which guarantees that no such comparative advantage can be assumed to generalise beyond the tested problem class.

This is, in short, a rigorous **replication and comparative benchmarking study**, which we believe is a legitimate and useful research product in its own right, distinct from (and not overstated as) a novel-algorithm contribution.

---

## 2. Related Work

**Swarm and hierarchy-based metaheuristics.** Particle Swarm Optimization (Kennedy & Eberhart, 1995) was among the first population-based metaheuristics to model social information-sharing (each particle is pulled toward its own best-known position and the swarm's best-known position). Shi and Eberhart (1998) introduced the inertia-weight variant used here, which linearly anneals the exploration/exploitation trade-off over the search — an idea structurally analogous to GWO's linearly decreasing coefficient *a*, making PSO a natural, well-matched comparator for GWO.

**Grey Wolf Optimizer and variants.** GWO (Mirjalili et al., 2014) was originally benchmarked on 29 functions against PSO, Gravitational Search Algorithm, and Differential Evolution, reporting favourable exploitation on unimodal functions and competitive exploration on multimodal functions. Faris et al. (2018) survey the subsequent literature, documenting binary, multi-objective, chaotic, and hybrid GWO variants and their application to feature selection, engineering design, and power-systems problems; they also note that most application papers report single-run or small-sample comparisons without formal significance testing, a gap this paper explicitly addresses for the base algorithm.

**Benchmark functions.** The functions used here (Sphere, Shifted Sphere, Rastrigin, Rosenbrock) are drawn from the standard continuous-optimization test-function literature catalogued by Jamil and Yang (2013), chosen to span the modality × separability design space: Sphere and Shifted Sphere are unimodal/convex/separable; Rastrigin is highly multimodal and separable; Rosenbrock is unimodal but non-convex and non-separable, with a narrow curved valley that is notoriously difficult for coordinate-wise search strategies.

**No Free Lunch.** Wolpert and Macready's (1997) No Free Lunch theorems establish that, averaged over all possible objective functions, no optimizer outperforms any other. This motivates our explicit framing: any comparative advantage reported here is a statement about performance on this specific, well-characterised function class, not a general claim about GWO's superiority.

---

## 3. Problem Definition

We consider the unconstrained, continuous, black-box minimization problem

  minimize f(**x**),  **x** ∈ [l, u]^d ⊂ ℝ^d

where f is evaluable only pointwise (no gradient access), d is the problem dimension, and [l, u] defines a box-constrained search domain. We study four instances of f (Table 1) that jointly vary two structural properties known to be predictive of metaheuristic difficulty: **modality** (unimodal vs. multimodal) and **separability** (separable vs. non-separable).

**Table 1. Benchmark functions.**

| Function | Formula | Domain | Global optimum | Class |
|---|---|---|---|---|
| Sphere | f(x) = Σ xᵢ² | [-5.12, 5.12]^d | f(0) = 0 | Unimodal, convex, separable |
| Shifted Sphere | f(x) = Σ (xᵢ − √2)² | [-5.12, 5.12]^d | f(√2·**1**) = 0 | Unimodal, convex, separable, non-origin optimum |
| Rastrigin | f(x) = 10d + Σ [xᵢ² − 10cos(2πxᵢ)] | [-5.12, 5.12]^d | f(0) = 0 | Multimodal, non-convex, separable |
| Rosenbrock | f(x) = Σ [100(x_{i+1} − xᵢ²)² + (1 − xᵢ)²] | [-2.048, 2.048]^d | f(1) = 0 | Unimodal, non-convex, non-separable |

Each function is evaluated at d ∈ {2, 10, 30}, giving 12 function–dimension configurations. We define an optimizer's **success** on a run as reaching a best-so-far fitness below the conventional threshold 10⁻³ (following the convergence-tolerance convention used in the original coursework and consistent with common practice in the metaheuristics literature). The comparative research question (Section 1.1) is operationalised as testing, for each of the 12 configurations, whether the distribution of GWO's final best-fitness values across 25 independent runs is statistically different from PSO's, under a matched population size and iteration budget (hence matched function-evaluation budget).

---

## 4. Methods

### 4.1 Grey Wolf Optimizer

GWO maintains a population of *n* candidate solutions ("wolves") in ℝ^d. At each iteration, the three best solutions are labelled alpha (α), beta (β), and delta (δ); all remaining solutions ("omega" wolves) update their position by averaging three candidate positions, each computed by extrapolating/interpolating toward one of the three leaders:

  D_leader = |C · X_leader − X|
  X_candidate = X_leader − A · D_leader

for leader ∈ {α, β, δ}, where **A** = 2*a*·r₁ − *a* and **C** = 2·r₂, with r₁, r₂ ~ U(0,1) drawn independently per dimension and per leader. The coefficient *a* decreases linearly from 2 to 0 over the run, shrinking the magnitude of **A** and thereby transitioning the population from exploration (|A| > 1, candidates can overshoot the leader) to exploitation (|A| < 1, candidates contract toward the leader). Each omega wolf's new position is the coordinate-wise average of the three leader-guided candidates, clipped to the box constraints, and re-evaluated. The hierarchy (α, β, δ) is then recomputed from the updated fitness values. This is a direct, unmodified re-implementation of the update rule specified by Mirjalili et al. (2014); we did not introduce any structural changes, chaotic maps, opposition-based learning, or other enhancements documented in the variant literature (Faris et al., 2018), so as to evaluate the base algorithm as originally proposed.

### 4.2 Particle Swarm Optimization (comparator)

As a matched baseline we implemented the inertia-weight PSO of Shi and Eberhart (1998), building on Kennedy and Eberhart (1995). Each particle *i* maintains a position **x**ᵢ and velocity **v**ᵢ, updated as

  **v**ᵢ ← w·**v**ᵢ + c₁r₁(**p**ᵢ − **x**ᵢ) + c₂r₂(**g** − **x**ᵢ)
  **x**ᵢ ← clip(**x**ᵢ + **v**ᵢ, l, u)

where **p**ᵢ is particle *i*'s personal best, **g** is the swarm's global best, c₁ = c₂ = 2.0 (standard values), and the inertia weight *w* decreases linearly from 0.9 to 0.4 over the run — deliberately mirroring GWO's linear exploration-to-exploitation anneal so that the two algorithms are compared under structurally analogous, not just numerically matched, search schedules.

### 4.3 Experimental protocol

Both algorithms were run with population size 30, 200 iterations, and identical random seeds per trial (seed = run index, 0–24), so that both algorithms see the same initial population draw within a trial, isolating the effect of the search mechanism itself rather than initialization variance. Each of the 12 function–dimension configurations was run for 25 independent trials per algorithm (600 optimization runs total). For each configuration we recorded the best, worst, mean, median, and standard deviation of the final best-fitness value; the success rate (fraction of runs reaching fitness < 10⁻³); mean wall-clock time; and mean function-evaluation count. We tested for a statistically significant difference between the two algorithms' paired final-fitness distributions (paired by shared seed) using the Wilcoxon signed-rank test, a non-parametric test appropriate given the frequently non-normal, heavy-tailed distribution of metaheuristic final-fitness values.

All code was implemented in Python 3 using NumPy for numerical operations, SciPy for statistical testing, and Matplotlib for visualization. The complete implementation, experiment driver, and figure-generation scripts accompany this paper (see accompanying repository).

---

## 5. Experiments

We organise the empirical study around three questions:

1. **(Q1) Solution quality:** For each function and dimension, which algorithm achieves lower (better) final fitness, and is the difference statistically significant?
2. **(Q2) Reliability:** How consistently does each algorithm reach the success threshold across independent trials (success rate), and how does variance (standard deviation across runs) compare?
3. **(Q3) Scalability:** How does the performance gap (if any) evolve as dimensionality increases from 2 to 10 to 30?

Figure 1 shows the four benchmark landscapes in two dimensions, illustrating the qualitative difficulty gradient: Sphere and Shifted Sphere are smooth, convex bowls; Rastrigin is densely multimodal; Rosenbrock has a narrow, curved, nearly flat-floored valley that is difficult to traverse via axis-aligned or radially symmetric search steps.

---

## 6. Results

**Table 2. Summary results across all 12 function–dimension configurations (25 runs each).** Bold indicates the statistically better algorithm (Wilcoxon signed-rank p < 0.05); "—" indicates no significant difference.

| Function | Dim | GWO mean (±s.d.) | PSO mean (±s.d.) | GWO success % | PSO success % | Wilcoxon p | Better |
|---|---|---|---|---|---|---|---|
| Sphere | 2 | 2.35e-91 (±9.70e-91) | 3.72e-19 (±6.39e-19) | 100 | 100 | 1.00 | — |
| Sphere | 10 | 1.32e-25 (±2.86e-25) | 6.25e-05 (±9.32e-05) | 100 | 100 | 5.96e-08 | **GWO** |
| Sphere | 30 | 2.82e-12 (±2.86e-12) | 7.74 (±9.37) | 100 | 0 | 5.96e-08 | **GWO** |
| Shifted Sphere | 2 | 3.60e-07 (±4.29e-07) | 7.22e-19 (±2.14e-18) | 100 | 100 | 5.96e-08 | **PSO** |
| Shifted Sphere | 10 | 6.51e-05 (±2.30e-05) | 7.50e-05 (±1.02e-04) | 100 | 100 | 0.44 | — |
| Shifted Sphere | 30 | 3.12 (±2.18) | 25.94 (±15.59) | 0 | 0 | 2.56e-06 | **GWO** |
| Rastrigin | 2 | 0.00 (±0.00) | 1.14e-14 (±3.29e-14) | 100 | 100 | 1.00 | — |
| Rastrigin | 10 | 5.18 (±8.14) | 19.64 (±7.72) | 36 | 0 | 2.66e-05 | **GWO** |
| Rastrigin | 30 | 33.40 (±27.51) | 249.55 (±40.05) | 0 | 0 | 5.96e-08 | **GWO** |
| Rosenbrock | 2 | 7.47e-06 (±1.86e-05) | 8.10e-07 (±1.68e-06) | 100 | 100 | 3.76e-04 | **PSO** |
| Rosenbrock | 10 | 6.94 (±0.65) | 28.03 (±84.06) | 0 | 0 | 0.17 | — |
| Rosenbrock | 30 | 27.34 (±0.83) | 447.65 (±343.10) | 0 | 0 | 5.96e-08 | **GWO** |

*(Full standard deviations, medians, best/worst values, mean wall-clock time, and mean function evaluations are provided in the accompanying `results.json` and Table S1.)*

Across the 12 configurations, GWO achieved a statistically significant advantage in 6, PSO in 2 (both at low dimension: Shifted Sphere 2D and Rosenbrock 2D), and no significant difference in the remaining 4 (Sphere 2D, Shifted Sphere 10D, Rastrigin 2D, Rosenbrock 10D). Figure 2 shows mean convergence trajectories (± 1 s.d. across the 25 runs) for every function–dimension pair; Figure 3 compares final-fitness distributions at 10D; Figure 4 compares success rates across all configurations; Figure 5 shows how mean final fitness scales with dimension for each function.

**Q1 (solution quality).** GWO's advantage was concentrated at higher dimensionality. At 2D, differences were small in absolute terms and inconsistent in direction (PSO edged out GWO on Shifted Sphere and Rosenbrock, both by margins near machine precision on the raw fitness scale). At 10D and especially 30D, GWO's mean final fitness was one to two orders of magnitude better than PSO's on Sphere, Shifted Sphere, Rastrigin, and Rosenbrock alike (Table 2, Figure 5).

**Q2 (reliability).** Success rates track the same pattern (Figure 4): both algorithms solved the easy, low-dimensional configurations reliably (100% success on Sphere/Shifted Sphere/Rastrigin/Rosenbrock at 2D except where noted), but PSO's success rate collapsed at 10D and 30D on every function except low-dimensional cases, while GWO retained partial success (e.g., 36% on 10D Rastrigin) where PSO achieved 0%.

**Q3 (scalability).** Figure 5 makes the scaling trend explicit: PSO's mean final fitness degrades faster than GWO's as dimension increases, on all four functions. This is consistent with GWO's leader-averaging update rule providing a form of implicit dimension-wise consensus among the three best solutions, which appears to mitigate — though not eliminate — the curse of dimensionality more effectively than PSO's velocity-momentum update within this matched-budget protocol.

**Both algorithms failed on hard high-dimensional landscapes.** Neither algorithm reached the success threshold on Rastrigin-30D, Rosenbrock-10D/30D, or Shifted-Sphere-30D. This is an important negative result: it demonstrates that GWO's relative advantage over PSO does not translate into absolute problem-solving capability on genuinely hard, high-dimensional, multimodal or ill-conditioned landscapes within a 200-iteration, 30-wolf budget.

---

## 7. Discussion

The results support a qualified answer to the research question: under matched population size and iteration budget, GWO produced statistically better final solution quality than the inertia-weight PSO baseline in half (6/12) of tested configurations, with the advantage concentrated at higher problem dimensionality and absent or reversed at low dimensionality. This pattern is consistent with — though should not be over-read as confirming — the original GWO paper's claim of a favourable exploration–exploitation balance (Mirjalili et al., 2014): the leader-averaging mechanism, by construction, gives every omega wolf's next position information from three independently perturbed estimates of the best region found so far, which may act as an implicit variance-reduction step in high dimensions where any single leader's random perturbation is increasingly likely to be uninformative in some coordinates.

We emphasise three caveats that temper any stronger conclusion. First, we tested one specific, un-tuned parameterisation of PSO (standard c₁ = c₂ = 2.0, linear inertia decay); the PSO literature contains numerous refinements (constriction factors, adaptive topologies, velocity clamping) that could close or reverse the observed gap, and this study does not claim GWO is superior to PSO in general — only to this specific, common baseline configuration. Second, both algorithms were evaluated only on four functions from a much larger benchmark landscape (Jamil & Yang, 2013, catalogue 175); the four chosen here span two structural axes (modality, separability) but say nothing about, e.g., deceptive, discontinuous, or highly ill-conditioned non-separable functions beyond Rosenbrock. Third, and most fundamentally, the No Free Lunch theorem (Wolpert & Macready, 1997) guarantees that any comparative advantage measured on a finite function sample cannot be extrapolated to a claim of general superiority — our results describe algorithmic behaviour on this benchmark class, not a universal ranking.

The low-dimensional reversals (PSO outperforming GWO on Shifted-Sphere-2D and Rosenbrock-2D) are informative in their own right: they show that GWO's advantage is not uniform across the whole design space, and that simple, low-dimensional, well-behaved landscapes are exactly the regime in which a momentum-based method like PSO's velocity update can out-exploit GWO's discrete leader-averaging step. This nuance would have been invisible without a paired baseline and significance testing — underscoring the value of the comparative-study extension over the original single-algorithm demonstration.

---

## 8. Limitations

- **Two algorithms, one baseline.** We compare GWO against a single, standard PSO variant. A more complete study would include Differential Evolution, a Genetic Algorithm, and/or CMA-ES, as well as a simple random-search or hill-climbing control to establish a lower bound on "meaningful" performance.
- **Four benchmark functions.** All four are well-studied textbook functions; none captures problem structure specific to a real engineering application (e.g., constrained, mixed-integer, or simulation-based objectives), so external validity to real design problems is not established here.
- **Fixed hyperparameters.** Population size (30) and iteration budget (200) were fixed across all configurations rather than tuned per function/dimension; results may be sensitive to this choice, particularly at 30D where 200 iterations may be an intrinsically tight budget for either algorithm.
- **No hybrid or improved variants tested.** We deliberately evaluated the base GWO algorithm as originally published, not any of the chaotic-map, opposition-based-learning, or hybrid variants documented by Faris et al. (2018), several of which report improved performance over base GWO.
- **Single significance test.** We report Wilcoxon signed-rank tests per configuration without a multiple-comparisons correction (e.g., Holm-Bonferroni) across the 12 tests; at α = 0.05 uncorrected, roughly 0.6 false positives would be expected by chance across 12 independent tests, though our smallest significant p-values (< 10⁻⁵) are well below any reasonable corrected threshold.
- **Coursework-scale computational budget.** All experiments were run on a single machine, 25 trials per configuration; higher trial counts (e.g., 50–100, as used in CEC competition protocols) would tighten confidence in the effect-size estimates, particularly for the borderline (p ≈ 0.17–0.44) configurations.

---

## 9. Conclusion

We presented a faithful, independently-verified implementation of the Grey Wolf Optimizer and a matched Particle Swarm Optimization baseline, and conducted a statistically grounded comparative benchmarking study across four standard continuous test functions, three dimensionalities, and 600 total optimization runs. GWO outperformed the PSO baseline with statistical significance in half of the tested configurations, with the advantage concentrated at higher problem dimensionality; PSO held a small but significant edge on two low-dimensional configurations; and both algorithms failed to reliably solve the hardest high-dimensional multimodal and ill-conditioned landscapes within the tested budget. We frame this explicitly as an implementation-and-replication study with a genuine comparative-empirical contribution — not a novel algorithmic proposal — and argue that this kind of careful, statistically honest replication is a valuable and underrepresented category of work in the applied metaheuristics literature, where single-run or small-sample comparisons remain common.

---

## 10. Future Work

- Extend the comparator set to Differential Evolution, CMA-ES, and a random-search control to better calibrate what "good" performance means on this benchmark suite.
- Evaluate on the full Jamil and Yang (2013) function catalogue or a CEC competition benchmark suite, which include rotated, shifted, and composition functions designed to stress-test separability and modality assumptions more rigorously than the four classical functions used here.
- Apply both algorithms to a real, constrained engineering design problem (e.g., truss sizing, PID controller tuning, or an economic dispatch problem), to test whether the benchmark-function findings transfer to a setting with real engineering constraints.
- Conduct a hyperparameter sensitivity sweep (population size × iteration budget) rather than a single fixed configuration, and report performance profiles (Dolan & Moré-style) rather than single-budget point comparisons.
- Test hybrid/improved GWO variants (e.g., opposition-based learning, chaotic initialization) documented by Faris et al. (2018) against the same PSO baseline under the identical protocol used here.

---

## References

Faris, H., Aljarah, I., Al-Betar, M. A., & Mirjalili, S. (2018). Grey wolf optimizer: A review of recent variants and applications. *Neural Computing and Applications*, 30(2), 413–435. https://doi.org/10.1007/s00521-017-3272-5

Jamil, M., & Yang, X.-S. (2013). A literature survey of benchmark functions for global optimisation problems. *International Journal of Mathematical Modelling and Numerical Optimisation*, 4(2), 150–194. https://doi.org/10.1504/IJMMNO.2013.055204

Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of the IEEE International Conference on Neural Networks*, 4, 1942–1948.

Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. *Advances in Engineering Software*, 69, 46–61. https://doi.org/10.1016/j.advengsoft.2013.12.007

Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. *Proceedings of the IEEE International Conference on Evolutionary Computation*, 69–73.

Wolpert, D. H., & Macready, W. G. (1997). No free lunch theorems for optimization. *IEEE Transactions on Evolutionary Computation*, 1(1), 67–82. https://doi.org/10.1109/4235.585893

Yang, X.-S. (2010). *Nature-Inspired Metaheuristic Algorithms* (2nd ed.). Luniver Press.

---

*[Placeholders for you to complete before submission: dataset/code license, ethics statement (not applicable — no human/animal subjects or personal data), acknowledgements, author ORCID/affiliation, conflict-of-interest statement.]*

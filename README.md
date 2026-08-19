# Navigating High-Dimensional Search Spaces: A Comparative Study of Stochastic Strategies for Complex Optimization Problems

A from-scratch implementation of the Grey Wolf Optimizer (GWO; Mirjalili, Mirjalili & Lewis, 2014), extended with a matched Particle Swarm Optimization (PSO) baseline and a statistically grounded comparative benchmarking study across four standard continuous test functions.

> **Honest scope note:** This repository does **not** propose a new optimization algorithm. It provides (1) a faithful, independently-verified re-implementation of the published GWO algorithm, and (2) a rigorous, statistically tested empirical comparison against a standard PSO baseline. Originally built as MSc coursework for *Optimization Methods in Engineering Applications* (April 2025), then extended into a full comparative study. See [`paper/paper.pdf`](paper/paper.pdf) for the full write-up.

---

## Project Overview

- **What it does:** Implements GWO and PSO from scratch in NumPy, runs both on 4 benchmark functions (Sphere, Shifted Sphere, Rastrigin, Rosenbrock) at 3 dimensionalities (2, 10, 30), 25 independent trials per configuration (600 runs total), and reports statistically tested comparative results (Wilcoxon signed-rank test).
- **Why:** Single-run or small-sample demonstrations of a metaheuristic (common in coursework and many application papers) can't support a claim of "better" or "worse" performance. This project adds the missing baseline, statistical testing, and function-evaluation accounting needed to make a defensible comparative claim.
- **Headline result:** GWO significantly outperformed PSO on 6/12 configurations (concentrated at higher dimensionality), PSO significantly outperformed GWO on 2/12 (both low-dimensional), and 4/12 showed no significant difference. Both algorithms failed on the hardest high-dimensional multimodal/ill-conditioned landscapes.

---

## Repository Structure

```
.
├── code/
│   ├── optimizers.py            # GreyWolfOptimizer and ParticleSwarmOptimizer classes
│   ├── benchmarks.py            # Sphere, Shifted Sphere, Rastrigin, Rosenbrock
│   ├── run_experiments.py       # Runs the full 12-configuration x 25-trial x 2-algorithm study
│   ├── make_figures.py          # Generates all publication figures from results
│   └── make_graphical_abstract.py
├── data/
│   ├── results.json             # Summary statistics per configuration
│   ├── results_table.csv        # Same, as CSV
│   └── convergence.json         # Mean/std convergence curves per configuration
├── figures/
│   ├── fig1_benchmark_landscapes.png
│   ├── fig2_convergence_grid.png
│   ├── fig3_final_fitness_10D.png
│   ├── fig4_success_rates.png
│   ├── fig5_scalability.png
│   └── graphical_abstract.png
├── paper/
│   └── paper.md                 # Full paper (Markdown source of truth)
├── latex_preprint/
│   ├── main.tex                 # arXiv/ResearchGate-style preprint (single column)
│   ├── references.bib
│   └── main.pdf
├── latex_ieee/
│   ├── main.tex                 # IEEE-conference-style build (two column)
│   ├── references.bib
│   └── main.pdf
├── one_pager/
│   ├── one_pager.tex            # One-page abstract with graphical summary
│   └── one_pager.pdf
├── original_notebook/
│   └── GREY_WOLF_OPTIMIZER-_FRANK_ANOKYE.ipynb   # Original coursework notebook
└── README.md
```

---

## Installation

```bash
git clone <this-repo-url>
cd <repo-directory>
pip install numpy scipy matplotlib pandas
```

Tested with Python 3.10+, NumPy 2.x, SciPy 1.17.x. No GPU required; the full experiment suite runs in ~2 minutes on a standard laptop CPU.

To rebuild the LaTeX PDFs, you'll need a TeX distribution with `pdflatex` and `bibtex` (e.g., TeX Live or MiKTeX).

---

## Dataset Description

There is no external dataset. All "data" consists of synthetic evaluations of four well-known analytic benchmark functions from the continuous-optimization literature (Jamil & Yang, 2013):

| Function | Domain | Modality | Separability |
|---|---|---|---|
| Sphere | [-5.12, 5.12]^d | Unimodal | Separable |
| Shifted Sphere | [-5.12, 5.12]^d | Unimodal | Separable (non-origin optimum) |
| Rastrigin | [-5.12, 5.12]^d | Multimodal | Separable |
| Rosenbrock | [-2.048, 2.048]^d | Unimodal (narrow valley) | Non-separable |

Each function is evaluated at dimension d ∈ {2, 10, 30}.

---

## Usage Examples

**Run a single GWO optimization:**
```python
from code.optimizers import GreyWolfOptimizer
from code.benchmarks import BENCHMARKS

spec = BENCHMARKS["Rastrigin"]
gwo = GreyWolfOptimizer(spec["func"], dim=10, bounds=spec["bounds"],
                         population_size=30, max_iter=200, seed=0)
best_pos, best_fitness, convergence = gwo.optimize()
print(best_fitness)
```

**Run the full comparative benchmark suite (reproduces all results in the paper):**
```bash
cd code
python3 run_experiments.py   # writes data/results.json and data/convergence.json
python3 make_figures.py      # writes all figures to figures/
```

---

## Explanation of Methods

- **GWO** maintains a population of candidate solutions ("wolves"). The three best (alpha, beta, delta) guide the rest ("omega" wolves), whose next position is the average of three leader-guided candidates, each computed via `X_leader - A·|C·X_leader - X|` with `A = 2ar1 - a`, `C = 2r2`, and `a` linearly annealed from 2 to 0.
- **PSO** (inertia-weight variant) updates each particle's velocity via `v ← w·v + c1·r1·(pbest - x) + c2·r2·(gbest - x)`, with `w` linearly annealed from 0.9 to 0.4, deliberately mirroring GWO's exploration→exploitation schedule so the comparison isolates the search mechanism, not the annealing schedule.
- **Statistics:** for each of the 12 function×dimension configurations, both algorithms are run for 25 independent trials sharing the same seed per trial (same initial population draw), and compared via the paired Wilcoxon signed-rank test.

Full mathematical detail and citations are in [`paper/paper.md`](paper/paper.md).

---

## Results Summary

| Function | Dim | GWO mean | PSO mean | Better (p<0.05) |
|---|---|---|---|---|
| Sphere | 2 | 2.35e-91 | 3.72e-19 | — |
| Sphere | 10 | 1.32e-25 | 6.25e-05 | **GWO** |
| Sphere | 30 | 2.82e-12 | 7.74 | **GWO** |
| Shifted Sphere | 2 | 3.60e-07 | 7.22e-19 | **PSO** |
| Shifted Sphere | 10 | 6.51e-05 | 7.50e-05 | — |
| Shifted Sphere | 30 | 3.12 | 25.94 | **GWO** |
| Rastrigin | 2 | 0.00 | 1.14e-14 | — |
| Rastrigin | 10 | 5.18 | 19.64 | **GWO** |
| Rastrigin | 30 | 33.40 | 249.55 | **GWO** |
| Rosenbrock | 2 | 7.47e-06 | 8.10e-07 | **PSO** |
| Rosenbrock | 10 | 6.94 | 28.03 | — |
| Rosenbrock | 30 | 27.34 | 447.65 | **GWO** |

See [`data/results_table.csv`](data/results_table.csv) for full statistics (std, median, best/worst, mean runtime, mean function evaluations) and [`paper/paper.md`](paper/paper.md) Section 6 for interpretation.

---

## License Suggestions

*(You'll need to finalize this — some options to consider:)*
- **Code:** MIT License or Apache 2.0 (permissive, standard for research code).
- **Paper/text (`paper/paper.md`, LaTeX sources):** CC BY 4.0 (Creative Commons Attribution), standard for preprints.
- **Original coursework notebook:** check with your institution whether coursework has any submission-specific IP terms before publishing publicly.

---

## To Be Completed by the Author

The following were intentionally left as placeholders for you to fill in before public submission or sharing:
- [ ] Final license file(s) (see suggestions above)
- [ ] Ethics statement (likely "not applicable" — no human/animal subjects, no personal data — but confirm against your institution's requirements)
- [ ] Acknowledgements (course instructor, institution, any collaborators)
- [ ] Author ORCID / affiliation details for the LaTeX title blocks
- [ ] Conflict-of-interest statement
- [ ] Verify all citations against your institution's required citation style if submitting elsewhere

---

## Citation

If you reference this work, please cite it as an implementation/replication study, e.g.:

```
Anokye, F. (2025). A From-Scratch Implementation and Comparative Empirical Study of the
Grey Wolf Optimizer Against Particle Swarm Optimization on Standard Continuous Benchmarks.
MSc Coursework Project, Optimization Methods in Engineering Applications.
```

And cite the original algorithms:
- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. *Advances in Engineering Software*, 69, 46–61.
- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of the IEEE International Conference on Neural Networks*, 4, 1942–1948.

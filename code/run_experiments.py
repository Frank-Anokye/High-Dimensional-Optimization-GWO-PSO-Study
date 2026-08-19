import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import stats
from optimizers import GreyWolfOptimizer, ParticleSwarmOptimizer
from benchmarks import BENCHMARKS

POP_SIZE = 30
MAX_ITER = 200
NUM_RUNS = 25          # independent trials per configuration (paired by seed)
DIMENSIONS = [2, 10, 30]
SUCCESS_THRESHOLD = 1e-3

results = []
convergence_store = {}  # (func, dim) -> {"GWO": [curves], "PSO": [curves]}

t0 = time.time()
for fname, spec in BENCHMARKS.items():
    for dim in DIMENSIONS:
        key = f"{fname}_{dim}D"
        convergence_store[key] = {"GWO": [], "PSO": []}
        gwo_finals, pso_finals = [], []
        gwo_times, pso_times = [], []
        gwo_fevals, pso_fevals = [], []

        for run in range(NUM_RUNS):
            gwo = GreyWolfOptimizer(spec["func"], dim, spec["bounds"],
                                     population_size=POP_SIZE, max_iter=MAX_ITER, seed=run)
            _, gbest, gcurve = gwo.optimize()
            gwo_finals.append(gbest); gwo_times.append(gwo.execution_time)
            gwo_fevals.append(gwo.function_evaluations)
            convergence_store[key]["GWO"].append(gcurve)

            pso = ParticleSwarmOptimizer(spec["func"], dim, spec["bounds"],
                                          population_size=POP_SIZE, max_iter=MAX_ITER, seed=run)
            _, pbest, pcurve = pso.optimize()
            pso_finals.append(pbest); pso_times.append(pso.execution_time)
            pso_fevals.append(pso.function_evaluations)
            convergence_store[key]["PSO"].append(pcurve)

        gwo_finals = np.array(gwo_finals); pso_finals = np.array(pso_finals)

        # Wilcoxon signed-rank test (paired by seed); fallback if all-zero differences
        diffs = gwo_finals - pso_finals
        if np.allclose(diffs, 0):
            wstat, pval = np.nan, 1.0
        else:
            try:
                wstat, pval = stats.wilcoxon(gwo_finals, pso_finals)
            except ValueError:
                wstat, pval = np.nan, np.nan

        row = {
            "function": fname, "dimension": dim, "problem_class": spec["class"],
            "gwo_best": float(np.min(gwo_finals)), "gwo_worst": float(np.max(gwo_finals)),
            "gwo_mean": float(np.mean(gwo_finals)), "gwo_std": float(np.std(gwo_finals)),
            "gwo_median": float(np.median(gwo_finals)),
            "gwo_success_rate": float(np.mean(gwo_finals < SUCCESS_THRESHOLD) * 100),
            "gwo_mean_time_s": float(np.mean(gwo_times)),
            "gwo_mean_fevals": float(np.mean(gwo_fevals)),

            "pso_best": float(np.min(pso_finals)), "pso_worst": float(np.max(pso_finals)),
            "pso_mean": float(np.mean(pso_finals)), "pso_std": float(np.std(pso_finals)),
            "pso_median": float(np.median(pso_finals)),
            "pso_success_rate": float(np.mean(pso_finals < SUCCESS_THRESHOLD) * 100),
            "pso_mean_time_s": float(np.mean(pso_times)),
            "pso_mean_fevals": float(np.mean(pso_fevals)),

            "wilcoxon_stat": float(wstat) if wstat == wstat else None,
            "wilcoxon_p": float(pval) if pval == pval else None,
        }
        results.append(row)
        print(f"{fname:16s} {dim:3d}D | GWO mean={row['gwo_mean']:.4e} PSO mean={row['pso_mean']:.4e} "
              f"p={row['wilcoxon_p']}")

print(f"\nTotal experiment time: {time.time()-t0:.1f}s")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
with open(os.path.join(DATA_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Save convergence curves (mean +/- std across runs) compactly for plotting
compact_conv = {}
for key, d in convergence_store.items():
    compact_conv[key] = {}
    for algo, curves in d.items():
        arr = np.array(curves)  # (runs, iters+1)
        compact_conv[key][algo] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
        }
with open(os.path.join(DATA_DIR, "convergence.json"), "w") as f:
    json.dump(compact_conv, f)

print("Saved results.json and convergence.json")

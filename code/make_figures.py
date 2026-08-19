import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from benchmarks import BENCHMARKS

plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(BASE_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(FIG, exist_ok=True)

results = json.load(open(os.path.join(DATA_DIR, "results.json")))
conv = json.load(open(os.path.join(DATA_DIR, "convergence.json")))

COLOR_GWO = "#2C6E49"
COLOR_PSO = "#C1440E"

# ---------- Figure 1: benchmark function landscapes ----------
fig = plt.figure(figsize=(11, 9))
gs = GridSpec(2, 2, figure=fig)
for idx, (name, spec) in enumerate(BENCHMARKS.items()):
    ax = fig.add_subplot(gs[idx // 2, idx % 2], projection="3d")
    b = spec["bounds"]
    x = np.linspace(b[0], b[1], 80)
    y = np.linspace(b[0], b[1], 80)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = spec["func"](np.array([X[i, j], Y[i, j]]))
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title(f"{name}\n({spec['class']})", fontsize=9)
    ax.set_xlabel("x1", fontsize=8); ax.set_ylabel("x2", fontsize=8); ax.set_zlabel("f(x)", fontsize=8)
    ax.tick_params(labelsize=6)
fig.suptitle("Figure 1. Two-dimensional landscapes of the benchmark test functions", y=1.0, fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_benchmark_landscapes.png")
plt.close(fig)

# ---------- Figure 2: convergence curves (mean +/- std), one panel per function, 3 dims each ----------
dims = [2, 10, 30]
functions = list(BENCHMARKS.keys())
fig, axes = plt.subplots(len(functions), len(dims), figsize=(12, 12), sharex=True)
for fi, fname in enumerate(functions):
    for di, dim in enumerate(dims):
        ax = axes[fi, di]
        key = f"{fname}_{dim}D"
        FLOOR = 1e-18
        for algo, color in [("GWO", COLOR_GWO), ("PSO", COLOR_PSO)]:
            m = np.array(conv[key][algo]["mean"])
            s = np.array(conv[key][algo]["std"])
            it = np.arange(len(m))
            m_plot = np.clip(m, FLOOR, None)
            ax.plot(it, m_plot, label=algo, color=color, linewidth=1.4)
            ax.fill_between(it, np.clip(m - s, FLOOR, None), np.clip(m + s, FLOOR, None), color=color, alpha=0.15)
        ax.set_yscale("log")
        ax.set_ylim(bottom=FLOOR)
        if fi == 0:
            ax.set_title(f"{dim}D", fontsize=10)
        if di == 0:
            ax.set_ylabel(f"{fname}\nBest fitness (log)", fontsize=9)
        if fi == len(functions) - 1:
            ax.set_xlabel("Iteration")
        if fi == 0 and di == 0:
            ax.legend(fontsize=8, loc="upper right")
fig.suptitle("Figure 2. Convergence trajectories (mean \u00b1 1 s.d. over 25 independent runs): GWO vs PSO", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{FIG}/fig2_convergence_grid.png")
plt.close(fig)

# ---------- Figure 3: final-fitness distributions (boxplots) at 10D ----------
import matplotlib.patches as mpatches
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for i, fname in enumerate(functions):
    ax = axes[i]
    row = [r for r in results if r["function"] == fname and r["dimension"] == 10][0]
    means = [row["gwo_mean"], row["pso_mean"]]
    stds = [row["gwo_std"], row["pso_std"]]
    bars = ax.bar(["GWO", "PSO"], means, yerr=stds, color=[COLOR_GWO, COLOR_PSO], alpha=0.85, capsize=4)
    ax.set_yscale("log")
    ax.set_title(fname, fontsize=10)
    if i == 0:
        ax.set_ylabel("Final best fitness\n(mean \u00b1 s.d., log scale)")
fig.suptitle("Figure 3. Final solution quality at 10D (25 independent runs)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{FIG}/fig3_final_fitness_10D.png")
plt.close(fig)

# ---------- Figure 4: success-rate comparison heatmap-style grouped bars ----------
fig, ax = plt.subplots(figsize=(11, 5))
labels = [f"{r['function']} {r['dimension']}D" for r in results]
gwo_sr = [r["gwo_success_rate"] for r in results]
pso_sr = [r["pso_success_rate"] for r in results]
x = np.arange(len(labels))
w = 0.38
ax.bar(x - w/2, gwo_sr, width=w, label="GWO", color=COLOR_GWO)
ax.bar(x + w/2, pso_sr, width=w, label="PSO", color=COLOR_PSO)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Success rate (%)\n(runs reaching fitness < 1e-3)")
ax.legend()
ax.set_title("Figure 4. Success rate by function, dimension, and algorithm", fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/fig4_success_rates.png")
plt.close(fig)

# ---------- Figure 5: scalability (mean final fitness vs dimension) ----------
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for i, fname in enumerate(functions):
    ax = axes[i]
    rows = sorted([r for r in results if r["function"] == fname], key=lambda r: r["dimension"])
    d = [r["dimension"] for r in rows]
    g = [max(r["gwo_mean"], 1e-300) for r in rows]
    p = [max(r["pso_mean"], 1e-300) for r in rows]
    ax.plot(d, g, "o-", color=COLOR_GWO, label="GWO")
    ax.plot(d, p, "s-", color=COLOR_PSO, label="PSO")
    ax.set_yscale("log")
    ax.set_title(fname, fontsize=10)
    ax.set_xlabel("Dimension")
    if i == 0:
        ax.set_ylabel("Mean final fitness (log)")
        ax.legend(fontsize=8)
fig.suptitle("Figure 5. Scalability of solution quality with problem dimension", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(f"{FIG}/fig5_scalability.png")
plt.close(fig)

print("Figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    print(" -", f)

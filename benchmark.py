r"""Runtime / scaling benchmarks for the shadowprice instruments (L1.3).

Times each estimator's library-side cost as a function of dimension d (the
per-example backward pass that feeds the gradient-based instruments is
framework-dependent and excluded; we measure the toolkit's own overhead).
Produces benchmark_scaling.png and prints a table.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shadowprice as sp

rng = np.random.default_rng(0)
dims = [10, 20, 50, 100, 200, 500, 1000]
batch = 256
REPS = 5


def timeit(fn, reps=REPS):
    fn()  # warm up
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return float(np.median(t)) * 1e3  # ms


rows = []
for d in dims:
    G = rng.normal(size=(batch, d))
    A = rng.normal(size=(d, d)); H = 0.5 * (A + A.T)
    r = {"d": d}
    r["participation_ratio"] = timeit(lambda: sp.participation_ratio(G, axis=1))
    r["price_gradnorm"] = timeit(lambda: sp.price_gradnorm(lambda X: G, None, n_boot=200))
    r["kappa_abs"] = timeit(lambda: sp.kappa_abs(H))
    r["kappa_lower"] = timeit(lambda: sp.kappa_lower(H, n_samples=500))
    r["kappa_sdp"] = timeit(lambda: sp.kappa_sdp(H), reps=2) if d <= 200 else np.nan
    price = lambda e: 0.2 + 0.8 * e
    Pb = lambda e: np.exp(-1.5 * e)
    r["optimal_radius"] = timeit(lambda: sp.optimal_radius(price, Pb, K=10.0, eps_max=5.0))
    rows.append(r)

# ---- table ----
cols = ["participation_ratio", "price_gradnorm", "kappa_abs", "kappa_lower",
        "kappa_sdp", "optimal_radius"]
print(f"{'d':>6} | " + " | ".join(f"{c:>19}" for c in cols))
print("-" * (8 + 22 * len(cols)))
for r in rows:
    cells = []
    for c in cols:
        v = r[c]
        cells.append("        --         " if (isinstance(v, float) and np.isnan(v)) else f"{v:>17.3f}ms")
    print(f"{r['d']:>6} | " + " | ".join(cells))

# ---- figure ----
fig, ax = plt.subplots(1, 1, figsize=(7, 4.6))
for c in cols:
    ys = [r[c] for r in rows]
    ax.plot(dims, ys, "-o", ms=3, label=c)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("dimension $d$"); ax.set_ylabel("time per call (ms)")
ax.set_title("Instrument runtime vs dimension (batch=256)")
ax.legend(fontsize=8, ncol=2); ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "benchmark_scaling.png")
plt.savefig(out, dpi=130)

# machine-readable summary for the paper table
import json
summ = {c: {r["d"]: (None if (isinstance(r[c], float) and np.isnan(r[c])) else round(r[c], 3))
            for r in rows} for c in cols}
print("\nsaved", out)
print("SCALING:", "participation_ratio & price O(bd); kappa_abs O(d^2); "
      "kappa_lower O(d^2 * samples); kappa_sdp O(d^3+) (SDP); optimal_radius O(1) grid+root")

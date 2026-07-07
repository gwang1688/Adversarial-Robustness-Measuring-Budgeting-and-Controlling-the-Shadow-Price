r"""Stage-A validation of the control + diagnostics layer (batch 2).

Produces two figures on the Paper A linear-Gaussian model:
  control_validation.png : (a) controller trajectories -> eps** ; (b) the
                           retraining-free proxy vs the true retrained price,
                           with the measurement wall.
  six_observables.png    : the Stage-A dashboard over an eps-sweep.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shadowprice as sp

rng = np.random.default_rng(0)
d, sigma = 200, 7.0
mu = np.abs(rng.normal(0, 1, d)) + 0.4
st = lambda e: np.sign(mu) * np.maximum(np.abs(mu) - e, 0.0)
tof = lambda w, e: (e * np.abs(w).sum() - w @ mu) / (sigma * np.linalg.norm(w))
price = lambda e: np.sqrt(sp.participation_ratio(st(e))) / sigma * norm.pdf(tof(st(e), e))
risk = lambda e: norm.cdf(tof(st(e), e))

K, rate = 1.2, 1.8
Pbreach = lambda e: np.exp(-rate * e)
mb = lambda e: K * rate * np.exp(-rate * e)
eps_cert = sp.cert_radius(st(0.05), mu)
res = sp.optimal_radius(price, Pbreach, K=K, eps_max=eps_cert)

# ---- (a) controller convergence from several starts ----
ctl = sp.BudgetController(price, mb, gain=0.15, eps_max=eps_cert)
traj = {e0: ctl.run(eps0=e0, n_steps=400).trajectory for e0 in (0.05, 0.6, 1.4)}

# ---- (b) retraining-free proxy vs true price ----
# realistic: the single-pass gradient measurement of the signal is noisy, so the
# extrapolation is exact at small eps but diverges once the surviving support
# becomes noise-sensitive -- the measurement wall.
w0_meas = mu + 0.12 * rng.normal(size=d)
proxy = sp.RetrainFreeProxy(w0=w0_meas, sigma=sigma, M=3.5, min_active_frac=0.45)
egrid = np.linspace(0.05, 1.45, 60)
pout = proxy.price(egrid)
true = np.array([price(e) for e in egrid])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for e0, tr in traj.items():
    ax[0].plot(tr, lw=1.6, label=f"start $\\varepsilon_0$={e0}")
ax[0].axhline(res.eps_star, ls="--", color="k", lw=1)
ax[0].text(len(max(traj.values(), key=len)) * 0.5, res.eps_star,
           f"  $\\varepsilon^{{**}}$={res.eps_star:.2f}", va="bottom", fontsize=9)
ax[0].set_xlabel("control step"); ax[0].set_ylabel(r"budget $\varepsilon$")
ax[0].set_title("(a) closed-loop controller"); ax[0].legend(fontsize=8)

below = ~pout["beyond_wall"]
ax[1].plot(egrid, true, "k-", lw=2, label="true (retrained) price")
ax[1].plot(egrid[below], pout["lambda"][below], "o", ms=3, color="teal",
           label="retrain-free proxy")
ax[1].plot(egrid[~below], pout["lambda"][~below], "x", ms=4, color="crimson",
           label="beyond wall")
ax[1].axvline(pout["eps_wall"], ls="--", color="crimson", lw=1)
ax[1].text(pout["eps_wall"], ax[1].get_ylim()[1] * 0.7,
           f" wall $\\varepsilon$={pout['eps_wall']:.2f}", color="crimson", fontsize=8)
ax[1].set_xlabel(r"$\varepsilon$"); ax[1].set_ylabel(r"$\lambda_\varepsilon$")
ax[1].set_title("(b) retraining-free proxy + measurement wall"); ax[1].legend(fontsize=8)
plt.tight_layout()
p1 = os.path.join(os.path.dirname(__file__), "control_validation.png")
plt.savefig(p1, dpi=130); plt.close(fig)

# ---- six-observable dashboard over the sweep ----
Hsmall = None  # curvature panel needs a Hessian; linear model has H=0, skip
recs = [sp.collect_observables(e, w=st(e), mu=mu, sigma=sigma, risk_fn=risk)
        for e in np.linspace(0.1, 1.3, 40)]
p2 = os.path.join(os.path.dirname(__file__), "six_observables.png")
sp.Dashboard(recs).figure(path=p2)

# ---- console summary ----
rel = np.abs(pout["lambda"][below] - true[below]) / np.maximum(true[below], 1e-9)
maxres = max(r["collapse_residual"] for r in recs)
# ---- controller noise-tolerance, quantified (L1.4): interior eps** ----
noise_tol = {}
for s_ in (0.05, 0.15, 0.30):
    rngn = np.random.default_rng(1)
    noisy = lambda e, sc=s_: max(price(e) * (1 + sc * rngn.standard_normal()), 1e-6)
    trn = sp.BudgetController(noisy, mb, gain=0.15, eps_max=eps_cert).run(0.1, n_steps=600, tol=0.0)
    noise_tol[s_] = abs(float(trn.trajectory[-100:].mean()) - res.eps_star)
print("controller noise-tolerance |eps_tail - eps**| at interior eps**={:.3f}:".format(res.eps_star))
for s_ in (0.05, 0.15, 0.30):
    print(f"  {int(s_*100)}% price noise -> {noise_tol[s_]:.4f}")

print(f"controller eps** = {[round(float(t[-1]),3) for t in traj.values()]} "
      f"(solver {res.eps_star:.3f})")
print(f"retrain-free proxy max rel-err below wall = {rel.max():.2e}; "
      f"wall at eps = {pout['eps_wall']:.2f}")
print(f"six observables: max collapse residual = {maxres:.2e}")
print(f"saved {p1}\nsaved {p2}")

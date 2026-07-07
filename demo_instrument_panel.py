r"""Instrument-panel demo on the Paper A linear-Gaussian model.

Runs the four ready instruments end-to-end and produces a one-figure dashboard:
  (a) price meter: finite-difference lambda_eps vs the closed form (sqrt k_eps/sigma) phi(t);
  (b) effective support k_eps(eps) of the soft-threshold discriminant;
  (c) decision: the net value V-C and the optimal radius eps**.
This is synthetic (Stage A). Real-model measurement is the torch adapter (batch 2).
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
mu = np.abs(rng.normal(0, 1, d)) + 0.4            # dense signal

def soft_threshold(eps):
    return np.sign(mu) * np.maximum(np.abs(mu) - eps, 0.0)

def t_of(w, eps):
    l1, l2 = np.abs(w).sum(), np.linalg.norm(w)
    return (eps * l1 - w @ mu) / (sigma * l2)

def risk(eps):                                     # R(eps) at the eps-optimal w
    w = soft_threshold(eps)
    return norm.cdf(t_of(w, eps))

eps_grid = np.linspace(0.02, 1.2, 60)

# (a) price meter: finite difference vs closed form
price_fd, price_cf, keps = [], [], []
for e in eps_grid:
    w = soft_threshold(e)
    k = sp.participation_ratio(w); keps.append(k)
    price_fd.append(sp.price_finite_diff(risk, e, h=1e-3))
    price_cf.append(np.sqrt(k) / sigma * norm.pdf(t_of(w, e)))
price_fd, price_cf, keps = map(np.array, (price_fd, price_cf, keps))

# (c) decision calculus: pick eps** given a deployment breach model
K = 1.2
Pbreach = lambda e: np.exp(-1.8 * e)
price_fn = lambda e: np.sqrt(sp.participation_ratio(soft_threshold(e))) / sigma \
                     * norm.pdf(t_of(soft_threshold(e), e))
w0 = soft_threshold(0.05)
eps_cert = sp.cert_radius(w0, mu)
res = sp.optimal_radius(price_fn, Pbreach, K=K, eps_max=eps_cert)
netval = [sp.radius._net_value(price_fn, Pbreach, K, e) for e in eps_grid]

# ---- dashboard ----
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(eps_grid, price_cf, "k-", lw=2, label=r"closed form $(\sqrt{k_\varepsilon}/\sigma)\phi(t)$")
ax[0].plot(eps_grid, price_fd, "o", ms=3, color="crimson", label="finite-diff meter")
ax[0].set_xlabel(r"$\varepsilon$"); ax[0].set_ylabel(r"marginal price $\lambda_\varepsilon$")
ax[0].set_title("(a) price meter"); ax[0].legend(fontsize=8)

ax[1].plot(eps_grid, keps, "b-", lw=2)
ax[1].axhline(d, ls="--", color="gray", lw=1); ax[1].text(0.05, d*0.93, f"$d={d}$", fontsize=8)
ax[1].set_xlabel(r"$\varepsilon$"); ax[1].set_ylabel(r"$k_\varepsilon$")
ax[1].set_title("(b) effective support")

ax[2].plot(eps_grid, netval, "g-", lw=2)
ax[2].axvline(res.eps_star, ls="--", color="crimson")
ax[2].text(res.eps_star, min(netval), f"  $\\varepsilon^{{**}}={res.eps_star:.2f}$",
           color="crimson", fontsize=9, va="bottom")
ax[2].set_xlabel(r"$\varepsilon$"); ax[2].set_ylabel(r"net value $V-C$")
ax[2].set_title("(c) decision calculus")

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "instrument_panel.png")
plt.savefig(out, dpi=130)
print(f"price meter: mean |fd - closed| = {np.mean(np.abs(price_fd-price_cf)):.2e}")
print(f"k_eps range: {keps.min():.1f} .. {keps.max():.1f}  (d={d})")
print(f"eps_cert = {eps_cert:.3f}   eps** = {res.eps_star:.3f}  (interior={res.interior})")
print(f"saved {out}")

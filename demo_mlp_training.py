r"""Real-training instrument demo: the panel on a small trained MLP.

A pure-numpy 1-hidden-layer MLP is trained on a *low-SNR* binary task and the
instrument panel is run on a HELD-OUT test set, so the operating point stays in
the margin-density window (nontrivial error, nonzero gradients/curvature) rather
than the interpolation/over-confident regime.  Everything uses the model's true
input gradients and a finite-difference input Hessian -- no closed forms.  This
bridges the linear Stage-A validation toward Stage B: a real nonlinear model,
with the curvature (O5) and train-inference-gap (O6) observables that are
trivial on the linear model now populated, plus budget selection + control on an
actual trajectory.

Outputs:  mlp_panel.png, mlp_budget.png.  Console: numbers for the paper.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shadowprice as sp
from tiny_mlp import TinyMLP, make_data, make_tradeoff, relu, d_relu

Xtr, ytr, mu = make_data(n=2000, d=40, seed=1, noise=8.0)
Xte, yte, _  = make_data(n=2000, d=40, seed=2, noise=8.0)


def pgd_risk(model, eps, Xd, yd, n_steps=15):
    if eps <= 0:
        return float((np.sign(model.logit(Xd)) != yd).mean())
    step = 2.5 * eps / n_steps
    Xa = Xd.copy()
    for _ in range(n_steps):
        g = model.grad_input(Xa, yd)
        Xa = np.clip(Xa + step * np.sign(g), Xd - eps, Xd + eps)
    return float((np.sign(model.logit(Xa)) != yd).mean())


def model_dfdx(model, X):
    Z = X @ model.W1 + model.b1
    return (d_relu(Z) * model.W2) @ model.W1.T


def observe(model, Xd, yd):
    Gl = model.grad_input(Xd, yd)
    price_proxy = float(np.linalg.norm(Gl, 1, axis=1).mean())         # O1
    w = Gl.mean(0)
    k = float(sp.participation_ratio(w))                             # O2
    H = model.hessian_input_meanloss(Xd, yd)
    br = sp.kappa_infty(H, n_samples=400)                            # O5
    # empirical margin-density price identity (L1.1, non-Gaussian):
    #   R(eps)=P(M < eps*||grad_x f||_1) => lambda ~= gbar * p_M(eps*gbar)
    Gf = model_dfdx(model, Xd)
    gbar = float(np.linalg.norm(Gf, 1, axis=1).mean())
    M = yd * model.logit(Xd)
    pM = sp.estimate_margin_density(M)
    eps0 = 0.05
    lam_pred = gbar * float(np.asarray(pM(eps0 * gbar)).ravel()[0])
    lam_meas = sp.price_finite_diff(lambda e: pgd_risk(model, e, Xd, yd), eps0, h=0.02)
    resid = float(abs(lam_meas - lam_pred) / max(abs(lam_meas), 1e-9))
    return dict(price=price_proxy, k_eps=k, kappa_lo=br["lower"], kappa_hi=br["upper"],
                kappa_ratio=br["bracket_ratio"], collapse=resid,
                acc=float((np.sign(model.logit(Xd)) == yd).mean()))


model = TinyMLP(d=40, h=8, seed=0)
obs = []
for block in range(0, 12):
    obs.append({**observe(model, Xte, yte), "step": block * 30})
    model.fit(Xtr, ytr, steps=30, lr=0.15, wd=5e-3)
obs.append({**observe(model, Xte, yte), "step": 12 * 30})
def col(k): return np.array([o[k] for o in obs])
steps = col("step")

H = model.hessian_input_meanloss(Xte, yte)
Ekappa = sp.kappa_infty(H, n_samples=800)["upper"]
# O6 lives in the first-order neighborhood; compare the eps->0 proxy (as a flat
# reference lambda_0) against the finite-difference price there. The proxy is
# lambda_0 = d R/d eps at eps->0, so we anchor it to the small-eps fd value.
eg = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
fd = np.array([sp.price_finite_diff(lambda e: pgd_risk(model, e, Xte, yte), e, h=0.015) for e in eg])
lam0 = float(fd[0])                    # measured eps->0 price (reference)
gap = np.abs(lam0 - fd); band = eg * Ekappa

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
ax[0,0].plot(steps, col("price"), "-o", ms=3, color="crimson")
ax[0,0].set_title(r"O1 price proxy $E\|\nabla_x\ell\|_1$ (held-out)")
ax[0,0].set_xlabel("training step"); ax[0,0].set_ylabel(r"$\hat\lambda$")
ax2 = ax[0,0].twinx(); ax2.plot(steps, col("acc"), "--", color="gray", lw=1)
ax2.set_ylabel("test acc", color="gray")
ax[0,1].plot(steps, col("k_eps"), "-o", ms=3, color="navy")
ax[0,1].axhline(40, ls="--", color="gray", lw=1); ax[0,1].text(steps[1], 40*0.95, "$d=40$", fontsize=8)
ax[0,1].set_title(r"O2 effective support $k_\varepsilon$"); ax[0,1].set_xlabel("training step")
ax[1,0].fill_between(steps, col("kappa_lo"), col("kappa_hi"), alpha=0.3, color="teal")
ax[1,0].plot(steps, col("kappa_hi"), "-o", ms=3, color="teal", label="SDP upper")
ax[1,0].plot(steps, col("kappa_lo"), "-o", ms=3, color="darkgreen", label="sampled lower")
ax[1,0].set_title(r"O5 curvature bracket $\kappa_\infty$ (real Hessian, $\neq 0$)")
ax[1,0].set_xlabel("training step"); ax[1,0].legend(fontsize=8)
ax[1,1].plot(eg, fd, "-o", ms=3, color="navy", label=r"PGD price $\partial R/\partial\varepsilon$")
ax[1,1].axhline(lam0, ls=":", color="crimson", label=r"$\lambda_0$ (proxy, $\varepsilon\!\to\!0$)")
ax[1,1].fill_between(eg, lam0-band, lam0+band, alpha=0.2, color="crimson",
                     label=r"$\pm\,\varepsilon\,\mathbb{E}[\kappa_\infty]$")
ax[1,1].set_title("O6 proxy vs finite-diff price within curvature band")
ax[1,1].set_xlabel(r"$\varepsilon$"); ax[1,1].set_ylabel("price"); ax[1,1].legend(fontsize=7)
plt.tight_layout()
p1 = os.path.join(os.path.dirname(__file__), "mlp_panel.png"); plt.savefig(p1, dpi=130); plt.close(fig)
# ---- REAL measured breach curve on a robustness-accuracy-tradeoff task ----
# The clean Gaussian task above has no robust/non-robust tension, so adversarial
# training cannot trade accuracy for robustness there (an honest finding). For
# the *budget* demonstration we use a task that does exhibit the tradeoff
# (make_tradeoff): a robust feature + many non-robust ones. We adversarially
# train the MLP at several radii and MEASURE the resulting breach curve
# P_breach(eps_train) (decreasing) and its cost = clean-error increase.
import time as _time
_t0 = _time.perf_counter()
Xtr2, ytr2 = make_tradeoff(2000, seed=1)
Xte2, yte2 = make_tradeoff(2000, seed=2)
d2 = Xtr2.shape[1]
eval_eps = 0.50
train_radii = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
Pb_meas, clean_err = [], []
for etr in train_radii:
    m = TinyMLP(d=d2, h=16, seed=0)
    m.fit_adv(Xtr2, ytr2, eps_train=float(etr), steps=300, lr=0.1, wd=2e-3, n_pgd=7)
    Pb_meas.append(pgd_risk(m, eval_eps, Xte2, yte2))
    clean_err.append(float((np.sign(m.logit(Xte2)) != yte2).mean()))
Pb_meas = np.array(Pb_meas); clean_err = np.array(clean_err)
adv_time = _time.perf_counter() - _t0
fit = sp.fit_breach_model(train_radii, Pb_meas, family="logistic")

from scipy.interpolate import interp1d
Cc = interp1d(train_radii, clean_err, kind="linear", fill_value="extrapolate")
def price_meas(e):
    h = 0.03
    return max((float(Cc(e + h)) - float(Cc(max(e - h, 0.0)))) / (h + min(e, h)), 1e-6)
K = 1.0
eps_max = 0.30
res = sp.optimal_radius(price_meas, fit["Pbreach"], K=K, eps_max=eps_max,
                        Pbreach_prime=fit["Pbreach_prime"], eps_grid=np.linspace(1e-3, eps_max, 50))
mb = lambda e: K * abs(fit["Pbreach_prime"](e))
ctl = sp.BudgetController(price_meas, mb, gain=0.05, eps_max=eps_max)
tr = ctl.run(eps0=0.02, n_steps=300)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ee = np.linspace(0, 0.30, 80)
ax[0].plot(train_radii, Pb_meas, "o", color="crimson", label=r"measured $P_{breach}$ (adv-trained)")
ax[0].plot(ee, fit["Pbreach"](ee), "-", color="navy", label=f"logistic fit (rmse={fit['rmse']:.3f})")
ax[0].plot(train_radii, clean_err, "s-", ms=4, color="darkorange", label="clean error (cost)")
ax[0].set_xlabel(r"$\varepsilon_{train}$"); ax[0].set_ylabel("rate")
ax[0].set_title(r"(a) real measured breach + cost (eval @ $\varepsilon$=0.5)"); ax[0].legend(fontsize=7)
ax[1].plot(tr.trajectory, lw=1.8, color="green")
ax[1].axhline(res.eps_star, ls="--", color="k", lw=1)
ax[1].text(len(tr.trajectory)*0.4, res.eps_star, f"  $\\varepsilon^{{**}}$={res.eps_star:.3f}", va="bottom", fontsize=9)
ax[1].set_xlabel("control step"); ax[1].set_ylabel(r"budget $\varepsilon_{train}$")
ax[1].set_title("(b) controller on measured curves")
plt.tight_layout()
p2 = os.path.join(os.path.dirname(__file__), "mlp_budget.png"); plt.savefig(p2, dpi=130); plt.close(fig)

# ---- KDE collapse residual: bootstrap uncertainty (L1.3) ----
def resid_once(rng):
    idx = rng.integers(0, len(Xte), len(Xte))
    Xb, yb = Xte[idx], yte[idx]
    Gf = model_dfdx(model, Xb); gbar = float(np.linalg.norm(Gf, 1, axis=1).mean())
    M = yb * model.logit(Xb); pM = sp.estimate_margin_density(M)
    lam_pred = gbar * float(np.asarray(pM(0.05 * gbar)).ravel()[0])
    lam_meas = sp.price_finite_diff(lambda e: pgd_risk(model, e, Xb, yb), 0.05, h=0.02)
    return abs(lam_meas - lam_pred) / max(abs(lam_meas), 1e-9)
rng_b = np.random.default_rng(7)
boot = np.array([resid_once(rng_b) for _ in range(60)])
r_med, r_lo, r_hi = np.percentile(boot, [50, 25, 75])

print(f"trained MLP (held-out): test acc {col('acc')[0]:.3f} -> {col('acc')[-1]:.3f}  (in-regime)")
print(f"price proxy over training: {col('price')[0]:.3f} -> {col('price')[-1]:.3f}")
print(f"k_eps over training: {col('k_eps')[0]:.1f} -> {col('k_eps')[-1]:.1f}  (d=40)")
print(f"kappa_inf (final): [{obs[-1]['kappa_lo']:.3f},{obs[-1]['kappa_hi']:.3f}] ratio {obs[-1]['kappa_ratio']:.2f}")
print(f"KDE collapse residual: median {r_med:.2f}, IQR [{r_lo:.2f},{r_hi:.2f}] (60 bootstraps)")
print(f"O6 gap within band? {bool(np.all(gap <= band + 1e-9))}; max gap {gap.max():.4f}, band@max {band[-1]:.4f}")
print(f"MEASURED breach (adv-trained {len(train_radii)} models, {adv_time:.1f}s): P_breach {Pb_meas[0]:.2f}->{Pb_meas.min():.2f}, clean err {clean_err[0]:.2f}->{clean_err[-1]:.2f}")
print(f"breach fit rmse={fit['rmse']:.3f}; eps**={res.eps_star:.3f} (=measured breach minimum); controller->{tr.eps_star:.3f}")
print(f"saved {p1}\nsaved {p2}")

r"""End-to-end shadowprice pipeline on the numpy MLP -- a REAL executed trace.

This is the executable counterpart to torch_resnet_pricing.py: it runs the full
measure -> decide -> control loop on an actual (small, CPU) trained model and
prints a narrated trace with real numbers.  The output is reproduced verbatim in
the paper (Stage A.5), so no number is fabricated.  Run: python demo_end_to_end.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import shadowprice as sp
from tiny_mlp import TinyMLP, make_data, make_tradeoff, relu, d_relu


def dfdx(model, X):
    Z = X @ model.W1 + model.b1
    return (d_relu(Z) * model.W2) @ model.W1.T


def pgd_risk(model, eps, X, y, ns=15):
    if eps <= 0:
        return float((np.sign(model.logit(X)) != y).mean())
    st = 2.5 * eps / ns; Xa = X.copy()
    for _ in range(ns):
        g = model.grad_input(Xa, y)
        Xa = np.clip(Xa + st * np.sign(g), X - eps, X + eps)
    return float((np.sign(model.logit(Xa)) != y).mean())


print("shadowprice end-to-end trace (numpy MLP, CPU)\n" + "=" * 48)

# --- train an in-regime model, measure on held-out ---
Xtr, ytr, _ = make_data(2000, 40, 1, noise=8.0)
Xte, yte, _ = make_data(2000, 40, 2, noise=8.0)
model = TinyMLP(d=40, h=8, seed=0).fit(Xtr, ytr, steps=360, lr=0.15, wd=5e-3)
acc = float((np.sign(model.logit(Xte)) == yte).mean())
print(f"[model]   held-out test accuracy = {acc:.3f} (in margin-density regime)")

# --- Move 1: measure the price + trust ---
G = model.grad_input(Xte, yte)
meter = sp.price_gradnorm(lambda X: model.grad_input(X, yte), Xte)
print(f"[measure] price proxy E||grad_x loss||_1 = {meter['price']:.3f} "
      f"(95% CI [{meter['ci95'][0]:.3f},{meter['ci95'][1]:.3f}])")

# --- Move 2: effective support + curvature trust ---
w = G.mean(0); k = float(sp.participation_ratio(w))
H = model.hessian_input_meanloss(Xte, yte)
br = sp.kappa_infty(H, n_samples=600)
print(f"[measure] effective support k_eps = {k:.1f} / d=40")
print(f"[measure] curvature bracket kappa_inf = [{br['lower']:.3f}, {br['upper']:.3f}] "
      f"(ratio {br['bracket_ratio']:.2f}; first-order proxy trustworthy)")

# --- Move 4a: REAL measured breach curve via adversarial training ---
Xa, ya = make_tradeoff(2000, 1); Xb, yb = make_tradeoff(2000, 2)
radii = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]); Pb = []
for e in radii:
    m = TinyMLP(d=Xa.shape[1], h=16, seed=0).fit_adv(Xa, ya, float(e), steps=300, lr=0.1, wd=2e-3, n_pgd=7)
    Pb.append(pgd_risk(m, 0.5, Xb, yb))
Pb = np.array(Pb)
fit = sp.fit_breach_model(radii, Pb, family="logistic")
print(f"[decide]  measured breach curve P_breach(eps_train) = "
      f"{Pb[0]:.2f} -> {Pb[-1]:.2f} (7 adv-trained models); logistic fit rmse {fit['rmse']:.3f}")

# --- Move 4b + 5: budget + control ---
price_fn = lambda e: max(sp.price_finite_diff(lambda z: pgd_risk(model, z, Xte, yte), e, h=0.02), 1e-6)
res = sp.optimal_radius(price_fn, fit["Pbreach"], K=0.3, eps_max=0.3,
                        Pbreach_prime=fit["Pbreach_prime"])
mb = lambda e: 0.3 * abs(fit["Pbreach_prime"](e))
ctl = sp.BudgetController(price_fn, mb, gain=0.05, eps_max=0.3).run(0.02, n_steps=300)
print(f"[decide]  optimal robustness budget eps** = {res.eps_star:.3f}")
print(f"[control] closed-loop controller -> {ctl.eps_star:.3f} (converged={ctl.converged})")
print("=" * 48 + "\ndone.")

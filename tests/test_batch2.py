r"""Validate the batch-2 control / diagnostics / proxy layer against Paper A.

Run: python tests/test_batch2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import norm
import shadowprice as sp


# ---- shared synthetic linear-Gaussian model (Paper A) ----
def make_model(d=200, sigma=7.0, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.abs(rng.normal(0, 1, d)) + 0.4
    st = lambda e: np.sign(mu) * np.maximum(np.abs(mu) - e, 0.0)
    tof = lambda w, e: (e * np.abs(w).sum() - w @ mu) / (sigma * np.linalg.norm(w))
    price = lambda e: (np.sqrt(sp.participation_ratio(st(e))) / sigma
                       * norm.pdf(tof(st(e), e)))
    risk = lambda e: norm.cdf(tof(st(e), e))
    return dict(d=d, sigma=sigma, mu=mu, st=st, tof=tof, price=price, risk=risk)


def test_controller_converges_to_optimal_radius():
    m = make_model()
    K, rate = 1.2, 1.8
    Pbreach = lambda e: np.exp(-rate * e)
    mb = lambda e: K * rate * np.exp(-rate * e)         # K|P_breach'|
    eps_cert = sp.cert_radius(m["st"](0.05), m["mu"])
    # ground truth from the solver
    res = sp.optimal_radius(m["price"], Pbreach, K=K, eps_max=eps_cert)
    assert res.interior
    # controller should reach the same eps**
    ctl = sp.BudgetController(m["price"], mb, gain=0.15, eps_max=eps_cert)
    tr = ctl.run(eps0=0.1, n_steps=500)
    assert tr.converged, tr.foc_residual
    assert abs(tr.eps_star - res.eps_star) < 1e-2, (tr.eps_star, res.eps_star)
    # converges from the other side too
    tr2 = ctl.run(eps0=eps_cert * 0.95, n_steps=500)
    assert abs(tr2.eps_star - res.eps_star) < 1e-2


def test_retrainfree_proxy_matches_true_price_below_wall():
    m = make_model()
    # measure gradient geometry once at eps=0 (w0 ∝ mu); no robust retrain
    proxy = sp.RetrainFreeProxy(w0=m["mu"], sigma=m["sigma"], M=3.0)
    egrid = np.linspace(0.05, 1.2, 40)
    out = proxy.price(egrid)
    true = np.array([m["price"](e) for e in egrid])
    ok = ~out["beyond_wall"]
    # below the measurement wall, extrapolation matches the retrained price
    rel = np.abs(out["lambda"][ok] - true[ok]) / np.maximum(true[ok], 1e-9)
    assert rel.max() < 1e-6, rel.max()
    assert np.isfinite(out["eps_wall"])


def test_six_observables_collapse_and_gap():
    m = make_model()
    eps = 0.5
    w = m["st"](eps)
    # closed-form price mode: collapse residual must be ~0 (scale invariance holds)
    rec = sp.collect_observables(eps, w=w, mu=m["mu"], sigma=m["sigma"],
                                 risk_fn=m["risk"])
    assert rec["collapse_residual"] < 1e-3, rec["collapse_residual"]
    # k_eps in (1, d); cert radius positive
    assert 1.0 < rec["k_eps"] < m["d"]
    assert rec["eps_cert"] > eps
    # finite-diff price matches closed form here
    price_cf = np.sqrt(rec["k_eps"]) / m["sigma"] * norm.pdf(rec["t"])
    assert abs(rec["price"] - price_cf) / price_cf < 1e-3


def test_curvature_observable_bracket():
    rng = np.random.default_rng(5)
    d = 10
    A = rng.normal(size=(d, d)); H = 0.5 * (A + A.T)
    rec = sp.collect_observables(0.3, w=rng.normal(size=d), mu=rng.normal(size=d),
                                 sigma=1.0, H=H)
    assert rec["kappa_lower"] <= rec["kappa_upper"] + 1e-6
    assert rec["kappa_ratio"] >= 1.0 - 1e-9


def test_controller_noise_robustness():
    """Controller still converges near eps** when the measured price is noisy."""
    m = make_model()
    K, rate = 1.2, 1.8
    Pbreach = lambda e: np.exp(-rate * e)
    mb = lambda e: K * rate * np.exp(-rate * e)
    eps_cert = sp.cert_radius(m["st"](0.05), m["mu"])
    res = sp.optimal_radius(m["price"], Pbreach, K=K, eps_max=eps_cert)
    assert res.interior
    rng = np.random.default_rng(0)
    # inject multiplicative + additive measurement noise into the price
    for noise in (0.05, 0.15, 0.30):
        noisy = lambda e, s=noise: max(m["price"](e) * (1 + s * rng.standard_normal())
                                       + 0.02 * s * rng.standard_normal(), 1e-6)
        ctl = sp.BudgetController(noisy, mb, gain=0.05, eps_max=eps_cert)
        tr = ctl.run(eps0=0.1, n_steps=600, tol=0.0)   # no early stop under noise
        tail = tr.trajectory[-100:].mean()             # average out the jitter
        # tolerance scales with the injected noise level
        assert abs(tail - res.eps_star) < 0.05 + 0.6 * noise, (noise, tail, res.eps_star)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")

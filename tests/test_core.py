r"""Validate the shadowprice core against Paper A closed forms.

Run:  python -m pytest tests/  (or just: python tests/test_core.py)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import norm
import shadowprice as sp


def test_effective_support_extremes():
    d = 64
    onehot = np.zeros(d); onehot[3] = 2.7
    assert abs(sp.participation_ratio(onehot) - 1.0) < 1e-9      # localized -> 1
    uniform = np.ones(d) * 0.5
    assert abs(sp.participation_ratio(uniform) - d) < 1e-9       # delocalized -> d
    # soft-threshold participation ratio matches the manual formula
    mu = np.array([1.5, 1.2, 0.9, 0.3, 0.0]); eps = 0.5
    w = np.sign(mu) * np.maximum(np.abs(mu) - eps, 0.0)
    k_manual = (np.abs(w).sum() ** 2) / (w @ w)
    assert abs(sp.participation_ratio(w) - k_manual) < 1e-9


def test_cert_radius():
    rng = np.random.default_rng(0)
    w = rng.normal(size=32); mu = rng.normal(size=32)
    assert abs(sp.cert_radius(w, mu) - (w @ mu) / np.abs(w).sum()) < 1e-12


def test_optimal_radius_foc():
    # Synthetic: price rises linearly, breach prob is a decaying exponential.
    K = 10.0
    price = lambda e: 0.2 + 0.8 * e                     # lambda(eps)
    Pbreach = lambda e: np.exp(-1.5 * e)                # P_breach(eps), P(0)=1
    res = sp.optimal_radius(price, Pbreach, K=K, eps_max=5.0)
    assert res.interior, res
    e = res.eps_star
    # FOC: K |P'| = lambda  at eps*
    Pp = (Pbreach(e + 1e-6) - Pbreach(e - 1e-6)) / 2e-6
    assert abs(K * abs(Pp) - price(e)) < 1e-4
    # existence criterion used the true marginal cost lambda_0 = price(0+)
    assert res.marginal_cost_at_0 == price(1e-8)


def test_optimal_radius_corner():
    # If robustness is too expensive at 0, no interior optimum.
    res = sp.optimal_radius(price=lambda e: 100.0, Pbreach=lambda e: np.exp(-e),
                            K=1.0, eps_max=5.0)
    assert not res.interior and res.eps_star == 0.0


def test_kappa_bracket_bruteforce():
    # For small d, brute-force the true Boolean max and check lower<=true<=upper.
    rng = np.random.default_rng(1)
    d = 12
    A = rng.normal(size=(d, d)); H = 0.5 * (A + A.T)
    signs = np.array(np.meshgrid(*[[-1, 1]] * d)).T.reshape(-1, d)
    true = np.max(np.abs(np.einsum("ij,jk,ik->i", signs, H, signs)))
    lo = sp.kappa_lower(H, n_samples=500, seed=2)
    up_abs = sp.kappa_abs(H)
    assert lo <= true + 1e-6
    assert up_abs >= true - 1e-6
    # SDP upper bound (cvxpy) should also dominate the true max
    try:
        up_sdp = sp.kappa_sdp(H)
        assert up_sdp >= true - 1e-4
        # and be no looser than the trivial abs bound (usually much tighter)
        assert up_sdp <= up_abs + 1e-6
    except ImportError:
        pass


def test_price_gradnorm():
    # grad_fn returns a known batch of input gradients -> mean l1 norm.
    G = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    out = sp.price_gradnorm(lambda X: G, X=None, ord=1)
    assert abs(out["price"] - np.mean([3.5, 4.0])) < 1e-9


def test_finite_diff_matches_closed_form_price():
    # Paper A linear-Gaussian: R(eps)=Phi(t), t=(eps||w||_1-<w,mu>)/(sigma||w||_2),
    # closed-form price dR/deps = (sqrt(k_eps)/sigma) phi(t).
    rng = np.random.default_rng(3)
    d = 40
    mu = np.abs(rng.normal(size=d)) + 0.5
    eps = 0.3
    w = np.sign(mu) * np.maximum(np.abs(mu) - eps, 0.0)   # soft-threshold
    sigma = 1.0
    l1, l2 = np.abs(w).sum(), np.linalg.norm(w)
    t = lambda e: (e * l1 - w @ mu) / (sigma * l2)
    R = lambda e: norm.cdf(t(e))
    price_fd = sp.price_finite_diff(R, eps, h=1e-4)
    k_eps = (l1 / l2) ** 2
    price_closed = (np.sqrt(k_eps) / sigma) * norm.pdf(t(eps))
    assert abs(price_fd - price_closed) / price_closed < 1e-4


def test_fd_budget_delta_cubed():
    # required samples scale as delta^-3
    n1 = sp.fd_budget(delta=0.1)["n_required"]
    n2 = sp.fd_budget(delta=0.05)["n_required"]
    assert abs(np.log(n2 / n1) / np.log(0.1 / 0.05) - 3.0) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")

r"""The price meter: measuring the marginal robust price lambda_eps.

Two estimators, both framework-agnostic (they consume user-supplied callables,
so they work with numpy, torch, jax, ... via a thin adapter):

* **Gradient-norm proxy** (Paper A, Prop. 8.1): as eps -> 0,
      lambda_eps  ~=  E || grad_x loss(x, y) ||_1,
  obtainable from a single backward pass.  ``price_gradnorm`` averages the
  l_1 input-gradient norm over a batch and returns bootstrap error bars.

* **Finite-difference price** (the direct definition dR/deps): central
  difference of a robust-risk estimator ``risk_fn(eps)``.  Because ``risk_fn``
  is itself Monte-Carlo, the estimator has a bias ~ h^2 and variance
  ~ sigma^2 / (n h^2); optimizing the step gives the "delta^-3 budget"
  (Paper A, sec 7.2): to reach derivative accuracy ``delta`` one needs
  n ~ (sigma / delta)^... ~ delta^-3 samples.  ``fd_budget`` returns the
  optimal step and required sample count.

The gradient-norm proxy is the cheap leading term; ``kappa_infty`` bounds the
O(eps) correction.
"""
from __future__ import annotations
import numpy as np

__all__ = ["price_gradnorm", "price_finite_diff", "fd_budget", "PriceMeter"]


def price_gradnorm(grad_fn, X, ord=1, n_boot=1000, seed=0):
    r"""E || grad_x loss ||_ord over a batch, with bootstrap CI.

    Parameters
    ----------
    grad_fn : callable
        ``grad_fn(X) -> G`` where ``G`` has shape ``(batch, d)`` (the input
        gradients of the loss).  Any framework; just return an array.
    X : array (batch, d)
        Inputs at the operating point.
    ord : int
        Norm order (1 for the l_inf adversary's l_1 dual; default 1).

    Returns
    -------
    dict with ``price`` (mean l_ord gradient norm), ``ci95`` (tuple), ``std``.
    """
    G = np.asarray(grad_fn(X), float)
    if G.ndim == 1:
        G = G[None, :]
    G = G.reshape(G.shape[0], -1)
    norms = np.linalg.norm(G, ord=ord, axis=1)
    mean = float(norms.mean())
    rng = np.random.default_rng(seed)
    n = len(norms)
    boot = np.array([norms[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"price": mean, "ci95": (float(lo), float(hi)),
            "std": float(norms.std(ddof=1) if n > 1 else 0.0), "n": n}


def price_finite_diff(risk_fn, eps, h=None, sigma=None, n=None):
    r"""Central-difference marginal price  dR/deps  at ``eps``.

    ``risk_fn(eps) -> float`` is a (possibly Monte-Carlo) robust-risk estimate.
    If ``h`` is None and ``sigma`` (per-eval noise std) and ``n`` are given, the
    variance-optimal step is used (see :func:`fd_budget`).
    """
    if h is None:
        if sigma is not None and n is not None:
            h = fd_budget(sigma=sigma, n=n)["h_opt"]
        else:
            h = 1e-3 * max(1.0, abs(eps))
    r_plus = risk_fn(eps + h)
    r_minus = risk_fn(eps - h)
    return float((r_plus - r_minus) / (2 * h))


def fd_budget(delta=None, sigma=1.0, n=None, third_deriv=1.0):
    r"""The delta^-3 measurement budget for a finite-difference derivative.

    Central difference of a Monte-Carlo risk has bias ``~ (f''' / 6) h^2`` and
    standard error ``~ sqrt(2) sigma / (sqrt(n) h)``.  Balancing them:

    * given a sample budget ``n``: optimal step ``h_opt = (3 sigma / (f''' *
      sqrt(n)))^{1/3}`` yielding total error ``~ n^{-1/3}``;
    * to *reach* accuracy ``delta``: required ``n ~ (sigma^2 / (f''' * delta^3))``
      up to constants -- i.e. the characteristic ``delta^-3`` scaling.

    Returns a dict with whichever of ``h_opt`` / ``n_required`` / ``error`` is
    determined by the inputs.
    """
    out = {}
    if n is not None:
        h_opt = (3.0 * sigma / (max(third_deriv, 1e-12) * np.sqrt(n))) ** (1.0 / 3.0)
        err = np.sqrt(2) * sigma / (np.sqrt(n) * h_opt) + third_deriv / 6.0 * h_opt ** 2
        out.update(h_opt=float(h_opt), error=float(err))
    if delta is not None:
        # error ~ C * n^{-1/3}; invert for n ~ (C/delta)^3, C ~ sigma^{2/3} f'''^{1/3}
        C = (sigma ** 2 * max(third_deriv, 1e-12)) ** (1.0 / 3.0)
        out["n_required"] = float((C / delta) ** 3)
    return out


class PriceMeter:
    """Bundle the two estimators behind one operating point.

    ``grad_fn`` gives the cheap eps->0 proxy; ``risk_fn`` (optional) gives the
    finite-difference price at finite eps for cross-checking / the correction.
    """

    def __init__(self, grad_fn=None, risk_fn=None):
        self.grad_fn = grad_fn
        self.risk_fn = risk_fn

    def proxy(self, X, **kw):
        if self.grad_fn is None:
            raise ValueError("PriceMeter needs grad_fn for the proxy price.")
        return price_gradnorm(self.grad_fn, X, **kw)

    def finite_diff(self, eps, **kw):
        if self.risk_fn is None:
            raise ValueError("PriceMeter needs risk_fn for the finite-difference price.")
        return price_finite_diff(self.risk_fn, eps, **kw)

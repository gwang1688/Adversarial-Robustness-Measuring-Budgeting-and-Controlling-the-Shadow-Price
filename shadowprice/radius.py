r"""Certification radius and the economically-optimal robust radius.

Paper A gives two decision quantities:

* **Certification radius** (Cor. 8.3): for a linear discriminant ``w`` with
  clean signal ``mu``,
      eps_cert = <w, mu> / ||w||_1,
  the radius at which the robust risk reaches 1/2 (beyond it the classifier is
  worse than chance under the fixed binary-channel reduction).

* **Optimal robust radius** (Prop. 8.4): with a deployment value
  ``V(eps) = K (1 - P_breach(eps))`` and a robustness cost
  ``C(eps) = \int_0^eps lambda(eps') d eps'`` whose marginal is the shadow
  price ``lambda(eps)``, the optimum ``eps** = argmax [V - C]`` satisfies the
  first-order condition
      K * |P_breach'(eps)| = lambda(eps).
  Existence of an interior optimum requires ``K |P_breach'(0)| > lambda(0)``
  (the *true* marginal cost at 0, an over-confident-regime quantity), and
  ``eps** <= eps_cert``.

The solver treats ``lambda`` and ``P_breach`` as user-supplied callables (the
price meter can provide ``lambda``), so it is framework-agnostic.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

__all__ = ["cert_radius", "optimal_radius", "OptimalRadiusResult"]


def cert_radius(w, mu):
    r"""eps_cert = <w, mu> / ||w||_1 for a linear discriminant."""
    w = np.asarray(w, float)
    mu = np.asarray(mu, float)
    l1 = np.abs(w).sum()
    if l1 == 0:
        return np.inf
    return float(np.dot(w, mu) / l1)


class OptimalRadiusResult(dict):
    """dict with attribute access for the optimal-radius solution."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def optimal_radius(price, Pbreach, K, eps_max,
                   Pbreach_prime=None, eps_grid=None, tol=1e-8):
    r"""Solve for eps** = argmax_{eps in [0, eps_max]} V(eps) - C(eps).

    Parameters
    ----------
    price : callable
        ``lambda(eps) -> float``, the marginal robust price (>= 0).  Typically
        supplied by :mod:`shadowprice.price_meter`.
    Pbreach : callable
        ``P_breach(eps) -> float in [0,1]``, strictly decreasing, P(0)=1.
    K : float
        Single-breach loss (value scale).
    eps_max : float
        Upper end of the search interval (e.g. an ``eps_cert``).
    Pbreach_prime : callable, optional
        Derivative of ``P_breach``; if ``None`` a central finite difference is
        used.
    eps_grid : array, optional
        Grid for a robust bracket search before root-finding.

    Returns
    -------
    OptimalRadiusResult with keys:
        eps_star, interior (bool), foc_residual, net_value(eps_star),
        marginal_benefit_at_0, marginal_cost_at_0.
    """
    def Pp(e):
        if Pbreach_prime is not None:
            return Pbreach_prime(e)
        h = 1e-5 * max(1.0, e)
        return (Pbreach(e + h) - Pbreach(e - h)) / (2 * h)

    # F'(eps) = marginal net benefit = K|P_breach'| - lambda
    def Fprime(e):
        return K * abs(Pp(e)) - price(e)

    mb0 = K * abs(Pp(1e-8))          # marginal benefit at 0+
    mc0 = price(1e-8)                # true marginal cost at 0+ (lambda_0)

    # Interior optimum exists iff F'(0+) > 0.
    if Fprime(1e-8) <= 0:
        # corner: no robustness worth buying
        return OptimalRadiusResult(
            eps_star=0.0, interior=False, foc_residual=Fprime(1e-8),
            net_value=_net_value(price, Pbreach, K, 0.0),
            marginal_benefit_at_0=mb0, marginal_cost_at_0=mc0)

    # Bracket a sign change of F' on (0, eps_max].
    if eps_grid is None:
        eps_grid = np.linspace(1e-6, eps_max, 200)
    vals = np.array([Fprime(e) for e in eps_grid])
    sign_change = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(sign_change) == 0:
        # F' stays positive throughout -> optimum at the boundary eps_max.
        e_star = float(eps_max)
        return OptimalRadiusResult(
            eps_star=e_star, interior=False, foc_residual=Fprime(e_star),
            net_value=_net_value(price, Pbreach, K, e_star),
            marginal_benefit_at_0=mb0, marginal_cost_at_0=mc0)
    a, b = eps_grid[sign_change[0]], eps_grid[sign_change[0] + 1]
    e_star = float(brentq(Fprime, a, b, xtol=tol))
    return OptimalRadiusResult(
        eps_star=e_star, interior=True, foc_residual=Fprime(e_star),
        net_value=_net_value(price, Pbreach, K, e_star),
        marginal_benefit_at_0=mb0, marginal_cost_at_0=mc0)


def _net_value(price, Pbreach, K, eps):
    if eps <= 0:
        return float(K * (1 - Pbreach(0.0)))
    cost, _ = quad(price, 0.0, eps, limit=100)
    return float(K * (1 - Pbreach(eps)) - cost)

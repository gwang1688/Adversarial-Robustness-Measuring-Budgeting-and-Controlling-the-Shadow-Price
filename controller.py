r"""Closed-loop robustness-budget controller (v32 sec 7.6).

Drives the training-time robustness budget ``eps`` online toward the optimum
``eps**`` characterized by the first-order condition of Paper A (Prop. 8.4):

    F'(eps) := K |P_breach'(eps)| - lambda(eps) = 0,

i.e. marginal deployment benefit == marginal robust price.  Because the net
value ``V - C`` is strictly concave in the relevant range, ``F'`` is strictly
decreasing, and a damped fixed-point / proportional update converges:

    eps_{t+1} = clip( eps_t + gain * F'(eps_t), 0, eps_max ).

The controller only needs a *measured* price ``lambda(eps)`` (from the price
meter) and the deployment marginal benefit ``K |P_breach'(eps)|``; it never
retrains at every candidate eps.
"""
from __future__ import annotations
import numpy as np

__all__ = ["BudgetController", "ControlTrace"]


class ControlTrace(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class BudgetController:
    """Proportional (optionally damped-Newton) controller on F'(eps)=0.

    Parameters
    ----------
    price : callable
        ``lambda(eps) -> float`` measured marginal price (>=0).
    marginal_benefit : callable
        ``K |P_breach'(eps)| -> float``; the deployment marginal benefit.
    gain : float
        Proportional gain (step size on F').
    eps_max : float
        Upper clip (e.g. an ``eps_cert``).
    mode : {"proportional", "newton"}
        "newton" divides the step by a finite-difference F'' (faster, needs F'
        smooth); "proportional" is robust.
    """

    def __init__(self, price, marginal_benefit, gain=0.1, eps_max=np.inf,
                 mode="proportional"):
        self.price = price
        self.mb = marginal_benefit
        self.gain = gain
        self.eps_max = eps_max
        self.mode = mode

    def Fprime(self, eps):
        return self.mb(eps) - self.price(eps)

    def step(self, eps):
        g = self.Fprime(eps)
        if self.mode == "newton":
            h = 1e-4 * max(1.0, eps)
            fpp = (self.Fprime(eps + h) - self.Fprime(eps - h)) / (2 * h)
            delta = -g / fpp if abs(fpp) > 1e-9 else self.gain * g
        else:
            delta = self.gain * g
        return float(np.clip(eps + delta, 0.0, self.eps_max))

    def run(self, eps0, n_steps=200, tol=1e-6):
        """Iterate to convergence; return a ControlTrace."""
        eps = float(eps0)
        hist = [eps]
        for _ in range(n_steps):
            nxt = self.step(eps)
            hist.append(nxt)
            if abs(nxt - eps) < tol:
                eps = nxt
                break
            eps = nxt
        return ControlTrace(eps_star=eps, foc_residual=self.Fprime(eps),
                            n_steps=len(hist) - 1, trajectory=np.array(hist),
                            converged=abs(self.Fprime(eps)) < 1e-4)

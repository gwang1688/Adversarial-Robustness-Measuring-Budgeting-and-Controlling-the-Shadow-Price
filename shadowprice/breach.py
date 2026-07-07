r"""Fitting a breach model P_breach(eps) from red-team / evaluation data.

The decision calculus (Paper A, Prop. 8.4) needs a deployment breach
probability ``P_breach(eps)`` -- strictly decreasing, ``P_breach(0)=1``.  In
practice this is not known in closed form; it is *fit* to a handful of measured
points ``(eps_i, success_rate_i)`` from an attack suite (e.g. AutoAttack /
RobustBench evaluations at several radii).  ``fit_breach_model`` returns a
smooth callable and its derivative, ready for :func:`shadowprice.optimal_radius`.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit

__all__ = ["fit_breach_model"]


def _exp(e, rate):
    return np.exp(-rate * e)


def _logistic(e, k, e0):
    # decreasing logistic normalized to P(0)=1
    raw = 1.0 / (1.0 + np.exp(k * (e - e0)))
    raw0 = 1.0 / (1.0 + np.exp(k * (0.0 - e0)))
    return raw / raw0


def fit_breach_model(eps_points, success_rates, family="logistic"):
    r"""Fit ``P_breach(eps)`` to attack-success data.

    Parameters
    ----------
    eps_points : array
        Radii at which the attack suite was run.
    success_rates : array
        Measured attack success rate at each radius (= breach probability),
        in [0,1]; typically 1 at eps=0 and decreasing.
    family : {"logistic", "exp"}
        Functional form.  "logistic" is the flexible default; "exp" is the
        one-parameter exponential ``exp(-rate*eps)``.

    Returns
    -------
    dict with:
        Pbreach : callable  eps -> P_breach(eps)
        Pbreach_prime : callable  eps -> dP/deps
        params : fitted parameters
        rmse : fit residual on the given points
    """
    e = np.asarray(eps_points, float)
    y = np.clip(np.asarray(success_rates, float), 1e-6, 1 - 1e-6)
    if family == "exp":
        p, _ = curve_fit(_exp, e, y, p0=[1.0], bounds=(1e-6, 100), maxfev=10000)
        rate = float(p[0])
        P = lambda x: np.exp(-rate * np.asarray(x, float))
        Pp = lambda x: -rate * np.exp(-rate * np.asarray(x, float))
        params = {"rate": rate}
    elif family == "logistic":
        p, _ = curve_fit(_logistic, e, y, p0=[3.0, np.median(e) + 1e-3],
                         bounds=([1e-3, -10], [100, 100]), maxfev=20000)
        k, e0 = float(p[0]), float(p[1])
        raw0 = 1.0 / (1.0 + np.exp(k * (0.0 - e0)))
        def P(x):
            x = np.asarray(x, float)
            return (1.0 / (1.0 + np.exp(k * (x - e0)))) / raw0
        def Pp(x):
            x = np.asarray(x, float)
            s = 1.0 / (1.0 + np.exp(k * (x - e0)))
            return (-k * s * (1 - s)) / raw0
        params = {"k": k, "e0": e0}
    else:
        raise ValueError(f"unknown family '{family}'")
    rmse = float(np.sqrt(np.mean((np.asarray(P(e)) - y) ** 2)))
    return {"Pbreach": P, "Pbreach_prime": Pp, "params": params,
            "family": family, "rmse": rmse}

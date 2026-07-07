r"""Estimating the margin density p_S(t) in real networks (general case).

The dimensionless-collapse identity of Paper A is, in general,

    lambda_eps * (sigma * p_S(t)) ... ->   lambda_eps / (density factor) = sqrt(k_eps),

with ``p_S`` the density of the *standardized margin* t = y<w,x>/(sigma||w||_2).
The Gaussian ``phi(t)`` is only the linear-Gaussian special case; Paper A's
log-concave floor covers the general one.  In a real network there is no closed
form, so ``p_S(t)`` is estimated from the *empirical* distribution of
standardized margins over a batch (a kernel density estimate), and the collapse
residual becomes a genuine, model-agnostic self-check.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import gaussian_kde

__all__ = ["standardized_margins", "estimate_margin_density"]


def standardized_margins(scores, y=None):
    r"""Standardize margins to unit variance (mean-centered): t = m/std(m).

    ``scores`` are the (signed) per-example margins ``y*<w,x>`` or logit gaps;
    if ``y`` is given, ``scores`` are multiplied by ``y`` first.
    """
    m = np.asarray(scores, float)
    if y is not None:
        m = m * np.asarray(y, float)
    s = m.std()
    return m / (s if s > 0 else 1.0)


def estimate_margin_density(margins, bw=None):
    r"""Return a callable density ``p_S(t)`` estimated from a sample of
    standardized margins via Gaussian KDE.

    The returned function accepts scalars or arrays.  Use it in place of the
    Gaussian ``phi`` for the collapse residual on real models.
    """
    t = np.asarray(margins, float)
    kde = gaussian_kde(t, bw_method=bw)
    return lambda x: kde(np.atleast_1d(np.asarray(x, float)))

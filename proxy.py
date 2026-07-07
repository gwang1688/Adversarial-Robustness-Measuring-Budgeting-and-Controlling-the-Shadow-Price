r"""Retraining-free price proxy and the "measurement wall" (v32 sec 7.7).

Estimating ``lambda_eps`` at a target radius normally needs a robust model
*retrained at that eps*.  The retraining-free proxy avoids this: it measures
the input-gradient geometry once at the current operating point (one backward
pass, no robust retrain) and extrapolates the price across ``eps`` using the
soft-threshold structure of the robust-optimal discriminant (Paper A, Thm 4.1):

    w(eps) ~= soft_threshold_eps( w0 ),     w0 := measured |grad f|  (proxy for |mu|),
    lambda(eps) ~= (sqrt(k_eps(w(eps))) / sigma) * phi( t(eps) ).

The extrapolation is trustworthy while the active support ``S_eps`` is stable
and the operating point stays in the margin-density window; it breaks past the
**measurement wall** ``eps_wall`` -- the radius at which the support collapses
(few active coordinates) or ``|t|`` exits the window.  ``price`` returns the
extrapolated curve and flags points beyond the wall.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

__all__ = ["RetrainFreeProxy"]


class RetrainFreeProxy:
    r"""Extrapolate lambda(eps) from a single-point gradient measurement.

    Parameters
    ----------
    w0 : array
        Measured input-gradient magnitudes at the current model (proxy for the
        per-coordinate signal ``|mu|``), one backward pass, no robust retrain.
    sigma : float
        Noise scale of the projection (estimated from data once).
    signal : array, optional
        If the signed signal ``mu`` is known/estimated, pass it; otherwise
        ``w0`` is used as ``|mu|`` and signs are irrelevant to the price.
    M : float
        Margin-density window half-width (``|t|<=M``); sets the wall.
    min_active_frac : float
        Support-collapse threshold for the wall (fraction of the initial
        active set).
    """

    def __init__(self, w0, sigma, signal=None, M=3.0, min_active_frac=0.1):
        self.absmu = np.abs(np.asarray(w0, float)) if signal is None \
            else np.abs(np.asarray(signal, float))
        self.mu = self.absmu if signal is None else np.asarray(signal, float)
        self.sigma = float(sigma)
        self.M = float(M)
        self.min_active_frac = float(min_active_frac)
        self._n0 = int((self.absmu > 0).sum())

    def _w(self, eps):
        return np.maximum(self.absmu - eps, 0.0)          # soft-threshold magnitudes

    def _t(self, eps):
        w = self._w(eps)
        l1, l2 = w.sum(), np.linalg.norm(w)
        if l2 == 0:
            return np.nan
        # <w,|mu|> since signs align on the active set
        return (eps * l1 - w @ self.absmu) / (self.sigma * l2)

    def price(self, eps):
        r"""Extrapolated lambda(eps) with a `beyond_wall` flag.

        Accepts a scalar or array eps; returns a dict of arrays.
        """
        eps = np.atleast_1d(np.asarray(eps, float))
        lam, keps, tt, wall = [], [], [], []
        for e in eps:
            w = self._w(e)
            l1, l2 = w.sum(), np.linalg.norm(w)
            n_active = int((w > 0).sum())
            if l2 == 0:
                lam.append(0.0); keps.append(0.0); tt.append(np.nan)
                wall.append(True); continue
            t = (e * l1 - w @ self.absmu) / (self.sigma * l2)
            k = (l1 / l2) ** 2
            lam.append(np.sqrt(k) / self.sigma * norm.pdf(t))
            keps.append(k); tt.append(t)
            wall.append((abs(t) > self.M) or
                        (n_active < self.min_active_frac * self._n0))
        out = {"eps": eps, "lambda": np.array(lam), "k_eps": np.array(keps),
               "t": np.array(tt), "beyond_wall": np.array(wall)}
        # eps_wall = first eps flagged beyond the wall
        idx = np.where(out["beyond_wall"])[0]
        out["eps_wall"] = float(eps[idx[0]]) if len(idx) else float(eps[-1])
        return out

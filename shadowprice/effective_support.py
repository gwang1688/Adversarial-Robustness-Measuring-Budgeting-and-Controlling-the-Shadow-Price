r"""Effective-support (participation-ratio) diagnostics.

The pricing unit of the adversarial shadow price is the *effective support*

    k_eps(w) = (||w||_1 / ||w||_2)^2            (participation ratio)

of the (robust-optimal / linearized) discriminant ``w`` -- see Paper A,
Theorem 4.1.  In practice ``w`` is an input-gradient ``grad_x f(x)`` at an
operating point, so k_eps is a cheap, per-example *effective dimension* that
tracks how the l_inf/l_1 dimensional tax scales.  Values range in [1, d]:
1 for a one-hot (maximally localized) gradient, d for a fully delocalized one.

Everything here is framework-agnostic (plain array math).
"""
from __future__ import annotations
import numpy as np

__all__ = [
    "participation_ratio",
    "effective_support",
    "sqrt_price_factor",
    "SupportTracker",
]


def _as2d(w):
    w = np.asarray(w, dtype=float)
    if w.ndim == 1:
        w = w[None, :]
    return w.reshape(w.shape[0], -1)


def participation_ratio(w, axis=-1, eps=1e-12):
    r"""k_eps = (||w||_1 / ||w||_2)^2 along ``axis``.

    Accepts a vector or a batch (rows = examples).  Returns a scalar for a
    single vector, else a per-row array.
    """
    w = np.asarray(w, dtype=float)
    l1 = np.abs(w).sum(axis=axis)
    l2 = np.sqrt((w * w).sum(axis=axis))
    k = (l1 / np.maximum(l2, eps)) ** 2
    return k


# Alias matching the paper's name.
effective_support = participation_ratio


def sqrt_price_factor(w, axis=-1):
    r"""sqrt(k_eps) = ||w||_1 / ||w||_2, the geometric price prefactor."""
    return np.sqrt(participation_ratio(w, axis=axis))


class SupportTracker:
    """Accumulate the effective support of gradients over training/eval.

    Usage::

        tr = SupportTracker()
        for step, g in enumerate(grad_stream):     # g: (batch, d) input grads
            tr.update(g, step=step)
        tr.summary()          # dict of running stats
        tr.history()          # (steps, mean_k_eps) for plotting
    """

    def __init__(self):
        self._steps: list[int] = []
        self._mean: list[float] = []
        self._p10: list[float] = []
        self._p90: list[float] = []

    def update(self, grads, step: int | None = None):
        g = _as2d(grads)
        k = participation_ratio(g, axis=1)
        self._steps.append(len(self._steps) if step is None else step)
        self._mean.append(float(np.mean(k)))
        self._p10.append(float(np.percentile(k, 10)))
        self._p90.append(float(np.percentile(k, 90)))
        return k

    def history(self):
        return (np.array(self._steps),
                np.array(self._mean),
                np.array(self._p10),
                np.array(self._p90))

    def summary(self):
        if not self._mean:
            return {}
        return {
            "n_updates": len(self._mean),
            "k_eps_mean_last": self._mean[-1],
            "k_eps_mean_overall": float(np.mean(self._mean)),
            "k_eps_trend": float(self._mean[-1] - self._mean[0]),
        }

r"""The six low-cost Stage-A observables + dashboard (v32 sec 9.1).

Stage A of the two-stage validation collects six cheap observables at an
operating point -- none requires a robust retrain or a large sweep:

  O1  price            lambda_eps            (finite-diff / gradient proxy)
  O2  effective support k_eps = (||w||_1/||w||_2)^2
  O3  collapse residual |lambda*sigma/phi(t) - sqrt(k_eps)| / sqrt(k_eps)
                        -- the scale-invariance / dimensionless-collapse check
                           (Paper A: the price factor must equal sqrt(k_eps))
  O4  cert radius       eps_cert = <w,mu>/||w||_1
  O5  curvature ratio   kappa_inf upper/lower bracket (proxy trustworthiness)
  O6  train-infer gap   |lambda_proxy - lambda_finite_diff|
                        -- bounded by eps*E[kappa_inf] (Prop 8.2)

``collect_observables`` returns whichever are computable from the supplied
inputs; ``Dashboard`` renders a sweep.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

from .effective_support import participation_ratio
from .radius import cert_radius
from .price_meter import price_finite_diff, price_gradnorm
from .kappa_infty import kappa_infty as _kappa_bracket

__all__ = ["collect_observables", "Dashboard"]


def _t(w, mu, sigma, eps):
    w = np.asarray(w, float)
    l2 = np.linalg.norm(w)
    return (eps * np.abs(w).sum() - w @ mu) / (sigma * l2)


def collect_observables(eps, w, mu, sigma, H=None, risk_fn=None,
                        grad_fn=None, X=None, M=3.0,
                        density_fn=None, margins=None):
    """Compute the six Stage-A observables at radius ``eps``.

    ``density_fn`` (callable ``t -> p_S(t)``) or ``margins`` (a sample of
    standardized margins, from which a KDE is built) generalize the collapse
    residual beyond the Gaussian ``phi``; if neither is given, the Gaussian
    density is used (linear-Gaussian special case).
    """
    w = np.asarray(w, float)
    rec = {"eps": float(eps)}

    # O2 effective support
    k = float(participation_ratio(w)); rec["k_eps"] = k

    # margin density p_S at the operating point (general -> Gaussian fallback)
    t = _t(w, mu, sigma, eps)
    if density_fn is not None:
        pS = float(np.asarray(density_fn(t)).ravel()[0])
    elif margins is not None:
        from .margin_density import estimate_margin_density
        pS = float(np.asarray(estimate_margin_density(margins)(t)).ravel()[0])
    else:
        pS = float(norm.pdf(t))
    rec["margin_density"] = pS

    # O1 price: finite-diff if a risk_fn is given, else closed form
    price_cf = float(np.sqrt(k) / sigma * pS)
    if risk_fn is not None:
        rec["price"] = float(price_finite_diff(risk_fn, eps, h=1e-3))
    else:
        rec["price"] = price_cf

    # O3 dimensionless-collapse residual: price*sigma/p_S(t) should equal sqrt(k)
    if pS > 1e-12:
        factor = rec["price"] * sigma / pS
        rec["collapse_residual"] = float(abs(factor - np.sqrt(k)) / np.sqrt(k))
    else:
        rec["collapse_residual"] = np.nan

    # O4 certification radius
    rec["eps_cert"] = float(cert_radius(w, mu))

    # O5 curvature bracket ratio
    if H is not None:
        br = _kappa_bracket(H)
        rec["kappa_lower"], rec["kappa_upper"] = br["lower"], br["upper"]
        rec["kappa_ratio"] = br["bracket_ratio"]

    # O6 train-inference marginal gap
    if grad_fn is not None and X is not None and risk_fn is not None:
        proxy = price_gradnorm(grad_fn, X)["price"]
        rec["price_proxy"] = float(proxy)
        rec["train_infer_gap"] = float(abs(proxy - rec["price"]))

    rec["t"] = float(t)
    return rec


class Dashboard:
    """Render the six observables over an eps-sweep (matplotlib)."""

    def __init__(self, records):
        self.recs = list(records)

    def _col(self, key):
        return np.array([r.get(key, np.nan) for r in self.recs])

    def figure(self, path=None, dpi=130):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e = self._col("eps")
        panels = [
            ("price", r"$\lambda_\varepsilon$", "O1 price"),
            ("k_eps", r"$k_\varepsilon$", "O2 effective support"),
            ("collapse_residual", "residual", "O3 collapse residual"),
            ("kappa_ratio", "upper/lower", "O5 curvature bracket"),
            ("train_infer_gap", "|proxy-fd|", "O6 train-infer gap"),
        ]
        fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
        ax = ax.ravel()
        for i, (key, yl, ti) in enumerate(panels):
            y = self._col(key)
            ax[i].plot(e, y, "-o", ms=3)
            ax[i].set_xlabel(r"$\varepsilon$"); ax[i].set_ylabel(yl); ax[i].set_title(ti)
        # O4 cert radius as a vertical marker on the price panel
        ec = np.nanmedian(self._col("eps_cert"))
        ax[0].axvline(ec, ls="--", color="crimson")
        ax[0].text(ec, np.nanmin(self._col("price")),
                   f"  O4 $\\varepsilon_{{cert}}={ec:.2f}$", color="crimson", fontsize=8)
        ax[5].axis("off")
        ax[5].text(0.05, 0.5,
                   "O4 cert radius (dashed, O1)\n\nStage A: six cheap observables,\n"
                   "no robust retrain, single sweep.", fontsize=10, va="center")
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=dpi)
        return fig

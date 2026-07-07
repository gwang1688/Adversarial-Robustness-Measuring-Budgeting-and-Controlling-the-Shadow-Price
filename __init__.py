r"""shadowprice -- a measurement / decision / control toolkit for the
marginal price of adversarial robustness (companion to Paper A).

Layers
------
Measure  : price_meter, kappa_infty, effective_support   (+ proxy, batch 2)
Decide   : radius        (eps_cert, eps**)
Control  : controller     (batch 2)
Report   : diagnostics    (batch 2)

The core is framework-agnostic: estimators consume user-supplied ``grad_fn`` /
``risk_fn`` callables, so numpy/torch/jax all work.  Heavy deps (cvxpy for the
SDP curvature bound, torch for the adapter) are optional.
"""
from __future__ import annotations

from .effective_support import (
    participation_ratio, effective_support, sqrt_price_factor, SupportTracker,
)
from .radius import cert_radius, optimal_radius, OptimalRadiusResult
from .kappa_infty import (
    kappa_abs, kappa_spec, kappa_sdp, kappa_lower, kappa_infty,
)
from .price_meter import price_gradnorm, price_finite_diff, fd_budget, PriceMeter

__version__ = "0.1.0-dev"

__all__ = [
    "participation_ratio", "effective_support", "sqrt_price_factor",
    "SupportTracker", "cert_radius", "optimal_radius", "OptimalRadiusResult",
    "kappa_abs", "kappa_spec", "kappa_sdp", "kappa_lower", "kappa_infty",
    "price_gradnorm", "price_finite_diff", "fd_budget", "PriceMeter",
]

# batch 2: control / diagnostics / proxy
from .controller import BudgetController, ControlTrace
from .proxy import RetrainFreeProxy
from .diagnostics import collect_observables, Dashboard
__all__ += ["BudgetController", "ControlTrace", "RetrainFreeProxy",
            "collect_observables", "Dashboard"]

# batch 4 (revision): breach fitting + general margin density
from .breach import fit_breach_model
from .margin_density import standardized_margins, estimate_margin_density
__all__ += ["fit_breach_model", "standardized_margins", "estimate_margin_density"]

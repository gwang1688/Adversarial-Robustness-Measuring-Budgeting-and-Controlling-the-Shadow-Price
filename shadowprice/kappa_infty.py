r"""l_inf quadratic-form curvature ``kappa_inf`` (curvature early-warning).

Paper A (Prop. 8.2, Rmk. 8.3) controls the gap between the cheap first-order
price proxy and the true price by

    kappa_inf(x) = max_{s in {+-1}^d} | s^T H s |,   H = Hessian_x f(x).

Computing it exactly is a Boolean quadratic program (NP-hard), but the
engineering bridge only needs a computable *upper* bound (kappa_inf enters an
O(eps) correction term).  We provide, in increasing tightness / cost:

* ``kappa_abs``   : sum_ij |H_ij|            -- always valid, O(d^2), trivial.
* ``kappa_spec``  : d * ||H||_2              -- spectral, loose.
* ``kappa_sdp``   : SDP relaxation           -- dimension-free-quality upper
                    bound via max <H,X>, X>=0, diag(X)=1 (needs cvxpy).
* ``kappa_lower`` : random +-1 sampling      -- a valid *lower* bound.

``kappa_infty`` bundles a cheap upper bound and a sampled lower bound so a
practitioner can see the bracket in one call.
"""
from __future__ import annotations
import numpy as np

__all__ = ["kappa_abs", "kappa_spec", "kappa_sdp", "kappa_lower", "kappa_infty"]


def _sym(H):
    H = np.asarray(H, float)
    return 0.5 * (H + H.T)


def kappa_abs(H):
    r"""Trivial valid upper bound  sum_ij |H_ij| >= max_s s^T H s."""
    H = _sym(H)
    return float(np.abs(H).sum())


def kappa_spec(H):
    r"""Spectral upper bound  d * ||H||_2  (loose; the sqrt(d) factor of Rmk 8.3)."""
    H = _sym(H)
    d = H.shape[0]
    return float(d * np.linalg.norm(H, 2))


def kappa_lower(H, n_samples=2000, seed=0):
    r"""Lower bound on max_{s in {+-1}^d} |s^T H s| by random sign sampling."""
    H = _sym(H)
    d = H.shape[0]
    rng = np.random.default_rng(seed)
    best = 0.0
    # include a greedy 1-opt refinement of the best random draw
    for _ in range(max(1, n_samples // 50)):
        S = rng.choice([-1.0, 1.0], size=(50, d))
        vals = np.einsum("ij,jk,ik->i", S, H, S)
        j = int(np.argmax(np.abs(vals)))
        if abs(vals[j]) > best:
            best = float(abs(vals[j]))
            s = S[j].copy()
    # 1-opt local search from the incumbent
    improved = True
    Hs = H @ s
    while improved:
        improved = False
        cur = float(s @ Hs)
        for i in range(d):
            # flipping s_i changes s^T H s by -4 s_i (Hs_i) + 4 H_ii
            delta = -4 * s[i] * Hs[i] + 4 * H[i, i]
            if abs(cur + delta) > abs(cur) + 1e-12:
                s[i] = -s[i]
                Hs = H @ s
                cur = float(s @ Hs)
                improved = True
    return max(best, abs(cur))


def kappa_sdp(H, solver=None):
    r"""SDP-relaxation upper bound on  max_{s in {+-1}^d} s^T H s.

    Solves  maximize <H, X>  s.t.  X >= 0 (PSD), diag(X) = 1.  This is the
    standard MaxQP / Goemans-Williamson relaxation and is a valid upper bound
    on the Boolean maximum (Rmk. 8.3; the pi/2-theorem gives a 2/pi ratio in
    the PSD case).  Requires ``cvxpy``; raises ImportError otherwise.
    """
    try:
        import cvxpy as cp
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "kappa_sdp needs cvxpy (pip install cvxpy). Use kappa_abs/kappa_spec "
            "for a dependency-free upper bound.") from e
    H = _sym(H)
    d = H.shape[0]

    def _sdp(G):
        X = cp.Variable((d, d), symmetric=True)
        constraints = [X >> 0] + [X[i, i] == 1 for i in range(d)]
        prob = cp.Problem(cp.Maximize(cp.trace(G @ X)), constraints)
        prob.solve(solver=solver)
        return float(prob.value)

    # kappa_inf = max_s |s^T H s| = max( max_s s^T H s , max_s s^T(-H) s ).
    return max(_sdp(H), _sdp(-H))


def kappa_infty(H, upper="auto", n_samples=2000, seed=0):
    r"""Return a validated bracket ``{lower, upper, method}`` for kappa_inf.

    ``upper='auto'`` uses the SDP bound when cvxpy is available (tightest),
    else falls back to ``sum|H_ij|``.  The lower bound is always the sampled
    Boolean value.  A large upper/lower ratio warns that the first-order price
    proxy may be unreliable at the current eps.
    """
    H = _sym(H)
    lo = kappa_lower(H, n_samples=n_samples, seed=seed)
    if upper == "auto":
        try:
            up = kappa_sdp(H)
            method = "sdp"
        except ImportError:
            up = kappa_abs(H)
            method = "abs"
    elif upper == "sdp":
        up, method = kappa_sdp(H), "sdp"
    elif upper == "abs":
        up, method = kappa_abs(H), "abs"
    elif upper == "spec":
        up, method = kappa_spec(H), "spec"
    else:
        raise ValueError(f"unknown upper='{upper}'")
    return {"lower": lo, "upper": up, "method": method,
            "bracket_ratio": up / max(lo, 1e-12)}

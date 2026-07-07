# shadowprice

**Measure, budget, and control the marginal price of adversarial robustness.**

`shadowprice` is a small, framework-agnostic, open-source toolkit that turns the
theory of the *adversarial shadow price* into a practical **measure → decide →
control** workflow. It answers three questions that a robustness *evaluation*
(attack/benchmark) does not: **what does robustness cost on my model right now**,
**how much should I buy**, and **how do I steer the training budget there** —
without an adversarial retraining sweep.

<!-- Suggested GitHub topics: adversarial-robustness, adversarial-machine-learning,
shadow-price, robustness-certification, trustworthy-ml, deep-learning, pytorch,
robustness-budget, ml-safety -->

---

## Why

A companion paper (**Paper A**) proves that the marginal price of adversarial
robustness — the subgradient `λ_ε = ∂L*/∂ε` of a constrained value function in
the adversarial radius `ε` — has a floor whose correct pricing unit is the
**effective support** of the robust-optimal discriminant,

```
λ_ε  ≥  c_M · √k_ε / σ ,        k_ε = ( ‖w*‖₁ / ‖w*‖₂ )²
```

Paper A says the price *exists* and identifies its unit `√k_ε` (which specializes
to `√d`, `√k`, or `√r_eff`). It does **not** tell a practitioner what the price
*is* on their model, how much robustness to buy, or how to steer the budget.
`shadowprice` closes that gap.

## What it does — three layers + a self-check

| layer | module | what it does |
|---|---|---|
| **measure** | `price_meter` | price meter `λ_ε ≈ E‖∇ₓℓ‖₁` (one backward pass/batch, no adversarial inner loop) + finite-difference price + the `δ⁻³` sampling budget |
| | `kappa_infty` | curvature early-warning `κ∞`: SDP upper bound (optional `cvxpy`) + random-sign lower bound; a wide bracket flags an untrustworthy first-order price |
| | `effective_support` | `k_ε = (‖w‖₁/‖w‖₂)²` effective-dimension diagnostic + training-time tracker |
| | `proxy` | **retraining-free** price extrapolation across `ε`, with a **measurement wall** |
| **decide** | `radius` | certification radius `ε_cert` and economically-optimal radius `ε**` (+ comparative statics) |
| **control** | `controller` | closed-loop budget controller that drives training to `ε**` (proportional / damped-Newton) |
| **report** | `diagnostics` | the six Stage-A observables + a dimensionless-collapse **runtime self-check** |
| **adapt** | `adapters.torch` | optional PyTorch `grad_fn` / `risk_fn` (lazy import) |

The **dimensionless-collapse residual** (`λ_ε·σ/p_S(t) = √k_ε`) is an always-on
consistency monitor: it trips *before* a mis-measured price corrupts a budget.

## Install

```bash
pip install shadowprice                  # core: numpy, scipy
pip install "shadowprice[sdp,plots]"     # + cvxpy (tightest curvature bound), matplotlib
pip install "shadowprice[torch]"         # + PyTorch adapter
```

Nothing in the import path hard-requires `cvxpy` or `torch`; the curvature bound
degrades gracefully to a dependency-free formula and the adapter is imported lazily.

## Quick start

```python
import shadowprice as sp

# --- measure ---
k   = sp.participation_ratio(grad_x)          # effective dimension k_eps in [1, d]
out = sp.price_gradnorm(grad_fn, X)           # one-pass price + bootstrap CI
br  = sp.kappa_infty(H)                        # curvature bracket {lower, upper}

# --- decide ---
ec  = sp.cert_radius(w, mu)                    # certification radius
res = sp.optimal_radius(price_fn, Pbreach, K, eps_max=ec)   # res.eps_star == eps**

# --- control ---
ctl = sp.BudgetController(price_fn, marginal_benefit, eps_max=ec)
eps_star = ctl.run(eps0=0.1).eps_star          # closed-loop to eps**
```

Fit the breach curve from a few red-team points (e.g. AutoAttack/RobustBench),
then read off the budget:

```python
eps_pts   = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
successes = [0.90, 0.74, 0.58, 0.36, 0.21, 0.09]
fit = sp.fit_breach_model(eps_pts, successes, family="logistic")
res = sp.optimal_radius(price_fn, fit["Pbreach"], K=2.0,
                        eps_max=ec, Pbreach_prime=fit["Pbreach_prime"])
res.eps_star   # -> the training budget
```

## Validation status

| stage | model | establishes |
|---|---|---|
| **A** (done) | linear–Gaussian, closed forms | estimators are correct — **13/13** exact checks |
| **A.5** (done) | small trained MLP (CPU, real gradients + Hessian) | instruments run & cohere on a real model; the `κ∞` / train–inference observables populate |
| **B** (specified) | deep nets + AutoAttack/RobustBench | real-scale behavior + falsification protocol (runnable template provided; not executed) |

Reproduce everything: `pytest` runs the 13 closed-form checks; the `examples/`
scripts regenerate all figures deterministically (seeded, CPU, pinned versions).

## Positioning

`shadowprice` does **not** attack, benchmark, or certify — it **prices**. It is a
complementary layer that consumes the output of tools that do:

| tool | attacks / evaluates | certifies | reads price `λ_ε` | budget `ε**` |
|---|:--:|:--:|:--:|:--:|
| Foolbox / AutoAttack / RobustBench | ✓ | | | |
| Randomized smoothing | | ✓ | | |
| **shadowprice** (this work) | | (coarse `ε_cert`) | ✓ | ✓ |

An attack suite answers *“how robust is this model?”*; `shadowprice` answers
*“what does more robustness cost, and how much should I buy?”*

## Papers

- **Paper A (theory).** *The Effective-Support Floor of the Adversarial Shadow
  Price: An Ω(√kₑ) Lower Bound.* Proves the price floor and identifies the
  pricing unit `√k_ε`.
- **Paper B (this toolkit).** *The Instrument Panel for Adversarial Robustness:
  Measuring, Budgeting, and Controlling the Shadow Price.* / 中文版:*对抗鲁棒性的
  仪表盘:影子价格的测量、预算与控制。*

## Citation

If you use `shadowprice` or build on these results, please cite the papers.
<!-- Update year / venue / arXiv id / DOI once published. -->

```bibtex
@misc{wang2026shadowprice_theory,
  title  = {The Effective-Support Floor of the Adversarial Shadow Price:
            An {$\Omega(\sqrt{k_\varepsilon})$} Lower Bound},
  author = {Wang, Guangyu},
  year   = {2026},
  note   = {Manuscript}
}

@misc{wang2026shadowprice_toolkit,
  title  = {The Instrument Panel for Adversarial Robustness:
            Measuring, Budgeting, and Controlling the Shadow Price},
  author = {Wang, Guangyu},
  year   = {2026},
  note   = {Manuscript; software: \url{https://github.com/billgywang/shadowprice}}
}
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

## Contact

Guangyu Wang (王广宇) · billgywang@gmail.com

---

<sub>Keywords: adversarial robustness · adversarial machine learning · shadow price ·
marginal price of robustness · robustness budget · certification radius ·
effective support / participation ratio · trustworthy ML · ML safety · PyTorch.</sub>

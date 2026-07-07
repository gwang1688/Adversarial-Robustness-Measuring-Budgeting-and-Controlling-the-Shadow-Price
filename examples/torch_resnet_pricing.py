r"""End-to-end shadowprice pricing on a real PyTorch model + attack suite.

    >>> NOT EXECUTED IN THE PAPER'S CI <<<
This script requires PyTorch (and optionally the `autoattack` / `robustbench`
packages), which are not installed in the paper's build environment.  It is a
complete, runnable reference for Stage B: it trains a small CNN, runs the
instrument panel through the torch adapter on *real* input gradients and a
Hessian-vector curvature probe, obtains a breach curve from an attack suite,
and solves for the optimal robustness budget.  Run it in a torch environment.

Reproducibility: set the seeds below; results depend on the model/data, which
is why we report no numbers here (see the paper's Stage-A / A.5 evidence).
"""
import numpy as np
import shadowprice as sp
from shadowprice.adapters.torch import torch_grad_fn, torch_risk_fn

# torch is imported lazily by the adapter; guard the top-level demo too.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None


def build_model(d_in=3 * 32 * 32, n_classes=10):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(d_in, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, n_classes),
    )


def hvp_kappa_probe(model, loss_fn, X, y, n_probes=32, seed=0):
    r"""Estimate the l_inf curvature kappa_inf via random-sign Hessian-vector
    products (a torch-native lower bound; pair with an SDP audit on a subspace).

    kappa_inf = max_{s in {+-1}^d} |s^T H s|; each HVP gives s^T H s for one s.
    """
    g = torch.Generator().manual_seed(seed)
    X = X.clone().requires_grad_(True)
    loss = loss_fn(model(X), y)
    grad, = torch.autograd.grad(loss, X, create_graph=True)
    best = 0.0
    flat = grad.reshape(-1)
    for _ in range(n_probes):
        s = (torch.randint(0, 2, flat.shape, generator=g).float() * 2 - 1)
        Hs, = torch.autograd.grad((flat * s).sum(), X, retain_graph=True)
        val = float((Hs.reshape(-1) * s).sum())
        best = max(best, abs(val))
    return best


def main():
    assert torch is not None, "install torch to run this example"
    torch.manual_seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data (substitute CIFAR-10 / your loader) ---
    # from torchvision import datasets, transforms  # real loader
    n, d = 512, 3 * 32 * 32
    X = torch.randn(n, 3, 32, 32, device=device)
    yv = torch.randint(0, 10, (n,), device=device)

    model = build_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    for _ in range(200):                       # train (use a real loop/epochs)
        opt.zero_grad(); loss_fn(model(X), yv).backward(); opt.step()

    # --- Move 1: measure the price (one backward pass per batch) ---
    grad_fn = torch_grad_fn(model, loss_fn, y=yv, device=device)
    meter = sp.price_gradnorm(grad_fn, X.detach().cpu().numpy())
    print("price proxy E||grad_x loss||_1:", meter["price"], "CI", meter["ci95"])

    # --- Move 2: effective support + curvature trust ---
    G = grad_fn(X.detach().cpu().numpy())
    k_eps = float(np.mean(sp.participation_ratio(G, axis=1)))
    kappa = hvp_kappa_probe(model, loss_fn, X, yv)      # torch-native lower bound
    print("k_eps:", k_eps, " kappa_inf (HVP lower):", kappa)

    # --- Move 4a: breach curve from an attack suite (AutoAttack / RobustBench) ---
    # from autoattack import AutoAttack                      # real suite
    # aa = AutoAttack(model, norm='Linf', eps=e, version='standard')
    # success = 1 - clean_accuracy(aa.run_standard_evaluation(X, yv))
    # Here we use the adapter's minimal PGD as a stand-in:
    radii = [0.0, 2/255, 4/255, 8/255, 16/255, 32/255]
    breach = []
    for e in radii:
        risk_fn = torch_risk_fn(model, loss_fn, X.detach().cpu().numpy(),
                                yv.cpu().numpy(), device=device, n_pgd=20)
        breach.append(risk_fn(e))
    fit = sp.fit_breach_model(radii, breach, family="logistic")
    print("breach fit:", fit["params"], "rmse", fit["rmse"])

    # --- Move 4b: price schedule + optimal budget ---
    risk_fn = torch_risk_fn(model, loss_fn, X.detach().cpu().numpy(),
                            yv.cpu().numpy(), device=device, n_pgd=20)
    price_fn = lambda e: max(sp.price_finite_diff(risk_fn, e, h=1/255), 1e-6)
    eps_max = 32/255
    res = sp.optimal_radius(price_fn, fit["Pbreach"], K=1.0, eps_max=eps_max,
                            Pbreach_prime=fit["Pbreach_prime"])
    print("optimal robustness radius eps** =", res.eps_star, "interior", res.interior)

    # --- Move 5: closed-loop control (drives training-time budget) ---
    mb = lambda e: 1.0 * abs(fit["Pbreach_prime"](e))
    ctl = sp.BudgetController(price_fn, mb, gain=1.0, eps_max=eps_max)
    print("controller ->", ctl.run(eps0=2/255, n_steps=100).eps_star)


if __name__ == "__main__":
    main()

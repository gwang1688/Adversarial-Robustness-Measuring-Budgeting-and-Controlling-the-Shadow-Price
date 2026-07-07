r"""Torch adapter: turn a torch model + loss into the framework-agnostic
callables consumed by :mod:`shadowprice.price_meter` and the diagnostics.

Torch is imported lazily, so importing :mod:`shadowprice` never requires it.

Example
-------
>>> from shadowprice.adapters.torch import torch_grad_fn, torch_risk_fn
>>> grad_fn = torch_grad_fn(model, loss_fn)          # X (np) -> input grads (np)
>>> pm = PriceMeter(grad_fn=grad_fn,
...                 risk_fn=torch_risk_fn(model, loss_fn, y, attack="linf"))
>>> pm.proxy(X)                                       # E||grad_x loss||_1
"""
from __future__ import annotations
import numpy as np

__all__ = ["torch_grad_fn", "torch_risk_fn"]


def _torch():
    try:
        import torch
        return torch
    except Exception as e:  # pragma: no cover
        raise ImportError("adapters.torch needs PyTorch installed.") from e


def torch_grad_fn(model, loss_fn, y=None, device="cpu"):
    r"""Return ``grad_fn(X_np) -> input-gradients (np, shape (batch, d))``.

    Computes ``grad_x loss(model(x), y)`` for each row via one backward pass.
    """
    torch = _torch()

    def grad_fn(X_np):
        X = torch.as_tensor(np.asarray(X_np, np.float32), device=device)
        X.requires_grad_(True)
        out = model(X)
        yy = out.detach().argmax(-1) if y is None else torch.as_tensor(y, device=device)
        loss = loss_fn(out, yy)
        g, = torch.autograd.grad(loss, X)
        return g.reshape(g.shape[0], -1).detach().cpu().numpy()

    return grad_fn


def torch_risk_fn(model, loss_fn, X_np, y, device="cpu", n_pgd=20, step=None):
    r"""Return ``risk_fn(eps) -> robust 0/1 risk`` under an l_inf PGD attack.

    A minimal PGD loop; the finite-difference price meter differentiates this.
    """
    torch = _torch()
    X0 = torch.as_tensor(np.asarray(X_np, np.float32), device=device)
    yy = torch.as_tensor(np.asarray(y), device=device)

    def risk_fn(eps):
        if eps <= 0:
            with torch.no_grad():
                pred = model(X0).argmax(-1)
            return float((pred != yy).float().mean())
        st = step if step is not None else 2.5 * eps / n_pgd
        X = X0.clone()
        for _ in range(n_pgd):
            X.requires_grad_(True)
            loss = loss_fn(model(X), yy)
            g, = torch.autograd.grad(loss, X)
            with torch.no_grad():
                X = X + st * g.sign()
                X = torch.max(torch.min(X, X0 + eps), X0 - eps)
            X = X.detach()
        with torch.no_grad():
            pred = model(X).argmax(-1)
        return float((pred != yy).float().mean())

    return risk_fn

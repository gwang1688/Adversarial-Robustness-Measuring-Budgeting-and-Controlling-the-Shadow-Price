r"""A tiny pure-numpy MLP + trainer for the shadowprice real-training demo.

Not part of the core library -- it exists so the paper's empirical section can
run an *actual* nonlinear model (real input gradients, a real nonzero Hessian)
on a CPU, bridging the linear closed forms (Stage A) toward Stage B.
"""
import numpy as np


def relu(z): return np.maximum(z, 0.0)
def d_relu(z): return (z > 0).astype(float)


class TinyMLP:
    """1-hidden-layer MLP for binary classification with logistic loss.

    Exposes input gradients and (finite-difference) input Hessians of the loss,
    which is what the instrument panel consumes.
    """

    def __init__(self, d, h=32, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(d), (d, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, 1 / np.sqrt(h), (h,))
        self.b2 = 0.0

    def logit(self, X):
        Z = X @ self.W1 + self.b1
        return relu(Z) @ self.W2 + self.b2

    def loss(self, X, y):
        # y in {-1,+1}; logistic loss on margins
        m = y * self.logit(X)
        return np.mean(np.log1p(np.exp(-m)))

    def grad_params(self, X, y):
        Z = X @ self.W1 + self.b1
        A = relu(Z)
        f = A @ self.W2 + self.b2
        m = y * f
        g = -y * (1.0 / (1.0 + np.exp(m)))          # dL/df per example
        gW2 = A.T @ g / len(y)
        gb2 = g.mean()
        dA = np.outer(g, self.W2) * d_relu(Z)
        gW1 = X.T @ dA / len(y)
        gb1 = dA.mean(0)
        return gW1, gb1, gW2, gb2

    def grad_input(self, X, y):
        """dL/dx per example (batch, d) -- one 'backward pass'."""
        Z = X @ self.W1 + self.b1
        A = relu(Z)
        f = A @ self.W2 + self.b2
        m = y * f
        g = -y * (1.0 / (1.0 + np.exp(m)))          # (batch,)
        dfdx = (d_relu(Z) * self.W2) @ self.W1.T     # (batch, d)
        return g[:, None] * dfdx

    def hessian_input_meanloss(self, X, y, h=1e-4):
        """Finite-difference input Hessian of the mean loss (d x d)."""
        d = X.shape[1]
        g0 = self.grad_input(X, y).mean(0)
        Hm = np.zeros((d, d))
        for i in range(d):
            Xp = X.copy(); Xp[:, i] += h
            gi = self.grad_input(Xp, y).mean(0)
            Hm[:, i] = (gi - g0) / h
        return 0.5 * (Hm + Hm.T)

    def fit(self, X, y, steps=400, lr=0.5, wd=0.0):
        for _ in range(steps):
            gW1, gb1, gW2, gb2 = self.grad_params(X, y)
            self.W1 -= lr * (gW1 + wd * self.W1); self.b1 -= lr * gb1
            self.W2 -= lr * (gW2 + wd * self.W2); self.b2 -= lr * gb2
        return self

    def fit_adv(self, X, y, eps_train, steps=200, lr=0.15, wd=5e-3, n_pgd=7):
        """PGD adversarial training at radius eps_train (l_inf)."""
        for _ in range(steps):
            if eps_train > 0:
                st = 2.5 * eps_train / n_pgd
                Xa = X.copy()
                for _ in range(n_pgd):
                    g = self.grad_input(Xa, y)
                    Xa = np.clip(Xa + st * np.sign(g), X - eps_train, X + eps_train)
            else:
                Xa = X
            gW1, gb1, gW2, gb2 = self.grad_params(Xa, y)
            self.W1 -= lr * (gW1 + wd * self.W1); self.b1 -= lr * gb1
            self.W2 -= lr * (gW2 + wd * self.W2); self.b2 -= lr * gb2
        return self


def make_data(n=1500, d=40, seed=1, noise=3.0):
    rng = np.random.default_rng(seed)
    mu = np.abs(rng.normal(0, 1, d)) + 0.4
    y = rng.choice([-1.0, 1.0], size=n)
    X = y[:, None] * mu[None, :] + rng.normal(0, noise, (n, d))
    return X, y, mu


def make_tradeoff(n=2000, seed=1, mnr=0.35, mr=1.2, d_nr=38):
    """A task with a robustness-accuracy tradeoff (Tsipras et al. style):
    feature 0 is *robust* (strong per-feature signal, hard to flip); features
    1..d_nr are *non-robust* (weak per-feature signal but jointly predictive,
    easily flipped by a small l_inf perturbation). Standard training leans on
    the non-robust features (high clean accuracy, breachable); adversarial
    training shifts weight to the robust feature (lower clean accuracy, lower
    breach rate) -- so P_breach(eps_train) genuinely decreases.
    """
    rng = np.random.default_rng(seed)
    y = rng.choice([-1.0, 1.0], size=n)
    xr = (y * mr + rng.normal(0, 1.0, n))[:, None]
    xnr = y[:, None] * mnr + rng.normal(0, 1.0, (n, d_nr))
    return np.concatenate([xr, xnr], 1), y

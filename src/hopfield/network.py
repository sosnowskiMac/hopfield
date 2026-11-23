from dataclasses import dataclass

import numpy as np

EPS = 1e-9


@dataclass
class RecallResult:
    history: list
    status: str  # "stable" | "cycle" | "max_steps"
    steps: int
    period: int = 1

    @property
    def final(self):
        return self.history[-1]


class HopfieldNetwork:
    """Sieć Hopfielda z wagami w postaci niskiego rzędu: W = A @ Bt.T - diag(d).

    Dzięki temu bitmapy 200x300 (n=60000) nie wymagają macierzy n x n.
    Hebb: A = X.T, Bt = X.T / n.   Oja: A = Bt = Q (ortonormalna baza podprzestrzeni wzorców).
    """

    def __init__(self, shape, seed=None):
        self.shape = (shape, 1) if isinstance(shape, int) else tuple(shape)
        self.n = self.shape[0] * self.shape[1]
        self.rng = np.random.default_rng(seed)
        self.A = self.Bt = self.d = None
        self.state = self.rng.choice([-1.0, 1.0], self.n)

    # ---------- uczenie ----------
    def train(self, patterns, rule="hebb", zero_diag=None, **kw):
        X = np.atleast_2d(np.asarray(patterns, float))
        if X.shape[1] != self.n:
            raise ValueError(f"wzorce mają {X.shape[1]} elementów, sieć {self.n}")
        if rule == "hebb":
            self.A = np.ascontiguousarray(X.T)
            self.Bt = self.A / self.n
            zero_diag = True if zero_diag is None else zero_diag
        elif rule == "oja":
            self.A = self.Bt = np.ascontiguousarray(self._oja(X, **kw))
            zero_diag = False if zero_diag is None else zero_diag
        else:
            raise ValueError(rule)
        self.d = (self.A * self.Bt).sum(1) if zero_diag else np.zeros(self.n)
        return self

    def _oja(self, X, eta=None, epochs=3000, tol=1e-7, solver="iterative"):
        k = np.linalg.matrix_rank(X)
        if solver == "svd":  # granica reguły Oji: ortonormalna baza span(X)
            return np.linalg.svd(X.T, full_matrices=False)[0][:, :k]
        p = len(X)
        eta = 0.1 / p if eta is None else eta
        Xs = X * np.sqrt(p / self.n)
        Q = self.rng.normal(size=(self.n, k)) / np.sqrt(self.n)
        for _ in range(epochs):
            Q0 = Q.copy()
            for j in self.rng.permutation(p):
                x = Xs[j]
                y = Q.T @ x
                Q += eta * np.outer(x - Q @ y, y)  # reguła podprzestrzeni Oji
            if np.abs(Q - Q0).max() < tol:
                break
        return Q

    # ---------- dynamika ----------
    def weights(self):
        return self.A @ self.Bt.T - np.diag(self.d)

    def field(self, x):
        return (x @ self.Bt) @ self.A.T - self.d * x

    @staticmethod
    def _sign(s, x):
        return np.where(np.abs(s) < EPS, x, np.sign(s))

    def update(self, x):
        return self._sign(self.field(x), x)

    def stable_mask(self, X):
        X = np.atleast_2d(X)
        return (self.update(X) == X).all(1)

    def energy(self, x):
        return -0.5 * x @ self.field(x)

    def set_state(self, x):
        self.state = np.asarray(x, float).copy()

    def recall(self, x0=None, mode="sync", max_steps=100, record_every=None):
        x = (self.state if x0 is None else np.asarray(x0, float)).copy()
        hist = [x.copy()]
        if mode == "sync":
            res = self._recall_sync(x, hist, max_steps)
        elif mode == "async":
            res = self._recall_async(x, hist, max_steps, record_every)
        else:
            raise ValueError(mode)
        self.state = res.final.copy()
        return res

    def _recall_sync(self, x, hist, max_steps):
        key = lambda v: np.packbits(v > 0).tobytes()
        seen = {key(x): 0}
        for t in range(1, max_steps + 1):
            x = self.update(x)
            hist.append(x.copy())
            k = key(x)
            if k in seen:
                period = t - seen[k]
                return RecallResult(hist, "stable" if period == 1 else "cycle", t, period)
            seen[k] = t
        return RecallResult(hist, "max_steps", max_steps, 0)

    def _recall_async(self, x, hist, max_steps, record_every):
        every = record_every or max(1, self.n // 20)
        h = x @ self.Bt
        count = 0
        for sweep in range(1, max_steps + 1):
            changed = False
            for i in self.rng.permutation(self.n):
                s = self.A[i] @ h - self.d[i] * x[i]
                new = x[i] if abs(s) < EPS else (1.0 if s > 0 else -1.0)
                if new != x[i]:
                    h += self.Bt[i] * (new - x[i])
                    x[i] = new
                    changed = True
                count += 1
                if count % every == 0:
                    hist.append(x.copy())
            if not changed:
                if not np.array_equal(hist[-1], x):
                    hist.append(x.copy())
                return RecallResult(hist, "stable", sweep - 1)
        if not np.array_equal(hist[-1], x):
            hist.append(x.copy())
        return RecallResult(hist, "max_steps", max_steps, 0)

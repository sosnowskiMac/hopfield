import re
from pathlib import Path

import numpy as np


def parse_shape(path, n):
    """Kształt (wiersze, kolumny) z nazwy pliku (np. 'x_25x50.csv'), inaczej kwadrat."""
    m = re.search(r"(\d+)x(\d+)", Path(path).stem)
    if m and int(m[1]) * int(m[2]) == n:
        return int(m[1]), int(m[2])
    r = int(round(np.sqrt(n)))
    if r * r == n:
        return r, r
    raise ValueError(f"Nie można ustalić kształtu dla {path} (n={n}); dodaj HxW do nazwy pliku.")


def load_patterns(path):
    X = np.loadtxt(path, delimiter=",", ndmin=2)
    if not np.isin(X, (-1, 1)).all():
        raise ValueError(f"{path}: oczekiwano wartości 1 i -1")
    return X, parse_shape(path, X.shape[1])


def corrupt(x, shape, kind, level, rng):
    y = x.copy()
    if kind == "noise":
        idx = rng.choice(x.size, int(round(level * x.size)), replace=False)
        y[idx] *= -1
    elif kind == "block":
        h, w = shape
        s = np.sqrt(level)
        bh, bw = int(round(h * s)), int(round(w * s))
        r, c = rng.integers(0, h - bh + 1), rng.integers(0, w - bw + 1)
        y.reshape(shape)[r:r + bh, c:c + bw] = rng.choice([-1.0, 1.0], (bh, bw))
    else:
        raise ValueError(kind)
    return y


def subset(X, p, rng):
    return X[rng.choice(len(X), p, replace=False)]


def pattern_stats(X):
    p, n = X.shape
    C = X @ X.T / n
    off = C[~np.eye(p, dtype=bool)]
    return dict(
        p=p, n=n, load=p / n, rank=int(np.linalg.matrix_rank(X)),
        mean_abs_overlap=float(np.abs(off).mean()) if off.size else 0.0,
        max_abs_overlap=float(np.abs(off).max()) if off.size else 0.0,
        bias=float(abs(X.mean())),
    )

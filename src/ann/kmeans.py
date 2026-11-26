"""k-means with k-means++ seeding, used by IVF (coarse quantiser) and PQ (sub-codebooks).

Distances use ``‖x‖² − 2x·c + ‖c‖²`` in blocks, so memory stays bounded. An empty cluster
is re-seeded with the point that currently has the largest error, which is what FAISS
does too (it splits the biggest cluster).
"""

from __future__ import annotations

import numpy as np


def sq_dists(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """(n, k) squared Euclidean distances, clipped at 0 against rounding."""
    d = (x * x).sum(1)[:, None] - 2.0 * x @ c.T + (c * c).sum(1)[None]
    return np.maximum(d, 0.0)


def assign(x, c, block: int = 65536):
    idx = np.empty(len(x), dtype=np.int64)
    err = np.empty(len(x), dtype=np.float64)
    for i in range(0, len(x), block):
        d = sq_dists(x[i : i + block], c)
        idx[i : i + block] = d.argmin(1)
        err[i : i + block] = d[np.arange(len(d)), idx[i : i + block]]
    return idx, err


def kmeans_pp(x, k, rng):
    c = np.empty((k, x.shape[1]), dtype=x.dtype)
    c[0] = x[rng.integers(len(x))]
    d = sq_dists(x, c[:1])[:, 0]
    for i in range(1, k):
        p = d / d.sum() if d.sum() > 0 else None
        c[i] = x[rng.choice(len(x), p=p)]
        d = np.minimum(d, sq_dists(x, c[i : i + 1])[:, 0])
    return c


def kmeans(x, k, iters: int = 20, seed: int = 0, max_train: int | None = None):
    """Returns (centroids (k, d), inertia per iteration). ``max_train`` subsamples."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    if max_train is not None and len(x) > max_train:
        x = x[rng.choice(len(x), max_train, replace=False)]
    if k > len(x):
        raise ValueError(f"k = {k} > {len(x)} points")
    c = kmeans_pp(x, k, rng)
    hist = []
    for _ in range(iters):
        idx, err = assign(x, c)
        hist.append(float(err.sum()))
        counts = np.bincount(idx, minlength=k)
        sums = np.zeros_like(c, dtype=np.float64)
        np.add.at(sums, idx, x)
        nonempty = counts > 0
        c[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype(c.dtype)
        for j in np.flatnonzero(~nonempty):  # re-seed at the worst-served point
            far = int(err.argmax())
            c[j] = x[far]
            err[far] = 0.0
    return c, hist

"""Product quantisation (Jégou et al. 2011) and optimised PQ (Ge et al. 2013).

**PQ.** Split a d-dim vector into M sub-vectors of d/M dims and quantise each against its
own codebook of 2^nbits centroids. A vector becomes M bytes (nbits = 8). With d = 128 and
M = 16 that is 16 bytes instead of 512, a 32× compression.

**ADC (asymmetric distance computation).** For a query q, precompute the table
``T[m, j] = ‖q_m − C_m[j]‖²`` once (M × 256 entries). The distance to any encoded vector is
then M table lookups and adds, ``Σ_m T[m, code_m]``, with no decompression. This equals the
exact distance from q to the *decoded* vector, which the tests check.

**OPQ.** PQ assumes the sub-spaces are independent and equally informative. A rotation R
learned beforehand helps: OPQ alternates between training PQ on ``xR`` and solving the
orthogonal Procrustes problem ``min_R ‖xR − decode(encode(xR))‖`` (an SVD). R is orthogonal,
so distances are unchanged and the rotation can be applied to queries for free.
"""

from __future__ import annotations

import numpy as np

from .kmeans import assign, kmeans


class PQ:
    def __init__(self, d: int, M: int, nbits: int = 8):
        if d % M:
            raise ValueError("d must be divisible by M")
        self.d, self.M, self.ds, self.ksub = d, M, d // M, 2**nbits
        self.codebooks = None  # (M, ksub, ds)

    def train(self, x, iters=20, seed=0, max_train=65536):
        x = np.asarray(x, dtype=np.float32)
        books = []
        for m in range(self.M):
            sub = x[:, m * self.ds : (m + 1) * self.ds]
            k = min(self.ksub, len(np.unique(sub, axis=0)))
            c, _ = kmeans(sub, k, iters, seed + m, max_train)
            if k < self.ksub:  # pad unused entries far away so they are never chosen
                c = np.concatenate([c, np.full((self.ksub - k, self.ds), 1e18, np.float32)])
            books.append(c)
        self.codebooks = np.stack(books)
        return self

    def encode(self, x):
        x = np.asarray(x, dtype=np.float32)
        codes = np.empty((len(x), self.M), dtype=np.uint8 if self.ksub <= 256 else np.uint16)
        for m in range(self.M):
            codes[:, m] = assign(x[:, m * self.ds : (m + 1) * self.ds], self.codebooks[m])[0]
        return codes

    def decode(self, codes):
        return np.concatenate([self.codebooks[m][codes[:, m]] for m in range(self.M)], 1)

    def tables(self, q):
        """(nq, M, ksub) squared distances from each query sub-vector to every centroid."""
        q = np.asarray(q, dtype=np.float32).reshape(len(q), self.M, 1, self.ds)
        return ((q - self.codebooks[None]) ** 2).sum(-1)

    @staticmethod
    def adc(table, codes):
        """Distances from one query (its ``(M, ksub)`` table) to ``codes (n, M)``."""
        return table[np.arange(table.shape[0])[None], codes.astype(np.int64)].sum(1)


class OPQ:
    """A learned rotation R (d × d, orthogonal) followed by PQ on ``x @ R``."""

    def __init__(self, d, M, nbits=8):
        self.R = np.eye(d, dtype=np.float32)
        self.pq = PQ(d, M, nbits)
        self.errors = []

    def train(self, x, outer=8, iters=10, seed=0, max_train=32768):
        x = np.asarray(x, dtype=np.float32)
        rng = np.random.default_rng(seed)
        if len(x) > max_train:
            x = x[rng.choice(len(x), max_train, replace=False)]
        for it in range(outer):
            xr = x @ self.R
            self.pq.train(xr, iters, seed + it)
            y = self.pq.decode(self.pq.encode(xr))
            self.errors.append(float(((xr - y) ** 2).sum(1).mean()))
            u, _, vt = np.linalg.svd(x.T.astype(np.float64) @ y.astype(np.float64))
            self.R = (u @ vt).astype(np.float32)  # argmin_R ‖xR − y‖ over orthogonal R
        self.pq.train(x @ self.R, iters, seed + outer)
        xr = x @ self.R
        self.errors.append(float(((xr - self.pq.decode(self.pq.encode(xr))) ** 2).sum(1).mean()))
        return self

    def rotate(self, x):
        return np.asarray(x, dtype=np.float32) @ self.R

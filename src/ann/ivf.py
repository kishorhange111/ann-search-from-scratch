"""Exhaustive search, IVF-Flat and IVF-PQ.

**IVF (inverted file).** A coarse k-means with ``nlist`` centroids partitions the base
vectors. A query scans only the ``nprobe`` lists whose centroids are nearest, so the cost
falls from n to about n·nprobe/nlist distance computations. Recall is lost exactly when a
true neighbour lives in a list that was not probed.

**IVF-PQ.** Each list stores PQ codes of the **residual** ``x − c_list``, not of x itself.
Residuals are smaller and more uniform across lists than raw vectors, so the same 8 bits
per sub-vector buy a finer quantisation. For a query, the residual ``q − c_list`` differs
per list, so each probed list gets its own ADC table. ``rerank`` optionally re-scores the
best candidates with exact distances from the stored float vectors, as FAISS's
``IndexRefine`` does, trading memory for recall.
"""

from __future__ import annotations

import numpy as np

from .kmeans import assign, kmeans, sq_dists
from .pq import OPQ, PQ


def topk(d: np.ndarray, k: int):
    """Smallest-k per row, sorted: (dist, idx)."""
    k = min(k, d.shape[1])
    part = np.argpartition(d, k - 1, axis=1)[:, :k]
    pd = np.take_along_axis(d, part, 1)
    order = np.argsort(pd, 1, kind="stable")
    return np.take_along_axis(pd, order, 1), np.take_along_axis(part, order, 1)


class Flat:
    def __init__(self, x):
        self.x = np.asarray(x, dtype=np.float32)

    def search(self, q, k, block=1024):
        D, I = [], []
        for i in range(0, len(q), block):
            d, idx = topk(sq_dists(np.asarray(q[i : i + block], np.float32), self.x), k)
            D.append(d)
            I.append(idx)
        return np.concatenate(D), np.concatenate(I)

    def bytes_per_vector(self):
        return self.x.shape[1] * 4


class IVFFlat:
    def __init__(self, nlist, seed=0):
        self.nlist, self.seed = nlist, seed

    def train(self, x, iters=20, max_train=65536):
        self.centroids, _ = kmeans(x, self.nlist, iters, self.seed, max_train)
        return self

    def add(self, x):
        x = np.asarray(x, dtype=np.float32)
        lists, _ = assign(x, self.centroids)
        order = np.argsort(lists, kind="stable")
        self.ids = order
        self.vecs = x[order]
        self.offsets = np.searchsorted(lists[order], np.arange(self.nlist + 1))
        return self

    def probe(self, q, nprobe):
        return topk(sq_dists(np.asarray(q, np.float32), self.centroids), nprobe)[1]

    def search(self, q, k, nprobe=8):
        q = np.asarray(q, np.float32)
        P = self.probe(q, nprobe)
        D = np.full((len(q), k), np.inf, np.float32)
        I = np.full((len(q), k), -1, np.int64)
        for i in range(len(q)):
            rows = np.concatenate([np.arange(self.offsets[c], self.offsets[c + 1]) for c in P[i]])
            if len(rows) == 0:
                continue
            d = sq_dists(q[i : i + 1], self.vecs[rows])
            dd, jj = topk(d, k)
            D[i, : dd.shape[1]], I[i, : dd.shape[1]] = dd[0], self.ids[rows[jj[0]]]
        return D, I

    def bytes_per_vector(self):
        return self.vecs.shape[1] * 4 + 8  # vector + id


class IVFPQ(IVFFlat):
    def __init__(self, nlist, M, nbits=8, opq=False, seed=0):
        super().__init__(nlist, seed)
        self.M, self.nbits, self.use_opq = M, nbits, opq

    def train(self, x, iters=20, max_train=65536):
        x = np.asarray(x, dtype=np.float32)
        d = x.shape[1]
        if self.use_opq:
            self.opq = OPQ(d, self.M, self.nbits).train(x, seed=self.seed)
            x = self.opq.rotate(x)
        super().train(x, iters, max_train)
        lists, _ = assign(x, self.centroids)
        self.pq = PQ(d, self.M, self.nbits).train(x - self.centroids[lists], iters, self.seed)
        return self

    def _rot(self, x):
        return self.opq.rotate(x) if self.use_opq else np.asarray(x, np.float32)

    def add(self, x, keep_vectors=False):
        raw = np.asarray(x, dtype=np.float32)
        x = self._rot(raw)
        lists, _ = assign(x, self.centroids)
        order = np.argsort(lists, kind="stable")
        self.ids = order
        self.list_of = lists[order]
        self.codes = self.pq.encode(x[order] - self.centroids[lists[order]])
        self.offsets = np.searchsorted(self.list_of, np.arange(self.nlist + 1))
        self.vecs = raw[order] if keep_vectors else None
        return self

    def search(self, q, k, nprobe=8, rerank=0):
        raw = np.asarray(q, np.float32)
        q = self._rot(raw)
        P = self.probe(q, nprobe)
        kk = max(k, rerank)
        D = np.full((len(q), k), np.inf, np.float32)
        I = np.full((len(q), k), -1, np.int64)
        for i in range(len(q)):
            cand_d, cand_r = [], []
            tabs = self.pq.tables(q[i] - self.centroids[P[i]])  # one table per probed list
            for t, c in zip(tabs, P[i], strict=True):
                lo, hi = self.offsets[c], self.offsets[c + 1]
                if hi > lo:
                    cand_d.append(PQ.adc(t, self.codes[lo:hi]))
                    cand_r.append(np.arange(lo, hi))
            if not cand_d:
                continue
            d, r = np.concatenate(cand_d), np.concatenate(cand_r)
            dd, jj = topk(d[None], kk)
            rows = r[jj[0]]
            if rerank:
                exact = sq_dists(raw[i : i + 1], self.vecs[rows])
                dd, jj = topk(exact, k)
                rows = rows[jj[0]]
            n = min(k, len(rows))
            D[i, :n], I[i, :n] = dd[0, :n], self.ids[rows[:n]]
        return D, I

    def bytes_per_vector(self):
        b = self.M * self.nbits / 8 + 8
        return b + (self.pq.d * 4 if self.vecs is not None else 0)

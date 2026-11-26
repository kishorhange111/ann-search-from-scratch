"""HNSW (Malkov & Yashunin 2018): a hierarchy of proximity graphs.

Each point gets a level ``⌊−ln U · mL⌋`` with ``mL = 1/ln M``, so level ℓ holds about a
fraction M^−ℓ of the points. Upper layers are sparse "express lanes". Search descends
greedily (ef = 1) from the top entry point to layer 1, then runs a best-first beam search
with a candidate list of size ``ef`` on layer 0, where every point lives with up to
``M0 = 2M`` links.

**Neighbour selection.** Insertion connects a new point to the M nearest it found
(``heuristic=False``) or uses the paper's heuristic (Algorithm 4). The heuristic takes
candidates in order of distance and keeps one only if it is closer to the new point than to
every neighbour already kept. This spreads links across directions instead of spending them
all inside one dense cluster, which keeps clustered data navigable. When a neighbour's
list overflows it is re-pruned with the same rule.

Distances are batched with NumPy: each expansion scores all of a node's unvisited
neighbours in one call.
"""

from __future__ import annotations

import heapq
import math

import numpy as np


class HNSW:
    def __init__(self, dim, M=16, ef_construction=100, heuristic=True, seed=0):
        self.dim, self.M, self.M0 = dim, M, 2 * M
        self.efc, self.heuristic = ef_construction, heuristic
        self.mL = 1.0 / math.log(M)
        self.rng = np.random.default_rng(seed)
        self.x = np.empty((0, dim), np.float32)
        self.links: list[list[list[int]]] = []  # links[node][level] -> neighbour ids
        self.levels: list[int] = []
        self.entry, self.max_level = -1, -1
        self.n_dist = 0  # distance evaluations, for cost accounting

    # -- distances -------------------------------------------------------------------------
    def _d(self, q, ids):
        self.n_dist += len(ids)
        diff = self.x[ids] - q
        return np.einsum("ij,ij->i", diff, diff)

    # -- search on one layer ---------------------------------------------------------------
    def _search_layer(self, q, entries, ef, level):
        """Best-first search; returns up to ef (dist, id) pairs, nearest first."""
        visited = set(entries)
        d0 = self._d(q, np.array(entries))
        cand = [(float(d), e) for d, e in zip(d0, entries, strict=True)]  # min-heap
        best = [(-d, e) for d, e in cand]  # max-heap of the ef best so far
        heapq.heapify(cand)
        heapq.heapify(best)
        while len(best) > ef:
            heapq.heappop(best)
        while cand:
            d, c = heapq.heappop(cand)
            if d > -best[0][0]:
                break
            nb = [n for n in self.links[c][level] if n not in visited]
            if not nb:
                continue
            visited.update(nb)
            for dn, n in zip(self._d(q, np.array(nb)), nb, strict=True):
                dn = float(dn)
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(cand, (dn, n))
                    heapq.heappush(best, (-dn, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-d, e) for d, e in best)

    # -- neighbour selection ---------------------------------------------------------------
    def _select(self, base, cands, m):
        """``cands``: (dist to base, id), sorted. Returns ≤ m ids."""
        if not self.heuristic or len(cands) <= m:
            return [e for _, e in cands[:m]]
        kept, kept_ids = [], []
        for d, e in cands:
            if len(kept) >= m:
                break
            if kept_ids:
                if (self._d(self.x[e], np.array(kept_ids)) < d).any():
                    continue  # some kept neighbour is closer to e than base is
            kept.append((d, e))
            kept_ids.append(e)
        if len(kept) < m:  # keepPrunedConnections: top up with the nearest discarded
            chosen = set(kept_ids)
            kept_ids += [e for _, e in cands if e not in chosen][: m - len(kept)]
        return kept_ids

    # -- build -----------------------------------------------------------------------------
    def add(self, xs):
        xs = np.asarray(xs, dtype=np.float32)
        start = len(self.x)
        self.x = np.concatenate([self.x, xs]) if start else xs.copy()
        for i in range(start, len(self.x)):
            self._insert(i)
        return self

    def _insert(self, i):
        q = self.x[i]
        lvl = int(-math.log(1.0 - self.rng.random()) * self.mL)
        self.levels.append(lvl)
        self.links.append([[] for _ in range(lvl + 1)])
        if self.entry < 0:
            self.entry, self.max_level = i, lvl
            return
        ep = [self.entry]
        for level in range(self.max_level, lvl, -1):
            ep = [self._search_layer(q, ep, 1, level)[0][1]]
        for level in range(min(lvl, self.max_level), -1, -1):
            W = self._search_layer(q, ep, self.efc, level)
            m_max = self.M0 if level == 0 else self.M
            nbrs = self._select(i, W, self.M)
            self.links[i][level] = list(nbrs)
            for n in nbrs:
                ln = self.links[n][level]
                ln.append(i)
                if len(ln) > m_max:  # shrink the neighbour's list with the same rule
                    dn = self._d(self.x[n], np.array(ln))
                    order = np.argsort(dn)
                    self.links[n][level] = self._select(
                        n, [(float(dn[j]), ln[j]) for j in order], m_max
                    )
            ep = [e for _, e in W]
        if lvl > self.max_level:
            self.entry, self.max_level = i, lvl

    # -- query -----------------------------------------------------------------------------
    def search_one(self, q, k, ef=None):
        q = np.asarray(q, np.float32)
        ef = max(ef or k, k)
        ep = [self.entry]
        for level in range(self.max_level, 0, -1):
            ep = [self._search_layer(q, ep, 1, level)[0][1]]
        W = self._search_layer(q, ep, ef, 0)[:k]
        return np.array([d for d, _ in W], np.float32), np.array([e for _, e in W], np.int64)

    def search(self, Q, k, ef=None):
        D = np.full((len(Q), k), np.inf, np.float32)
        I = np.full((len(Q), k), -1, np.int64)
        for i, q in enumerate(Q):
            d, e = self.search_one(q, k, ef)
            D[i, : len(d)], I[i, : len(e)] = d, e
        return D, I

    def bytes_per_vector(self):
        links = sum(len(level) for node in self.links for level in node)
        return self.dim * 4 + 4 * links / max(1, len(self.links))

    def reachable_from_entry(self):
        """Fraction of nodes reachable from the entry point along layer-0 links."""
        seen, stack = {self.entry}, [self.entry]
        while stack:
            for n in self.links[stack.pop()][0]:
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        return len(seen) / len(self.links)

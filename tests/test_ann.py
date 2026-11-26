import math

import numpy as np
import pytest

from ann.hnsw import HNSW
from ann.ivf import IVFPQ, Flat, IVFFlat, topk
from ann.kmeans import kmeans, sq_dists
from ann.pq import OPQ, PQ


def data(n=2000, d=16, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def clustered(n=2000, d=16, k=20, seed=0, spread=0.05):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, d)) * 3
    return (centres[rng.integers(k, size=n)] + spread * rng.normal(size=(n, d))).astype(np.float32)


def recall(I, gt, k):
    return np.mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(I, gt, strict=True)])


def test_sq_dists_and_topk_match_brute_force():
    x, q = data(300), data(7, seed=1)
    want = ((q[:, None] - x[None]) ** 2).sum(-1)
    np.testing.assert_allclose(sq_dists(q, x), want, rtol=1e-4, atol=1e-4)
    D, I = topk(want, 5)
    np.testing.assert_array_equal(I, np.argsort(want, 1, kind="stable")[:, :5])
    assert (np.diff(D, axis=1) >= 0).all()


def test_kmeans_inertia_never_increases_and_finds_separated_clusters():
    x = clustered(1000, 8, k=5, spread=0.01)
    c, hist = kmeans(x, 5, iters=10)
    assert all(b <= a + 1e-3 for a, b in zip(hist, hist[1:], strict=False))
    # every point is within the tight spread of its nearest centroid
    assert sq_dists(x, c).min(1).max() < 0.01 * 8 * 9


def test_kmeans_reseeds_empty_clusters():
    x = np.repeat(np.eye(3, dtype=np.float32), 50, axis=0)  # only 3 distinct points
    x = np.concatenate([x, np.array([[5, 5, 5]], np.float32)])
    c, _ = kmeans(x, 4, iters=5)
    assert len({tuple(np.round(r, 3)) for r in c}) == 4


def test_pq_adc_equals_exact_distance_to_decoded_vectors():
    x, q = data(3000, 32), data(5, 32, seed=1)
    pq = PQ(32, M=8, nbits=6).train(x, iters=8)
    codes = pq.encode(x)
    assert codes.shape == (3000, 8) and codes.dtype == np.uint8 and codes.max() < 64
    T = pq.tables(q)
    for i in range(len(q)):
        np.testing.assert_allclose(
            PQ.adc(T[i], codes), sq_dists(q[i : i + 1], pq.decode(codes))[0], rtol=1e-4, atol=1e-3
        )


def test_pq_is_lossless_when_each_subspace_has_few_values():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, size=(500, 8)).astype(np.float32)  # 4^2 = 16 values per 2-d sub
    pq = PQ(8, M=4, nbits=4).train(x, iters=10)
    np.testing.assert_allclose(pq.decode(pq.encode(x)), x, atol=1e-5)


def test_opq_rotation_is_orthogonal_and_helps_on_correlated_data():
    rng = np.random.default_rng(0)
    d = 16
    # variance concentrated in a few random directions, unevenly spread over subspaces
    basis = np.linalg.qr(rng.normal(size=(d, d)))[0]
    x = (rng.normal(size=(4000, d)) * np.geomspace(4, 0.05, d)) @ basis.T
    x = x.astype(np.float32)
    opq = OPQ(d, M=4, nbits=4).train(x, outer=6, iters=8)
    np.testing.assert_allclose(opq.R @ opq.R.T, np.eye(d), atol=1e-4)
    pq = PQ(d, M=4, nbits=4).train(x, iters=8)
    pq_err = ((x - pq.decode(pq.encode(x))) ** 2).sum(1).mean()
    assert opq.errors[-1] < 0.8 * pq_err
    # rotation preserves distances, so the rotated database answers the same queries
    np.testing.assert_allclose(
        sq_dists(opq.rotate(x[:5]), opq.rotate(x[5:50])),
        sq_dists(x[:5], x[5:50]),
        rtol=1e-3,
        atol=1e-3,
    )


def test_ivf_with_every_list_probed_is_exact():
    x, q = data(1500), data(20, seed=1)
    gt_d, gt = Flat(x).search(q, 10)
    ivf = IVFFlat(16).train(x).add(x)
    D, I = ivf.search(q, 10, nprobe=16)
    np.testing.assert_array_equal(I, gt)
    np.testing.assert_allclose(D, gt_d, rtol=1e-4, atol=1e-4)
    assert sorted(ivf.ids.tolist()) == list(range(1500))  # every vector stored once


def test_ivf_recall_grows_with_nprobe():
    x, q = clustered(3000, 16, k=40, spread=0.5), clustered(100, 16, k=40, seed=1, spread=0.5)
    gt = Flat(x).search(q, 10)[1]
    ivf = IVFFlat(32).train(x).add(x)
    r = [recall(ivf.search(q, 10, p)[1], gt, 10) for p in (1, 4, 32)]
    assert r[0] <= r[1] <= r[2] == 1.0


def test_ivfpq_scores_residuals_and_rerank_restores_exact_order():
    x, q = data(3000, 32), data(30, 32, seed=1)
    gt = Flat(x).search(q, 10)[1]
    ix = IVFPQ(8, M=8, nbits=8).train(x).add(x, keep_vectors=True)
    # the ADC score of a stored vector is its distance to centroid + decoded residual
    i, row = 0, 5
    c = ix.list_of[row]
    approx = ix.centroids[c] + ix.pq.decode(ix.codes[row : row + 1])[0]
    t = ix.pq.tables((q[i] - ix.centroids[c])[None])[0]
    assert PQ.adc(t, ix.codes[row : row + 1])[0] == pytest.approx(
        ((q[i] - approx) ** 2).sum(), rel=1e-4
    )
    plain = recall(ix.search(q, 10, nprobe=8)[1], gt, 10)
    rr = recall(ix.search(q, 10, nprobe=8, rerank=200)[1], gt, 10)
    assert rr > plain and rr > 0.9
    D, I = ix.search(q, 10, nprobe=8, rerank=3000)  # rerank everything ⇒ exact
    np.testing.assert_array_equal(I, gt)


def test_ivfpq_with_opq_runs_and_is_competitive():
    x, q = data(3000, 32), data(30, 32, seed=1)
    gt = Flat(x).search(q, 10)[1]
    a = IVFPQ(8, M=8).train(x).add(x)
    b = IVFPQ(8, M=8, opq=True).train(x).add(x)
    ra, rb = (recall(ix.search(q, 10, 8)[1], gt, 10) for ix in (a, b))
    assert rb > ra - 0.1


def test_hnsw_levels_are_geometric():
    h = HNSW(4, M=16, seed=0)
    h.add(np.random.default_rng(0).normal(size=(4000, 4)))
    frac1 = np.mean(np.array(h.levels) >= 1)
    assert abs(frac1 - 1 / 16) < 0.015  # P(level ≥ 1) = M^-1
    assert h.max_level == max(h.levels) and h.levels[h.entry] == h.max_level


def test_hnsw_link_limits_and_symmetry_of_new_links():
    h = HNSW(8, M=6, ef_construction=40).add(data(800, 8))
    for node in h.links:
        assert len(node[0]) <= 12
        assert all(len(level) <= 6 for level in node[1:])
        assert len(set(node[0])) == len(node[0])


def test_hnsw_recall_and_exactness_with_large_ef():
    x, q = data(2000, 16), data(50, 16, seed=1)
    gt = Flat(x).search(q, 10)[1]
    h = HNSW(16, M=12, ef_construction=80).add(x)
    assert recall(h.search(q, 10, ef=20)[1], gt, 10) > 0.8
    assert recall(h.search(q, 10, ef=200)[1], gt, 10) > 0.99
    D, I = h.search(q[:3], 10, ef=2000)  # beam ≥ n reaches everything: exact
    np.testing.assert_array_equal(I, gt[:3])


def test_heuristic_keeps_clustered_data_navigable():
    """On tight, well-separated clusters, linking each point to its M nearest neighbours
    spends every link inside its own cluster; the heuristic keeps long-range links."""
    x = clustered(1500, 8, k=30, spread=0.02, seed=3)
    q = clustered(100, 8, k=30, spread=0.02, seed=3)[::1] + 0.01
    gt = Flat(x).search(q, 10)[1]
    res = {}
    for heur in (True, False):
        h = HNSW(8, M=4, ef_construction=32, heuristic=heur, seed=1).add(x)
        res[heur] = (h.reachable_from_entry(), recall(h.search(q, 10, ef=16)[1], gt, 10))
    assert res[True][0] == 1.0
    assert res[True][0] >= res[False][0]
    assert res[True][1] > res[False][1]


def test_hnsw_cost_is_sublinear():
    h = HNSW(16, M=12, ef_construction=60).add(data(4000, 16))
    h.n_dist = 0
    h.search(data(20, 16, seed=1), 10, ef=40)
    per_query = h.n_dist / 20
    assert per_query < 4000 / 4
    assert math.isfinite(per_query)

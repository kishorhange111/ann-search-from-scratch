"""Recall@10 against queries per second on SIFT, compared with FAISS and hnswlib.

    python -m ann.bench --out results --n 50000 --nq 500

The data is ann-benchmarks' ``sift-128-euclidean.hdf5``. Its first ``--n`` base vectors are
the database, and its first ``--nq`` test vectors are queries. Ground truth is recomputed
exactly on that subset. All timings are **single-threaded**: FAISS is pinned to one OpenMP
thread, and this repo's code is NumPy plus Python loops. The interesting comparison is
recall at a given amount of work, and how the curves bend. Absolute QPS mostly measures
Python against C++.

Sections, each appended to ``bench.jsonl`` and skipped when already present:
``flat``, ``ivf``, ``ivfpq``, ``hnsw``, ``hnsw_heuristic`` (the selection-rule ablation on a
smaller subset), ``faiss``, ``hnswlib``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

import numpy as np

from .hnsw import HNSW
from .ivf import IVFPQ, Flat, IVFFlat

URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"


def load_sift(path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    import h5py

    with h5py.File(path, "r") as f:
        return np.asarray(f["train"], np.float32), np.asarray(f["test"], np.float32)


def recall(I, gt, k=10):
    return float(np.mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(I, gt, strict=True)]))


def timed_search(fn, Q):
    t = time.perf_counter()
    D, I = fn(Q)
    return I, len(Q) / (time.perf_counter() - t)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default="sift-128-euclidean.hdf5")
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--nq", type=int, default=500)
    ap.add_argument("--n-ablation", type=int, default=20_000)
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--sections", default="flat,ivf,ivfpq,hnsw,hnsw_heuristic,faiss,hnswlib")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "bench.jsonl")
    done = {json.loads(s)["section"] for s in open(path)} if os.path.exists(path) else set()
    base, test = load_sift(a.data)
    X, Q = base[: a.n], test[: a.nq]
    k = 10
    gt = Flat(X).search(Q, 100)[1]

    def emit(section, **row):
        row = {"section": section, **row}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    secs = [s for s in a.sections.split(",") if s not in done]
    if "flat" in secs:
        I, qps = timed_search(lambda q: Flat(X).search(q, k), Q)
        emit("flat", index="Flat", param="", recall=recall(I, gt), qps=qps, bytes=512)
    if "ivf" in secs:
        t = time.perf_counter()
        ivf = IVFFlat(a.nlist).train(X).add(X)
        build = time.perf_counter() - t
        for nprobe in (1, 2, 4, 8, 16, 32, 64):
            I, qps = timed_search(lambda q, p=nprobe: ivf.search(q, k, p), Q)
            emit(
                "ivf", index=f"IVF{a.nlist},Flat", param=f"nprobe={nprobe}",
                recall=recall(I, gt), qps=qps, bytes=ivf.bytes_per_vector(), build_s=build,
            )  # fmt: skip
    if "ivfpq" in secs:
        for opq in (False, True):
            t = time.perf_counter()
            ix = IVFPQ(a.nlist, M=16, opq=opq).train(X).add(X, keep_vectors=True)
            build = time.perf_counter() - t
            name = f"{'OPQ16,' if opq else ''}IVF{a.nlist},PQ16"
            for nprobe in (4, 16, 64):
                for rr in (0, 100):
                    I, qps = timed_search(lambda q, p=nprobe, r=rr, ix=ix: ix.search(q, k, p, r), Q)
                    b = ix.M + 8 + (512 if rr else 0)
                    emit("ivfpq", index=name + (" + rerank" if rr else ""),
                         param=f"nprobe={nprobe}" + (f", rerank={rr}" if rr else ""),
                         recall=recall(I, gt), qps=qps, bytes=b, build_s=build,
                         opq_error=ix.opq.errors if opq else None)  # fmt: skip
    if "hnsw" in secs:
        t = time.perf_counter()
        h = HNSW(128, M=16, ef_construction=100).add(X)
        build = time.perf_counter() - t
        for ef in (10, 20, 40, 80, 160, 320):
            h.n_dist = 0
            I, qps = timed_search(lambda q, e=ef: h.search(q, k, e), Q)
            emit(
                "hnsw", index="HNSW16", param=f"ef={ef}", recall=recall(I, gt), qps=qps,
                bytes=h.bytes_per_vector(), build_s=build, dist_per_query=h.n_dist / len(Q),
            )  # fmt: skip
    if "hnsw_heuristic" in secs:
        Xa = X[: a.n_ablation]
        gta = Flat(Xa).search(Q, k)[1]
        for heur in (True, False):
            t = time.perf_counter()
            h = HNSW(128, M=8, ef_construction=64, heuristic=heur).add(Xa)
            build = time.perf_counter() - t
            for ef in (10, 20, 40, 80):
                h.n_dist = 0
                I, qps = timed_search(lambda q, e=ef, h=h: h.search(q, k, e), Q)
                emit("hnsw_heuristic", index=f"HNSW8 {'heuristic' if heur else 'nearest-M'}",
                     param=f"ef={ef}", recall=recall(I, gta), qps=qps, build_s=build,
                     dist_per_query=h.n_dist / len(Q), n=len(Xa),
                     reachable=h.reachable_from_entry())  # fmt: skip
    if "faiss" in secs:
        import faiss

        faiss.omp_set_num_threads(1)
        flat_q = faiss.IndexFlatL2(128)
        ivf = faiss.IndexIVFFlat(flat_q, 128, a.nlist)
        ivf.train(X)
        ivf.add(X)
        for nprobe in (1, 4, 16, 64):
            ivf.nprobe = nprobe
            I, qps = timed_search(lambda q: ivf.search(q, k), Q)
            emit("faiss", index=f"faiss IVF{a.nlist},Flat", param=f"nprobe={nprobe}",
                 recall=recall(I, gt), qps=qps)  # fmt: skip
        flat_q2 = faiss.IndexFlatL2(128)
        pq = faiss.IndexIVFPQ(flat_q2, 128, a.nlist, 16, 8)
        pq.train(X)
        pq.add(X)
        for nprobe in (4, 16, 64):
            pq.nprobe = nprobe
            I, qps = timed_search(lambda q: pq.search(q, k), Q)
            emit("faiss", index=f"faiss IVF{a.nlist},PQ16", param=f"nprobe={nprobe}",
                 recall=recall(I, gt), qps=qps)  # fmt: skip
        hn = faiss.IndexHNSWFlat(128, 16)
        hn.hnsw.efConstruction = 100
        t = time.perf_counter()
        hn.add(X)
        build = time.perf_counter() - t
        for ef in (10, 20, 40, 80, 160, 320):
            hn.hnsw.efSearch = ef
            I, qps = timed_search(lambda q: hn.search(q, k), Q)
            emit("faiss", index="faiss HNSW16", param=f"ef={ef}", recall=recall(I, gt),
                 qps=qps, build_s=build)  # fmt: skip
    if "hnswlib" in secs:
        import hnswlib

        p = hnswlib.Index(space="l2", dim=128)
        p.init_index(max_elements=len(X), M=16, ef_construction=100, random_seed=0)
        p.set_num_threads(1)
        t = time.perf_counter()
        p.add_items(X)
        build = time.perf_counter() - t
        for ef in (10, 20, 40, 80, 160, 320):
            p.set_ef(max(ef, k))
            I, qps = timed_search(lambda q: p.knn_query(q, k)[::-1], Q)
            emit("hnswlib", index="hnswlib M=16", param=f"ef={ef}", recall=recall(I, gt),
                 qps=qps, build_s=build)  # fmt: skip
    rows = [json.loads(s) for s in open(path)]
    lines = [
        "| index | setting | recall@10 | QPS (1 thread) | bytes/vector |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        b = f"{r['bytes']:.0f}" if r.get("bytes") else "–"
        lines.append(f"| {r['index']} | {r['param']} | {r['recall']:.3f} | {r['qps']:.0f} | {b} |")
    with open(os.path.join(a.out, "bench.md"), "w", encoding="utf-8") as f:
        f.write(f"n = {a.n}, {a.nq} queries, SIFT-128\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

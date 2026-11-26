# ann-search-from-scratch

[![CI](https://github.com/kishorhange111/ann-search-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/ann-search-from-scratch/actions/workflows/ci.yml)

The index structures inside FAISS and vector databases, written from scratch in NumPy and
benchmarked against FAISS and hnswlib on SIFT:
- **IVF**: an inverted file over a k-means coarse quantiser;
- **PQ**: product quantisation with asymmetric-distance lookup tables;
- **IVF-PQ**: residual encoding, with exact re-ranking;
- **OPQ**: a rotation learned by alternating k-means and orthogonal Procrustes;
- **HNSW**: a hierarchy of navigable small-world graphs, with the paper's
  neighbour-selection heuristic.

## Pieces

**[`kmeans.py`](src/ann/kmeans.py).** k-means++ seeding, Lloyd iterations with blocked
`‖x‖² − 2x·c + ‖c‖²` distances, and re-seeding of empty clusters at the worst-served point.

**[`pq.py`](src/ann/pq.py): PQ and OPQ.**
- A 128-d float vector becomes 16 bytes: 16 sub-vectors, each a 1-byte index into its own
  256-entry codebook.
- A query builds one `16 × 256` table of sub-distances. Its distance to any code is then 16
  lookups and adds (**ADC**), which is exactly the distance to the decoded vector.
- OPQ learns an orthogonal rotation first, so variance is spread across the sub-spaces.
  Each round trains PQ on `xR`, then solves `min_R ‖xR − decode(encode(xR))‖` with one SVD.

**[`ivf.py`](src/ann/ivf.py): Flat, IVF-Flat and IVF-PQ.**
- The coarse k-means splits the base into `nlist` lists, and a query scans only the
  `nprobe` nearest lists.
- IVF-PQ encodes each vector's **residual** from its list centroid, so each probed list
  needs its own ADC table.
- Optional re-ranking re-scores the best candidates exactly, as FAISS's `IndexRefine` does.

**[`hnsw.py`](src/ann/hnsw.py): HNSW.**
- Levels are drawn as `⌊−ln U / ln M⌋`. Search descends greedily through the sparse upper
  layers, then runs a best-first beam of width `ef` on layer 0.
- Insertion links the new point either to its M nearest candidates or, by default, through
  **Algorithm 4's heuristic**. A candidate is kept only if it is closer to the new point than
  to every neighbour already kept, which spreads links in different directions.
- Overflowing neighbour lists are re-pruned with the same rule.
- Distance evaluations are counted, so cost can be compared independently of Python speed.

## Tests

There are 15, on CPU. They check:
- blocked distances and top-k against brute force;
- k-means never increases inertia, finds separated clusters, and re-seeds empty ones;
- **ADC equals the exact distance to the decoded vector**, and PQ is lossless when each
  sub-space has at most 2^nbits distinct values;
- **OPQ's R is orthogonal and preserves distances, and it cuts quantisation error by more
  than 20%** on anisotropic data;
- **IVF with every list probed equals exhaustive search**, and recall grows with nprobe;
- IVF-PQ's score is the distance to centroid plus decoded residual, and **re-ranking the
  whole candidate set restores the exact answer**;
- HNSW levels are geometric with P(level ≥ 1) = 1/M, the link limits hold, recall exceeds
  0.99 at ef = 200, and the search is exact when ef ≥ n;
- **on tight, well-separated clusters, the heuristic keeps every node reachable and gives
  higher recall than linking the M nearest**;
- a search costs far fewer distance evaluations than n.

```
pip install -e ".[dev]"
pytest -q                                   # 15 tests, CPU
pip install -e ".[bench]"
python -m ann.bench --out results           # downloads SIFT (501 MB) from ann-benchmarks
```

## Results

**Pending** the SIFT run.

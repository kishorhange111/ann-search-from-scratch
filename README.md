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

There are 16, on CPU. They check:
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
- a search costs far fewer distance evaluations than n;
- the reachability diagnostics are consistent (layer 0 ⊆ all layers, the descent lands on
  a real node).

```
pip install -e ".[dev]"
pytest -q                                   # 16 tests, CPU
pip install -e ".[bench]"
python -m ann.bench --out results           # downloads SIFT (501 MB) from ann-benchmarks
```

## Results

SIFT-128: the first 50,000 base vectors and 500 queries. Ground truth is recomputed
exactly on that subset. Everything runs on one thread of a Colab Xeon @ 2.00 GHz, with
FAISS pinned to one OpenMP thread. The full table is in [results/bench.md](results/bench.md).

| index | setting | recall@10 | QPS | FAISS / hnswlib recall at the same setting |
|---|---|---|---|---|
| Flat | | 1.000 | 1,595 | |
| IVF256,Flat | nprobe = 4 | 0.735 | 3,970 | 0.730 |
| IVF256,Flat | nprobe = 16 | 0.964 | 1,288 | 0.957 |
| IVF256,PQ16 (24 B/vector) | nprobe = 16 | 0.630 | 226 | 0.635 |
| IVF256,PQ16 + re-rank 100 | nprobe = 16 | 0.959 | 237 | |
| IVF256,PQ16 + re-rank 100 | nprobe = 64 | 0.994 | 71 | |
| HNSW16 | ef = 10 | 0.844 | 1,821 | 0.803 / 0.807 |
| HNSW16 | ef = 40 | 0.983 | 795 | 0.972 / 0.975 |
| HNSW16 | ef = 80 | 0.997 | 358 | 0.995 / 0.995 |

- **Recall matches the reference libraries.** IVF-Flat is within 0.007 of FAISS at every
  nprobe, and IVF-PQ within 0.005. QPS is another matter: FAISS IVF-Flat is 6× faster at
  nprobe = 16, and hnswlib is 23× faster at ef = 80 and builds 53× faster (7.7 s against
  412 s). That gap is Python loops against C++, not the algorithm.
- **HNSW reaches 0.983 recall with 1.5% of the work.** At ef = 40 a query evaluates 756
  distances out of 50,000. At small ef our recall is higher than FAISS's and hnswlib's, by
  up to 0.04 at ef = 10. The three agree from ef = 160. A likely cause is that this implementation keeps pruned connections, topping
  each list back up to M (26.8 links per vector on average), and spends more distance
  evaluations per hop. In this Python build, the 66× saving in distances still loses to
  a single BLAS call: Flat runs 1,595 QPS, against 795 for HNSW at ef = 40.
- **PQ has a recall ceiling, and re-ranking removes it.** With 16-byte codes, recall stops
  at 0.64 however many lists are probed, because the code cannot separate the true top 10
  from its near neighbours. Re-scoring the best 100 candidates exactly gives 0.959 at
  nprobe = 16 and 0.994 at nprobe = 64.
- **OPQ barely helps on SIFT.** Nine alternating rounds cut the quantisation error by only
  3.8%, and recall moves by at most 0.008, for 4× the build time. This suggests SIFT's
  variance is already spread fairly evenly over the 16 sub-spaces. The synthetic
  anisotropic test, by contrast, shows the more-than-20% cut OPQ is designed for.

**The neighbour-selection heuristic** (M = 8, on the first 20,000 vectors):

| ef | recall, heuristic | recall, M nearest | distances per query, heuristic / nearest |
|---|---|---|---|
| 10 | 0.763 | 0.632 | 185 / 158 |
| 40 | 0.957 | 0.861 | 416 / 344 |
| 80 | 0.986 | 0.932 | 677 / 552 |

The heuristic gives +0.05 to +0.13 recall at every ef, for 17–23% more distance
evaluations. One measurement is **not yet explained**. Along layer-0 links from the entry
point, only 54% of nodes are reachable with the heuristic, against 98% with M-nearest
linking. That cannot bound recall: a search enters layer 0 wherever the descent through
the upper layers lands. Yet the same check on synthetic low-rank data (8,000 vectors, M = 8)
gives 100% with the heuristic and 99.6% without.
The benchmark now records the reachability over all layers, the share of nodes with no
incoming layer-0 link, how many of each query's true neighbours can be reached from its
landing node, and the number of distinct vectors. A rerun will settle it.

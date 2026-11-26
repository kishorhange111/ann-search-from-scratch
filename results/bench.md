n = 50000, 500 queries, SIFT-128

| index | setting | recall@10 | QPS (1 thread) | bytes/vector |
|---|---|---|---|---|
| Flat |  | 1.000 | 1595 | 512 |
| IVF256,Flat | nprobe=1 | 0.391 | 8340 | 520 |
| IVF256,Flat | nprobe=2 | 0.564 | 5937 | 520 |
| IVF256,Flat | nprobe=4 | 0.735 | 3970 | 520 |
| IVF256,Flat | nprobe=8 | 0.872 | 2140 | 520 |
| IVF256,Flat | nprobe=16 | 0.964 | 1288 | 520 |
| IVF256,Flat | nprobe=32 | 0.995 | 512 | 520 |
| IVF256,Flat | nprobe=64 | 1.000 | 284 | 520 |
| IVF256,PQ16 | nprobe=4 | 0.555 | 1083 | 24 |
| IVF256,PQ16 + rerank | nprobe=4, rerank=100 | 0.735 | 955 | 536 |
| IVF256,PQ16 | nprobe=16 | 0.630 | 226 | 24 |
| IVF256,PQ16 + rerank | nprobe=16, rerank=100 | 0.959 | 237 | 536 |
| IVF256,PQ16 | nprobe=64 | 0.637 | 81 | 24 |
| IVF256,PQ16 + rerank | nprobe=64, rerank=100 | 0.994 | 71 | 536 |
| OPQ16,IVF256,PQ16 | nprobe=4 | 0.556 | 1171 | 24 |
| OPQ16,IVF256,PQ16 + rerank | nprobe=4, rerank=100 | 0.734 | 977 | 536 |
| OPQ16,IVF256,PQ16 | nprobe=16 | 0.636 | 310 | 24 |
| OPQ16,IVF256,PQ16 + rerank | nprobe=16, rerank=100 | 0.962 | 257 | 536 |
| OPQ16,IVF256,PQ16 | nprobe=64 | 0.645 | 74 | 24 |
| OPQ16,IVF256,PQ16 + rerank | nprobe=64, rerank=100 | 0.997 | 77 | 536 |
| faiss IVF256,Flat | nprobe=1 | 0.387 | 51391 | – |
| faiss IVF256,Flat | nprobe=4 | 0.730 | 22375 | – |
| faiss IVF256,Flat | nprobe=16 | 0.957 | 7688 | – |
| faiss IVF256,Flat | nprobe=64 | 1.000 | 2081 | – |
| faiss IVF256,PQ16 | nprobe=4 | 0.559 | 25552 | – |
| faiss IVF256,PQ16 | nprobe=16 | 0.635 | 9536 | – |
| faiss IVF256,PQ16 | nprobe=64 | 0.642 | 2586 | – |
| faiss HNSW16 | ef=10 | 0.803 | 32341 | – |
| faiss HNSW16 | ef=20 | 0.916 | 21847 | – |
| faiss HNSW16 | ef=40 | 0.972 | 12328 | – |
| faiss HNSW16 | ef=80 | 0.995 | 7086 | – |
| faiss HNSW16 | ef=160 | 0.999 | 3663 | – |
| faiss HNSW16 | ef=320 | 0.999 | 2146 | – |
| hnswlib M=16 | ef=10 | 0.807 | 37498 | – |
| hnswlib M=16 | ef=20 | 0.918 | 18194 | – |
| hnswlib M=16 | ef=40 | 0.975 | 14032 | – |
| hnswlib M=16 | ef=80 | 0.995 | 8253 | – |
| hnswlib M=16 | ef=160 | 0.999 | 4304 | – |
| hnswlib M=16 | ef=320 | 0.999 | 2455 | – |
| HNSW16 | ef=10 | 0.844 | 1821 | 619 |
| HNSW16 | ef=20 | 0.939 | 1203 | 619 |
| HNSW16 | ef=40 | 0.983 | 795 | 619 |
| HNSW16 | ef=80 | 0.997 | 358 | 619 |
| HNSW16 | ef=160 | 0.999 | 179 | 619 |
| HNSW16 | ef=320 | 1.000 | 137 | 619 |
| HNSW8 heuristic | ef=10 | 0.763 | 2000 | – |
| HNSW8 heuristic | ef=20 | 0.882 | 1430 | – |
| HNSW8 heuristic | ef=40 | 0.957 | 906 | – |
| HNSW8 heuristic | ef=80 | 0.986 | 343 | – |
| HNSW8 nearest-M | ef=10 | 0.632 | 1973 | – |
| HNSW8 nearest-M | ef=20 | 0.764 | 1502 | – |
| HNSW8 nearest-M | ef=40 | 0.861 | 968 | – |
| HNSW8 nearest-M | ef=80 | 0.932 | 574 | – |

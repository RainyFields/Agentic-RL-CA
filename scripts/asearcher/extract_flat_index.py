#!/usr/bin/env python3
"""One-time: extract raw vectors from a faiss FLAT index into an fp16 .npy memmap
(+ meta json). Runs under the retriever conda env (CPU faiss). B200 path: no faiss
build exists for sm_100 (conda 1.10 and pypi 1.14 both top out at sm_90/sm_80, no PTX),
so retrieval runs as a torch matmul server instead (torch_retrieval_server.py).
Usage: python extract_flat_index.py <index> <out.npy> <out_meta.json>
"""
import json
import sys

import faiss
import numpy as np

index_path, out_npy, out_meta = sys.argv[1], sys.argv[2], sys.argv[3]
idx = faiss.read_index(index_path)
n, d = idx.ntotal, idx.d
print(f"index: ntotal={n} dim={d}", flush=True)
mm = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.float16, shape=(n, d))
CH = 200_000
for s in range(0, n, CH):
    e = min(n, s + CH)
    mm[s:e] = idx.reconstruct_n(s, e - s).astype(np.float16)
    if (s // CH) % 20 == 0:
        print(f"  {e}/{n}", flush=True)
mm.flush()
del mm
json.dump({"n": n, "d": d}, open(out_meta, "w"))
print("DONE", flush=True)

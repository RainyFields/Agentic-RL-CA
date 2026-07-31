#!/usr/bin/env python3
"""Torch-based flat retrieval server (Blackwell-safe drop-in for the faiss server).

Same HTTP contract as examples/search/retriever/retrieval_server.py:
  POST /retrieve {"queries": [...], "topk": k, "return_scores": bool}
  -> {"result": [ [ {"document": {...,"contents":...}, "score": float}, ... ], ... ]}
Same e5 encoding path: "query: " prefix, mean pooling, L2 normalize, fp16 encoder.
Index = fp16 corpus-embedding matrix (from extract_flat_index.py) sharded across all
GPUs; scoring = inner-product matmul + merged top-k (matches faiss FLAT-IP with
useFloat16 sharding). Runs under the MAIN training venv (CUDA torch). Requests are
handled on the event loop (async) -> naturally serialized single-query searches
(sub-ms each); health probes interleave.
"""
import argparse
import json

import datasets
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer

app = FastAPI()
STATE = {}


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


def mean_pool(last_hidden, mask):
    m = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


@torch.no_grad()
def encode(queries: List[str]) -> torch.Tensor:
    tok, model = STATE["tok"], STATE["enc"]
    inputs = tok([f"query: {q}" for q in queries], max_length=256, padding=True,
                 truncation=True, return_tensors="pt").to(STATE["enc_dev"])
    out = model(**inputs)
    emb = mean_pool(out.last_hidden_state, inputs["attention_mask"])
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.half()


@torch.no_grad()
def search(queries: List[str], k: int):
    q = encode(queries)  # (B, d) fp16 on enc_dev
    all_scores, all_idxs = [], []
    for shard, dev, off in STATE["shards"]:
        qs = q.to(dev, non_blocking=True)
        sc = qs @ shard.T                     # (B, n_shard) fp16
        s, i = torch.topk(sc.float(), k, dim=1)
        all_scores.append(s.cpu())
        all_idxs.append((i + off).cpu())
    sc = torch.cat(all_scores, dim=1)          # (B, k*nshards)
    ix = torch.cat(all_idxs, dim=1)
    top_s, top_pos = torch.topk(sc, k, dim=1)
    top_i = torch.gather(ix, 1, top_pos)
    return top_s.tolist(), top_i.tolist()


@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    k = request.topk or STATE["topk"]
    scores, idxs = search(request.queries, k)
    corpus = STATE["corpus"]
    resp = []
    for qs, qi in zip(scores, idxs):
        docs = [corpus[int(i)] for i in qi]
        if request.return_scores:
            resp.append([{"document": d, "score": float(s)} for d, s in zip(docs, qs)])
        else:
            resp.append([{"document": d} for d in docs])
    return {"result": resp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npy", required=True)
    ap.add_argument("--emb_meta", required=True)
    ap.add_argument("--corpus_path", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--retriever_model", default="intfloat/e5-base-v2")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    meta = json.load(open(args.emb_meta))
    n, d = meta["n"], meta["d"]
    ng = torch.cuda.device_count()
    assert ng >= 1, "no CUDA devices"
    print(f"loading {n}x{d} fp16 embeddings across {ng} GPUs", flush=True)
    mm = np.load(args.emb_npy, mmap_mode="r")
    assert mm.shape == (n, d)
    shards = []
    per = (n + ng - 1) // ng
    for g in range(ng):
        s, e = g * per, min(n, (g + 1) * per)
        if s >= e:
            break
        t = torch.from_numpy(np.ascontiguousarray(mm[s:e])).to(f"cuda:{g}")
        shards.append((t, f"cuda:{g}", s))
        print(f"  shard {g}: rows [{s},{e}) on cuda:{g} ({t.element_size()*t.nelement()/2**30:.1f} GiB)", flush=True)
    STATE["shards"] = shards

    print("loading corpus...", flush=True)
    STATE["corpus"] = datasets.load_dataset("json", data_files=args.corpus_path,
                                            split="train", num_proc=4)
    enc_dev = "cuda:0"
    model = AutoModel.from_pretrained(args.retriever_model, trust_remote_code=True).eval().half().to(enc_dev)
    tok = AutoTokenizer.from_pretrained(args.retriever_model, use_fast=True, trust_remote_code=True)
    STATE.update(tok=tok, enc=model, enc_dev=enc_dev, topk=args.topk)

    # smoke: one search end-to-end before serving
    s, i = search(["who won the first nobel prize in physics"], args.topk)
    print("smoke topk scores:", [round(x, 3) for x in s[0]], flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

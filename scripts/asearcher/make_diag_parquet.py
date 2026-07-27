#!/usr/bin/env python3
"""Build the stage-2 pass-rate diagnostic parquet from a closed-book keep-set.

Each kept question is replicated --rep times ADJACENT in the val split so the trainer's
val_only pass (val_kwargs n=1, do_sample, temp 1.0) yields rep independent in-setup
rollouts per question. Replication happens in DATA (not val_kwargs.n) because
SearchMultiProcessEnv.reset raises if the repeated batch exceeds the val env pool
(env_num = data.val_batch_size, group_n=1).

Writes <out-dir>/val_2048.parquet (replicated) + <out-dir>/train.parquet (small dummy —
never trained on, only needed so require_data / dataloader init pass).

Run: python make_diag_parquet.py --keep <keep_ge2.jsonl> --split base|lrm --out-dir <dir> [--rep 8]
"""
import argparse, importlib.util, json, sys
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "convert_base_to_parquet", Path(__file__).parent / "convert_base_to_parquet.py")
_conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", required=True, help="closed-book keep_ge2.jsonl (original rows)")
    ap.add_argument("--split", required=True, choices=["base", "lrm"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rep", type=int, default=8)
    ap.add_argument("--dummy-train", type=int, default=256)
    args = ap.parse_args()

    data_source = f"asearcher_{args.split}"
    rows = [json.loads(l) for l in open(args.keep, encoding="utf-8")]
    n_raw = len(rows)
    # base keep-set still contains the invalid slice (closed-book filter ran on raw jsonl)
    rows = [r for r in rows if r.get("source") not in _conv.DROP_SOURCES]
    rows = [r for r in rows if _conv.targets(r)]
    print(f"[{args.split}] keep-set {n_raw} -> {len(rows)} after invalid/empty-target drop")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    val_recs = []
    for i, r in enumerate(rows):
        rec = _conv.make_row(r, i, "val", data_source)
        val_recs.extend([rec] * args.rep)
    pd.DataFrame(val_recs).to_parquet(out / "val_2048.parquet", index=False)

    dummy = [_conv.make_row(r, i, "train", data_source)
             for i, r in enumerate(rows[: args.dummy_train])]
    pd.DataFrame(dummy).to_parquet(out / "train.parquet", index=False)
    print(f"val_2048.parquet: {len(val_recs)} rows ({len(rows)} q x {args.rep}) | "
          f"train.parquet (dummy): {len(dummy)} rows -> {out}")


if __name__ == "__main__":
    main()

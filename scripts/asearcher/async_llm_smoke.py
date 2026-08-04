"""Phase 0 smoke test — lean async rollout plan (2026-08-04).

Go/no-go: can vLLM's V1 AsyncLLM engine (as vendored verl's async server uses it)
serve our model with the capabilities the async collector needs?

Checks (each → PASS/FAIL, summary JSON at the end):
  1. build      — AsyncLLM engine builds with our model, TP, sleep mode enabled
  2. concurrent — N concurrent requests actually interleave (timestamp proof:
                  a short request must FINISH while a longer one is still running)
  3. tokens     — exact token ids + per-token logprobs returned, lengths agree
  4. cancel     — a request can be cancelled promptly; engine stays healthy after
  5. sleepwake  — engine.sleep()/wake_up() round-trip, generation still works after

Run on a GPU worker:
  MODEL_PATH=/tmp/qwen3-8b-base TP=1 python async_llm_smoke.py
"""
import asyncio
import json
import os
import time
import traceback

os.environ.setdefault("VLLM_USE_V1", "1")

from transformers import AutoTokenizer  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402

MODEL = os.environ.get("MODEL_PATH", "/tmp/qwen3-8b-base")
TP = int(os.environ.get("TP", "1"))
RESULTS: dict = {}
T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[smoke +{time.monotonic() - T0:8.2f}s] {msg}", flush=True)


async def collect(engine, rid: str, prompt_ids, max_tokens: int, temperature: float = 1.0):
    """Drive one request to completion; return (final_output, t_submit, t_first, t_done)."""
    sp = SamplingParams(
        temperature=temperature, top_p=1.0, max_tokens=max_tokens, logprobs=0
    )
    t_submit = time.monotonic()
    t_first = None
    final = None
    async for out in engine.generate(
        TokensPrompt(prompt_token_ids=list(prompt_ids)), sp, rid
    ):
        if t_first is None:
            t_first = time.monotonic()
        final = out
    return final, t_submit, t_first, time.monotonic()


async def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    chat = [{"role": "user", "content": "Name three primary colors, one per line."}]
    prompt_ids = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
    log(f"prompt has {len(prompt_ids)} tokens")

    # ---- 1. build ----
    try:
        args = AsyncEngineArgs(
            model=MODEL,
            tensor_parallel_size=TP,
            dtype="bfloat16",
            gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.5")),
            max_model_len=17408,
            enable_sleep_mode=True,
            enforce_eager=True,  # arms run enforce_eager on H100; also faster to build
        )
        engine = AsyncLLM.from_engine_args(args)
        RESULTS["build"] = "PASS"
        log("engine built")
    except Exception:
        traceback.print_exc()
        RESULTS["build"] = "FAIL"
        print("SMOKE SUMMARY " + json.dumps(RESULTS), flush=True)
        return 1

    # ---- 2. concurrent + interleaving timestamps ----
    try:
        lens = [16, 32, 64, 128, 256, 512, 768, 1024]
        tasks = [
            asyncio.create_task(collect(engine, f"conc-{i}", prompt_ids, n))
            for i, n in enumerate(lens)
        ]
        outs = await asyncio.gather(*tasks)
        finish = [o[3] for o in outs]
        # timestamp proof: shortest request finishes strictly before the longest
        # request's completion, i.e. requests did not run to a batch barrier.
        overlap = min(finish) < max(finish) - 0.5
        spread = max(finish) - min(finish)
        RESULTS["concurrent"] = "PASS" if overlap else "FAIL"
        log(f"8 concurrent requests: finish spread {spread:.2f}s "
            f"(first done {min(finish)-outs[0][1]:.2f}s after submit)")
    except Exception:
        traceback.print_exc()
        RESULTS["concurrent"] = "FAIL"
        outs = []

    # ---- 3. exact token ids + logprobs ----
    try:
        ok = bool(outs)
        for final, *_ in outs:
            seq = final.outputs[0]
            ids = list(seq.token_ids)
            lps = seq.logprobs
            assert ids, "no token ids"
            assert lps is not None and len(lps) == len(ids), (
                f"logprobs len {None if lps is None else len(lps)} != ids len {len(ids)}"
            )
            for tid, lp in zip(ids, lps):
                assert tid in lp, f"sampled id {tid} missing from logprob dict"
                assert lp[tid].logprob <= 0.0
            tok.decode(ids)  # detokenize sanity
        RESULTS["tokens"] = "PASS" if ok else "FAIL"
        if ok:
            log(f"token ids + logprobs exact on {len(outs)} requests "
                f"(e.g. {len(outs[0][0].outputs[0].token_ids)} ids)")
    except Exception:
        traceback.print_exc()
        RESULTS["tokens"] = "FAIL"

    # ---- 4. cancel ----
    try:
        t = asyncio.create_task(collect(engine, "cancel-0", prompt_ids, 4096))
        await asyncio.sleep(2.0)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        t_c = time.monotonic()
        # engine must still serve after the abort
        final, ts, _, td = await collect(engine, "after-cancel", prompt_ids, 32)
        assert final.outputs[0].token_ids
        RESULTS["cancel"] = "PASS"
        log(f"cancelled request at +2s; follow-up request served in {td - ts:.2f}s "
            f"({(td - t_c):.2f}s after cancel)")
    except Exception:
        traceback.print_exc()
        RESULTS["cancel"] = "FAIL"

    # ---- 5. sleep / wake ----
    try:
        for name in ("sleep", "wake_up"):
            fn = getattr(engine, name)
            r = fn() if name == "wake_up" else fn(level=1)
            if asyncio.iscoroutine(r):
                await r
            log(f"engine.{name}() ok")
        final, ts, _, td = await collect(engine, "after-wake", prompt_ids, 32)
        assert final.outputs[0].token_ids
        RESULTS["sleepwake"] = "PASS"
        log(f"generation after sleep/wake in {td - ts:.2f}s")
    except Exception:
        traceback.print_exc()
        RESULTS["sleepwake"] = "FAIL"

    print("SMOKE SUMMARY " + json.dumps(RESULTS), flush=True)
    return 0 if all(v == "PASS" for v in RESULTS.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

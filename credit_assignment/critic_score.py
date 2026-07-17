# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""F8a extension — score diag prefix states with a trained critic (offline, single GPU).

Loads a model_merger-merged critic (AutoModelForTokenClassification, num_labels=1) and
computes V_phi(s_t) for every state dumped by diag_runner (--dump requires a
prefix_values.json that contains "states"; runs from 2026-07-16 onward). The state prompt
is tokenized EXACTLY like diag_runner.Generator.prompt_ids (chat template, no generation
tokens beyond the template, protocol truncation side) and the value is read at the last
prompt token — the turn-level critic's V(s_t) convention (value of the state the turn is
generated from).

  python -m credit_assignment.critic_score \
      --critic outputs/diag/<label>/critic_merged \
      --prefix-values outputs/diag/<label>/prefix_values.json \
      --out outputs/diag/<label>/critic_values.json \
      [--max-prompt-length 4096] [--truncation left] [--batch 16]

Output: {"critic": <dir>, "values": {traj: {depth: v}}}, feeding
diagnostic.build_pairs(prefix_values=vhat, assigned_advantages=critic deltas) for the
lambda-sweep gate (docs/plan_lambda_sweep.md §2a).
"""
import argparse
import json

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def prompt_ids(tokenizer, obs_text, max_prompt_length, truncation, enable_thinking):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": obs_text}],
        add_generation_prompt=True, tokenize=False,
        enable_thinking=enable_thinking,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_prompt_length:
        ids = ids[-max_prompt_length:] if truncation == "left" else ids[:max_prompt_length]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", required=True, help="merged critic dir (TokenClassification)")
    ap.add_argument("--prefix-values", required=True, help="prefix_values.json WITH states")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-prompt-length", type=int, default=4096)
    ap.add_argument("--truncation", default="left")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    payload = json.load(open(args.prefix_values))
    states = payload.get("states")
    assert states, ("prefix_values.json has no 'states' — re-run diag_runner (state dumping "
                    "landed 2026-07-16); older runs persisted vhat only.")
    enable_thinking = bool(payload["config"].get("enable_thinking", True))

    tokenizer = AutoTokenizer.from_pretrained(args.critic)
    model = AutoModelForTokenClassification.from_pretrained(
        args.critic, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    items = [(t, d, s) for t, per in states.items() for d, s in per.items()]
    values = {}
    with torch.no_grad():
        for start in range(0, len(items), args.batch):
            chunk = items[start:start + args.batch]
            ids = [prompt_ids(tokenizer, s, args.max_prompt_length, args.truncation,
                              enable_thinking) for _, _, s in chunk]
            maxlen = max(len(x) for x in ids)
            pad = tokenizer.pad_token_id or 0
            input_ids = torch.full((len(ids), maxlen), pad, dtype=torch.long)
            attn = torch.zeros((len(ids), maxlen), dtype=torch.long)
            for r, x in enumerate(ids):  # left-pad so the last token sits at -1
                input_ids[r, maxlen - len(x):] = torch.tensor(x, dtype=torch.long)
                attn[r, maxlen - len(x):] = 1
            out = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda())
            v = out.logits[:, -1, 0].float().cpu()
            for (t, d, _), val in zip(chunk, v.tolist()):
                values.setdefault(t, {})[d] = val
            print(f"[critic_score] {min(start + args.batch, len(items))}/{len(items)}",
                  flush=True)

    with open(args.out, "w") as fh:
        json.dump({"critic": args.critic, "values": values}, fh)
    print(f"[critic_score] wrote {args.out} ({len(items)} states)", flush=True)


if __name__ == "__main__":
    main()

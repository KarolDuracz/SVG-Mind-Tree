#!/usr/bin/env python3
"""
build_plan_from_text.py

Read a text file, tokenize with GPT-4o tokenizer (via tiktoken),
and output a plan.json file suitable for write_plan_to_db.py.

Each node now has (x, y) coordinates starting from a random origin
and incremented by +10 per step.
"""

import argparse
import json
import random
from collections import Counter
import tiktoken


def tokenize_text(text: str):
    # Use GPT-4o tokenizer
    enc = tiktoken.get_encoding("o200k_base")  # GPT-4o family
    tokens = enc.encode(text, disallowed_special=())
    return tokens, enc


def build_plan(text: str):
    tokens, enc = tokenize_text(text)

    # start positions (randomized origin)
    start_x = random.randint(0, 100)
    start_y = random.randint(0, 100)

    instances = []
    for pos, tokid in enumerate(tokens):
        toktext = enc.decode([tokid])
        x = start_x + pos * 10
        y = start_y + pos * 10
        instances.append({
            "pos": pos,
            "token_id": tokid,
            "token_text": toktext,
            "x": x,
            "y": y
        })

    # adjacency edges: link pos -> pos+1
    edges = [[i, i+1] for i in range(len(tokens)-1)]

    # canonical IDs = unique set
    canonical_ids = sorted(set(str(tid) for tid in tokens), key=int)

    plan = {
        "canonical_token_ids": canonical_ids,
        "instances": instances,
        "adjacency_edges": edges
    }
    return plan, tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input text file (output.txt)")
    p.add_argument("--out", required=True, help="Output plan.json")
    p.add_argument("--stats", action="store_true", help="Print top token frequency stats")
    args = p.parse_args()

    # read file (robust to encodings)
    with open(args.input, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")

    plan, tokens = build_plan(text)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=2)

    print(f"[done] wrote plan with {len(plan['instances'])} instances, {len(plan['canonical_token_ids'])} canonical IDs to {args.out}")

    if args.stats:
        c = Counter(tokens)
        top10 = c.most_common(10)
        print(f"[stats] unique token ids: {len(c)}")
        print("Top 10 tokens:")
        enc = tiktoken.get_encoding("o200k_base")
        for tid, count in top10:
            print(f"  id={tid:<6} text={repr(enc.decode([tid])):<10} count={count}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
extract_and_build.py

1) Extract lines matching a pattern from an input file and write them to an output file.
2) Tokenize the extracted lines using tiktoken.
3) Plan a recursive graph (canonical nodes per token-id, instance nodes per token position,
   adjacency edges for substrings up to --maxlen).
4) (optionally) POST the plan to the server:
     - create canonical nodes if missing,
     - create instance nodes (with connect_to canonical_node_id when available),
     - create adjacency connections between instance nodes.

Usage:
    pip install requests tiktoken
    python extract_and_build.py --input src.js --pattern "if\\s*\\(" --out matches.txt --maxlen 6
"""

import argparse, re, sys, time, json
from collections import Counter
from typing import List, Dict, Tuple, Set

# external libs
try:
    import requests
except Exception:
    raise RuntimeError("This script requires 'requests'. Install with: pip install requests")

try:
    import tiktoken
except Exception:
    tiktoken = None

# optional chardet to better guess encodings
try:
    import chardet
except Exception:
    chardet = None

API_BASE = "http://localhost:5000"
API_NODES = API_BASE.rstrip('/') + "/api/nodes"
API_CONNS = API_BASE.rstrip('/') + "/api/connections"
API_GET_NODES = API_BASE.rstrip('/') + "/api/nodes"
API_GET_CONNS = API_BASE.rstrip('/') + "/api/connections"

DEFAULT_MAXLEN = 6
DEFAULT_DELAY = 0.01

# ---------- helpers ----------

def read_file_flexible(path: str) -> str:
    """Robustly read a file with fallback encodings."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except Exception as e:
        raise RuntimeError(f"Failed to open file {path}: {e}")

    # try chardet if available
    if chardet is not None:
        try:
            guess = chardet.detect(raw)
            enc = guess.get("encoding")
            conf = guess.get("confidence", 0)
            if enc:
                try:
                    data = raw.decode(enc)
                    print(f"[read] decoded using chardet: {enc} (conf={conf})")
                    return data
                except Exception:
                    pass
        except Exception:
            pass

    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            data = raw.decode(enc)
            print(f"[read] decoded using {enc}")
            return data
        except Exception:
            pass

    # fallback
    data = raw.decode("utf-8", errors="replace")
    print("[read] fallback decode utf-8 (errors replaced)")
    return data

import re
from typing import List

def extract_lines_matching(text: str, pattern: str, *, regex=True, ignore_case=False) -> List[str]:
    """
    Return list of entire lines that match the pattern.

    Behavior:
      - If regex=True, try to compile the provided pattern.
      - If compilation fails (bad regex), automatically fall back to literal substring matching
        using the raw pattern (no regex meta-characters). A warning is printed.
      - If regex=False, treat pattern as plain substring (no regex).
      - ignore_case toggles case-insensitive matching for both regex and literal modes.
    """
    flags = re.MULTILINE
    if ignore_case:
        flags |= re.IGNORECASE

    lines = text.splitlines(keepends=True)

    if not regex:
        # plain substring match (case sensitive / insensitive)
        if ignore_case:
            pat_lower = pattern.lower()
            return [ln for ln in lines if pat_lower in ln.lower()]
        else:
            return [ln for ln in lines if pattern in ln]

    # regex True: attempt compile; if fails, fall back to literal substring search
    try:
        pat = re.compile(pattern, flags)
    except re.error as err:
        # warn and fall back
        print(f"[warn] pattern compile failed: {err}. Falling back to literal substring search using the raw pattern.")
        if ignore_case:
            pat_lower = pattern.lower()
            return [ln for ln in lines if pat_lower in ln.lower()]
        else:
            return [ln for ln in lines if pattern in ln]

    # compiled successfully — perform regex search
    return [ln for ln in lines if pat.search(ln)]


# ---------- tokenizer & planning (based on previous scripts) ----------

def get_tokenizer(preferred_model="gpt4o"):
    if tiktoken is None:
        raise RuntimeError("tiktoken is required. Install: pip install tiktoken")
    try:
        enc = tiktoken.encoding_for_model(preferred_model)
        print(f"[tokenizer] using encoding_for_model('{preferred_model}')")
        return enc
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
        print("[tokenizer] fallback to 'cl100k_base'")
        return enc

def tokenize_and_decode(enc, text: str) -> Tuple[List[int], List[str]]:
    token_ids = enc.encode(text)
    token_texts = []
    for tid in token_ids:
        try:
            token_texts.append(enc.decode([tid]))
        except Exception:
            try:
                b = enc.decode_single_token_bytes(tid)
                token_texts.append(b.decode("utf-8", errors="replace"))
            except Exception:
                token_texts.append(f"<tok:{tid}>")
    return token_ids, token_texts

def plan_graph(token_ids: List[int], token_texts: List[str], maxlen: int):
    """Plan canonical token ids, instance nodes per position and adjacency edges (substrings up to maxlen)."""
    N = len(token_ids)
    canonical_token_ids = set(str(t) for t in set(token_ids))
    instances = [{"pos": pos, "token_id": int(token_ids[pos]), "token_text": token_texts[pos]} for pos in range(N)]
    adjacency_edges = set()
    for i in range(N):
        for L in range(1, maxlen+1):
            j = i + L
            if j > N:
                break
            for k in range(i, j-1):
                adjacency_edges.add((k, k+1))
    plan = {
        "canonical_token_ids": canonical_token_ids,
        "instances": instances,
        "adjacency_edges": adjacency_edges
    }
    return plan

# ---------- server helpers (create canonical, instances, connections) ----------

def get_server_nodes(session: requests.Session) -> Dict[int, Dict]:
    try:
        r = session.get(API_GET_NODES, timeout=30)
        r.raise_for_status()
        arr = r.json()
        return {int(n.get("id")): n for n in arr}
    except Exception as e:
        print(f"[warn] GET /api/nodes failed: {e}")
        return {}

def find_canonical_by_description(session: requests.Session) -> Dict[str,int]:
    """Map node.description -> node.id for server-side reuse of canonical token class nodes."""
    nodes = get_server_nodes(session)
    mapping = {}
    for nid, node in nodes.items():
        desc = node.get("description")
        if desc is None: continue
        key = str(desc)
        if key not in mapping:
            mapping[key] = nid
    return mapping

def post_node(session: requests.Session, payload: dict) -> dict:
    r = session.post(API_NODES, json=payload, headers={"Content-Type":"application/json"}, timeout=30)
    if r.status_code not in (200,201):
        raise RuntimeError(f"POST /api/nodes failed {r.status_code}: {r.text}")
    try:
        return r.json()
    except Exception:
        return {}

def post_connection(session: requests.Session, source: int, target: int, metadata="auto"):
    payload = {"source": int(source), "target": int(target), "metadata": metadata}
    r = session.post(API_CONNS, json=payload, headers={"Content-Type":"application/json"}, timeout=30)
    if r.status_code not in (200,201):
        return {"ok": False, "status": r.status_code, "text": r.text}
    try:
        return {"ok": True, "json": r.json()}
    except Exception:
        return {"ok": True, "text": r.text}

def execute_plan(session: requests.Session, plan: dict, delay: float = DEFAULT_DELAY, dry_run=False):
    tokenid_to_canonical = {}
    pos_to_instance = {}
    # discover existing canonical
    existing = find_canonical_by_description(session)
    print(f"[server] found {len(existing)} canonical nodes by description")
    # create canonical nodes for missing token ids
    for tidstr in sorted(plan["canonical_token_ids"], key=lambda x:int(x)):
        if tidstr in existing:
            tokenid_to_canonical[tidstr] = existing[tidstr]
            print(f"[reuse canonical] token_id {tidstr} -> node #{existing[tidstr]}")
            continue
        payload = {"name": f"token_class_{tidstr}", "text": f"token_class_{tidstr}", "description": tidstr, "sequence": True}
        if dry_run:
            print(f"[dry-run] would create canonical node for token {tidstr}")
            tokenid_to_canonical[tidstr] = None
            continue
        created = post_node(session, payload)
        nid = int(created.get("id") or created.get("node_id") or 0)
        if nid:
            tokenid_to_canonical[tidstr] = nid
            print(f"[created canonical] token_id {tidstr} -> node #{nid}")
        else:
            print(f"[warn] canonical create returned no id for token {tidstr}")
        time.sleep(delay)

    # create instance nodes (per position), with connect_to canonical if available
    for inst in plan["instances"]:
        pos = inst["pos"]; tid = inst["token_id"]; tok = inst["token_text"]; tidstr = str(tid)
        payload = {"name": tok, "text": tok, "description": tidstr, "sequence": True}
        if tidstr in tokenid_to_canonical and tokenid_to_canonical[tidstr]:
            payload["connect_to"] = int(tokenid_to_canonical[tidstr])
        if dry_run:
            print(f"[dry-run] would create instance pos={pos} token_id={tid} repr={repr(tok)} connect_to={payload.get('connect_to')}")
            pos_to_instance[pos] = None
            continue
        try:
            created = post_node(session, payload)
        except Exception as e:
            print(f"[error] creating instance pos={pos}: {e}")
            continue
        nid = int(created.get("id") or created.get("node_id") or 0)
        if nid:
            pos_to_instance[pos] = nid
            print(f"[created instance] pos={pos} -> node #{nid} token_id={tid} repr={repr(tok[:40])}")
        else:
            print(f"[warn] no id returned for instance pos={pos}, response: {created}")
        time.sleep(delay)

    # create adjacency connections (dedup)
    edges = plan["adjacency_edges"]
    seen = set()
    created_conns = []
    failed_conns = []
    for (src_pos, tgt_pos) in sorted(edges):
        if src_pos not in pos_to_instance or tgt_pos not in pos_to_instance:
            # maybe missing from earlier failure or dry-run (pos_to_instance[pos]=None)
            if dry_run:
                print(f"[dry-run] would create connection pos {src_pos} -> {tgt_pos}")
            else:
                print(f"[skip] missing instance ids for edge {src_pos}->{tgt_pos}")
            continue
        src_id = pos_to_instance[src_pos]
        tgt_id = pos_to_instance[tgt_pos]
        # skip in dry-run if we don't have real ids
        if dry_run:
            print(f"[dry-run] planned connection: {src_pos} -> {tgt_pos} (pos-ids not created in dry-run)")
            continue
        if (src_id, tgt_id) in seen:
            continue
        res = post_connection(session, src_id, tgt_id, metadata="adjacency")
        if res.get("ok"):
            created_conns.append((src_id, tgt_id))
            seen.add((src_id, tgt_id))
            print(f"[created conn] {src_id} -> {tgt_id}")
        else:
            failed_conns.append((src_id, tgt_id, res))
            print(f"[failed conn] {src_id} -> {tgt_id} -> {res}")
        time.sleep(delay)

    return {
        "canonical_map": tokenid_to_canonical,
        "instances_map": pos_to_instance,
        "connections_created": created_conns,
        "connections_failed": failed_conns
    }

# ---------- main flow ----------

def main():
    ap = argparse.ArgumentParser(description="Extract lines matching pattern and build recursive graph from them.")
    ap.add_argument("--input", "-i", required=True, help="Input file (source code or text)")
    ap.add_argument("--pattern", "-p", default="if", help="Regex pattern to match lines (default 'if')")
    ap.add_argument("--out", "-o", default="matched_lines.txt", help="Output file with matched lines")
    ap.add_argument("--maxlen", type=int, default=DEFAULT_MAXLEN, help="Max substring length per start index")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between HTTP calls (s)")
    ap.add_argument("--plain", action="store_true", help="Treat pattern as plain substring (not regex)")
    ap.add_argument("--ignore-case", action="store_true", help="Case-insensitive matching")
    ap.add_argument("--dry-run", action="store_true", help="Do not send POST/PUT requests; only write matched file and show plan")
    ap.add_argument("--confirm", action="store_true", help="Skip confirmation (auto proceed)")
    args = ap.parse_args()

    # read input
    try:
        src = read_file_flexible(args.input)
    except Exception as e:
        print("Failed to read input file:", e)
        sys.exit(1)

    # extract lines
    matched = extract_lines_matching(src, args.pattern, regex=(not args.plain), ignore_case=args.ignore_case)
    print(f"[extract] found {len(matched)} matching lines (pattern={args.pattern!r})")

    if len(matched) == 0:
        print("No matches found — exiting.")
        sys.exit(0)

    # write out file
    try:
        with open(args.out, "w", encoding="utf-8") as fh:
            for ln in matched:
                fh.write(ln)
        print(f"[write] wrote {len(matched)} lines to {args.out}")
    except Exception as e:
        print("Failed to write output file:", e)
        sys.exit(1)

    # Now process matched text (tokenize and plan)
    matched_text = "".join(matched)
    try:
        enc = get_tokenizer()
    except Exception as e:
        print("Tokenizer error:", e)
        sys.exit(1)

    token_ids, token_texts = tokenize_and_decode(enc, matched_text)
    print(f"[tokenize] tokens: {len(token_ids)} from matched lines")

    # stats
    text_counter = Counter(token_texts)
    id_counter = Counter(token_ids)
    print(f"[stats] unique token texts: {len(text_counter)}, unique token ids: {len(id_counter)}")
    top_texts = text_counter.most_common(10)
    print("\nTop 10 token texts (repr, token id, count):")
    text_to_first_id = {}
    for pos, t in enumerate(token_texts):
        if t not in text_to_first_id:
            text_to_first_id[t] = token_ids[pos]
    for t,cnt in top_texts:
        tid = text_to_first_id.get(t)
        repr_t = t.replace("\n","\\n").replace("\r","\\r").replace("\t","\\t")
        print(f"  {repr_t!r}   id={tid}   count={cnt}")
    top_ids = id_counter.most_common(10)
    print("\nTop 10 token ids (id, text, count):")
    id_to_text = {}
    for pos, tid in enumerate(token_ids):
        if tid not in id_to_text:
            try:
                id_to_text[tid] = enc.decode([tid])
            except Exception:
                try:
                    b = enc.decode_single_token_bytes(tid)
                    id_to_text[tid] = b.decode("utf-8", errors="replace")
                except Exception:
                    id_to_text[tid] = f"<tok:{tid}>"
    for tid, cnt in top_ids:
        decoded = id_to_text.get(tid, f"<tok:{tid}>")
        print(f"  id={tid}   text={decoded!r}   count={cnt}")

    # Plan graph
    plan = plan_graph(token_ids, token_texts, args.maxlen)
    print(f"\n[plan] canonical token ids: {len(plan['canonical_token_ids'])}, instances: {len(plan['instances'])}, adjacency edges: {len(plan['adjacency_edges'])}")

    if args.dry_run:
        print("[dry-run] not sending any POST requests. Exiting after planning.")
        sys.exit(0)

    # confirm
    if not args.confirm:
        ok = input("Proceed to POST planned canonical nodes, instance nodes and connections? (y/N) ").strip().lower()
        if ok != "y":
            print("Aborted by user.")
            sys.exit(0)

    # execute
    session = requests.Session()
    result = execute_plan(session, plan, delay=args.delay, dry_run=args.dry_run)

    print("\n=== RESULT SUMMARY ===")
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()

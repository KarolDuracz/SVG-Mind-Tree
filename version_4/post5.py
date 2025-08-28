#!/usr/bin/env python3
"""
post_nodes.py

Build recursive sequence graphs from text with branching on repeated tokens.

Behavior:
 - Tokenize input with tiktoken (token ids + token text per position).
 - Plan everything in-memory:
    * Create one canonical node per unique token id (if not already present on the server).
    * Create one instance node per token position in the text.
      - For each instance:
         - If canonical node exists for that token_id, create the instance node with payload `connect_to: canonical_node_id`
           (this produces a branch from canonical -> instance).
         - Always set `name` and `text` to the token text, and `description` to the token id (string).
    * Create adjacency connections between successive instance nodes (position p -> p+1).
 - After plan ready, POST canonical nodes, POST instance nodes (including connect_to where planned),
   then POST adjacency connections (deduped).
 - Prints tokenization stats (top-10 tokens) before sending.

Usage:
  python post_nodes.py --file input.txt --maxlen 6 --delay 0.02
  python post_nodes.py --text "some code ..." --maxlen 6
"""

from typing import List, Dict, Tuple, Set
import argparse, json, sys, time
from collections import Counter, defaultdict

try:
    import requests
except Exception:
    raise RuntimeError("requests required: pip install requests")

try:
    import tiktoken
except Exception:
    tiktoken = None

# optional chardet
try:
    import chardet
except Exception:
    chardet = None

API_BASE = "http://localhost:5000"
API_NODES = f"{API_BASE}/api/nodes"
API_CONNS = f"{API_BASE}/api/connections"
API_GET_NODES = f"{API_BASE}/api/nodes"
API_GET_CONNS = f"{API_BASE}/api/connections"

DEFAULT_MAXLEN = 6
DEFAULT_DELAY = 0.01

# ---------- file reading ----------
def read_file_flexible(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except Exception as e:
        raise RuntimeError(f"Failed to open file {path}: {e}")

    # try chardet
    if chardet is not None:
        try:
            guess = chardet.detect(raw)
            enc = guess.get("encoding")
            conf = guess.get("confidence", 0)
            if enc:
                try:
                    text = raw.decode(enc)
                    print(f"[read_file] decoded using chardet: {enc} (conf={conf})")
                    return text
                except Exception:
                    pass
        except Exception:
            pass

    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            print(f"[read_file] decoded using {enc}")
            return text
        except Exception:
            pass

    # fallback replace
    text = raw.decode("utf-8", errors="replace")
    print("[read_file] fallback decode utf-8 with replacement")
    return text

# ---------- tokenizer ----------
def get_tokenizer(preferred_model="gpt4o"):
    if tiktoken is None:
        raise RuntimeError("tiktoken required (pip install tiktoken)")
    try:
        enc = tiktoken.encoding_for_model(preferred_model)
        print(f"[tokenizer] using encoding_for_model('{preferred_model}')")
        return enc
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
        print("[tokenizer] fallback to cl100k_base")
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

# ---------- server helpers ----------
def get_server_nodes(session: requests.Session) -> Dict[int, Dict]:
    """Return dict of node_id -> node dict from GET /api/nodes"""
    try:
        r = session.get(API_GET_NODES)
        if r.status_code != 200:
            print(f"[warn] GET /api/nodes returned {r.status_code}")
            return {}
        arr = r.json()
        return {int(n["id"]): n for n in arr}
    except Exception as e:
        print(f"[warn] could not GET nodes: {e}")
        return {}

def find_canonical_nodes_by_tokenid(session: requests.Session) -> Dict[str,int]:
    """
    Query server nodes and map description (token_id string) -> node_id.
    This lets us reuse canonical nodes created earlier (server side).
    """
    server_nodes = get_server_nodes(session)
    mapping = {}
    for nid, node in server_nodes.items():
        desc = node.get("description")
        if desc is None: continue
        # description stored as string token_id in our protocol
        key = str(desc)
        if key not in mapping:
            mapping[key] = nid
    return mapping

def post_node(session: requests.Session, payload: dict) -> dict:
    r = session.post(API_NODES, json=payload, headers={"Content-Type":"application/json"})
    if r.status_code not in (200,201):
        raise RuntimeError(f"POST /api/nodes failed {r.status_code}: {r.text}")
    try:
        return r.json()
    except Exception:
        return {}

def post_connection(session: requests.Session, source: int, target: int, metadata: str="manual link") -> dict:
    payload = {"source": int(source), "target": int(target), "metadata": metadata}
    r = session.post(API_CONNS, json=payload, headers={"Content-Type":"application/json"})
    if r.status_code not in (200,201):
        return {"ok": False, "status": r.status_code, "text": r.text}
    try:
        return {"ok": True, "json": r.json()}
    except Exception:
        return {"ok": True, "text": r.text}

# ---------- planning logic ----------
def plan_graph(token_ids: List[int], token_texts: List[str], maxlen: int):
    """
    Plan the scratchpad graph:
     - canonical_token_ids: set(token_id strings)
     - instance_positions: nodes for each position index
     - adjacency edges between consecutive positions (pos -> pos+1) for substring recursion (we keep edges for all adjacencies up to maxlen windows)
     - recurrence: we will use canonical nodes to branch to instances when token repeats
    Returns:
       plan = {
         "canonical_token_ids": set of token_id strings,
         "instances": list of {pos, token_id, token_text},
         "adjacency_edges": set((src_pos,tgt_pos))
       }
    """
    N = len(token_ids)
    canonical_token_ids = set(str(tid) for tid in set(token_ids))
    # Instances: one per position
    instances = [{"pos": pos, "token_id": int(token_ids[pos]), "token_text": token_texts[pos]} for pos in range(N)]

    # adjacency edges: for each start i and length L=1..maxlen, add pos->pos+1 for adjacent positions
    adjacency_edges = set()
    for i in range(N):
        for L in range(1, maxlen+1):
            j = i + L
            if j > N: break
            for k in range(i, j-1):
                adjacency_edges.add((k, k+1))

    plan = {
        "canonical_token_ids": canonical_token_ids,
        "instances": instances,
        "adjacency_edges": adjacency_edges
    }
    return plan

# ---------- execution (send plan) ----------
def execute_plan(session: requests.Session, plan: dict, delay: float = DEFAULT_DELAY):
    """
    Execute the planned graph:
     1) Ensure canonical nodes exist on server (for each token_id string). Reuse existing if found by description==token_id.
     2) Create instance nodes for each position. If canonical exists for the token_id, send connect_to: canonical_node_id in the payload.
     3) After instance nodes created, create adjacency connections between instance nodes (pos->pos+1).
    """
    tokenid_to_canonical_nodeid: Dict[str,int] = {}
    pos_to_instance_nodeid: Dict[int,int] = {}

    # 1) discover existing canonical nodes (by description token id)
    existing_map = find_canonical_nodes_by_tokenid(session)
    print(f"[server] found {len(existing_map)} canonical nodes on server (by description token id)")

    # Plan: create canonical nodes for any token id not present on server
    canonical_token_ids = sorted(plan["canonical_token_ids"], key=lambda x: int(x))
    cnt = 0
    for tidstr in canonical_token_ids:
        if tidstr in existing_map:
            tokenid_to_canonical_nodeid[tidstr] = existing_map[tidstr]
            print(f"[reuse canonical] token_id {tidstr} -> node #{existing_map[tidstr]}")
            continue
        # create canonical node: name = f"token_class_{tidstr}" (we can include decoded text later)
        payload = {
            "name": f"token_class_{tidstr}",
            "text": f"token_class_{tidstr}",
            "description": tidstr,
            "sequence": True
        }
        created = post_node(session, payload)
        nid = int(created.get("id") or created.get("node_id") or 0)
        if nid == 0:
            # warn and skip
            print(f"[warn] canonical create returned no id for token {tidstr} -> response: {created}")
            continue
        tokenid_to_canonical_nodeid[tidstr] = nid
        print(f"[created canonical] {cnt} token_id {tidstr} -> node #{nid}")
        cnt += 1
        time.sleep(delay)

    # 2) Create instance nodes (one per position)
    # Instances ordered by position
    instances = plan["instances"]
    for inst in instances:
        pos = inst["pos"]
        tid = inst["token_id"]
        tok = inst["token_text"]
        tidstr = str(tid)

        # payload: name/text = token_text, description = token id
        payload = {
            "name": tok,
            "text": tok,
            "description": tidstr,
            "sequence": True
        }

        # If canonical exists, include connect_to to canonical so server will add canonical -> instance
        if tidstr in tokenid_to_canonical_nodeid:
            payload["connect_to"] = int(tokenid_to_canonical_nodeid[tidstr])

        try:
            created = post_node(session, payload)
        except Exception as e:
            print(f"[ERROR] creating instance node pos={pos} token_id={tid} repr={repr(tok)}: {e}")
            continue

        nid = int(created.get("id") or created.get("node_id") or 0)
        if nid == 0:
            print(f"[warn] instance create returned no id for pos {pos}: {created}")
            continue
        pos_to_instance_nodeid[pos] = nid
        repr_tok = tok.replace("\n","\\n").replace("\t","\\t").replace("\r","\\r")
        ct = f"connect_to=canonical#{tokenid_to_canonical_nodeid[tidstr]}" if tidstr in tokenid_to_canonical_nodeid else "(no canonical)"
        print(f"[created instance] pos={pos} -> node #{nid} token_id={tid} repr={repr_tok} {ct}")
        time.sleep(delay)

    # 3) Create adjacency connections between instance nodes
    edges = plan["adjacency_edges"]
    created_conns = []
    failed_conns = []
    seen = set()
    print(f"[execute] creating adjacency connections: planned {len(edges)} edges")
    count_conn = 0
    for src_pos, tgt_pos in sorted(edges):
        if src_pos not in pos_to_instance_nodeid or tgt_pos not in pos_to_instance_nodeid:
            # node creation may have failed for a pos
            print(f"[skip edge] missing instance node for {src_pos}->{tgt_pos}")
            continue
        src_id = pos_to_instance_nodeid[src_pos]
        tgt_id = pos_to_instance_nodeid[tgt_pos]
        if (src_id, tgt_id) in seen:
            continue
        res = post_connection(session, src_id, tgt_id, metadata="adjacency")
        if res.get("ok"):
            created_conns.append((src_id, tgt_id))
            seen.add((src_id, tgt_id))
            print(f"[created conn] {count_conn} : {src_id} -> {tgt_id}")
            count_conn += 1
        else:
            failed_conns.append((src_id, tgt_id, res))
            print(f"[failed conn] {src_id}->{tgt_id} -> {res}")
        time.sleep(delay)

    return {
        "canonical_map": tokenid_to_canonical_nodeid,
        "instances_map": pos_to_instance_nodeid,
        "connections_created": created_conns,
        "connections_failed": failed_conns
    }

# ---------- orchestration ----------
def run(text: str, maxlen: int = DEFAULT_MAXLEN, delay: float = DEFAULT_DELAY):
    if not text:
        raise ValueError("empty input")

    enc = get_tokenizer()
    token_ids, token_texts = tokenize_and_decode(enc, text)
    total_tokens = len(token_ids)
    print(f"[tokenize] tokens: {total_tokens}")

    # Stats
    text_counter = Counter(token_texts)
    id_counter = Counter(token_ids)
    print(f"[stats] unique token texts: {len(text_counter)}, unique token ids: {len(id_counter)}")
    top_texts = text_counter.most_common(10)
    print("\nTop 10 token texts (repr, token id, count):")
    text_to_first_id = {}
    for pos, t in enumerate(token_texts):
        if t not in text_to_first_id:
            text_to_first_id[t] = token_ids[pos]
    for t, cnt in top_texts:
        tid = text_to_first_id.get(t)
        print(f"  {t.replace(chr(10),'\\n').replace(chr(9),'\\t')!r}   id={tid}   count={cnt}")

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
        print(f"  id={tid} text={id_to_text.get(tid)!r} count={cnt}")

    # Plan graph scratchpad
    plan = plan_graph(token_ids, token_texts, maxlen)
    print(f"\n[plan] canonical token ids: {len(plan['canonical_token_ids'])}, instances: {len(plan['instances'])}, adjacency edges: {len(plan['adjacency_edges'])}")
    print("Ready to send planned graph to server.")

    # Ask for confirmation before sending (safe-guard)
    proceed = input("Proceed to POST planned canonical nodes, instances and connections? (y/N) ").strip().lower()
    if proceed != "y":
        print("Aborted by user.")
        return None

    session = requests.Session()
    result = execute_plan(session, plan, delay=delay)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"canonical nodes created/reused: {len(result['canonical_map'])}")
    print(f"instance nodes created: {len(result['instances_map'])}")
    print(f"connections created: {len(result['connections_created'])}")
    if result['connections_failed']:
        print(f"connections failed: {len(result['connections_failed'])}")
    return result

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Build recursive graphs from text with branching on repeated tokens")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="input file path")
    group.add_argument("--text", help="inline text")
    parser.add_argument("--maxlen", type=int, default=DEFAULT_MAXLEN, help="max substring length per start index")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="delay between HTTP calls (s)")
    args = parser.parse_args()

    if args.file:
        try:
            text = read_file_flexible(args.file)
        except Exception as e:
            print("Failed to read file:", e)
            sys.exit(1)
    else:
        text = args.text

    try:
        res = run(text, maxlen=args.maxlen, delay=args.delay)
        if res is not None:
            print(json.dumps(res, indent=2))
    except Exception as e:
        print("[ERROR] run failed:", e)
        raise

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
write_plan_to_db.py

Write a planned graph directly into a SQLite database file.

Usage:
    python write_plan_to_db.py --db path/to/app.db --plan plan.json [--dry-run] [--verbose]

Plan file format (JSON):
{
  "canonical_token_ids": ["284","220", ...],
  "instances": [{"pos":0,"token_id":284,"token_text":" if"}, ...],
  "adjacency_edges": [[0,1], [1,2], ...]
}

Behavior:
 - For each canonical token id: reuse server node if a node exists with description == token_id (string),
   otherwise create a new canonical node (name/text = token_class_<id>, description = id).
 - For each instance (position) create a node (name/text = token_text, description = token_id).
   If canonical exists, add a connection canonical_node -> instance_node (like connect_to).
 - For each adjacency edge (src_pos -> tgt_pos), create a connection row (source_id,target_id).
 - All DB writes happen in a single transaction for speed.
"""

import argparse
import sqlite3
import json
import os
from datetime import datetime, date
from typing import Dict, Any, Tuple, List

def now_iso():
    return datetime.utcnow().isoformat(timespec='seconds')

def today_iso():
    return date.today().isoformat()

def read_json(path: str):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)

def find_node_by_description(cur, desc: str) -> int:
    cur.execute("SELECT id FROM nodes WHERE description = ? LIMIT 1;", (desc,))
    r = cur.fetchone()
    return r[0] if r else None

def insert_node(cur, name: str, x: float = None, y: float = None, description: str = None, text: str = None) -> int:
    """
    Insert node and return new id.
    Columns used (matching server's api_create_node):
      name, x, y, created_at, date, time, description, seq_angle, branch_side, text
    """
    created = now_iso()
    date_field = today_iso()
    time_field = datetime.utcnow().strftime("%H:%M:%S")
    # Use None for optional seq_angle and branch_side
    cur.execute(
        "INSERT INTO nodes (name, x, y, created_at, date, time, description, seq_angle, branch_side, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
        (name, x, y, created, date_field, time_field, description, None, None, text)
    )
    return cur.lastrowid

def connection_exists(cur, src_id: int, tgt_id: int) -> bool:
    cur.execute("SELECT 1 FROM connections WHERE source_id = ? AND target_id = ? LIMIT 1;", (src_id, tgt_id))
    return cur.fetchone() is not None

def insert_connection(cur, src_id: int, tgt_id: int, metadata: str = None):
    created = now_iso()
    cur.execute("INSERT INTO connections (source_id, target_id, created_at, metadata) VALUES (?, ?, ?, ?);", (src_id, tgt_id, created, metadata))

def main():
    p = argparse.ArgumentParser(description="Write planned graph data directly to SQLite DB.")
    p.add_argument("--db", required=True, help="Path to the sqlite .db file used by your app")
    p.add_argument("--plan", required=True, help="Path to the JSON plan file (see script header for format)")
    p.add_argument("--dry-run", action="store_true", help="Do not modify DB; show planned actions")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args()

    db_path = args.db
    plan_path = args.plan
    dry_run = args.dry_run
    verbose = args.verbose

    if not os.path.isfile(db_path):
        print("Error: DB file not found:", db_path)
        return

    plan = read_json(plan_path)
    canonical_ids = plan.get("canonical_token_ids", [])
    instances = plan.get("instances", [])
    adjacency_edges = plan.get("adjacency_edges", [])

    print(f"[plan] canonical tokens: {len(canonical_ids)}, instances: {len(instances)}, adjacency edges: {len(adjacency_edges)}")
    if dry_run:
        print("[dry-run] No DB changes will be made. Showing planned steps...")

    conn = sqlite3.connect(db_path)
    # return rows as dict-like
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        # Ensure foreign_keys is on (if schema uses it)
        cur.execute("PRAGMA foreign_keys = ON;")

        # Mapping token_id string -> canonical_node_id
        tokenid_to_canonical = {}

        # 1) Check existing nodes for canonical token ids
        for tid in canonical_ids:
            tidstr = str(tid)
            existing = find_node_by_description(cur, tidstr)
            if existing:
                tokenid_to_canonical[tidstr] = existing
                if verbose: print(f"[reuse] description={tidstr} -> node id {existing}")
            else:
                if dry_run:
                    print(f"[dry-run] would create canonical node for token_id {tidstr}")
                else:
                    name = f"token_class_{tidstr}"
                    created_id = insert_node(cur, name=name, x=None, y=None, description=tidstr, text=name)
                    tokenid_to_canonical[tidstr] = created_id
                    if verbose: print(f"[create canonical] token_id={tidstr} -> node #{created_id}")

        # 2) Create instance nodes (one per position). We'll create them in ascending pos order to be deterministic.
        #    For each instance, if its token_id has a canonical node, we'll also create a connection canonical -> instance
        pos_to_nodeid: Dict[int, int] = {}
        instances_sorted = sorted(instances, key=lambda x: int(x.get("pos", 0)))
        for inst in instances_sorted:
            pos = int(inst["pos"])
            token_id = int(inst["token_id"])
            token_text = inst.get("token_text", None)
            tidstr = str(token_id)

            if dry_run:
                print(f"[dry-run] would create instance node pos={pos} token_id={token_id} repr={repr(token_text)} connect_to canonical={tokenid_to_canonical.get(tidstr)}")
                pos_to_nodeid[pos] = None
            else:
                # name/text stored as token text; description stores token id
                nid = insert_node(cur, name=token_text, x=None, y=None, description=tidstr, text=token_text)
                pos_to_nodeid[pos] = nid
                if verbose: print(f"[create instance] pos={pos} -> node #{nid} token_id={token_id}")

                # if canonical exists, create connection canonical -> instance (if not exists)
                canid = tokenid_to_canonical.get(tidstr)
                if canid:
                    if not connection_exists(cur, canid, nid):
                        insert_connection(cur, canid, nid, metadata="connect_to_canonical")
                        if verbose: print(f"[create conn] canonical {canid} -> instance {nid}")
                    else:
                        if verbose: print(f"[exists conn] canonical {canid} -> instance {nid} already exists")

        # 3) Create adjacency edges between instance nodes (pos -> pos)
        # deduplicate and use existing pos_to_nodeid mapping
        created_conn_count = 0
        for (src_pos, tgt_pos) in adjacency_edges:
            src_pos = int(src_pos); tgt_pos = int(tgt_pos)
            src_nid = pos_to_nodeid.get(src_pos)
            tgt_nid = pos_to_nodeid.get(tgt_pos)
            if src_nid is None or tgt_nid is None:
                # maybe dry-run or failed earlier; skip
                if dry_run:
                    print(f"[dry-run] would create adjacency connection pos {src_pos} -> {tgt_pos}")
                else:
                    print(f"[skip] missing instance node for edge {src_pos}->{tgt_pos} (src_nid={src_nid}, tgt_nid={tgt_nid})")
                continue

            if dry_run:
                print(f"[dry-run] would create adjacency connection node {src_nid} -> {tgt_nid}")
                continue

            if not connection_exists(cur, src_nid, tgt_nid):
                insert_connection(cur, src_nid, tgt_nid, metadata="adjacency")
                created_conn_count += 1
                if verbose: print(f"[create conn] {src_nid} -> {tgt_nid}")
            else:
                if verbose: print(f"[exists conn] {src_nid} -> {tgt_nid} already exists")

        # commit when not dry-run
        if dry_run:
            print("[dry-run] finished. No changes committed.")
            conn.rollback()
        else:
            conn.commit()
            print(f"[done] committed changes. Created canonical nodes: {len(tokenid_to_canonical)}, created instance nodes: {len([p for p in pos_to_nodeid.values() if p])}, created adjacency connections: {created_conn_count}")

    except Exception as e:
        conn.rollback()
        print("[error] exception occurred – rolled back transaction:", e)
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()

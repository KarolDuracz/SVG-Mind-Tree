#!/usr/bin/env python3
"""
scale_nodes_space.py

Fetch all nodes, compute bounding box, scale node positions away from bounding-box center
by a linear factor (default 10x), and PUT updated x,y back to the server.

Writes two CSVs:
  - <out_before> (original positions)
  - <out_after>  (new positions after scaling)

Options:
  --url: base URL of the API server (default http://localhost:5000)
  --scale: linear scale factor (default 10)
  --dry-run: simulate only; do NOT send PUT requests
  --out-before / --out-after: CSV filenames
  --delay: seconds sleep between PUT requests (default 0.02)
"""

import argparse
import csv
import json
import sys
import time
from typing import List, Dict

import requests


def fetch_nodes(base_url: str) -> List[Dict]:
    url = base_url.rstrip('/') + '/api/nodes'
    print(f"[fetch] GET {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    nodes = r.json()
    if not isinstance(nodes, list):
        raise RuntimeError("Expected JSON list from /api/nodes")
    print(f"[fetch] got {len(nodes)} nodes")
    return nodes


def compute_bbox(nodes: List[Dict]):
    xs = []
    ys = []
    missing = []
    for n in nodes:
        try:
            x = n.get('x', None)
            y = n.get('y', None)
            if x is None or y is None:
                missing.append(n.get('id'))
                continue
            xs.append(float(x))
            ys.append(float(y))
        except Exception:
            missing.append(n.get('id'))
    if not xs or not ys:
        raise RuntimeError("No nodes with valid x/y coordinates found.")
    minx = min(xs); maxx = max(xs)
    miny = min(ys); maxy = max(ys)
    return (minx, maxx, miny, maxy, missing)


def scale_point(x: float, y: float, cx: float, cy: float, scale: float):
    nx = cx + (x - cx) * scale
    ny = cy + (y - cy) * scale
    return nx, ny


def put_node_position(base_url: str, node_id: int, x: float, y: float, session: requests.Session):
    url = base_url.rstrip('/') + f'/api/nodes/{node_id}'
    payload = {"x": float(x), "y": float(y)}
    headers = {"Content-Type": "application/json"}
    r = session.put(url, headers=headers, json=payload, timeout=30)
    return r


def save_csv(filename: str, rows: List[Dict], fieldnames: List[str]):
    with open(filename, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})


def main():
    p = argparse.ArgumentParser(description="Scale node positions by expanding space about bounding-box center.")
    p.add_argument('--url', default='http://localhost:5000', help='API base URL (default http://localhost:5000)')
    p.add_argument('--scale', type=float, default=10.0, help='Linear scale factor (default 10)')
    p.add_argument('--dry-run', action='store_true', help='Do not send PUT requests; just simulate')
    p.add_argument('--out-before', default='nodes_before.csv', help='CSV file for original nodes')
    p.add_argument('--out-after', default='nodes_after.csv', help='CSV file for updated nodes')
    p.add_argument('--delay', type=float, default=0.02, help='Delay (s) between PUT requests')
    args = p.parse_args()

    base_url = args.url
    scale = args.scale
    dry_run = args.dry_run
    out_before = args.out_before
    out_after = args.out_after
    delay = args.delay

    print(f"[start] base_url={base_url} scale={scale} dry_run={dry_run}")

    try:
        nodes = fetch_nodes(base_url)
    except Exception as e:
        print("[error] failed to fetch nodes:", e)
        sys.exit(1)

    # Save original table
    before_rows = []
    for n in nodes:
        before_rows.append({
            "id": n.get("id"),
            "name": n.get("name"),
            "x": n.get("x"),
            "y": n.get("y"),
            "description": n.get("description")
        })
    save_csv(out_before, before_rows, ["id", "name", "x", "y", "description"])
    print(f"[save] original nodes saved to {out_before}")

    # Extract coords and compute bbox
    try:
        minx, maxx, miny, maxy, missing = compute_bbox(nodes)
    except Exception as e:
        print("[error] cannot compute bounding box:", e)
        sys.exit(1)

    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    width = maxx - minx
    height = maxy - miny

    print("[bbox] minx=%.3f maxx=%.3f miny=%.3f maxy=%.3f" % (minx, maxx, miny, maxy))
    print("[bbox] center_x=%.3f center_y=%.3f width=%.3f height=%.3f" % (cx, cy, width, height))
    if missing:
        print(f"[warn] {len(missing)} nodes missing x/y (skipped): sample ids: {missing[:10]}")

    # Build updates
    updates = []
    for n in nodes:
        nid = n.get("id")
        x = n.get("x", None)
        y = n.get("y", None)
        if x is None or y is None:
            continue
        try:
            x = float(x); y = float(y)
        except Exception:
            print(f"[warn] node {nid} has non-numeric x/y, skipping")
            continue
        nx, ny = scale_point(x, y, cx, cy, scale)
        updates.append({
            "id": nid,
            "old_x": x,
            "old_y": y,
            "new_x": nx,
            "new_y": ny,
            "name": n.get("name")
        })

    print(f"[plan] prepared {len(updates)} updates (scale factor {scale})")

    # If dry-run, show sample and exit without sending
    if dry_run:
        print("[dry-run] sample of planned updates:")
        for u in updates[:10]:
            print(f" id={u['id']}  {u['old_x']:.2f},{u['old_y']:.2f} -> {u['new_x']:.2f},{u['new_y']:.2f}")
        # save after CSV with new coords (not applied to server)
        after_rows = []
        for u in updates:
            after_rows.append({
                "id": u["id"],
                "name": u["name"],
                "old_x": u["old_x"],
                "old_y": u["old_y"],
                "new_x": u["new_x"],
                "new_y": u["new_y"]
            })
        save_csv(out_after, after_rows, ["id", "name", "old_x", "old_y", "new_x", "new_y"])
        print(f"[save] planned updates saved to {out_after}")
        print("[dry-run] finished (no PUTs were made).")
        sys.exit(0)

    # Confirm
    print(f"[confirm] About to send {len(updates)} PUT requests to update node positions. This will modify the server DB.")
    ok = input("Proceed? (y/N) ").strip().lower()
    if ok != 'y':
        print("Aborted by user.")
        sys.exit(0)

    # Send PUT requests sequentially
    session = requests.Session()
    after_rows = []
    success = 0
    failed = 0
    for idx, u in enumerate(updates, start=1):
        nid = u["id"]
        nx = u["new_x"]
        ny = u["new_y"]
        try:
            r = put_node_position(base_url, nid, nx, ny, session)
            if r.status_code in (200, 201):
                success += 1
                print(f"[{idx}/{len(updates)}] updated node {nid} -> {nx:.2f},{ny:.2f}")
            else:
                failed += 1
                print(f"[{idx}/{len(updates)}] FAILED updating node {nid} -> {r.status_code} {r.text}")
        except Exception as e:
            failed += 1
            print(f"[{idx}/{len(updates)}] ERROR updating node {nid}: {e}")

        after_rows.append({
            "id": nid,
            "name": u.get("name"),
            "old_x": u.get("old_x"),
            "old_y": u.get("old_y"),
            "new_x": nx,
            "new_y": ny
        })
        time.sleep(delay)

    save_csv(out_after, after_rows, ["id", "name", "old_x", "old_y", "new_x", "new_y"])
    print(f"[save] wrote updated node table to {out_after}")
    print(f"[done] PUTs succeeded: {success}, failed: {failed}")

if __name__ == '__main__':
    main()

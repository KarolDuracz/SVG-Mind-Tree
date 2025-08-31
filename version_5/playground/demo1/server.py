#!/usr/bin/env python3
"""
Token Vector Animator — Backend (groups + vocab + persistence)

Endpoints:
- GET  /                  -> serves index.html
- POST /tokenize          -> tokenizes text using tiktoken and returns tokens + centroid + prompt_key
- POST /token_at          -> nearest vocab token lookup
- GET  /vocab_info        -> readiness and basic stats
- GET  /groups            -> list groups
- POST /groups            -> create group
- POST /groups/clear      -> delete all groups
- POST /groups/<gid>/add  -> add a prompt (prompt_key + centroid) into group
- POST /groups/<gid>/move -> move a member inside group (persist)
- GET  /groups/<gid>/members -> list group members
- POST /groups/<gid>/update_location -> update group's world_x/world_y
"""
import os
import json
import time
import hashlib
import logging
import threading
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, send_from_directory, abort

# ---------- Configuration ----------
VOCAB_MAX = 100_000       # number of token ids to include (0..VOCAB_MAX-1)
GRID_SIZE = 200           # grid resolution for spatial buckets (GRID_SIZE x GRID_SIZE)
GROUPS_FILE = "groups.json"
INDEX_HTML = "index.html"
HOST = "127.0.0.1"
PORT = 5000
# -----------------------------------

app = Flask(__name__, static_folder='.')

# Logging
logger = logging.getLogger("token_viz")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
fh = RotatingFileHandler("tokens.log", maxBytes=5_000_000, backupCount=2)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
sh.setFormatter(fmt); fh.setFormatter(fmt)
logger.addHandler(sh); logger.addHandler(fh)

# Load tokenizer
try:
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    logger.info("Loaded tiktoken encoding: cl100k_base")
except Exception as e:
    logger.exception("Failed to load tiktoken. Install via `pip install tiktoken`.")
    raise

# Data structures
vocab_tokens = []       # list of dicts { token_id, token_text, x, y }
grid_buckets = {}       # (ix,iy) -> list of indices into vocab_tokens
vocab_ready = False
vocab_build_seconds = None
vocab_lock = threading.Lock()

# Groups persistence
groups_lock = threading.Lock()

def deterministic_position_for_key(key: str):
    """Return deterministic normalized coords [0..1) from string key."""
    h = hashlib.sha256(key.encode("utf-8")).digest()
    x_int = int.from_bytes(h[0:8], "big")
    y_int = int.from_bytes(h[8:16], "big")
    x = (x_int % 10_000_000) / 10_000_000
    y = (y_int % 10_000_000) / 10_000_000
    return x, y

def bucket_index(x_norm, y_norm):
    ix = min(max(int(x_norm * GRID_SIZE), 0), GRID_SIZE - 1)
    iy = min(max(int(y_norm * GRID_SIZE), 0), GRID_SIZE - 1)
    return ix, iy

def add_to_bucket(ix, iy, idx):
    key = (ix, iy)
    if key not in grid_buckets:
        grid_buckets[key] = []
    grid_buckets[key].append(idx)

def build_vocab_and_grid():
    """Compute deterministic positions for token ids 0..VOCAB_MAX-1 and populate grid_buckets."""
    global vocab_tokens, grid_buckets, vocab_ready, vocab_build_seconds
    with vocab_lock:
        start = time.time()
        vocab_tokens = []
        grid_buckets = {}
        logger.info("Building vocab positions for token ids 0..%d (this may take a few seconds)...", VOCAB_MAX - 1)
        for tid in range(VOCAB_MAX):
            token_text = ""
            try:
                token_text = encoding.decode([tid])
            except Exception:
                token_text = ""
            key = f"{tid}:{token_text}"
            x, y = deterministic_position_for_key(key)
            idx = len(vocab_tokens)
            vocab_tokens.append({"token_id": int(tid), "token_text": token_text, "x": float(x), "y": float(y)})
            ix, iy = bucket_index(x, y)
            add_to_bucket(ix, iy, idx)
            if tid and (tid % 10000) == 0:
                logger.info("Processed %d tokens...", tid)
        vocab_build_seconds = time.time() - start
        vocab_ready = True
        logger.info("Vocab grid built: tokens=%d grid=%dx%d time=%.2fs", len(vocab_tokens), GRID_SIZE, GRID_SIZE, vocab_build_seconds)

# Build on startup (blocking)
build_vocab_and_grid()

# Groups persistence helpers
def load_groups():
    if os.path.exists(GROUPS_FILE):
        try:
            with open(GROUPS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info("Loaded groups from %s", GROUPS_FILE)
                return data
        except Exception:
            logger.exception("Failed to load groups.json, starting with empty data")
    return {"next_id": 1, "groups": []}

def save_groups(data):
    """
    Persist groups to disk.

    IMPORTANT: callers are expected to acquire groups_lock when appropriate.
    Avoid acquiring groups_lock here to prevent nested-lock deadlocks.
    """
    try:
        with open(GROUPS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved groups to %s", GROUPS_FILE)
    except Exception:
        logger.exception("Failed to save groups.json")

groups_data = load_groups()

# Utility: compute centroid of a token list
def centroid_of_tokens(tokens):
    if not tokens:
        return {"x": 0.5, "y": 0.5}
    sx = sum(t.get("x", 0.0) for t in tokens)
    sy = sum(t.get("y", 0.0) for t in tokens)
    n = len(tokens)
    return {"x": sx / n, "y": sy / n}

# ---------- Flask endpoints ----------

@app.route("/")
def index():
    if os.path.exists(INDEX_HTML):
        return send_from_directory(".", INDEX_HTML)
    return "index.html not found on server", 404

@app.route("/tokenize", methods=["POST"])
def tokenize_endpoint():
    """
    Tokenize arbitrary text and return the tokens with deterministic positions.
    Used for manual tokenization and realtime chunk tokenization.
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "") or ""
    logger.info("Tokenize request len(text)=%d", len(text))
    try:
        token_ids = encoding.encode(text)
    except Exception as e:
        logger.exception("Tokenization failed")
        return jsonify({"error": "tokenization failed", "details": str(e)}), 500
    tokens_out = []
    for idx, tid in enumerate(token_ids):
        try:
            tok_text = encoding.decode([tid])
        except Exception:
            tok_text = ""
        key = f"{tid}:{tok_text}"
        x, y = deterministic_position_for_key(key)
        tokens_out.append({"index": idx, "token_id": int(tid), "token_text": tok_text, "x": x, "y": y})
    prompt_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    centroid = centroid_of_tokens(tokens_out)
    logger.info("Tokenized chunk -> %d tokens prompt_key=%s centroid=(%.4f,%.4f)", len(tokens_out), prompt_key, centroid["x"], centroid["y"])
    return jsonify({"tokens": tokens_out, "count": len(tokens_out), "prompt_key": prompt_key, "centroid": centroid})

@app.route("/token_at", methods=["POST"])
def token_at():
    if not vocab_ready:
        return jsonify({"error": "vocab not ready"}), 503
    data = request.get_json(silent=True) or {}
    try:
        x = float(data.get("x", 0.5))
        y = float(data.get("y", 0.5))
    except Exception:
        return jsonify({"error": "invalid coordinates"}), 400
    ix, iy = bucket_index(x, y)
    # expand neighbor search until candidates found
    radius = 0
    candidates = []
    max_rad = max(GRID_SIZE // 2, 10)
    while radius <= max_rad and len(candidates) == 0:
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx = ix + dx; ny = iy + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    key = (nx, ny)
                    if key in grid_buckets:
                        candidates.extend(grid_buckets[key])
        radius += 1
    if not candidates:
        tok = vocab_tokens[0]
        return jsonify({"token_id": tok["token_id"], "token_text": tok["token_text"], "x": tok["x"], "y": tok["y"], "distance": 0.0})
    best = None
    best_d2 = float("inf")
    for idx in candidates:
        tok = vocab_tokens[idx]
        dx = tok["x"] - x; dy = tok["y"] - y
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_d2 = d2; best = tok
    return jsonify({"token_id": best["token_id"], "token_text": best["token_text"], "x": best["x"], "y": best["y"], "distance": best_d2 ** 0.5})

@app.route("/vocab_info", methods=["GET"])
def vocab_info():
    return jsonify({"ready": vocab_ready, "vocab_size": len(vocab_tokens), "grid_size": GRID_SIZE, "build_seconds": vocab_build_seconds})

# Groups endpoints
@app.route("/groups", methods=["GET"])
def list_groups():
    return jsonify(groups_data)

@app.route("/groups", methods=["POST"])
def create_group():
    body = request.get_json(silent=True) or {}
    name = body.get("name", "Group")
    world_x = float(body.get("world_x", 200.0))
    world_y = float(body.get("world_y", 200.0))
    with groups_lock:
        gid = groups_data.get("next_id", 1)
        g = {"id": gid, "name": name, "world_x": float(world_x), "world_y": float(world_y), "w": 1100, "h": 700, "members": []}
        groups_data["groups"].append(g)
        groups_data["next_id"] = gid + 1
        save_groups(groups_data)
    logger.info("Created group %s id=%d", name, gid)
    return jsonify(g), 201

@app.route("/groups/clear", methods=["POST"])
def clear_groups():
    """Clear all groups (delete)."""
    with groups_lock:
        groups_data.clear()
        groups_data.update({"next_id": 1, "groups": []})
        save_groups(groups_data)
    logger.info("Cleared all groups")
    return jsonify({"ok": True, "groups_cleared": True})

@app.route("/groups/<int:gid>/add", methods=["POST"])
def add_to_group(gid):
    body = request.get_json(silent=True) or {}
    prompt_key = body.get("prompt_key")
    centroid = body.get("centroid") or {"x": 0.5, "y": 0.5}
    local_x = float(body.get("local_x", centroid.get("x", 0.5)))
    local_y = float(body.get("local_y", centroid.get("y", 0.5)))
    with groups_lock:
        for g in groups_data["groups"]:
            if g["id"] == gid:
                member_id = (g["members"][-1]["member_id"] + 1) if g["members"] else 1
                item = {"member_id": member_id, "prompt_key": prompt_key, "local_x": float(local_x), "local_y": float(local_y), "centroid": centroid}
                g["members"].append(item)
                save_groups(groups_data)
                logger.info("Added prompt %s to group %d at local (%.4f,%.4f)", prompt_key, gid, local_x, local_y)
                return jsonify(item), 201
    return jsonify({"error": "group not found"}), 404

@app.route("/groups/<int:gid>/move", methods=["POST"])
def move_member(gid):
    body = request.get_json(silent=True) or {}
    try:
        member_id = int(body.get("member_id"))
        lx = float(body.get("local_x"))
        ly = float(body.get("local_y"))
    except Exception:
        return jsonify({"error": "invalid payload"}), 400
    with groups_lock:
        for g in groups_data["groups"]:
            if g["id"] == gid:
                for m in g["members"]:
                    if m["member_id"] == member_id:
                        m["local_x"] = float(lx); m["local_y"] = float(ly)
                        save_groups(groups_data)
                        logger.info("Moved member %d in group %d to local (%.4f,%.4f)", member_id, gid, lx, ly)
                        return jsonify(m)
    return jsonify({"error": "not found"}), 404

@app.route("/groups/<int:gid>/members", methods=["GET"])
def get_members(gid):
    for g in groups_data["groups"]:
        if g["id"] == gid:
            return jsonify(g["members"])
    return jsonify({"error": "not found"}), 404

@app.route("/groups/<int:gid>/update_location", methods=["POST"])
def update_group_location(gid):
    body = request.get_json(silent=True) or {}
    try:
        wx = float(body.get("world_x"))
        wy = float(body.get("world_y"))
    except Exception:
        return jsonify({"error": "invalid payload"}), 400
    with groups_lock:
        for g in groups_data["groups"]:
            if g["id"] == gid:
                g["world_x"] = float(wx)
                g["world_y"] = float(wy)
                save_groups(groups_data)
                logger.info("Updated group %d location to (%.2f,%.2f)", gid, wx, wy)
                return jsonify(g)
    return jsonify({"error": "not found"}), 404

# Serve static assets if any (index.html is in the same folder)
@app.route("/<path:path>")
def static_proxy(path):
    if os.path.exists(path):
        return send_from_directory(".", path)
    return abort(404)

if __name__ == "__main__":
    logger.info("Starting app on http://%s:%d", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=True)

from flask import Flask, request, jsonify, send_file
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import tiktoken
import time

app = Flask(__name__)

# logging
logger = logging.getLogger("vocab")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
fh = RotatingFileHandler("tokens.log", maxBytes=5_000_000, backupCount=2)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
sh.setFormatter(fmt); fh.setFormatter(fmt)
logger.addHandler(sh); logger.addHandler(fh)

# Load tiktoken encoding
try:
    encoding = tiktoken.get_encoding("cl100k_base")
    logger.info("Loaded encoding: cl100k_base")
except Exception as e:
    logger.exception("Failed to load tiktoken encoding; install tiktoken.")
    raise

# Configuration: how many token ids to consider for the 'vocabulary'
VOCAB_MAX = 100_000   # will attempt ids 0..VOCAB_MAX-1
GRID_SIZE = 200       # grid dimension for spatial hashing (GRID_SIZE x GRID_SIZE buckets)

# Data structures (populated at startup)
vocab_tokens = []     # list of dicts: { token_id, token_text, x, y }
grid_buckets = {}     # mapping (ix,iy)->list of token indices into vocab_tokens
vocab_ready = False
vocab_build_time = None

def deterministic_position_for_key(key: str):
    """Return deterministic normalized (x,y) [0..1) for string key."""
    h = hashlib.sha256(key.encode("utf8")).digest()
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
    """Build vocab_tokens list and spatial grid buckets for quick nearest-token lookup."""
    global vocab_tokens, grid_buckets, vocab_ready, vocab_build_time
    start = time.time()
    logger.info("Building vocab positions for token ids 0..%d (this may take a few seconds)...", VOCAB_MAX - 1)
    vocab_tokens = []
    grid_buckets = {}

    for tid in range(VOCAB_MAX):
        token_text = ""
        try:
            # decode single id; tiktoken may raise for some ids but often returns data
            token_text = encoding.decode([tid])
        except Exception:
            token_text = ""
        # create stable key using id and decoded text
        key = f"{tid}:{token_text}"
        x, y = deterministic_position_for_key(key)
        idx = len(vocab_tokens)
        vocab_tokens.append({
            "token_id": int(tid),
            "token_text": token_text,
            "x": float(x),
            "y": float(y)
        })
        ix, iy = bucket_index(x, y)
        add_to_bucket(ix, iy, idx)
        # occasionally log progress
        if tid and (tid % 10000) == 0:
            logger.info("Built positions for %d tokens...", tid)

    vocab_build_time = time.time() - start
    vocab_ready = True
    logger.info("Vocabulary grid built: tokens=%d grid=%dx%d time=%.2fs", len(vocab_tokens), GRID_SIZE, GRID_SIZE, vocab_build_time)

# Build at startup
build_vocab_and_grid()

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/tokenize", methods=["POST"])
def tokenize():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "") or ""
    logger.info("Tokenize request: len(text)=%d", len(text))
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
        tokens_out.append({
            "index": idx,
            "token_id": int(tid),
            "token_text": tok_text,
            "x": x,
            "y": y
        })

    prompt_key = hashlib.sha256(text.encode("utf8")).hexdigest()
    logger.info("Tokenized prompt -> %d tokens (prompt_key=%s)", len(tokens_out), prompt_key)
    return jsonify({"tokens": tokens_out, "count": len(tokens_out), "prompt_key": prompt_key})

def neighbors_in_buckets(ix, iy, radius_cells=1):
    """Return list of token indices in square neighborhood of radius_cells around (ix,iy)."""
    res = []
    for dx in range(-radius_cells, radius_cells+1):
        for dy in range(-radius_cells, radius_cells+1):
            nx = ix + dx; ny = iy + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                key = (nx, ny)
                if key in grid_buckets:
                    res.extend(grid_buckets[key])
    return res

@app.route("/token_at", methods=["POST"])
def token_at():
    """
    POST JSON { x: float, y: float }  (normalized 0..1)
    Returns nearest token from the full vocab: { token_id, token_text, x, y, distance }
    """
    if not vocab_ready:
        return jsonify({"error": "vocab not ready"}), 503

    data = request.get_json(silent=True) or {}
    x = float(data.get("x", 0.5))
    y = float(data.get("y", 0.5))

    ix, iy = bucket_index(x, y)

    # expand radius of grid cells until we find candidates
    radius = 0
    candidates = []
    while radius <= max(GRID_SIZE//2, 10) and len(candidates) == 0:
        candidates = neighbors_in_buckets(ix, iy, radius)
        radius += 1

    if not candidates:
        # fallback: return a random token (actually first)
        tok = vocab_tokens[0]
        return jsonify({"token_id": tok["token_id"], "token_text": tok["token_text"], "x": tok["x"], "y": tok["y"], "distance": 0.0})

    # find nearest among candidates
    best = None
    best_d2 = float("inf")
    for idx in candidates:
        tok = vocab_tokens[idx]
        dx = tok["x"] - x
        dy = tok["y"] - y
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_d2 = d2
            best = tok

    return jsonify({"token_id": best["token_id"], "token_text": best["token_text"], "x": best["x"], "y": best["y"], "distance": best_d2 ** 0.5})

@app.route("/vocab_info")
def vocab_info():
    return jsonify({
        "ready": vocab_ready,
        "vocab_size": len(vocab_tokens),
        "grid_size": GRID_SIZE,
        "build_seconds": vocab_build_time
    })

if __name__ == "__main__":
    app.run(debug=True)

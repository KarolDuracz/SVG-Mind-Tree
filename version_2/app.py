# app.py
import os
import sqlite3
from flask import Flask, g, jsonify, request, render_template, send_from_directory
from datetime import datetime
import random
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "graph.db")

app = Flask(__name__, template_folder="templates", static_folder="static")


# ---------- SQLite helpers ----------
def sqlite_row_factory(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.row_factory = sqlite_row_factory
        db.execute("PRAGMA foreign_keys = ON;")
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db:
        db.close()


def init_db():
    """
    Create tables and add missing columns if necessary.
    Nodes are 2D (x,y). We keep seq_angle and branch_side fields for optional layout propagation.
    """
    db = get_db()
    cur = db.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS nodes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT DEFAULT 'Node',
      x REAL DEFAULT 0.0,
      y REAL DEFAULT 0.0,
      created_at TEXT,
      date TEXT,
      time TEXT,
      description TEXT,
      updated_at TEXT,
      seq_angle REAL,
      branch_side TEXT,    -- 'left'/'right'/NULL
      text TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS connections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source_id INTEGER,
      target_id INTEGER,
      created_at TEXT,
      metadata TEXT,
      FOREIGN KEY(source_id) REFERENCES nodes(id) ON DELETE CASCADE,
      FOREIGN KEY(target_id) REFERENCES nodes(id) ON DELETE CASCADE,
      UNIQUE(source_id, target_id)
    );
    """)
    db.commit()

    # back-compat / ensure columns exist
    cur.execute("PRAGMA table_info(nodes);")
    cols = [r["name"] for r in cur.fetchall()]
    extras = {"seq_angle": "REAL", "branch_side": "TEXT", "text": "TEXT", "updated_at": "TEXT"}
    for c, ctype in extras.items():
        if c not in cols:
            try:
                cur.execute(f"ALTER TABLE nodes ADD COLUMN {c} {ctype};")
            except Exception:
                pass
    db.commit()


def seed_if_empty():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT COUNT(1) AS cnt FROM nodes;")
    if cur.fetchone().get("cnt", 0) == 0:
        now_iso = datetime.utcnow().isoformat()
        now_date = datetime.utcnow().strftime("%Y-%m-%d")
        now_time = datetime.utcnow().strftime("%H:%M:%S")

        # some seeds for demo
        cur.execute("INSERT INTO nodes (name,x,y,created_at,date,time,description,text) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                    ("Root", 0.0, 0.0, now_iso, now_date, now_time, "Root node", "Root long text"))
        cur.execute("INSERT INTO nodes (name,x,y,created_at,date,time,description,text) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                    ("Branch A", 160.0, -20.0, now_iso, now_date, now_time, "A short desc", "Long text A"))
        cur.execute("INSERT INTO nodes (name,x,y,created_at,date,time,description,text) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                    ("Branch B", 160.0, 80.0, now_iso, now_date, now_time, "B short", "Long text B"))
        db.commit()

        # connect Root -> Branch A, Root -> Branch B
        cur.execute("SELECT id FROM nodes WHERE name = ?;", ("Root",))
        root = cur.fetchone()["id"]
        cur.execute("SELECT id FROM nodes WHERE name = ?;", ("Branch A",))
        a = cur.fetchone()["id"]
        cur.execute("SELECT id FROM nodes WHERE name = ?;", ("Branch B",))
        b = cur.fetchone()["id"]

        cur.execute("INSERT OR IGNORE INTO connections (source_id,target_id,created_at,metadata) VALUES (?, ?, ?, ?);",
                    (root, a, now_iso, "seed root->A"))
        cur.execute("INSERT OR IGNORE INTO connections (source_id,target_id,created_at,metadata) VALUES (?, ?, ?, ?);",
                    (root, b, now_iso, "seed root->B"))
        db.commit()


with app.app_context():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    init_db()
    seed_if_empty()


# ---------- Helper utility ----------
def now_iso():
    return datetime.utcnow().isoformat()


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
def admin_panel():
    return render_template("admin.html")


# --- Admin clear ---
@app.route("/api/admin/clear", methods=["POST"])
def api_admin_clear():
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM connections;")
    cur.execute("DELETE FROM nodes;")
    db.commit()
    try:
        cur.execute("VACUUM;")
    except Exception:
        pass
    return jsonify({"cleared": True})


# --- Nodes ---
@app.route("/api/nodes", methods=["GET"])
def api_get_nodes():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, name, x, y, created_at, date, time, description, updated_at, seq_angle, branch_side, text FROM nodes ORDER BY id;")
    return jsonify(cur.fetchall())


@app.route("/api/nodes/<int:node_id>", methods=["GET"])
def api_get_node(node_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, name, x, y, created_at, date, time, description, updated_at, seq_angle, branch_side, text FROM nodes WHERE id = ?;", (node_id,))
    node = cur.fetchone()
    if not node:
        return jsonify({"error": "not found"}), 404
    # include connections (incoming and outgoing)
    cur.execute("SELECT id, source_id AS source, target_id AS target, metadata, created_at FROM connections WHERE source_id = ? OR target_id = ?;", (node_id, node_id))
    node["connections"] = cur.fetchall()
    return jsonify(node)


@app.route("/api/nodes", methods=["POST"])
def api_create_node():
    """
    Create a node. Accepts:
      - name, x, y, date, time, description, text
      - connect_to (existing node id)
      - sequence (bool) -> place as 'sequence' child (to the right of parent)
      - branch (bool)   -> place as branch child (above/below)
      - branch_side ('left'|'right') influences side for branch placement (optional)
      - distance, offset (optional placement tuning)
    """
    data = request.get_json(force=True) or {}
    name = data.get("name", "Node")
    x = data.get("x", None)
    y = data.get("y", None)
    date_field = data.get("date", None)
    time_field = data.get("time", None)
    description = data.get("description", None)
    text = data.get("text", None)
    connect_to = data.get("connect_to", None)
    is_sequence = bool(data.get("sequence", False))
    is_branch = bool(data.get("branch", False))
    branch_side = data.get("branch_side", None)  # 'left' or 'right' (affects layout)

    db = get_db()
    cur = db.cursor()

    # compute placement if connecting and coords missing
    if connect_to is not None and (x is None or y is None):
        cur.execute("SELECT id,x,y,seq_angle,branch_side FROM nodes WHERE id = ?;", (connect_to,))
        parent = cur.fetchone()
        if parent:
            px = float(parent.get("x") or 0.0)
            py = float(parent.get("y") or 0.0)

            # default distances in px
            seq_distance = float(data.get("distance", 140.0))
            branch_distance = float(data.get("distance", 100.0))

            # For sequence: place to the right (x increasing) using parent's seq_angle if present
            if is_sequence:
                angle = parent.get("seq_angle")
                if angle is None:
                    # default horizontal to the right
                    angle = 0.0
                # angle stored in degrees, 0 degrees -> +x
                rad = math.radians(float(angle))
                dx = math.cos(rad) * seq_distance
                dy = math.sin(rad) * seq_distance
                x = px + dx
                y = py + dy
            elif is_branch:
                # branch: default above/below depending on branch_side
                side = branch_side or parent.get("branch_side") or random.choice(["above", "below"])
                # vertical offset and slight horizontal offset
                vertical = branch_distance if side in ("below", "right") else -branch_distance
                horizontal = float(data.get("offset_x", 40.0)) * (1 if side in ("below", "right") else -1)
                # allow explicit small random jitter to separate siblings
                jitter_x = (random.random() - 0.5) * 30.0
                jitter_y = (random.random() - 0.5) * 30.0
                x = px + horizontal + jitter_x
                y = py + vertical + jitter_y
            else:
                # fallback: small random offset
                x = px + (random.random() - 0.5) * 80.0
                y = py + (random.random() - 0.5) * 80.0

    # if still missing, assign random placement
    if x is None or y is None:
        x = (random.random() - 0.5) * 800
        y = (random.random() - 0.5) * 600

    # defaults for date/time
    if not date_field:
        date_field = datetime.utcnow().strftime("%Y-%m-%d")
    if not time_field:
        time_field = datetime.utcnow().strftime("%H:%M:%S")

    created = now_iso()
    cur.execute("INSERT INTO nodes (name,x,y,created_at,date,time,description,seq_angle,branch_side,text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (name, float(x), float(y), created, date_field, time_field, description, data.get("seq_angle"), branch_side, text))
    db.commit()
    nid = cur.lastrowid

    # create connection if requested
    if connect_to is not None:
        try:
            cur.execute("INSERT INTO connections (source_id,target_id,created_at,metadata) VALUES (?, ?, ?, ?);",
                        (int(connect_to), int(nid), now_iso(), data.get("connection_metadata")))
            db.commit()
        except sqlite3.IntegrityError:
            # duplicate connection — ignore
            pass

    cur.execute("SELECT id, name, x, y, created_at, date, time, description, updated_at, seq_angle, branch_side, text FROM nodes WHERE id = ?;", (nid,))
    return jsonify(cur.fetchone()), 201


@app.route("/api/nodes/<int:node_id>", methods=["PUT"])
def api_update_node(node_id):
    data = request.get_json(force=True) or {}
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id FROM nodes WHERE id = ?;", (node_id,))
    if not cur.fetchone():
        return jsonify({"error": "not found"}), 404

    fields = []
    values = []
    allowed = ("name", "x", "y", "date", "time", "description", "text", "seq_angle", "branch_side")
    for k in allowed:
        if k in data:
            fields.append(f"{k} = ?")
            val = data[k]
            if k in ("x", "y", "seq_angle") and val is not None:
                val = float(val)
            values.append(val)
    updated = now_iso()
    fields.append("updated_at = ?")
    values.append(updated)

    if fields:
        sql = "UPDATE nodes SET " + ", ".join(fields) + " WHERE id = ?;"
        values.append(node_id)
        cur.execute(sql, tuple(values))
        db.commit()

    cur.execute("SELECT id, name, x, y, created_at, date, time, description, updated_at, seq_angle, branch_side, text FROM nodes WHERE id = ?;", (node_id,))
    return jsonify(cur.fetchone())


@app.route("/api/nodes/<int:node_id>", methods=["DELETE"])
def api_delete_node(node_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id FROM nodes WHERE id = ?;", (node_id,))
    if not cur.fetchone():
        return jsonify({"error": "not found"}), 404
    cur.execute("DELETE FROM nodes WHERE id = ?;", (node_id,))
    db.commit()
    return jsonify({"deleted": node_id})


# --- Connections ---
@app.route("/api/connections", methods=["GET"])
def api_get_connections():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, source_id AS source, target_id AS target, metadata, created_at FROM connections;")
    return jsonify(cur.fetchall())


@app.route("/api/connections", methods=["POST"])
def api_create_connection():
    data = request.get_json(force=True) or {}
    source = data.get("source")
    target = data.get("target")
    metadata = data.get("metadata", None)
    if source is None or target is None:
        return jsonify({"error": "source and target are required"}), 400
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT 1 FROM nodes WHERE id = ?;", (source,))
    if not cur.fetchone():
        return jsonify({"error": "source not found"}), 404
    cur.execute("SELECT 1 FROM nodes WHERE id = ?;", (target,))
    if not cur.fetchone():
        return jsonify({"error": "target not found"}), 404
    try:
        cur.execute("INSERT INTO connections (source_id,target_id,created_at,metadata) VALUES (?, ?, ?, ?);",
                    (int(source), int(target), now_iso(), metadata))
        db.commit()
        cur.execute("SELECT id, source_id AS source, target_id AS target, metadata, created_at FROM connections WHERE id = ?;", (cur.lastrowid,))
        return jsonify(cur.fetchone()), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "duplicate connection"}), 409


@app.route("/api/connections/<int:conn_id>", methods=["DELETE"])
def api_delete_connection(conn_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id FROM connections WHERE id = ?;", (conn_id,))
    if not cur.fetchone():
        return jsonify({"error": "not found"}), 404
    cur.execute("DELETE FROM connections WHERE id = ?;", (conn_id,))
    db.commit()
    return jsonify({"deleted": conn_id})


# --- Random generator (lightweight) ---
@app.route("/api/generate_random", methods=["POST"])
def api_generate_random():
    data = request.get_json(force=True) or {}
    count = int(data.get("count", 0))
    if count <= 0:
        return jsonify({"error": "invalid count"}), 400

    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id FROM nodes;")
    existing = [r["id"] for r in cur.fetchall()]
    now = now_iso()
    added = []
    for i in range(count):
        name = f"Rnd-{random.randint(1000,9999)}"
        x = (random.random() - 0.5) * 1000
        y = (random.random() - 0.5) * 700
        cur.execute("INSERT INTO nodes (name,x,y,created_at,date,time) VALUES (?, ?, ?, ?, ?, ?);",
                    (name, x, y, now, datetime.utcnow().strftime("%Y-%m-%d"), datetime.utcnow().strftime("%H:%M:%S")))
        db.commit()
        added.append({"id": cur.lastrowid, "name": name, "x": x, "y": y})

    # try to add a few random connections
    cur.execute("SELECT id FROM nodes;")
    all_ids = [r["id"] for r in cur.fetchall()]
    for nid in [a["id"] for a in added]:
        attempts = 0
        while attempts < 8:
            s = random.choice(all_ids)
            t = random.choice(all_ids)
            if s == t:
                attempts += 1
                continue
            try:
                cur.execute("INSERT INTO connections (source_id,target_id,created_at,metadata) VALUES (?, ?, ?, ?);",
                            (s, t, now, f"auto {s}->{t}"))
                db.commit()
                break
            except sqlite3.IntegrityError:
                attempts += 1
                continue
    return jsonify({"added_count": len(added), "added": added}), 201


if __name__ == "__main__":
    app.run(debug=True, port=5000)

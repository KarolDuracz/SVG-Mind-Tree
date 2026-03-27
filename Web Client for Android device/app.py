from __future__ import annotations

import json
import math
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from collections import Counter, deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Deque, Dict, List, Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

app = Flask(__name__)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TILE_ROWS = 3
TILE_COLS = 3
TILE_COUNT = TILE_ROWS * TILE_COLS
CHART_WINDOW = 200
EVENT_WINDOW = 180
PATTERN_WINDOW = 60
APP_ROOT = Path(__file__).resolve().parent
CHART_DATA_DIR = APP_ROOT / "chart_data"
CHART_MANIFEST_PATH = CHART_DATA_DIR / "manifest.json"
CHART_CHUNK_PREFIX = "chart_chunk_"
CHART_CHUNK_MAX_EVENTS = 200  # tune this if you want 200-400 point chunks later
CHART_RETAIN_CHUNKS = 4
# Client-facing grid tuning constants. Adjust these if the board feels too large/small.
CLIENT_BOARD_SIDE_PX = 320
CLIENT_BOARD_SIDE_RATIO = 0.78
CLIENT_BOARD_MIN_SIDE_PX = 240
CLIENT_BOARD_MAX_SIDE_PX = 420
CLIENT_BOARD_MARGIN_PX = 18


def _now() -> float:
    return time.time()


def _fnv1a_32(text: str) -> str:
    h = 0x811C9DC5
    for b in text.encode("utf-8", "replace"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def _tile_index(row: int, col: int) -> int:
    return row * TILE_COLS + col


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _angle_deg(dx: float, dy: float) -> float:
    # Screen coordinates: x grows right, y grows down.
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360.0
    return angle


def _sequence_text(values: List[int]) -> str:
    return "".join(str(v) for v in values)


def _board_hash(board: Dict[str, object]) -> str:
    return _fnv1a_32(json.dumps(board, sort_keys=True, separators=(",", ":")))


@dataclass
class FrameStats:
    started_at: float = field(default_factory=_now)
    lock: threading.Lock = field(default_factory=threading.Lock)

    latest_png: bytes = b""
    latest_id: int = 0
    latest_size: int = 0
    latest_at: float = 0.0

    received_frames: int = 0
    total_bytes: int = 0
    last_frame_at: Optional[float] = None
    interval_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    size_samples: Deque[int] = field(default_factory=lambda: deque(maxlen=300))
    processing_samples_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    duplicate_png_skips: int = 0

    last_error: str = ""
    last_hash: str = ""
    last_frame_hash: str = ""
    last_client_elapsed_ms: Optional[float] = None
    last_frame_latency_ms: Optional[float] = None
    last_frame_event_type: str = ""

    layout_count: int = 0
    board_x: float = 0.0
    board_y: float = 0.0
    board_size: int = 360
    board_rows: int = TILE_ROWS
    board_cols: int = TILE_COLS
    board_tile_w: float = 120.0
    board_tile_h: float = 120.0
    board_margin: float = 0.0
    last_layout_hash: str = ""
    last_layout_at: Optional[float] = None

    tile_event_count: int = 0
    press_count: int = 0
    hold_count: int = 0
    move_count: int = 0
    release_count: int = 0
    active_tile_index: Optional[int] = None
    active_tile_row: Optional[int] = None
    active_tile_col: Optional[int] = None
    active_tile_state: str = "idle"
    last_tile_index: Optional[int] = None
    last_tile_row: Optional[int] = None
    last_tile_col: Optional[int] = None
    last_tile_state: str = ""
    last_tile_hold_ms: Optional[float] = None
    last_tile_down_at: Optional[float] = None
    last_tile_up_at: Optional[float] = None
    first_event_after_press_ms: Optional[float] = None
    press_started_at: Optional[float] = None
    last_press_hash: str = ""
    last_press_client_elapsed_ms: Optional[float] = None
    last_release_client_elapsed_ms: Optional[float] = None

    current_sequence: List[int] = field(default_factory=list)
    current_sequence_started_at: Optional[float] = None
    current_sequence_updated_at: Optional[float] = None
    last_completed_sequence: str = ""
    last_completed_sequence_len: int = 0
    last_completed_sequence_hold_ms: Optional[float] = None
    recent_sequences: Deque[str] = field(default_factory=lambda: deque(maxlen=PATTERN_WINDOW))
    pattern_counts: Counter = field(default_factory=Counter)
    sequence_length_counts: Counter = field(default_factory=Counter)

    tile_press_counts: List[int] = field(default_factory=lambda: [0] * TILE_COUNT)
    tile_hold_counts: List[int] = field(default_factory=lambda: [0] * TILE_COUNT)
    tile_move_counts: List[int] = field(default_factory=lambda: [0] * TILE_COUNT)
    tile_release_counts: List[int] = field(default_factory=lambda: [0] * TILE_COUNT)

    def _set_board_locked(self, board: Dict[str, object]) -> None:
        self.board_x = float(board.get("x", self.board_x) or self.board_x)
        self.board_y = float(board.get("y", self.board_y) or self.board_y)
        self.board_size = int(board.get("size", self.board_size) or self.board_size)
        self.board_rows = int(board.get("rows", TILE_ROWS) or TILE_ROWS)
        self.board_cols = int(board.get("cols", TILE_COLS) or TILE_COLS)
        self.board_tile_w = float(board.get("tile_w", self.board_tile_w) or self.board_tile_w)
        self.board_tile_h = float(board.get("tile_h", self.board_tile_h) or self.board_tile_h)
        self.board_margin = float(board.get("margin", self.board_margin) or self.board_margin)
        self.last_layout_hash = str(board.get("hash", self.last_layout_hash))
        self.last_layout_at = _now()

    def _update_sequence_locked(self, state: str, index: Optional[int], hold_ms: Optional[float]) -> None:
        if index is None:
            return

        tile_num = int(index) + 1

        if state == "press":
            self.current_sequence = [tile_num]
            self.current_sequence_started_at = _now()
            self.current_sequence_updated_at = self.current_sequence_started_at
            self.first_event_after_press_ms = None
            self.press_started_at = self.current_sequence_started_at
            return

        if state in {"hold", "move"}:
            if not self.current_sequence:
                self.current_sequence = [tile_num]
                self.current_sequence_started_at = _now()
                self.press_started_at = self.current_sequence_started_at
            elif self.current_sequence[-1] != tile_num:
                self.current_sequence.append(tile_num)
            self.current_sequence_updated_at = _now()
            if hold_ms is not None and self.first_event_after_press_ms is None:
                self.first_event_after_press_ms = float(hold_ms)
            return

        if state == "release":
            if self.current_sequence:
                seq = _sequence_text(self.current_sequence)
                self.last_completed_sequence = seq
                self.last_completed_sequence_len = len(self.current_sequence)
                self.recent_sequences.appendleft(seq)
                self.pattern_counts[seq] += 1
                self.sequence_length_counts[len(self.current_sequence)] += 1
                self.last_completed_sequence_hold_ms = (
                    round((_now() - self.press_started_at) * 1000.0, 2)
                    if self.press_started_at is not None
                    else None
                )
            self.current_sequence = []
            self.current_sequence_started_at = None
            self.current_sequence_updated_at = None
            self.press_started_at = None

    def register_layout(self, board: Dict[str, object]) -> Dict[str, object]:
        with self.lock:
            self.layout_count += 1
            self._set_board_locked(board)
            snapshot = self._snapshot_locked()
            snapshot["layout"] = board
            snapshot["event_type"] = "layout"
            return snapshot

    def register_tile_event(self, payload: Dict[str, object]) -> Dict[str, object]:
        now = _now()
        with self.lock:
            self.tile_event_count += 1
            event_type = str(payload.get("state", ""))[:24]
            index = payload.get("index")
            row = payload.get("row")
            col = payload.get("col")
            board = payload.get("board")
            client_elapsed_ms = payload.get("elapsed_ms")
            hold_ms = payload.get("hold_ms")
            press_hash = str(payload.get("hash", ""))[:64]
            pointer_x = payload.get("pointer_x")
            pointer_y = payload.get("pointer_y")
            delta_x = payload.get("delta_x")
            delta_y = payload.get("delta_y")

            if isinstance(board, dict):
                self._set_board_locked(board)

            self.last_hash = press_hash
            self.last_tile_state = event_type
            self.last_tile_index = int(index) if index is not None else None
            self.last_tile_row = int(row) if row is not None else None
            self.last_tile_col = int(col) if col is not None else None
            self.last_tile_hold_ms = float(hold_ms) if hold_ms is not None else None
            self.last_frame_event_type = event_type
            if isinstance(client_elapsed_ms, (int, float)):
                self.last_client_elapsed_ms = float(client_elapsed_ms)

            if isinstance(index, int) and 0 <= index < TILE_COUNT:
                self.active_tile_index = index
                self.active_tile_row = int(row) if row is not None else None
                self.active_tile_col = int(col) if col is not None else None

                if event_type == "press":
                    self.press_count += 1
                    self.active_tile_state = "pressed"
                    self.tile_press_counts[index] += 1
                    self.press_started_at = now
                    self.first_event_after_press_ms = None
                    self.last_tile_down_at = now
                    self.last_press_hash = press_hash
                    self.last_press_client_elapsed_ms = float(client_elapsed_ms) if isinstance(client_elapsed_ms, (int, float)) else None
                    self.current_sequence = [index + 1]
                    self.current_sequence_started_at = now
                    self.current_sequence_updated_at = now

                elif event_type == "hold":
                    self.hold_count += 1
                    self.active_tile_state = "held"
                    self.tile_hold_counts[index] += 1
                    self._update_sequence_locked("hold", index, float(client_elapsed_ms) if isinstance(client_elapsed_ms, (int, float)) else None)
                    if self.press_started_at is not None and self.first_event_after_press_ms is None:
                        self.first_event_after_press_ms = round((now - self.press_started_at) * 1000.0, 2)

                elif event_type == "move":
                    self.move_count += 1
                    self.active_tile_state = "moving"
                    self.tile_move_counts[index] += 1
                    self._update_sequence_locked("move", index, float(client_elapsed_ms) if isinstance(client_elapsed_ms, (int, float)) else None)
                    if self.press_started_at is not None and self.first_event_after_press_ms is None:
                        self.first_event_after_press_ms = round((now - self.press_started_at) * 1000.0, 2)

                elif event_type == "release":
                    self.release_count += 1
                    self.active_tile_state = "released"
                    self.tile_release_counts[index] += 1
                    self.last_tile_up_at = now
                    if self.press_started_at is not None:
                        self.last_tile_hold_ms = round((now - self.press_started_at) * 1000.0, 2)
                    self._update_sequence_locked("release", index, float(client_elapsed_ms) if isinstance(client_elapsed_ms, (int, float)) else None)
                    self.last_release_client_elapsed_ms = float(client_elapsed_ms) if isinstance(client_elapsed_ms, (int, float)) else None

                else:
                    self.active_tile_state = event_type

            snapshot = self._snapshot_locked()
            snapshot["event_type"] = event_type
            snapshot["event"] = payload
            snapshot["pointer_x"] = pointer_x
            snapshot["pointer_y"] = pointer_y
            snapshot["delta_x"] = delta_x
            snapshot["delta_y"] = delta_y
            return snapshot

    def register_frame(
        self,
        png_bytes: bytes,
        processing_ms: float,
        frame_hash: str = "",
        client_elapsed_ms: Optional[float] = None,
        event_type: str = "",
    ) -> Dict[str, object]:
        now = _now()
        with self.lock:
            if self.last_frame_at is not None:
                self.interval_samples.append((now - self.last_frame_at) * 1000.0)
            self.last_frame_at = now

            self.received_frames += 1
            self.total_bytes += len(png_bytes)
            self.latest_png = png_bytes
            self.latest_id += 1
            self.latest_size = len(png_bytes)
            self.latest_at = now
            self.size_samples.append(len(png_bytes))
            self.processing_samples_ms.append(processing_ms)
            self.last_frame_hash = frame_hash
            self.last_frame_event_type = event_type
            if isinstance(client_elapsed_ms, (int, float)):
                self.last_client_elapsed_ms = float(client_elapsed_ms)

            snapshot = self._snapshot_locked()
            snapshot["latest_id"] = self.latest_id
            snapshot["latest_size"] = self.latest_size
            snapshot["last_frame_hash"] = self.last_frame_hash
            snapshot["event_type"] = "frame"
            return snapshot

    def skip_duplicate(self) -> None:
        with self.lock:
            self.duplicate_png_skips += 1

    def _snapshot_locked(self) -> Dict[str, object]:
        avg_interval = mean(self.interval_samples) if self.interval_samples else None
        avg_size = mean(self.size_samples) if self.size_samples else None
        avg_proc = mean(self.processing_samples_ms) if self.processing_samples_ms else None
        fps = 1000.0 / avg_interval if avg_interval and avg_interval > 0 else None
        elapsed = max(_now() - self.started_at, 0.001)
        throughput_kb_s = (self.total_bytes / 1024.0) / elapsed
        return {
            "uptime_s": round(elapsed, 2),
            "started_at_ms": round(self.started_at * 1000.0, 1),
            "received_frames": self.received_frames,
            "total_bytes": self.total_bytes,
            "throughput_kb_s": round(throughput_kb_s, 2),
            "latest_id": self.latest_id,
            "latest_size": self.latest_size,
            "latest_age_ms": round((_now() - self.latest_at) * 1000.0, 1) if self.latest_at else None,
            "avg_interval_ms": round(avg_interval, 2) if avg_interval is not None else None,
            "approx_fps": round(fps, 2) if fps is not None else None,
            "avg_png_size_bytes": round(avg_size, 2) if avg_size is not None else None,
            "avg_processing_ms": round(avg_proc, 2) if avg_proc is not None else None,
            "duplicate_png_skips": self.duplicate_png_skips,
            "last_error": self.last_error,
            "last_hash": self.last_hash,
            "last_frame_hash": self.last_frame_hash,
            "last_client_elapsed_ms": self.last_client_elapsed_ms,
            "last_frame_latency_ms": self.last_frame_latency_ms,
            "layout_count": self.layout_count,
            "board_x": round(self.board_x, 2),
            "board_y": round(self.board_y, 2),
            "board_size": self.board_size,
            "board_rows": self.board_rows,
            "board_cols": self.board_cols,
            "board_tile_w": round(self.board_tile_w, 2),
            "board_tile_h": round(self.board_tile_h, 2),
            "board_margin": round(self.board_margin, 2),
            "last_layout_hash": self.last_layout_hash,
            "last_layout_at": self.last_layout_at,
            "tile_event_count": self.tile_event_count,
            "press_count": self.press_count,
            "hold_count": self.hold_count,
            "move_count": self.move_count,
            "release_count": self.release_count,
            "active_tile_index": self.active_tile_index,
            "active_tile_row": self.active_tile_row,
            "active_tile_col": self.active_tile_col,
            "active_tile_state": self.active_tile_state,
            "last_tile_index": self.last_tile_index,
            "last_tile_row": self.last_tile_row,
            "last_tile_col": self.last_tile_col,
            "last_tile_state": self.last_tile_state,
            "last_tile_hold_ms": self.last_tile_hold_ms,
            "last_tile_down_at": self.last_tile_down_at,
            "last_tile_up_at": self.last_tile_up_at,
            "first_event_after_press_ms": self.first_event_after_press_ms,
            "press_started_at": self.press_started_at,
            "last_press_hash": self.last_press_hash,
            "last_press_client_elapsed_ms": self.last_press_client_elapsed_ms,
            "last_release_client_elapsed_ms": self.last_release_client_elapsed_ms,
            "current_sequence": list(self.current_sequence),
            "current_sequence_text": _sequence_text(self.current_sequence),
            "current_sequence_started_at": self.current_sequence_started_at,
            "current_sequence_updated_at": self.current_sequence_updated_at,
            "last_completed_sequence": self.last_completed_sequence,
            "last_completed_sequence_len": self.last_completed_sequence_len,
            "last_completed_sequence_hold_ms": self.last_completed_sequence_hold_ms,
            "recent_sequences": list(self.recent_sequences),
            "pattern_counts_top": self.pattern_counts.most_common(12),
            "sequence_length_counts": dict(self.sequence_length_counts),
            "tile_press_counts": list(self.tile_press_counts),
            "tile_hold_counts": list(self.tile_hold_counts),
            "tile_move_counts": list(self.tile_move_counts),
            "tile_release_counts": list(self.tile_release_counts),
        }

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            return self._snapshot_locked()


@dataclass
class ChartCache:
    lock: threading.Lock = field(default_factory=threading.Lock)
    version: int = 0
    updated_at: Optional[float] = None

    tile_series: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=EVENT_WINDOW))
    motion_series: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=EVENT_WINDOW))
    release_series: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=EVENT_WINDOW))

    def update(self, evt: Dict[str, object]) -> Dict[str, object]:
        with self.lock:
            state = str(evt.get("state", ""))
            idx = evt.get("index")
            t = float(evt.get("ts", _now()))
            seq = evt.get("sequence_text", "")
            response_ms = evt.get("response_ms")
            hold_ms = evt.get("hold_ms")
            dx = evt.get("delta_x")
            dy = evt.get("delta_y")
            angle = evt.get("direction_deg")
            speed = evt.get("speed")
            release_angle = evt.get("release_direction_deg")
            release_speed = evt.get("release_speed")
            board = evt.get("board") or {}

            if isinstance(idx, int):
                tile_point = {
                    "t": t,
                    "tile": idx + 1,
                    "state": state,
                    "response_ms": response_ms,
                    "hold_ms": hold_ms,
                    "sequence_len": len(seq) if isinstance(seq, str) else None,
                }
                self.tile_series.append(tile_point)

            if any(v is not None for v in (dx, dy, angle, speed)):
                motion_point = {
                    "t": t,
                    "state": state,
                    "dx": dx,
                    "dy": dy,
                    "angle": angle,
                    "speed": speed,
                    "tile": (idx + 1) if isinstance(idx, int) else None,
                    "board_x": board.get("x"),
                    "board_y": board.get("y"),
                }
                self.motion_series.append(motion_point)

            if state == "release" or release_angle is not None or release_speed is not None:
                release_point = {
                    "t": t,
                    "state": state,
                    "tile": (idx + 1) if isinstance(idx, int) else None,
                    "release_direction_deg": release_angle,
                    "release_speed": release_speed,
                    "sequence_len": len(seq) if isinstance(seq, str) else None,
                    "sequence_text": seq,
                }
                self.release_series.append(release_point)

            self.version += 1
            self.updated_at = t
            return self.snapshot()

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            return {
                "version": self.version,
                "updated_at": self.updated_at,
                "tile_series": list(self.tile_series),
                "motion_series": list(self.motion_series),
                "release_series": list(self.release_series),
                "window": EVENT_WINDOW,
            }


STATS = FrameStats()
CHARTS = ChartCache()
SUBSCRIBERS_LOCK = threading.Lock()
SUBSCRIBERS: List[queue.Queue[str]] = []
RAW_EVENT_QUEUE: "queue.Queue[Dict[str, object]]" = queue.Queue(maxsize=2048)
FILE_EVENT_QUEUE: "queue.Queue[Dict[str, object]]" = queue.Queue(maxsize=4096)
WORKER_STARTED = False
FILE_WORKER_STARTED = False
WORKER_LOCK = threading.Lock()
FILE_WORKER_LOCK = threading.Lock()


def publish(event: Dict[str, object]) -> None:
    payload = f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
    stale: List[queue.Queue[str]] = []
    with SUBSCRIBERS_LOCK:
        for q in SUBSCRIBERS:
            try:
                q.put_nowait(payload)
            except queue.Full:
                stale.append(q)
        for q in stale:
            try:
                SUBSCRIBERS.remove(q)
            except ValueError:
                pass


def _raw_event_worker() -> None:
    while True:
        try:
            evt = RAW_EVENT_QUEUE.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            charts_snapshot = CHARTS.update(evt)
            publish({"type": "charts", "charts": charts_snapshot, "stats": STATS.snapshot(), "ts": _now()})
        finally:
            RAW_EVENT_QUEUE.task_done()





def _atomic_write_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(obj, fp, separators=(",", ":"), ensure_ascii=False)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def _read_json_file(path: Path) -> Optional[Dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


@dataclass
class ChartFileStore:
    lock: threading.Lock = field(default_factory=threading.Lock)
    version: int = 0
    updated_at: Optional[float] = None
    chunk_index: int = 1
    current_events: List[Dict[str, object]] = field(default_factory=list)
    chunk_files: List[str] = field(default_factory=list)
    last_snapshot: Dict[str, object] = field(default_factory=dict)

    def _chunk_name(self, idx: int) -> str:
        return f"{CHART_CHUNK_PREFIX}{idx:06d}.json"

    def _chunk_path(self, idx: int) -> Path:
        return CHART_DATA_DIR / self._chunk_name(idx)

    def _serialize_event(self, evt: Dict[str, object]) -> Dict[str, object]:
        return {
            "ts": float(evt.get("ts", _now())),
            "state": str(evt.get("state", "")),
            "index": evt.get("index"),
            "row": evt.get("row"),
            "col": evt.get("col"),
            "elapsed_ms": evt.get("elapsed_ms"),
            "hold_ms": evt.get("hold_ms"),
            "response_ms": evt.get("response_ms"),
            "pointer_id": evt.get("pointer_id"),
            "client_ts": evt.get("client_ts"),
            "pointer_x": evt.get("pointer_x"),
            "pointer_y": evt.get("pointer_y"),
            "delta_x": evt.get("delta_x"),
            "delta_y": evt.get("delta_y"),
            "total_dx": evt.get("total_dx"),
            "total_dy": evt.get("total_dy"),
            "direction_deg": evt.get("direction_deg"),
            "speed": evt.get("speed"),
            "release_direction_deg": evt.get("release_direction_deg"),
            "release_speed": evt.get("release_speed"),
            "sequence_text": evt.get("sequence_text", ""),
            "frame_hash": evt.get("frame_hash", ""),
            "event_hash": evt.get("hash", ""),
            "board_hash": evt.get("board_hash", evt.get("hash", "")),
            "board": evt.get("board"),
        }

    def _flush_chunk_locked(self) -> None:
        if not self.current_events:
            return
        path = self._chunk_path(self.chunk_index)
        payload = {
            "chunk_index": self.chunk_index,
            "created_at": self.updated_at or _now(),
            "updated_at": self.updated_at or _now(),
            "event_count": len(self.current_events),
            "events": self.current_events,
        }
        _atomic_write_json(path, payload)
        name = path.name
        if name not in self.chunk_files:
            self.chunk_files.append(name)
        # Keep only a small rolling set of chunk files on disk.
        while len(self.chunk_files) > CHART_RETAIN_CHUNKS:
            old_name = self.chunk_files.pop(0)
            try:
                (CHART_DATA_DIR / old_name).unlink(missing_ok=True)
            except Exception:
                pass

    def _write_manifest_locked(self) -> None:
        manifest = {
            "version": self.version,
            "updated_at": self.updated_at,
            "chunk_index": self.chunk_index,
            "chunk_files": list(self.chunk_files),
            "chunk_max_events": CHART_CHUNK_MAX_EVENTS,
            "window": CHART_WINDOW,
            "retained_chunks": CHART_RETAIN_CHUNKS,
        }
        _atomic_write_json(CHART_MANIFEST_PATH, manifest)

    def _load_manifest_locked(self) -> None:
        manifest = _read_json_file(CHART_MANIFEST_PATH) or {}
        if not manifest:
            return
        files = manifest.get("chunk_files", [])
        if isinstance(files, list):
            self.chunk_files = [str(x) for x in files if isinstance(x, str)]
        try:
            self.chunk_index = int(manifest.get("chunk_index", self.chunk_index) or self.chunk_index)
        except (TypeError, ValueError):
            pass
        try:
            self.version = int(manifest.get("version", self.version) or self.version)
        except (TypeError, ValueError):
            pass
        self.updated_at = manifest.get("updated_at", self.updated_at)

    def ingest(self, evt: Dict[str, object]) -> Dict[str, object]:
        with self.lock:
            if self.current_events and len(self.current_events) >= CHART_CHUNK_MAX_EVENTS:
                self._flush_chunk_locked()
                self.chunk_index += 1
                self.current_events = []

            self.current_events.append(self._serialize_event(evt))
            self.version += 1
            self.updated_at = float(evt.get("ts", _now()))
            self._flush_chunk_locked()
            self._write_manifest_locked()
            snapshot = self._build_snapshot_locked(CHART_WINDOW)
            self.last_snapshot = snapshot
            return snapshot

    def _build_snapshot_locked(self, limit: int) -> Dict[str, object]:
        # Build from the current on-disk rolling files. This keeps /charts polling aligned
        # with the latest file state while remaining small and quick to load.
        events: List[Dict[str, object]] = []
        for name in self.chunk_files:
            path = CHART_DATA_DIR / name
            data = _read_json_file(path)
            if not data:
                continue
            chunk_events = data.get("events", [])
            if isinstance(chunk_events, list):
                events.extend(chunk_events)
        if self.current_events:
            events.extend(self.current_events)
        if limit and len(events) > limit:
            events = events[-limit:]

        tile_series: List[Dict[str, object]] = []
        motion_series: List[Dict[str, object]] = []
        release_series: List[Dict[str, object]] = []

        for evt in events:
            t = float(evt.get("ts") or _now())
            state = str(evt.get("state") or "")
            idx = evt.get("index")
            seq = str(evt.get("sequence_text") or "")
            if isinstance(idx, int):
                tile_series.append({
                    "t": t,
                    "tile": idx + 1,
                    "state": state,
                    "response_ms": evt.get("response_ms"),
                    "hold_ms": evt.get("hold_ms"),
                    "sequence_len": len(seq),
                })
            if evt.get("direction_deg") is not None or evt.get("speed") is not None:
                motion_series.append({
                    "t": t,
                    "state": state,
                    "angle": evt.get("direction_deg"),
                    "speed": evt.get("speed"),
                    "tile": (idx + 1) if isinstance(idx, int) else None,
                })
            if evt.get("release_direction_deg") is not None or evt.get("release_speed") is not None or state == "release":
                release_series.append({
                    "t": t,
                    "state": state,
                    "tile": (idx + 1) if isinstance(idx, int) else None,
                    "release_direction_deg": evt.get("release_direction_deg"),
                    "release_speed": evt.get("release_speed"),
                    "sequence_len": len(seq),
                    "sequence_text": seq,
                })

        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "tile_series": tile_series,
            "motion_series": motion_series,
            "release_series": release_series,
            "window": limit,
            "chunk_files": list(self.chunk_files),
            "chunk_max_events": CHART_CHUNK_MAX_EVENTS,
        }

    def snapshot(self, limit: int = CHART_WINDOW) -> Dict[str, object]:
        with self.lock:
            if not self.chunk_files and CHART_MANIFEST_PATH.exists():
                self._load_manifest_locked()
            snapshot = self._build_snapshot_locked(limit)
            self.last_snapshot = snapshot
            return snapshot


FILE_CHARTS = ChartFileStore()


def _file_event_worker() -> None:
    CHART_DATA_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            evt = FILE_EVENT_QUEUE.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            FILE_CHARTS.ingest(evt)
        except Exception as exc:  # pragma: no cover - background best effort
            STATS.last_error = f"chart file store: {exc}"
        finally:
            FILE_EVENT_QUEUE.task_done()


def _load_charts_from_files(limit: int = CHART_WINDOW) -> Dict[str, object]:
    try:
        return FILE_CHARTS.snapshot(limit=limit)
    except Exception:
        return {
            "version": 0,
            "updated_at": None,
            "tile_series": [],
            "motion_series": [],
            "release_series": [],
            "window": limit,
            "chunk_files": [],
            "chunk_max_events": CHART_CHUNK_MAX_EVENTS,
        }

def _start_background_threads() -> None:
    global WORKER_STARTED, FILE_WORKER_STARTED
    with WORKER_LOCK:
        if not WORKER_STARTED:
            t = threading.Thread(target=_raw_event_worker, daemon=True, name="raw-event-worker")
            t.start()
            WORKER_STARTED = True
    with FILE_WORKER_LOCK:
        if not FILE_WORKER_STARTED:
            t2 = threading.Thread(target=_file_event_worker, daemon=True, name="chart-file-worker")
            t2.start()
            FILE_WORKER_STARTED = True


_start_background_threads()

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/admin")
def admin():
    return render_template("admin.html")


@app.get("/charts")
def charts():
    return render_template("charts.html")


@app.get("/patterns")
def patterns():
    return render_template("patterns.html")


@app.post("/api/layout")
def api_layout():
    data = request.get_json(silent=True) or {}
    board = data.get("board") if isinstance(data, dict) else None
    if not isinstance(board, dict):
        return jsonify({"ok": False, "error": "missing board"}), 400

    board = dict(board)
    board["hash"] = _board_hash(board)
    snapshot = STATS.register_layout(board)
    publish({"type": "layout", "stats": snapshot, "board": board, "ts": _now()})
    return jsonify({"ok": True, "hash": board["hash"], "board": board})


@app.post("/api/tile")
def api_tile():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "bad payload"}), 400

    state = str(data.get("state", ""))[:24]
    index = data.get("index")
    row = data.get("row")
    col = data.get("col")
    board = data.get("board")
    elapsed_ms = data.get("elapsed_ms")
    hold_ms = data.get("hold_ms")
    press_hash = str(data.get("hash", ""))[:64]
    pointer_id = data.get("pointer_id")
    client_ts = data.get("client_ts")
    pointer_x = data.get("pointer_x")
    pointer_y = data.get("pointer_y")
    delta_x = data.get("delta_x")
    delta_y = data.get("delta_y")
    total_dx = data.get("total_dx")
    total_dy = data.get("total_dy")
    direction_deg = data.get("direction_deg")
    speed = data.get("speed")
    sequence = data.get("sequence")
    sequence_text = data.get("sequence_text", "")
    frame_hash = data.get("frame_hash", "")
    release_direction_deg = data.get("release_direction_deg")
    release_speed = data.get("release_speed")

    try:
        idx = int(index) if index is not None else None
    except (TypeError, ValueError):
        idx = None

    payload = {
        "state": state,
        "index": idx,
        "row": row,
        "col": col,
        "board": board,
        "elapsed_ms": elapsed_ms,
        "hold_ms": hold_ms,
        "hash": press_hash,
        "pointer_id": pointer_id,
        "client_ts": client_ts,
        "pointer_x": pointer_x,
        "pointer_y": pointer_y,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "total_dx": total_dx,
        "total_dy": total_dy,
        "direction_deg": direction_deg,
        "speed": speed,
        "sequence": sequence,
        "sequence_text": sequence_text,
        "frame_hash": frame_hash,
        "release_direction_deg": release_direction_deg,
        "release_speed": release_speed,
        "ts": _now(),
        "response_ms": elapsed_ms,
    }

    snapshot = STATS.register_tile_event(payload)
    publish({"type": "tile", "stats": snapshot, "tile": payload, "ts": _now()})

    # Feed the background workers after the server has acknowledged the tile.
    if idx is not None:
        try:
            RAW_EVENT_QUEUE.put_nowait(payload)
        except queue.Full:
            pass
        try:
            FILE_EVENT_QUEUE.put_nowait(payload)
        except queue.Full:
            pass

    if state == "release":
        publish({
            "type": "pattern",
            "stats": snapshot,
            "pattern": {
                "current_sequence": snapshot.get("current_sequence", []),
                "current_sequence_text": snapshot.get("current_sequence_text", ""),
                "last_completed_sequence": snapshot.get("last_completed_sequence", ""),
                "recent_sequences": snapshot.get("recent_sequences", []),
            },
            "ts": _now(),
        })

    return jsonify({
        "ok": True,
        "state": state,
        "index": idx,
        "hash": press_hash,
        "sequence_text": sequence_text,
        "frame_hash": frame_hash,
    })


@app.post("/api/frame")
def api_frame():
    start = time.perf_counter()
    raw = request.get_data(cache=False)

    if not raw:
        STATS.last_error = "empty upload"
        return jsonify({"ok": False, "error": "empty upload"}), 400

    if not raw.startswith(PNG_SIGNATURE):
        STATS.last_error = "invalid png signature"
        return jsonify({"ok": False, "error": "invalid png"}), 400

    frame_hash = request.headers.get("X-Board-Hash", "")[:64]
    client_elapsed_raw = request.headers.get("X-Client-Elapsed-Ms")
    event_type = request.headers.get("X-Event-Type", "")[:24]
    client_elapsed_ms: Optional[float] = None
    if client_elapsed_raw:
        try:
            client_elapsed_ms = float(client_elapsed_raw)
        except ValueError:
            client_elapsed_ms = None

    if frame_hash and frame_hash == STATS.last_frame_hash:
        STATS.skip_duplicate()

    processing_ms = (time.perf_counter() - start) * 1000.0
    snapshot = STATS.register_frame(
        raw,
        processing_ms,
        frame_hash=frame_hash,
        client_elapsed_ms=client_elapsed_ms,
        event_type=event_type,
    )
    publish({"type": "frame", "stats": snapshot, "ts": _now()})
    return jsonify({
        "ok": True,
        "frame_id": STATS.latest_id,
        "size": len(raw),
        "hash": frame_hash,
        "client_elapsed_ms": client_elapsed_ms,
    })


@app.get("/api/stats")
def api_stats():
    return jsonify(STATS.snapshot())


@app.get("/api/charts")
def api_charts():
    limit_raw = request.args.get("limit", str(CHART_WINDOW))
    try:
        limit = max(1, min(2000, int(limit_raw)))
    except (TypeError, ValueError):
        limit = CHART_WINDOW
    charts = _load_charts_from_files(limit=limit)
    return jsonify({
        "ok": True,
        "server_now_ms": int(_now() * 1000.0),
        "stats": STATS.snapshot(),
        "charts": charts,
    })


@app.get("/api/patterns")
def api_patterns():
    snap = STATS.snapshot()
    return jsonify({
        "ok": True,
        "stats": snap,
        "patterns": {
            "current_sequence": snap.get("current_sequence", []),
            "current_sequence_text": snap.get("current_sequence_text", ""),
            "last_completed_sequence": snap.get("last_completed_sequence", ""),
            "last_completed_sequence_len": snap.get("last_completed_sequence_len", 0),
            "last_completed_sequence_hold_ms": snap.get("last_completed_sequence_hold_ms"),
            "recent_sequences": snap.get("recent_sequences", []),
            "pattern_counts_top": snap.get("pattern_counts_top", []),
            "sequence_length_counts": snap.get("sequence_length_counts", {}),
        }
    })


@app.get("/api/events")
def api_events():
    def gen():
        q: queue.Queue[str] = queue.Queue(maxsize=64)
        with SUBSCRIBERS_LOCK:
            SUBSCRIBERS.append(q)
        try:
            yield f"data: {json.dumps({'type': 'hello', 'stats': STATS.snapshot(), 'charts': CHARTS.snapshot(), 'ts': _now()}, separators=(',', ':'))}\n\n"
            while True:
                try:
                    payload = q.get(timeout=15)
                    yield payload
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with SUBSCRIBERS_LOCK:
                try:
                    SUBSCRIBERS.remove(q)
                except ValueError:
                    pass

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/latest.png")
def latest_png():
    with STATS.lock:
        if not STATS.latest_png:
            return Response(status=204)
        data = STATS.latest_png
        frame_id = STATS.latest_id
    return Response(
        data,
        mimetype="image/png",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Frame-Id": str(frame_id),
        },
    )


@app.get("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    _start_background_threads()
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)

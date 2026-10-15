"""SQLite for parsed orders. Ephemeral on Render free tier (demo), a file on the laptop."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.environ.get("DABBA_DB", Path(__file__).resolve().parents[1] / "dabba.db"))


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, raw_text TEXT, sent_at TEXT,
        action TEXT, customer TEXT, date TEXT, meal TEXT, items TEXT, prefs TEXT, time TEXT, repeat TEXT,
        hints TEXT, latency_ms INTEGER, backend TEXT)""")
    return conn


def insert(raw: str, label: dict, hints: list[str], latency_ms: int, backend: str) -> dict:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO orders (created_at, raw_text, action, customer, date, meal, items, prefs, time, repeat, hints, latency_ms, backend) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(timespec="seconds"), raw, label["action"], label["customer"], label["date"], label["meal"],
             json.dumps(label["items"]), json.dumps(label["prefs"]), label["time"], label["repeat"], json.dumps(hints), latency_ms, backend))
        conn.commit()
        return _row(conn.execute("SELECT * FROM orders WHERE id=?", (cur.lastrowid,)).fetchone())


def _row(r) -> dict:
    d = dict(r)
    d["items"] = json.loads(d["items"] or "[]")
    d["prefs"] = json.loads(d["prefs"] or "[]")
    d["hints"] = json.loads(d["hints"] or "[]")
    return d


def list_orders(date: str | None = None, limit: int = 300) -> list[dict]:
    with connect() as conn:
        if date:
            rows = conn.execute("SELECT * FROM orders WHERE date=? ORDER BY meal, customer, id", (date, )).fetchall()
        else:
            rows = conn.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def update(order_id: int, items=None, meal=None, action=None) -> dict:
    with connect() as conn:
        if items is not None:
            conn.execute("UPDATE orders SET items=? WHERE id=?", (json.dumps(items), order_id))
        if meal is not None:
            conn.execute("UPDATE orders SET meal=? WHERE id=?", (meal, order_id))
        if action is not None:
            conn.execute("UPDATE orders SET action=? WHERE id=?", (action, order_id))
        conn.commit()
        return _row(conn.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone())


def delete(order_id: int):
    with connect() as conn:
        conn.execute("DELETE FROM orders WHERE id=?", (order_id,))
        conn.commit()


def clear():
    with connect() as conn:
        conn.execute("DELETE FROM orders")
        conn.commit()


def advance_counts(date: str) -> dict:
    """Net dabbas already ordered for `date` per (meal, dish): orders minus cancels of the same customer+meal."""
    counts: dict = {}
    cancelled = set()
    rows = list_orders(date)
    for o in rows:
        if o["action"] in ("cancel", "pause"):
            cancelled.add((o["customer"], o["meal"]))
            if o["meal"] is None:
                cancelled.add((o["customer"], "lunch")); cancelled.add((o["customer"], "dinner"))
    for o in rows:
        if o["action"] not in ("order", "modify") or (o["customer"], o["meal"]) in cancelled:
            continue
        for it in o["items"]:
            counts[(o["meal"] or "lunch", it["dish"])] = counts.get((o["meal"] or "lunch", it["dish"]), 0) + int(it["qty"])
    return counts

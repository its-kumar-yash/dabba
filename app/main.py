"""Dabba API + UI.  PARSER_BACKEND=tinker uvicorn app.main:app --port 8000"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app import db, planner  # noqa: E402
from app.memory import Memory  # noqa: E402
from app.parser import ParseError, get_backend  # noqa: E402
from data.schema import DISHES, MENU, PREFS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dabba")
app = FastAPI(title="Dabba", version="0.1.0")
STATIC = ROOT / "app" / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

memory = Memory()
_backend, _lock = None, threading.Lock()
LINE_RE = re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),? (\d{1,2}:\d{2}(?::\d{2})?\s?[APap][Mm])\] ([^:]+): (.+)$")


def backend():
    global _backend
    with _lock:
        if _backend is None:
            _backend = get_backend()
            log.info("parser backend: %s", _backend.name)
    return _backend


def _prewarm():
    try:
        backend()
        log.info("prewarm done")
    except Exception as e:
        log.warning("prewarm skipped: %s", e)


@app.on_event("startup")
def _startup():
    threading.Thread(target=_prewarm, daemon=True).start()


class IngestIn(BaseModel):
    text: str
    use_memory: bool = True


class RuleIn(BaseModel):
    customer: str
    usual: str
    prefs: list[str] = []


class EditIn(BaseModel):
    id: int
    items: list[dict] | None = None
    meal: str | None = None
    action: str | None = None
    remember: bool = True


def split_chat(text: str) -> list[str]:
    """A WhatsApp export has one message per line, continuation lines without the [ts] prefix."""
    msgs = []
    for line in text.splitlines():
        line = line.strip().replace(" ", " ").replace("‎", "")
        if not line:
            continue
        if LINE_RE.match(line):
            msgs.append(line)
        elif msgs:
            msgs[-1] += " " + line
        else:
            # bare message without a header: stamp it now, sender unknown
            now = datetime.now()
            h = now.hour % 12 or 12
            msgs.append(f"[{now.day:02d}/{now.month:02d}/{now.year % 100:02d}, {h}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}] Unknown: {line}")
    return msgs


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "backend": os.environ.get("PARSER_BACKEND", "tinker"), "memory": memory.enabled,
            "tabpfn": bool(os.environ.get("TABPFN_TOKEN")), "sample": (ROOT / "data" / "sample_chat.txt").read_text()[:2000]}


@app.get("/menu")
def menu():
    return {"dishes": [{"id": k, **{kk: vv for kk, vv in v.items() if kk != "aliases"}} for k, v in MENU.items()], "prefs": PREFS}


@app.post("/ingest")
def ingest(inp: IngestIn):
    msgs = split_chat(inp.text)
    if not msgs:
        raise HTTPException(400, "no messages found")
    be = backend()

    def one(m: str):
        hints = memory.hints_for(m) if inp.use_memory else []
        try:
            label, ms = be.parse(m, hints)
        except ParseError as e:
            return {"raw_text": m, "error": str(e)[:160]}
        except Exception as e:
            log.exception("parse failed")
            return {"raw_text": m, "error": f"parser backend error: {e}"[:160]}
        return (m, label, hints, ms)

    # Tinker sampling is network-bound; a few in flight at once turns 12 x 4 s into ~15 s
    with ThreadPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(one, msgs[:60]))
    results = []
    for o in outs:  # insert in message order so ids follow the chat
        results.append(o if isinstance(o, dict) else db.insert(*o, be.name))
    return {"orders": results, "n": len(msgs)}


@app.get("/orders")
def orders(date: str | None = None):
    return db.list_orders(date)


@app.post("/orders/edit")
def edit(inp: EditIn):
    o = db.update(inp.id, items=inp.items, meal=inp.meal, action=inp.action)
    remembered = None
    if inp.remember and inp.items and len(inp.items) == 1 and o["customer"] not in ("Unknown", None):
        try:
            remembered = memory.remember_rule(o["customer"], inp.items[0]["dish"], o["prefs"])
        except Exception as e:
            log.warning("remember failed: %s", e)
    return {"order": o, "remembered": remembered}


@app.delete("/orders/{order_id}")
def delete_order(order_id: int):
    db.delete(order_id)
    return {"ok": True}


@app.delete("/orders")
def clear_orders():
    db.clear()
    return {"ok": True}


@app.post("/rules")
def add_rule(inp: RuleIn):
    if inp.usual not in DISHES:
        raise HTTPException(400, "unknown dish")
    try:
        return {"remembered": memory.remember_rule(inp.customer, inp.usual, inp.prefs)}
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/rules")
def rules():
    return memory.list_memories()


@app.delete("/rules/{memory_id}")
def forget(memory_id: str):
    memory.forget(memory_id)
    return {"ok": True}


@app.get("/prep")
def prep(date_: str | None = None):
    day = date.fromisoformat(date_) if date_ else date.today() + timedelta(days=1)
    try:
        return planner.plan(day, db.advance_counts(day.isoformat()))
    except Exception as e:
        log.exception("plan failed")
        raise HTTPException(502, str(e))

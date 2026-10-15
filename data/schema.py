"""Dabba order schema: the single source of truth for training, inference and the app."""
from __future__ import annotations

import json

MENU = {
    "veg_thali": {"name": "Veg Thali", "price": 120, "aliases": ["veg thali", "thali", "dabba", "tiffin", "veg", "sabzi roti", "normal thali"]},
    "paneer_thali": {"name": "Paneer Thali", "price": 150, "aliases": ["paneer thali", "paneer", "paneer wala"]},
    "chicken_thali": {"name": "Chicken Thali", "price": 170, "aliases": ["chicken thali", "chicken", "non veg", "nonveg", "non-veg thali"]},
    "dal_chawal": {"name": "Dal Chawal", "price": 90, "aliases": ["dal chawal", "dal rice", "dal"]},
    "rajma_chawal": {"name": "Rajma Chawal", "price": 100, "aliases": ["rajma chawal", "rajma rice", "rajma"]},
    "chole_chawal": {"name": "Chole Chawal", "price": 100, "aliases": ["chole chawal", "chole rice", "chole"]},
    "roti_pack": {"name": "Roti Pack (5)", "price": 40, "aliases": ["roti pack", "extra roti", "rotis", "roti"]},
}
DISHES = list(MENU)
ACTIONS = ["order", "cancel", "modify", "pause", "resume", "query"]
MEALS = ["lunch", "dinner"]
PREFS = ["no_onion", "less_spicy", "extra_spicy", "jain", "extra_roti", "no_rice", "less_oil"]
REPEATS = ["daily", "weekdays", "weekends"]

FIELDS = ["action", "customer", "date", "meal", "items", "prefs", "time", "repeat"]

SYSTEM_PROMPT = (
    "You read one WhatsApp message sent to a home tiffin (dabba) service and return a single JSON object, nothing else.\n"
    "The message starts with its timestamp and sender like '[14/10/26, 9:12 PM] Name:'. Resolve relative dates (aaj, kal, parso, "
    "weekday names, '15 ko') against that timestamp; the sender name is the customer.\n"
    "Keys in this order: action, customer, date, meal, items, prefs, time, repeat.\n"
    "action: order|cancel|modify|pause|resume|query. customer: sender name. date: YYYY-MM-DD or null. meal: lunch|dinner|null. "
    "items: list of {dish, qty} with dish from the menu ids; empty list for cancel/pause/resume/query. "
    "prefs: subset of " + ", ".join(PREFS) + ". time: HH:MM (24h) or null. repeat: daily|weekdays|weekends|null.\n"
    "Menu ids: " + "; ".join(f"{k} = {', '.join(v['aliases'][:3])}" for k, v in MENU.items()) + ".\n"
    "A bare 'dabba', 'tiffin' or 'thali' means veg_thali unless the known rules say the customer's usual dish is different. "
    "'dopahar' is lunch, 'raat' is dinner."
)
HINTS_HEADER = "Known rules from the kitchen:"


def build_system_prompt(hints: list[str] | None = None) -> str:
    if not hints:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n" + HINTS_HEADER + "\n" + "\n".join(f"- {h}" for h in hints)


def to_json(label: dict) -> str:
    ordered = {k: label.get(k) for k in FIELDS}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def validate(label: dict) -> list[str]:
    errs = []
    if label.get("action") not in ACTIONS:
        errs.append("action")
    if label.get("meal") not in MEALS + [None]:
        errs.append("meal")
    items = label.get("items")
    if not isinstance(items, list):
        errs.append("items")
    else:
        for it in items:
            if not isinstance(it, dict) or it.get("dish") not in DISHES or not isinstance(it.get("qty"), int) or it["qty"] < 1:
                errs.append("items")
                break
    prefs = label.get("prefs")
    if not isinstance(prefs, list) or any(p not in PREFS for p in prefs):
        errs.append("prefs")
    if label.get("repeat") not in REPEATS + [None]:
        errs.append("repeat")
    d = label.get("date")
    if d is not None and not (isinstance(d, str) and len(d) == 10 and d[4] == "-" and d[7] == "-"):
        errs.append("date")
    t = label.get("time")
    if t is not None and not (isinstance(t, str) and len(t) == 5 and t[2] == ":"):
        errs.append("time")
    return errs

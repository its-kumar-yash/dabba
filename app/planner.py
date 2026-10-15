"""Tomorrow's prep list: advance orders from the parsed chat + TabPFN forecast of the final count."""
from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.schema import DISHES, MENU  # noqa: E402
from forecast.forecast import FEATURES, load_history, make_tabpfn, predict_tabpfn  # noqa: E402

log = logging.getLogger("dabba.planner")
HISTORY = Path(os.environ.get("DABBA_HISTORY", ROOT / "data" / "history.csv"))

# rough kitchen conversions so the list is in things you buy, not just dabbas
PER_DABBA = {
    "veg_thali": {"roti": 4, "rice_g": 150, "sabzi_g": 200, "dal_g": 150},
    "paneer_thali": {"roti": 4, "rice_g": 150, "paneer_g": 100, "dal_g": 100},
    "chicken_thali": {"roti": 4, "rice_g": 150, "chicken_g": 180},
    "dal_chawal": {"rice_g": 250, "dal_g": 200},
    "rajma_chawal": {"rice_g": 250, "rajma_g": 180},
    "chole_chawal": {"rice_g": 250, "chole_g": 180},
    "roti_pack": {"roti": 5},
}


@lru_cache(maxsize=1)
def history() -> pd.DataFrame:
    return load_history(HISTORY)


def _features_for(day: date, meal: str, dish: str, advance: int, hist: pd.DataFrame) -> dict:
    s = hist[(hist["dish"] == dish) & (hist["meal"] == meal)].sort_values("date")
    o = s["orders"].values
    a = s["advance_orders"].values
    roll7 = o[-7:].mean() if len(o) >= 3 else (o.mean() if len(o) else 0)
    adv7 = a[-7:].mean() if len(a) >= 3 else 0
    return {
        "dish_id": int(s["dish_id"].iloc[-1]) if len(s) else DISHES.index(dish),
        "meal_id": int(meal == "dinner"), "dow": day.weekday(), "dom": day.day,
        "is_holiday": 0, "navratri": int(date(2026, 10, 11) <= day <= date(2026, 10, 19)),
        "advance_orders": advance,
        "lag1": o[-1] if len(o) else 0, "lag7": o[-7] if len(o) >= 7 else roll7, "lag14": o[-14] if len(o) >= 14 else roll7,
        "roll7": roll7, "roll28": o[-28:].mean() if len(o) else 0, "adv_roll7": adv7,
        "adv_ratio7": (adv7 / roll7) if roll7 else 0.6,
        "days_since_start": int((pd.Timestamp(day) - hist["date"].min()).days),
    }


def plan(day: date, advance: dict, mode: str | None = None) -> dict:
    """advance: {(meal, dish): count}. Returns per meal/dish: advance, forecast median, lo, hi, cook."""
    hist = history()
    mode = mode or os.environ.get("TABPFN_MODE", "client")
    rows, keys = [], []
    for meal in ("lunch", "dinner"):
        for dish in DISHES:
            rows.append(_features_for(day, meal, dish, int(advance.get((meal, dish), 0)), hist))
            keys.append((meal, dish))
    X = pd.DataFrame(rows)[FEATURES].values
    tr = hist[hist["date"] < pd.Timestamp(day)]
    out = {"date": day.isoformat(), "mode": mode, "meals": {"lunch": [], "dinner": []}, "shopping": {}}
    try:
        model = make_tabpfn(mode)
        med, lo, hi = predict_tabpfn(model, tr[FEATURES].values, tr["orders"].values, X)
        out["forecast_ok"] = True
    except Exception as e:  # no token / offline: fall back to the owner's rule of thumb
        log.warning("TabPFN unavailable (%s); using advance / usual advance share", e)
        med = np.array([r["advance_orders"] / max(r["adv_ratio7"], 0.2) for r in rows])
        lo, hi = med * 0.85, med * 1.2
        out["forecast_ok"] = False
    for (meal, dish), r, m, l, h in zip(keys, rows, med, lo, hi):
        m, l, h = max(float(m), r["advance_orders"]), max(float(l), r["advance_orders"]), max(float(h), r["advance_orders"])
        cook = int(np.ceil(h)) if h - m <= 3 else int(np.ceil(m + (h - m) * 0.6))  # lean towards the upper band only when it's cheap
        if r["advance_orders"] == 0 and m < 0.5:
            continue
        entry = {"dish": dish, "name": MENU[dish]["name"], "advance": r["advance_orders"], "forecast": round(m, 1),
                 "lo": int(np.floor(l)), "hi": int(np.ceil(h)), "cook": cook, "usual": round(r["roll7"], 1)}
        out["meals"][meal].append(entry)
        for ing, per in PER_DABBA[dish].items():
            out["shopping"][ing] = out["shopping"].get(ing, 0) + per * cook
    out["total_cook"] = sum(e["cook"] for m in out["meals"].values() for e in m)
    out["total_advance"] = sum(e["advance"] for m in out["meals"].values() for e in m)
    return out

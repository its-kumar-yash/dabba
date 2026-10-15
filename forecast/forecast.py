"""Tomorrow's dabbas per dish, predicted at 9 pm tonight.

TabPFN (Prior Labs' tabular foundation model) sits inside a small prediction agent:
  features known at 9 pm  ->  TabPFNRegressor  ->  per-dish counts + 10/90% band  ->  prep list
Baselines the owner actually uses today: "same as yesterday", "last-week same weekday",
"advance orders + usual walk-ins", plus a plain gradient-boosting model for honesty.

  .venv/bin/python forecast/forecast.py              # backtest last 28 days, writes out/forecast.md
  .venv/bin/python forecast/forecast.py --mode client # Prior Labs API (TABPFN_TOKEN) instead of local weights
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT))

FEATURES = ["dish_id", "meal_id", "dow", "dom", "is_holiday", "navratri", "advance_orders",
            "lag1", "lag7", "lag14", "roll7", "roll28", "adv_roll7", "adv_ratio7", "days_since_start"]


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values(["dish", "meal", "date"]).reset_index(drop=True)
    df["dish_id"] = df["dish"].astype("category").cat.codes
    df["meal_id"] = (df["meal"] == "dinner").astype(int)
    g = df.groupby(["dish", "meal"], sort=False)
    df["lag1"] = g["orders"].shift(1)
    df["lag7"] = g["orders"].shift(7)
    df["lag14"] = g["orders"].shift(14)
    df["roll7"] = g["orders"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    df["roll28"] = g["orders"].transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())
    df["adv_roll7"] = g["advance_orders"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    df["adv_ratio7"] = (df["adv_roll7"] / df["roll7"].replace(0, np.nan)).fillna(0.6)
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
    # 'rain' is NOT a feature: nobody knows tomorrow's rain at 9 pm. It stays in the CSV as a nuisance variable.
    return df.dropna(subset=["lag14", "roll28"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- models
def make_tabpfn(mode: str):
    if mode == "client":
        from tabpfn_client import TabPFNRegressor
        return TabPFNRegressor()
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor(device=os.environ.get("TABPFN_DEVICE", "cpu"))


def predict_tabpfn(model, Xtr, ytr, Xte):
    model.fit(Xtr, ytr)
    try:
        q = model.predict(Xte, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
        q = np.asarray(q)
        if q.ndim == 2 and q.shape[0] == 3:
            lo, med, hi = q
        elif q.ndim == 2 and q.shape[1] == 3:
            lo, med, hi = q.T
        else:
            raise ValueError
        return med, lo, hi
    except Exception:
        p = np.asarray(model.predict(Xte))
        return p, p, p


def baselines(te: pd.DataFrame) -> dict:
    return {
        "same as yesterday": te["lag1"].values,
        "last week, same weekday": te["lag7"].values,
        "7-day average": te["roll7"].values,
        "advance orders / usual advance share": te["advance_orders"].values / np.clip(te["adv_ratio7"].values, 0.2, 1.0),
    }


def gbm(Xtr, ytr, Xte):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr, ytr).predict(Xte)
    except Exception:
        return None


# ----------------------------------------------------------------------------- backtest
def backtest(df: pd.DataFrame, horizon_days: int, mode: str, refit_every: int = 7):
    """Walk forward: for each test day, train on everything strictly before it (refit weekly to save calls)."""
    days = sorted(df["date"].unique())
    test_days = days[-horizon_days:]
    preds = []
    model = None
    t_fit = 0.0
    for i, day in enumerate(test_days):
        tr = df[df["date"] < day]
        te = df[df["date"] == day]
        Xtr, ytr, Xte = tr[FEATURES].values, tr["orders"].values, te[FEATURES].values
        if model is None or i % refit_every == 0:
            model = make_tabpfn(mode)
            t0 = time.perf_counter()
            med, lo, hi = predict_tabpfn(model, Xtr, ytr, Xte)
            t_fit += time.perf_counter() - t0
        else:
            med, lo, hi = predict_tabpfn(model, Xtr, ytr, Xte)  # cheap refit on CPU for small n
        out = te[["date", "dish", "meal", "orders", "advance_orders"]].copy()
        out["tabpfn"] = np.clip(np.rint(med), 0, None)
        out["tabpfn_lo"] = np.clip(np.floor(lo), 0, None)
        out["tabpfn_hi"] = np.clip(np.ceil(hi), 0, None)
        for name, b in baselines(te).items():
            out[name] = np.clip(np.rint(b), 0, None)
        g = gbm(Xtr, ytr, Xte)
        out["gradient boosting"] = np.clip(np.rint(g), 0, None) if g is not None else np.nan
        preds.append(out)
    res = pd.concat(preds, ignore_index=True)
    return res, t_fit


def metrics(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = res["orders"].values
    for col in ["tabpfn", "gradient boosting", "same as yesterday", "last week, same weekday", "7-day average", "advance orders / usual advance share"]:
        p = res[col].values
        if np.isnan(p).all():
            continue
        err = p - y
        rows.append({
            "model": col,
            "MAE (dabbas)": np.abs(err).mean(),
            "over-cooked / day": np.clip(err, 0, None).sum() / res["date"].nunique(),
            "under-cooked / day": np.clip(-err, 0, None).sum() / res["date"].nunique(),
            "exact days %": (err == 0).mean() * 100,
        })
    m = pd.DataFrame(rows).set_index("model")
    cover = ((res["orders"] >= res["tabpfn_lo"]) & (res["orders"] <= res["tabpfn_hi"])).mean() * 100
    m.loc["tabpfn", "80% band coverage"] = cover
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=str(ROOT / "data" / "history.csv"))
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--mode", default=os.environ.get("TABPFN_MODE", "local"), choices=["local", "client"])
    args = ap.parse_args()
    df = load_history(Path(args.history))
    print(f"history rows {len(df)}, series {df.groupby(['dish','meal']).ngroups}, test days {args.days}, mode {args.mode}")
    res, t_fit = backtest(df, args.days, args.mode)
    m = metrics(res)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "forecast_backtest.csv", index=False)
    md = ["# Dabba forecast backtest: last %d days, %d dish x meal series, predicted at 9 pm the night before" % (args.days, df.groupby(['dish','meal']).ngroups), "",
          m.round(2).to_markdown(), "",
          f"TabPFN fit+predict time total: {t_fit:.1f}s ({args.mode}). Over/under-cooked are summed across all dishes and meals per day.", ""]
    (OUT / "forecast.md").write_text("\n".join(md))
    print("\n".join(md))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tot = res.groupby("date")[["orders", "tabpfn", "tabpfn_lo", "tabpfn_hi", "7-day average", "advance orders / usual advance share"]].sum()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(tot.index, tot["tabpfn_lo"], tot["tabpfn_hi"], color="#2a9d8f", alpha=0.15, label="TabPFN 10-90%")
        ax.plot(tot.index, tot["orders"], "k-", lw=2, label="actual dabbas")
        ax.plot(tot.index, tot["tabpfn"], color="#2a9d8f", lw=2, label="TabPFN")
        ax.plot(tot.index, tot["7-day average"], color="#999", ls="--", label="7-day average")
        ax.set_ylabel("dabbas / day (all dishes)"); ax.legend(loc="upper left", fontsize=8); ax.set_title("Dabba: tomorrow's demand, predicted at 9 pm")
        fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "forecast.png", dpi=160)
        print("chart ->", OUT / "forecast.png")
    except Exception as e:
        print("chart skipped:", e)


if __name__ == "__main__":
    main()

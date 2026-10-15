"""Evaluate order parsers on data/messages_test.jsonl -> out/results.md, results.json, results.png.

  .venv/bin/python train/eval.py --systems base,big,lora
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import BASE_MODEL, PRICES, ROOT, load_jsonl, renderer_for

sys.path.insert(0, str(ROOT))
from app.parser import ParseError, extract_json  # noqa: E402
from data.schema import HINTS_HEADER  # noqa: E402

OUT = ROOT / "out"
KEYS = ["action", "customer", "date", "meal", "items", "prefs", "time", "repeat"]


def norm_items(items):
    if not isinstance(items, list):
        return None
    try:
        return sorted((it["dish"], int(it["qty"])) for it in items)
    except Exception:
        return None


def score(pred: dict | None, gold: dict) -> dict:
    if pred is None:
        return {k: 0 for k in ["valid"] + KEYS + ["all"]}
    s = {"valid": 1}
    for k in KEYS:
        if k == "items":
            s[k] = int(norm_items(pred.get(k)) == norm_items(gold[k]))
        elif k == "prefs":
            s[k] = int(sorted(pred.get(k) or []) == sorted(gold[k]))
        elif k == "customer":
            s[k] = int(re.sub(r"\W", "", str(pred.get(k) or "").lower()) == re.sub(r"\W", "", gold[k].lower()))
        else:
            s[k] = int((pred.get(k) or None) == gold[k])
    s["all"] = int(all(s[k] for k in KEYS))
    return s


class TinkerSampler:
    def __init__(self, base_model: str | None = None, model_path: str | None = None):
        import tinker
        from tinker_cookbook.renderers import get_renderer

        self.service = tinker.ServiceClient()
        self.client = self.service.create_sampling_client(base_model=base_model, model_path=model_path)
        self.model = base_model or self.client.get_base_model()
        if not isinstance(self.model, str):
            self.model = BASE_MODEL
        self.tok = self.client.get_tokenizer()
        self.renderer = get_renderer(renderer_for(self.model), self.tok)
        self.params = tinker.SamplingParams(max_tokens=300, temperature=0.0, stop=self.renderer.get_stop_sequences())

    def run(self, messages):
        prompt = self.renderer.build_generation_prompt(messages[:2])
        t0 = time.perf_counter()
        res = self.client.sample(prompt=prompt, num_samples=1, sampling_params=self.params).result()
        dt = (time.perf_counter() - t0) * 1000
        toks = res.sequences[0].tokens
        return self.tok.decode(toks), dt, prompt.length, len(toks)


def evaluate(name, sampler, rows, workers, price_model):
    def one(r):
        try:
            reply, ms, p_tok, c_tok = sampler.run(r["messages"])
        except Exception as e:
            return {"error": str(e)[:200], "ms": None, "pred": None, "p": 0, "c": 0}
        try:
            return {"pred": extract_json(reply), "ms": ms, "p": p_tok, "c": c_tok}
        except ParseError as e:
            return {"error": str(e)[:200], "ms": ms, "pred": None, "p": p_tok, "c": c_tok, "raw": reply[:300]}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(one, rows))
    scores = [score(o["pred"], r["label"]) for o, r in zip(outs, rows)]
    n = len(rows)
    agg = {k: sum(s[k] for s in scores) / n for k in scores[0]}
    lat = [o["ms"] for o in outs if o["ms"] is not None]
    agg["p50_ms"] = statistics.median(lat) if lat else None
    agg["p95_ms"] = sorted(lat)[int(len(lat) * 0.95) - 1] if len(lat) >= 20 else (max(lat) if lat else None)
    tok_p, tok_c = sum(o["p"] for o in outs), sum(o["c"] for o in outs)
    agg["usd_per_1k"] = (tok_p + tok_c) / n * 1000 / 1e6 * PRICES[price_model]["sample"] if price_model and tok_p + tok_c else None
    agg["n"] = n
    kinds = {}
    for s, r in zip(scores, rows):
        kinds.setdefault(r["kind"], []).append(s["all"])
    agg["by_kind"] = {k: sum(v) / len(v) for k, v in sorted(kinds.items())}
    hinted = [s["items"] for s, r in zip(scores, rows) if r.get("hints")]
    agg["hinted_items_acc"] = sum(hinted) / len(hinted) if hinted else None
    errors = [{"text": r["messages"][1]["content"], "gold": r["label"], "out": o} for o, r, s in zip(outs, rows, scores) if not s["all"]]
    return agg, errors


LABELS = {"base": "Qwen3.5-4B zero-shot", "big": "gpt-oss-120b zero-shot", "lora": "Qwen3.5-4B + Dabba LoRA"}


def write_report(results):
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    cols = ["valid"] + KEYS + ["all"]
    lines = ["# Dabba parser eval: 200 held-out WhatsApp messages", "",
             "| System | " + " | ".join(cols) + " | p50 ms | $/1k msgs |", "|---|" + "---|" * (len(cols) + 2)]
    for k, r in results.items():
        pct = " | ".join(f"{r[c]*100:.0f}%" for c in cols)
        p50 = f"{r['p50_ms']:.0f}" if r["p50_ms"] else "-"
        usd = f"${r['usd_per_1k']:.3f}" if r["usd_per_1k"] is not None else "-"
        lines.append(f"| {LABELS.get(k,k)} | {pct} | {p50} | {usd} |")
    lines += ["", "## Items accuracy on messages that needed a kitchen-memory hint (bare 'dabba' from a non-veg-thali regular)"]
    for k, r in results.items():
        if r.get("hinted_items_acc") is not None:
            lines.append(f"- {LABELS.get(k,k)}: {r['hinted_items_acc']*100:.0f}%")
    lines += ["", "## All-fields accuracy by intent", "| intent | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    for kk in sorted({x for r in results.values() for x in r["by_kind"]}):
        lines.append(f"| {kk} | " + " | ".join(f"{results[s]['by_kind'].get(kk, 0)*100:.0f}%" for s in results) + " |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [LABELS.get(k, k) for k in results]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(names))
        ax.bar([i - 0.2 for i in x], [results[k]["all"] * 100 for k in results], 0.4, label="all fields exact", color="#2a9d8f")
        ax.bar([i + 0.2 for i in x], [results[k]["items"] * 100 for k in results], 0.4, label="items correct", color="#e9c46a")
        ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=9); ax.set_ylim(0, 100); ax.set_ylabel("% of 200 test messages")
        ax.legend(loc="lower right"); ax.set_title("Dabba: WhatsApp order -> structured order")
        for i, k in enumerate(results):
            ax.text(i - 0.2, results[k]["all"] * 100 + 1, f"{results[k]['all']*100:.0f}", ha="center", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "results.png", dpi=160)
    except Exception as e:
        print("chart skipped:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="base,big,lora")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sampler-path", default=os.environ.get("DABBA_SAMPLER_PATH"))
    ap.add_argument("--big-model", default="openai/gpt-oss-120b")
    args = ap.parse_args()
    rows = load_jsonl(ROOT / "data" / "messages_test.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    sampler_path = args.sampler_path
    if not sampler_path and (OUT / "run.json").exists():
        sampler_path = json.loads((OUT / "run.json").read_text()).get("sampler_path")
    results = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    for name in args.systems.split(","):
        name = name.strip()
        print(f"\n=== {name} on {len(rows)} messages")
        if name == "base":
            s, price = TinkerSampler(base_model=BASE_MODEL), BASE_MODEL
        elif name == "big":
            s, price = TinkerSampler(base_model=args.big_model), args.big_model
        elif name == "lora":
            if not sampler_path:
                raise SystemExit("need --sampler-path or out/run.json")
            s, price = TinkerSampler(model_path=sampler_path), BASE_MODEL
        else:
            raise SystemExit(name)
        agg, errors = evaluate(name, s, rows, args.workers, price)
        results[name] = agg
        (OUT / f"errors_{name}.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False, default=str))
        print(json.dumps({k: v for k, v in agg.items() if k != "by_kind"}, indent=1, default=str))
    write_report(results)


if __name__ == "__main__":
    main()

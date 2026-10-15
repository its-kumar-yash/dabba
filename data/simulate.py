"""Simulate a home tiffin kitchen for ~5 months and emit two consistent datasets:

  data/history.csv       daily demand per dish x meal, with what was known by 9 pm the night before
  data/messages_*.jsonl  WhatsApp-style order messages with exact labels (for the Tinker fine-tune)
  data/sample_chat.txt   one evening of messages, pasteable into the demo

Every message is rendered from a structured intent, so the label is exact by construction.

  python data/simulate.py            # defaults: 80 customers, 2026-05-01 .. 2026-10-14
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schema import DISHES, MENU, build_system_prompt, to_json, validate  # noqa: E402

HERE = Path(__file__).resolve().parent

FIRST = ["Priya", "Rahul", "Anjali", "Vikas", "Neha", "Amit", "Pooja", "Rohit", "Sneha", "Karan", "Divya", "Arjun", "Meera", "Sahil",
         "Kavita", "Nikhil", "Ritu", "Manish", "Shreya", "Deepak", "Isha", "Varun", "Simran", "Harsh", "Tanvi", "Gaurav", "Ananya", "Mohit",
         "Swati", "Aakash", "Nidhi", "Yash", "Payal", "Siddharth", "Bhavna", "Kunal", "Jyoti", "Abhishek", "Megha", "Tarun"]
LAST = ["Sharma", "Verma", "Gupta", "Singh", "Mehra", "Kumar", "Agarwal", "Jain", "Yadav", "Kapoor", "Malhotra", "Saini", "Bansal", "Chauhan",
        "Mishra", "Tiwari", "Nair", "Reddy", "Iyer", "Das", "Bose", "Khan", "Sethi", "Arora"]
FESTIVALS = {  # date: (name, lunch multiplier, dinner multiplier, veg_shift)
    date(2026, 8, 15): ("Independence Day", 0.55, 0.7, 0.0),
    date(2026, 8, 28): ("Raksha Bandhan", 0.6, 0.65, 0.1),
    date(2026, 9, 4): ("Janmashtami", 0.8, 0.6, 0.3),
    date(2026, 9, 14): ("Ganesh Chaturthi", 0.75, 0.7, 0.2),
    date(2026, 10, 2): ("Gandhi Jayanti", 0.6, 0.75, 0.0),
}
NAVRATRI = (date(2026, 10, 11), date(2026, 10, 19))  # chicken drops, veg rises
PREF_POOL = ["no_onion", "less_spicy", "extra_spicy", "jain", "extra_roti", "no_rice", "less_oil"]


@dataclass
class Customer:
    cid: int
    name: str
    meals: list  # ["lunch"] / ["dinner"] / both
    usual: str  # dish id
    prefs: list
    style: str  # hinglish|english|hindi
    weekday_only: bool
    joined: date
    left: date | None
    subscribed: bool  # standing daily order vs ad-hoc
    advance_p: float  # probability of ordering the evening before (vs same morning)
    qty: int
    pauses: list = field(default_factory=list)  # (start, end)
    generic_p: float = 0.5  # says "dabba" instead of naming the dish

    def active(self, d: date) -> bool:
        if d < self.joined or (self.left and d >= self.left):
            return False
        if self.weekday_only and d.weekday() >= 5:
            return False
        return not any(a <= d <= b for a, b in self.pauses)


def make_customers(n: int, start: date, end: date, rng: random.Random) -> list[Customer]:
    out, names = [], set()
    for i in range(n):
        while True:
            name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            if name not in names:
                names.add(name)
                break
        r = rng.random()
        meals = ["lunch"] if r < 0.6 else (["dinner"] if r < 0.8 else ["lunch", "dinner"])
        usual = rng.choices(DISHES[:6], weights=[42, 18, 16, 10, 8, 6])[0]
        prefs = [p for p in PREF_POOL if rng.random() < {"no_onion": 0.15, "less_spicy": 0.18, "extra_spicy": 0.05, "jain": 0.05, "extra_roti": 0.1, "no_rice": 0.05, "less_oil": 0.06}[p]]
        if "jain" in prefs:
            prefs = [p for p in prefs if p != "no_onion"]
            if usual == "chicken_thali":
                usual = "veg_thali"
        joined = start + timedelta(days=int(rng.triangular(0, (end - start).days, 0))) if rng.random() < 0.5 else start
        left = joined + timedelta(days=rng.randint(20, 120)) if rng.random() < 0.3 else None
        if left and left > end:
            left = None
        pauses = []
        for _ in range(rng.choice([0, 1, 1, 2, 2, 3])):
            ps = joined + timedelta(days=rng.randint(5, max(6, (end - joined).days)))
            pauses.append((ps, ps + timedelta(days=rng.randint(2, 9))))
        out.append(Customer(
            cid=i, name=name, meals=meals, usual=usual, prefs=prefs,
            style=rng.choices(["hinglish", "english", "hindi"], weights=[65, 25, 10])[0],
            weekday_only=rng.random() < 0.55, joined=joined, left=left, subscribed=rng.random() < 0.45,
            advance_p=rng.uniform(0.45, 0.9), qty=rng.choices([1, 1, 1, 2, 2, 3], k=1)[0], pauses=pauses,
            generic_p=rng.uniform(0.2, 0.8)))
    return out


def day_multipliers(d: date):
    name, lm, dm, veg = None, 1.0, 1.0, 0.0
    if d in FESTIVALS:
        name, lm, dm, veg = FESTIVALS[d]
    if NAVRATRI[0] <= d <= NAVRATRI[1]:
        name = name or "Navratri"
        veg += 0.35
    if d.weekday() == 6:  # Sunday
        lm *= 0.7
        dm *= 0.85
    if d.day >= 27:  # month end: wallets thin
        lm *= 0.9
        dm *= 0.9
    return name, lm, dm, veg


# ----------------------------------------------------------------------------- simulation
def simulate(customers: list[Customer], start: date, end: date, rng: random.Random):
    """Returns (history rows, intents). An intent is a structured event that becomes one message."""
    rows, intents = [], []
    d = start
    growth = 1.0
    while d <= end:
        fname, lm, dm, veg_shift = day_multipliers(d)
        rain = rng.random() < 0.12  # rainy day: more orders (nobody wants to step out)
        for meal in ("lunch", "dinner"):
            mult = (lm if meal == "lunch" else dm) * growth * (1.15 if rain else 1.0)
            counts = {dish: [0, 0] for dish in DISHES}  # [advance, same-day]
            for c in customers:
                if meal not in c.meals or not c.active(d):
                    continue
                p = (0.92 if c.subscribed else 0.55) * mult
                if rng.random() > p:
                    if c.subscribed and rng.random() < 0.9:
                        intents.append(dict(kind="cancel", c=c, day=d, meal=meal, advance=rng.random() < c.advance_p))
                    continue
                dish = c.usual
                if dish == "chicken_thali" and rng.random() < veg_shift:
                    dish = "veg_thali"
                elif rng.random() < 0.12:
                    dish = rng.choice(DISHES[:6])
                qty = c.qty if rng.random() < 0.85 else max(1, c.qty + rng.choice([-1, 1]))
                advance = rng.random() < c.advance_p
                counts[dish][0 if advance else 1] += qty
                extra_roti = "extra_roti" in c.prefs and rng.random() < 0.5
                if extra_roti:
                    counts["roti_pack"][0 if advance else 1] += 1
                if not c.subscribed or rng.random() < 0.25 or dish != c.usual or qty != c.qty:
                    intents.append(dict(kind="order", c=c, day=d, meal=meal, dish=dish, qty=qty, advance=advance,
                                        extra_roti=extra_roti, prefs=[p for p in c.prefs if rng.random() < 0.5]))
            for dish in DISHES:
                adv, same = counts[dish]
                rows.append(dict(date=d.isoformat(), meal=meal, dish=dish, orders=adv + same, advance_orders=adv,
                                 dow=d.weekday(), dom=d.day, is_holiday=int(fname is not None and fname != "Navratri"),
                                 festival=fname or "", navratri=int(NAVRATRI[0] <= d <= NAVRATRI[1]), rain=int(rain)))
        # pause / resume / query / subscribe chatter
        for c in customers:
            for a, b in c.pauses:
                if a == d + timedelta(days=1):
                    intents.append(dict(kind="pause", c=c, day=d, until=b))
                if b == d:
                    intents.append(dict(kind="resume", c=c, day=d))
            if c.joined == d and c.subscribed:
                intents.append(dict(kind="subscribe", c=c, day=d))
            if rng.random() < 0.02:
                intents.append(dict(kind="query", c=c, day=d, meal=rng.choice(c.meals)))
            if rng.random() < 0.02 and c.active(d):
                intents.append(dict(kind="modify", c=c, day=d, meal=rng.choice(c.meals), dish=rng.choice(DISHES[:6]), qty=rng.choice([1, 2])))
        growth *= 1.0015
        d += timedelta(days=1)
    return rows, intents


# ----------------------------------------------------------------------------- rendering
QTY_WORDS = {1: ["1", "ek", "one"], 2: ["2", "do", "two"], 3: ["3", "teen", "three"], 4: ["4", "char"], 5: ["5", "paanch"]}
DAY_WORDS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HINDI_DAYS = ["somvaar", "mangalvaar", "budhvaar", "guruvaar", "shukravaar", "shanivaar", "ravivaar"]
PREF_TEXT = {
    "no_onion": ["no onion", "bina pyaaz", "pyaaz nahi", "without onion"],
    "less_spicy": ["less spicy", "kam mirch", "kam teekha", "mild"],
    "extra_spicy": ["extra spicy", "zyada mirch", "teekha"],
    "jain": ["jain", "jain wala", "no onion garlic"],
    "extra_roti": ["extra roti", "2 roti extra", "roti zyada"],
    "no_rice": ["no rice", "chawal nahi", "without rice"],
    "less_oil": ["less oil", "kam tel", "oil kam"],
}
MEAL_TEXT = {"lunch": ["lunch", "dopahar", "lunch me", "dopahar ko", "afternoon"], "dinner": ["dinner", "raat", "dinner me", "raat ko", "night"]}


def dish_text(dish: str, rng: random.Random, generic_ok: bool) -> str:
    al = MENU[dish]["aliases"]
    if dish == "veg_thali" and generic_ok:
        return rng.choice(["dabba", "tiffin", "thali", "veg thali"])
    return rng.choice(al[:3])


def rel_date(target: date, sent: date, rng: random.Random, style: str) -> str | None:
    delta = (target - sent).days
    if delta == 0:
        return rng.choice(["aaj", "today", "aj"] if style != "english" else ["today"])
    if delta == 1:
        return rng.choice(["kal", "tomorrow", "kl"] if style != "english" else ["tomorrow"])
    if delta == 2:
        return rng.choice(["parso", "day after tomorrow", "parson"] if style != "english" else ["day after tomorrow"])
    if 2 < delta <= 6:
        return rng.choice([DAY_WORDS[target.weekday()], HINDI_DAYS[target.weekday()] if style == "hindi" else DAY_WORDS[target.weekday()], f"{target.day} ko", f"{target.day} tarikh"])
    return f"{target.day}/{target.month}"


def fmt_ts(dt: datetime) -> str:
    h = dt.hour % 12 or 12
    return f"[{dt.day:02d}/{dt.month:02d}/{dt.year % 100:02d}, {h}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}]"


def render(intent: dict, rng: random.Random):
    """intent -> (text, label). Timestamp placement follows the WhatsApp export format."""
    c: Customer = intent["c"]
    day: date = intent["day"]
    kind = intent["kind"]
    style = c.style
    meal = intent.get("meal")
    label = {"action": None, "customer": c.name, "date": None, "meal": meal, "items": [], "prefs": [], "time": None, "repeat": None}

    if kind in ("order", "modify"):
        advance = intent.get("advance", True)
        sent_day = day - timedelta(days=1) if advance else day
        sent = datetime.combine(sent_day, time(rng.randint(18, 22) if advance else rng.randint(7, 10), rng.randint(0, 59)))
        when = rel_date(day, sent_day, rng, style)
        qty = intent["qty"]
        dish = intent["dish"]
        q = rng.choice(QTY_WORDS[qty])
        bare = kind == "order" and dish == c.usual and rng.random() < c.generic_p * 0.6
        dtxt = rng.choice(["dabba", "tiffin", "thali", "usual"]) if bare else dish_text(dish, rng, generic_ok=False)
        label["_needs_hint"] = bare and dish != "veg_thali"
        mtxt = rng.choice(MEAL_TEXT[meal])
        mplain = "lunch" if meal == "lunch" else "dinner"
        prefs = list(intent.get("prefs", []))
        ptxt = ", ".join(rng.choice(PREF_TEXT[p]) for p in prefs)
        ttxt, tval = "", None
        if rng.random() < 0.3:
            hour = rng.choice([12, 13, 13, 14]) if meal == "lunch" else rng.choice([19, 20, 20, 21])
            minute = rng.choice([0, 0, 30])
            tval = f"{hour:02d}:{minute:02d}"
            h12 = hour - 12 if hour > 12 else hour
            ttxt = rng.choice([f"{h12} bje", f"{h12}{':30' if minute else ''} baje", f"{h12}:{minute:02d} pm", f"by {h12}{':30' if minute else ''}", f"{h12} baje tak"])
        if kind == "modify":
            templates = [f"{when} {mtxt} me {q} {dtxt} kar dena instead", f"change {when} {mtxt} to {q} {dtxt}", f"{when} ka {mtxt} {q} {dtxt} kar do"]
        elif style == "english":
            templates = [f"Please send {q} {dtxt} {when} {mtxt} {ttxt}", f"{q} {dtxt} for {mtxt} {when} {ttxt}", f"{when} {mtxt}: {q} {dtxt} {ttxt}", f"Can I get {q} {dtxt} {when} for {mtxt}?"]
        elif style == "hindi":
            templates = [f"{when} {mtxt} {q} {dtxt} chahiye {ttxt}", f"{when} {q} {dtxt} bhej dena {mtxt} {ttxt}", f"{q} {dtxt} {when} {mtxt} ko"]
        else:
            templates = [f"{when} {q} {dtxt} {mtxt} {ttxt}", f"{when} {mtxt} {q} {dtxt} bhej dena {ttxt}", f"{q} {dtxt} {when} {mtxt} {ttxt}", f"{when} {mplain} ke liye {q} {dtxt} {ttxt}", f"bhaiya {when} {q} {dtxt} {mtxt} {ttxt} 🙏"]
        text = rng.choice(templates)
        if ptxt:
            text += rng.choice([f", {ptxt}", f" ({ptxt})", f" {ptxt}", f". {ptxt} please"])
        items = [{"dish": dish, "qty": qty}]
        if intent.get("extra_roti") and rng.random() < 0.7:
            text += rng.choice([" + 1 roti pack", ", ek roti pack extra", " and 1 extra roti pack"])
            items.append({"dish": "roti_pack", "qty": 1})
        elif intent.get("extra_roti"):
            prefs.append("extra_roti") if "extra_roti" not in prefs else None
            text += rng.choice([", extra roti", " roti zyada dena"])
        label.update(action=kind, date=day.isoformat(), items=items, prefs=sorted(set(prefs)), time=tval)
    elif kind == "cancel":
        advance = intent.get("advance", True)
        sent_day = day - timedelta(days=1) if advance else day
        sent = datetime.combine(sent_day, time(rng.randint(18, 22) if advance else rng.randint(6, 9), rng.randint(0, 59)))
        when = rel_date(day, sent_day, rng, style)
        mtxt = rng.choice(MEAL_TEXT[meal])
        text = rng.choice([f"{when} {mtxt} cancel", f"{when} ka {mtxt} cancel kar do", f"{when} {mtxt} nahi chahiye", f"please cancel {when} {mtxt}", f"{when} {mtxt} mat bhejna", f"{when} {mtxt} skip"])
        label.update(action="cancel", date=day.isoformat())
    elif kind == "pause":
        sent = datetime.combine(day, time(rng.randint(17, 22), rng.randint(0, 59)))
        until: date = intent["until"]
        text = rng.choice([f"kal se {(until - day).days} din ke liye band kar do, bahar ja rahe", f"pausing from tomorrow till {until.day}/{until.month}", f"{(until - day).days} din tiffin nahi chahiye, out of station", f"kal se {until.day} tak mat bhejna"])
        label.update(action="pause", date=(day + timedelta(days=1)).isoformat(), meal=None)
    elif kind == "resume":
        sent = datetime.combine(day, time(rng.randint(17, 22), rng.randint(0, 59)))
        text = rng.choice(["kal se phir se start kar do", "resume from tomorrow please", "wapas aa gaye, kal se chalu karo", "kal se tiffin start"])
        label.update(action="resume", date=(day + timedelta(days=1)).isoformat(), meal=None)
    elif kind == "subscribe":
        sent = datetime.combine(day - timedelta(days=1), time(rng.randint(17, 22), rng.randint(0, 59)))
        mtxt = rng.choice(MEAL_TEXT[c.meals[0]])
        rep = "weekdays" if c.weekday_only else "daily"
        rtxt = rng.choice(["Mon se Fri", "Monday to Friday", "weekdays", "mon-fri"]) if rep == "weekdays" else rng.choice(["daily", "roz", "every day", "har din"])
        q = rng.choice(QTY_WORDS[c.qty])
        bare = rng.random() < c.generic_p * 0.5
        dtxt = rng.choice(["dabba", "tiffin"]) if bare else dish_text(c.usual, rng, generic_ok=False)
        label["_needs_hint"] = bare and c.usual != "veg_thali"
        text = rng.choice([f"kal se {rtxt} {q} {dtxt} {mtxt}", f"{rtxt} {q} {dtxt} {mtxt} from tomorrow", f"start {rtxt} {q} {dtxt} for {mtxt} kal se"])
        prefs = list(c.prefs)
        if prefs and rng.random() < 0.7:
            text += ", " + ", ".join(rng.choice(PREF_TEXT[p]) for p in prefs)
        else:
            prefs = []
        label.update(action="order", date=day.isoformat(), meal=c.meals[0], items=[{"dish": c.usual, "qty": c.qty}], prefs=sorted(set(prefs)), repeat=rep)
    elif kind == "query":
        sent = datetime.combine(day, time(13 if meal == "lunch" else 20, rng.randint(30, 59)))
        text = rng.choice(["aaj ka dabba aaya nahi?", "tiffin kab tak aayega?", "where is my tiffin", "abhi tak nahi aaya", "kitna hua is mahine ka?", "bill bhej do"])
        label.update(action="query", date=day.isoformat() if "mahine" not in text and "bill" not in text else None, meal=meal if ("dabba" in text or "tiffin" in text or "aaya" in text) else None)
    else:
        raise ValueError(kind)

    text = f"{fmt_ts(sent)} {c.name}: " + " ".join(text.split())
    return text, label, sent


def noisify(text: str, rng: random.Random) -> str:
    head, _, body = text.partition(": ")
    body = body.replace("thali", rng.choice(["thali", "thaali", "thalli"])).replace("chahiye", rng.choice(["chahiye", "chaiye", "chahiye"]))
    if rng.random() < 0.3:
        body = body.lower()
    if rng.random() < 0.2:
        body = body.replace(",", "")
    return f"{head}: {body}"


def hint_for(c: Customer) -> str:
    h = f"{c.name}: usual dish {c.usual}"
    if c.prefs:
        h += ", prefs " + ", ".join(c.prefs)
    return h + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=80)
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-10-14")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--train", type=int, default=1400)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--test", type=int, default=200)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    customers = make_customers(args.customers, start, end, rng)
    rows, intents = simulate(customers, start, end, rng)

    with (HERE / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (HERE / "customers.json").open("w") as f:
        json.dump([dict(name=c.name, usual=c.usual, prefs=c.prefs, meals=c.meals, subscribed=c.subscribed, weekday_only=c.weekday_only) for c in customers], f, indent=1)

    rng.shuffle(intents)
    seen, examples = set(), []
    for it in intents:
        text, label, sent = render(it, rng)
        needs_hint = label.pop("_needs_hint", False)
        errs = validate(label)
        assert not errs, (errs, text, label)
        if text in seen:
            continue
        seen.add(text)
        c = it["c"]
        hints = None
        if needs_hint:
            hints = [hint_for(c)]  # the only way to know a bare 'dabba' means paneer_thali is the kitchen's memory
        elif rng.random() < 0.12:
            hints = [hint_for(c)]
        examples.append(dict(text=text, label=label, kind=it["kind"], hints=hints, sent=sent.isoformat()))
    print(f"{len(examples)} unique messages from {len(intents)} intents; history rows {len(rows)}")
    need = args.train + args.val + args.test
    if len(examples) < need:
        raise SystemExit(f"only {len(examples)} examples; lower --train or raise --customers")
    rare = [e for e in examples if e["kind"] != "order"]
    orders = [e for e in examples if e["kind"] == "order"]
    examples = (rare + orders[: max(0, need - len(rare))])[:need]
    rng.shuffle(examples)
    test, val, train = examples[: args.test], examples[args.test : args.test + args.val], examples[args.test + args.val :]
    for ex in train:
        if rng.random() < 0.25:
            ex["text"] = noisify(ex["text"], rng)
    for name, split in (("train", train), ("val", val), ("test", test)):
        with (HERE / f"messages_{name}.jsonl").open("w", encoding="utf-8") as f:
            for ex in split:
                rec = {"messages": [{"role": "system", "content": build_system_prompt(ex["hints"])}, {"role": "user", "content": ex["text"]},
                                    {"role": "assistant", "content": to_json(ex["label"])}], "label": ex["label"], "kind": ex["kind"], "hints": ex["hints"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kinds = {}
        for ex in split:
            kinds[ex["kind"]] = kinds.get(ex["kind"], 0) + 1
        print(f"{name}: {len(split)} {dict(sorted(kinds.items()))} hinted={sum(1 for e in split if e['hints'])}")

    # one evening of real-looking chat for the demo: everything sent on the last evening
    last = end
    evening = [render(it, random.Random(99 + i))[0] for i, it in enumerate(intents) if it["kind"] in ("order", "cancel") and it.get("advance") and it["day"] == last][:14]
    evening = [e for e in evening if "usual" not in e]
    (HERE / "sample_chat.txt").write_text("\n".join(sorted(evening)) + "\n")
    print("sample_chat.txt:", len(evening), "lines")


if __name__ == "__main__":
    main()

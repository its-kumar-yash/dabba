---
title: Dabba: a 4B model reads a tiffin kitchen's WhatsApp group and TabPFN says how many thalis to cook tomorrow
published: false
tags: devchallenge, hf26challenge
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 2: Side Hustle](https://dev.to/challenges/hacktoberfest-week2-2026-10-12)*

## What I Built

Every evening, in a few thousand Indian neighbourhoods, a WhatsApp group fills up with messages like these:

```
[14/10/26, 9:12 PM] Priya Sharma: kal 2 veg thali lunch, no onion
[14/10/26, 9:40 PM] Rahul Verma: parso ka dabba cancel kar do
[14/10/26, 10:05 PM] Megha Jain: kal se 6 din band, bahar ja rahe
[14/10/26, 10:31 PM] Harsh Khan: 1 dabba kl afternoon 12 bje
```

They are orders for a home tiffin service, a side hustle that feeds office-goers and students one dabba at a time. The owner reads the group at night, tallies on paper, and decides how much to cook at six the next morning. Harsh's "1 dabba" means a paneer thali because Harsh always takes paneer. Megha's "6 din band" means six fewer every day until she's back. Then the walk-in orders arrive at nine, after the shopping is done.

Cook too much and the margin goes into the bin. Cook too little and a regular eats elsewhere, and regulars are the whole business.

**Dabba** turns that WhatsApp group into tomorrow's prep list. Paste tonight's messages, and a fine-tuned 4B model turns each one into a structured order: who, what, how many, which meal, which day, with the kitchen's memory filling in what "dabba" means for that customer. Then [TabPFN](https://priorlabs.ai), a tabular foundation model, looks at the kitchen's history and the orders already booked and says how many of each dish to cook, with a band around it. The cook gets a shopping list in rotis and kilos.

Who pays: the owner. One over-cooked tray of chicken is more than a month of this.

## Demo

<!-- TODO: GIF -->

Hosted demo: **https://dabba-jgho.onrender.com** (free tier, about a minute to wake). Press *Use tonight's sample* to load twelve real-looking messages, parse them, and get the prep list.

## Code

{% embed https://github.com/its-kumar-yash/dabba %}

## How I Built It

Three open pieces, each doing one job.

### 1. The parser: Qwen3.5-4B + a LoRA trained on Tinker

The hard part of a tiffin message is not the vocabulary, it is the timestamp. "kal" is a different date on every line, "parso" is two days from whenever the message was sent, and "Monday" from a Thursday night is in four days. A general model has to count; a model that has seen 1,400 of these does not.

I generated the training set from a simulated kitchen rather than labelling by hand. Eighty customers with usual dishes, preferences, office days, travel pauses and churn produce five months of orders; each order is rendered into a message in one of three registers (Hinglish, English, roman Hindi) with the WhatsApp export prefix. **The program that writes the message also writes the label**, so the labels are exact. A quarter of the training messages get the typos and lowercase that real thumbs produce.

Training is the plain Tinker loop, three epochs, rank 16, 129 steps, under two dollars:

```python
tc = service.create_lora_training_client(base_model="Qwen/Qwen3.5-4B", rank=16)
renderer = get_renderer("qwen3_5_disable_thinking", tc.get_tokenizer())
for step in range(129):
    tc.forward_backward(batch(step), "cross_entropy")
    tc.optim_step(adam_params=tinker.AdamParams(learning_rate=lr(step)))
sampler_path = tc.save_weights_for_sampler(name="dabba-v2").result().path
```

**Did it beat the baseline?** Same 200 held-out messages, same system prompt, exact match on every field. Baselines are zero-shot: the untouched 4B and gpt-oss-120b.

| System | valid JSON | action | date | items | prefs | time | **all fields** | p50 latency | $ / 1k msgs |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B zero-shot | 72% | 56% | 56% | 53% | 67% | 66% | **30%** | 2.1 s | $0.52 |
| gpt-oss-120b zero-shot | 98% | 96% | 50% | 86% | 96% | 96% | **38%** | 2.8 s | $0.56 |
| Qwen3.5-4B + Dabba LoRA | 100% | 100% | 100% | 100% | 100% | 99% | **98%** | 2.3 s | $0.49 |

The 120B model writes valid JSON almost every time and still gets fewer than four messages in ten fully right, because it resolves "kal" against the wrong day. The fine-tuned 4B resolves every date in the test set and gets 98% of messages fully right. Of the four it misses, two are a template of mine whose delivery time never made it into the text (label noise, my fault), one is a dropped "jain", and one is a real slip: "kal" from the night of 30 May came out as 1 June.

![All-fields and items accuracy on 200 held-out messages](https://raw.githubusercontent.com/its-kumar-yash/dabba/main/results/results.png)

### 2. The memory: Backboard

"1 dabba" from Harsh means a paneer thali. No model can know that; the kitchen can. When the owner corrects a parsed order once, Dabba writes one Backboard memory:

```
POST /assistants/{id}/memories   {"content": "Harsh Khan: usual dish paneer_thali, prefs no_onion."}
```

Before parsing a message, Dabba searches those memories with the message, keeps only the ones whose subject is the sender, and prepends them to the system prompt as a *Known rules from the kitchen* block. The model was trained with that block present on 18% of examples, so it obeys it. On the 35 test messages where a bare "dabba" came from a non-veg-thali regular, the zero-shot 120B picked the right dish 69% of the time with the same rule in its prompt; the fine-tune picked it 100%.

Memory writes and reads are plain API calls, no LLM in the loop, which is what you want for a fact as boring as "Harsh takes paneer".

### 3. The forecast: TabPFN inside a small prediction agent

At 9 pm the kitchen knows three things: what it sold on each day of the last five months, what today's weekday and date are, and how many dabbas are *already* booked for tomorrow. That last one is the strong signal, and it is a known-future covariate, which is why I used TabPFN as a tabular regressor with hand-built features rather than a pure time-series model. For each dish and meal:

```
dow, day of month, holiday, Navratri, advance_orders (booked by 9 pm),
lag1, lag7, lag14, 7-day mean, 28-day mean, 7-day mean of advance orders, advance share
```

Rain is in the history and deliberately *not* a feature: nobody knows tomorrow's rain at 9 pm.

The model is `TabPFNRegressor` from Prior Labs, fitted on everything before the test day and asked for the 10/50/90 quantiles. I walked it forward over the last 28 days against the rules of thumb an owner actually uses, plus gradient boosting for honesty:

| Predictor (per dish, per meal) | MAE in dabbas | over-cooked / day | under-cooked / day | exactly right |
|---|---|---|---|---|
| Same as yesterday | 2.56 | 18.1 | 17.7 | 24% |
| 7-day average | 2.30 | 15.7 | 16.5 | 26% |
| Last week, same weekday | 1.79 | 12.6 | 12.5 | 32% |
| Advance orders ÷ usual advance share (what an owner does) | 1.55 | 12.4 | 9.3 | 36% |
| Gradient boosting on the same features | 1.07 | 7.3 | 7.7 | 39% |
| **TabPFN** | **0.90** | **3.9** | **8.7** | **51%** |

28 test days, 14 dish-meal series, 392 predictions, walk-forward with the model refitted weekly. "Over/under-cooked" are summed across all dishes for a day; "exactly right" is the share of dish-day predictions that hit the integer.

![Actual dabbas per day vs TabPFN median and 10–90% band over the 28 test days](https://raw.githubusercontent.com/its-kumar-yash/dabba/main/results/forecast.png)

The number that matters is the over- and under-cooked columns, because they are the money. The owner's own rule mis-cooks about 22 dabbas a day across the menu; TabPFN mis-cooks about 13, and most of its misses are on the under side, which is why the app's "cook" number leans towards the top of the band when the extra is cheap. Gradient boosting on identical features gets close on MAE, but TabPFN over-cooks half as much and needed no tuning at all.

One honest note on the band: it was asked for 10–90% and covered 97% of outcomes, so it is wider than it claims. For a kitchen that is the right failure mode, but it is a failure mode.

### 4. Render

One `render.yaml`, one Python web service, free tier. The parser is served from Tinker's sampling API because 512 MB cannot hold a 4B model; TabPFN runs through Prior Labs' API for the same reason. At home the same app points at Ollama and the local TabPFN weights, with one environment variable each.

## Why Does Open Innovation Matter?

**You can fine-tune.** No prompt teaches a frontier model that "parso" from a Tuesday night is Thursday and that a kitchen's "dabba" is whatever that customer always takes. Three epochs of LoRA on a 4B model did, for under two dollars, and the adapter is a file I own. When the kitchen adds a dish, I regenerate and retrain over lunch.

**Small is the right size.** A 4B model beat a 120B model by 60 points on this task. A tiffin kitchen's compute budget is a phone and, at best, an old laptop. A 4B adapter fits; a 120B API bill does not.

**The group chat stays in the house.** Those messages carry names, addresses, who is travelling and when. Run on a laptop with Ollama and the local TabPFN weights, nothing leaves. A closed model is a copy of the whole neighbourhood's routine on someone else's server.

**A tabular foundation model changes who gets a forecast.** Demand forecasting used to mean a data scientist and a month. TabPFN fits in one call on 2,000 rows and gives calibrated bands. The owner never sees it; the owner sees "cook 34, maybe 38".

Where a closed model would do better: nowhere I could find for this job. Where my setup is worse: the history is simulated, which I say plainly below.

## Honest limits

The kitchen is simulated. The customers, their habits and the festival dips are mine, so the forecast numbers measure the method on a realistic kitchen, not on a real one. The parser's messages are also generated, from 26 templates in three registers; the test set is held out but it is the same distribution. The real test is the next section.

## My First User

<!-- TODO: Yash -->

## My Agent Session

Built with Claude Code over three evenings. The session did the simulator, the Tinker and TabPFN scripts, the app, the deploy, and caught two of its own labelling bugs by reading the error files.

## Prize Categories

Thinking Machines (Tinker), Prior Labs (TabPFN), Render, Backboard.

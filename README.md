# Dabba · kal kitna banana hai

An ops tool for a home tiffin (dabba) kitchen. Paste tonight's WhatsApp messages, get tomorrow's
orders as a table and a prep list that says how many of each dish to cook, with a confidence band.

* **Order parsing**: Qwen3.5-4B LoRA fine-tuned on [Tinker](https://thinkingmachines.ai/tinker/) reads
  Hinglish messages like `kal 2 veg thali lunch, no onion` or `parso ka dabba cancel` and resolves
  relative dates against the WhatsApp timestamp.
* **Demand forecast**: [TabPFN](https://priorlabs.ai) (Prior Labs' tabular foundation model) predicts
  tomorrow's final count per dish from the kitchen's history and the orders already booked by 9 pm.
* **Kitchen memory**: [Backboard](https://backboard.io) remembers each customer's usual dish and prefs,
  so a bare "1 dabba" from a paneer regular becomes a paneer thali.
* **Hosted on [Render](https://render.com)**.

```bash
uv venv --python 3.12 && uv pip install -r requirements-train.txt
cp .env.example .env                      # TINKER_API_KEY, BACKBOARD_API_KEY, TABPFN_TOKEN
.venv/bin/python data/simulate.py         # kitchen simulator -> messages_*.jsonl + history.csv
.venv/bin/python train/sft_tinker.py      # LoRA on Tinker -> DABBA_SAMPLER_PATH
.venv/bin/python train/eval.py --systems base,big,lora
.venv/bin/python forecast/forecast.py --mode client
PARSER_BACKEND=tinker .venv/bin/uvicorn app.main:app --reload
```

Results live in `results/` and are reproduced in the DEV post (`POST.md`).

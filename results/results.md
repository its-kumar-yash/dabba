# Dabba parser eval: 200 held-out WhatsApp messages

| System | valid | action | customer | date | meal | items | prefs | time | repeat | all | p50 ms | $/1k msgs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B zero-shot | 72% | 56% | 72% | 56% | 68% | 53% | 67% | 66% | 71% | 30% | 2104 | $0.522 |
| gpt-oss-120b zero-shot | 98% | 96% | 98% | 50% | 88% | 86% | 96% | 96% | 98% | 38% | 2760 | $0.555 |
| Qwen3.5-4B + Dabba LoRA | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 99% | 100% | 98% | 2310 | $0.494 |

## Items accuracy on messages that needed a kitchen-memory hint (bare 'dabba' from a non-veg-thali regular)
- Qwen3.5-4B zero-shot: 51%
- gpt-oss-120b zero-shot: 69%
- Qwen3.5-4B + Dabba LoRA: 100%

## All-fields accuracy by intent
| intent | base | big | lora |
|---|---|---|---|
| cancel | 33% | 67% | 89% |
| modify | 38% | 38% | 100% |
| order | 37% | 40% | 98% |
| pause | 6% | 19% | 100% |
| query | 22% | 34% | 100% |
| resume | 0% | 33% | 100% |
| subscribe | 0% | 50% | 75% |

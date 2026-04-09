# Phi-4-mini-instruct — GSM8K eval comparison

| Configuration | KV compression | Samples | Accuracy | No answer | Avg latency | Throughput | Avg VRAM | Avg new tokens |
|---|---|---|---|---|---|---|---|---|
| fp16 base | 1.00x | 50 | 90.0% (45) | 0 | 2.75s | 69.8 tok/s | 7.72 GB | 192 |
| TQ 4-bit | 4.00x | 50 | 80.0% (40) | 2 | 5.47s | 33.7 tok/s | 7.93 GB | 184 |
| TQ 3-bit | 5.33x | 50 | 0.0% (0) | 46 | 7.86s | 34.9 tok/s | 8.06 GB | 274 |

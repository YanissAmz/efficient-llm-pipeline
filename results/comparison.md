# Phi-4-mini-instruct — GSM8K eval comparison

| Configuration | KV compression | Samples | Accuracy | No answer | Avg latency | Throughput | Avg VRAM | Avg new tokens |
|---|---|---|---|---|---|---|---|---|
| fp16 base | 1.00x | 50 | 90.0% (45) | 0 | 2.75s | 69.8 tok/s | 7.72 GB | 192 |
| TQ 4-bit | 4.00x | 50 | 80.0% (40) | 2 | 5.47s | 33.7 tok/s | 7.93 GB | 184 |
| TQ 3-bit | 5.33x | 50 | 0.0% (0) | 46 | 7.86s | 34.9 tok/s | 8.06 GB | 274 |
| fp16 base + LoRA-v1 | 1.00x | 50 | 76.0% (38) | 0 | 1.43s | 69.5 tok/s | 7.71 GB | 99 |
| TQ 4-bit + LoRA-v1 | 4.00x | 50 | 54.0% (27) | 3 | 3.37s | 34.5 tok/s | 7.89 GB | 116 |
| TQ 3-bit + LoRA-v1 | 5.33x | 50 | 0.0% (0) | 48 | 13.48s | 34.3 tok/s | 8.15 GB | 462 |
| fp16 base + LoRA-v2 | 1.00x | 50 | 76.0% (38) | 0 | 1.48s | 69.2 tok/s | 7.71 GB | 103 |
| TQ 4-bit + LoRA-v2 | 4.00x | 50 | 60.0% (30) | 2 | 3.38s | 34.5 tok/s | 7.89 GB | 117 |
| TQ 3-bit + LoRA-v2 | 5.33x | 50 | 0.0% (0) | 49 | 10.32s | 34.2 tok/s | 8.13 GB | 353 |

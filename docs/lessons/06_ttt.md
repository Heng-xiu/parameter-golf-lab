# Lesson 6: Score-First LoRA TTT

Run:

```bash
python scripts/run_ttt.py --checkpoint outputs/baseline/checkpoint.pt --config configs/ttt_lora.yaml
```

Expected output: per-chunk logs with `scored_before_update=true` plus aggregate no-TTT and TTT losses.

The rule is strict: score the current chunk before updating LoRA on that chunk. Updating first would leak information into the metric.

# Lesson 2: Progressive Context

Run:

```bash
python scripts/train.py --config configs/progressive_context.yaml
```

Expected output: logs where `seq_len` grows from `32` to `64` to `128`.

Long context costs more compute. The schedule teaches the model short contexts first and spends later steps on longer examples.

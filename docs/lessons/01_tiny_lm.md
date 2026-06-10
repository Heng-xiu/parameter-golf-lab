# Lesson 1: Tiny Causal LM

Run:

```bash
python scripts/make_tiny_corpus.py --out data/tiny.txt
python scripts/train.py --config configs/baseline.yaml
```

Expected output: training lines with `loss=...` and `bpb=...`, followed by `saved=outputs/baseline/checkpoint.pt`.

This lesson connects byte tokenization, shifted targets, causal attention, cross-entropy, and bits per byte.

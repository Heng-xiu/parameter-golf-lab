# Lesson 3: GQA and KV Cost

Run:

```bash
python scripts/train.py --config configs/gqa.yaml
```

Expected output: normal training logs with a model using four query heads and two KV heads.

Grouped-query attention keeps many query heads but shares fewer key/value heads. This reduces KV-cache size in long-context inference.

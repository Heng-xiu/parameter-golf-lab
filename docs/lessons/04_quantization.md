# Lesson 4: Quantization

Run:

```bash
python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8
python scripts/evaluate.py --checkpoint outputs/baseline/checkpoint.int8.pt
```

Expected output: artifact bytes and a loss delta.

This repo stores dequantized tensors in the teaching checkpoint so students can keep using ordinary PyTorch modules. Real low-bit inference needs specialized storage and kernels, which are outside the baseline.

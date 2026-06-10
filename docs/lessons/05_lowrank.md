# Lesson 5: Low-Rank Quantization-Error Correction

Run:

```bash
python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8-lowrank --rank 4
```

Expected output: `example_reconstruction_error int8=... corrected=...`.

The method quantizes a weight, computes the residual, and fits a small truncated-SVD approximation to that residual.

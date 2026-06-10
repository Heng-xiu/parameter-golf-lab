# PR Notes: Teaching Baseline Implementation

## Summary

Implements `mini-parameter-golf` as a small, typed, tested PyTorch teaching repository inspired by Parameter Golf and PR #2014's progressive context-growth lineage.

## Completed Phases

- Phase 0: repo skeleton, dependencies, `AGENTS.md`, Makefile, tests, linting, type checks.
- Phase 1: byte tokenizer, tiny corpus generator, dataset batcher, tiny causal Transformer, training/evaluation loops, checkpointing, BPB.
- Phase 2: MHA/GQA/MQA attention support, RoPE, progressive context schedule parser and training integration.
- Phase 3: model-size accounting, tensor/per-channel int8 quantization, fake int4 path, quantization CLI.
- Phase 4: simplified SVD low-rank quantization-error correction.
- Phase 5: LoRA injection helpers and score-first test-time training loop.
- Phase 6: README, six lesson docs, and experiment matrix.

## Skipped or Intentionally Simplified

- No CUDA-only kernels, FlashAttention, Triton, distributed training, Muon, GPTQ, full LQER, or competition-specific compression.
- Quantized checkpoints are teaching artifacts that dequantize weights for normal PyTorch forward passes.
- Low-rank correction is truncated SVD on residual weights, not a full LQER reproduction.
- TTT is document-local LoRA adaptation for demonstrating score-before-update behavior, not leaderboard tuning.

## Known Limitations

- The default tiny corpus is synthetic and meant for smoke tests, not meaningful language quality.
- TTT aggregate loss may match no-TTT on short smoke runs because scoring intentionally happens before updates.
- The checkpoint artifact can be larger after quantization because this baseline stores metadata/dequantized tensors for readability.
- `make setup` defaults to `/opt/homebrew/bin/python3.11`; override with `SYSTEM_PYTHON=/path/to/python3.11 make setup` on other machines.

## Verification

Ran successfully:

```bash
make setup
make format
make lint
make typecheck
make test
make smoke
python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8
python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8-lowrank --rank 4
python scripts/run_ttt.py --checkpoint outputs/baseline/checkpoint.pt --config configs/ttt_lora.yaml
```

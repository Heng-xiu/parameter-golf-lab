# AGENTS.md

This file defines rules for coding agents working on `mini-parameter-golf`.

## Mission

Build a teaching-oriented mini language-modeling repository inspired by Parameter Golf and PR #2014's progressive context-growth lineage.

The repo must be understandable, runnable, typed, tested, and useful for hands-on tutorials. Prefer clear educational code over leaderboard-style optimization.

## Operating Rules

1. Work in small, reviewable commits.
2. Keep the repo runnable after every phase.
3. Do not introduce CUDA-only requirements.
4. Do not vendor large datasets or model checkpoints.
5. Do not implement advanced competition features until the baseline, tests, and docs are stable.
6. Preserve deterministic smoke tests.
7. Keep all public APIs typed.
8. Add or update tests for every new feature.
9. Add docs when adding a teaching concept.
10. When making trade-offs, choose readability and correctness over speed.

## Required Local Commands

Before marking work complete, run:

```bash
make format
make lint
make typecheck
make test
make smoke
```

If a command fails, fix the issue or document the exact failure and blocker.

## Python Standards

- Python: 3.11+
- Package source: `src/mini_pgolf`
- Test framework: `pytest`
- Formatting: `black`
- Linting: `ruff`
- Type checking: `mypy`
- Tensor library: `torch`

## Code Style

- Use explicit tensor shape comments where helpful:
  - `x: Float[batch, seq, d_model]`
  - `q: Float[batch, n_heads, seq, head_dim]`
- Avoid implicit global state.
- Pass config objects explicitly.
- Prefer dataclasses for static configuration.
- Do not hide educational logic behind too many abstractions.
- Avoid notebook-only implementation; notebooks/docs may call the package.

## Testing Rules

Every major component needs a unit test:

- tokenizer round trip;
- causal mask correctness;
- MHA/GQA shape correctness;
- RoPE shape preservation;
- model forward shape;
- progressive context schedule;
- quantize/dequantize error bound sanity;
- low-rank correction improves reconstruction on a toy matrix;
- LoRA parameters are the only trainable parameters during TTT;
- score-first TTT does not update before scoring the current chunk.

## Documentation Rules

For each lesson doc, include:

1. learning objective;
2. concept overview;
3. command to run;
4. expected output;
5. common failure modes;
6. extension exercise.

## Safety and Data Policy

Use only synthetic data or small public-domain sample text in the default repo.
Do not commit private datasets, API keys, generated model checkpoints, or cached Hugging Face data.
Add `.gitignore` entries for outputs, checkpoints, logs, and cache directories.

## Scope Control

Implement the simplified teaching equivalents first:

| Upstream concept | Teaching equivalent |
|---|---|
| SP8192 CaseOps tokenizer | byte or char tokenizer |
| 11-layer 512d Transformer | 4-layer 256d Transformer |
| FlashAttention 3 | standard PyTorch attention |
| GPTQ int6/int7 | naive int8, per-channel int8, fake int4 |
| LQER | truncated-SVD low-rank error correction |
| phased TTT | single-pass score-first LoRA TTT |
| per-group lrzip compression | zip/gzip/brotli artifact-size reporting |
| Muon | AdamW |

Do not add full GPTQ, FlashAttention, Triton kernels, Muon, or large-data CaseOps prep unless the educational baseline is already complete.

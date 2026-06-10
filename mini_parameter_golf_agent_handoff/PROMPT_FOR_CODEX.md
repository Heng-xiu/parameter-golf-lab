# Master Prompt for Coding Agent: Mini Parameter Golf Teaching Repo

You are implementing a teaching-oriented repository named `mini-parameter-golf`.

The repository is inspired by OpenAI's Parameter Golf challenge and especially the PR #2014 lineage: progressive context growth, a compact Transformer backbone, grouped-query attention, quantization-aware compression experiments, and optional score-first test-time LoRA adaptation.

This is **not** a goal to reproduce the upstream leaderboard result. The upstream PR #2014 was a competition artifact using large hardware, FlashAttention 3, CaseOps SP8192 data, GPTQ int6/int7, LQER, per-group compression, and score-first phased TTT. This teaching repo must instead be:

- small enough for students to run on a laptop GPU, Colab, or modest single-GPU workstation;
- cleanly typed and tested;
- modular enough for classroom exercises;
- faithful to the engineering ideas, not to every competition-specific optimization;
- easy for a coding agent to extend phase by phase.

## Primary Goal

Build a compact educational language-modeling repository that teaches:

1. tiny causal Transformer implementation;
2. tokenizer/data pipeline basics;
3. bits-per-byte style evaluation;
4. model-size accounting;
5. progressive context growth;
6. GQA/MQA as long-context KV-cost control;
7. quantization from naive int8 to per-channel int8 and fake int4;
8. low-rank quantization-error correction;
9. score-first test-time LoRA adaptation;
10. reproducible repo hygiene: tests, type checks, configs, CI, and docs.

## Non-goals

Do not attempt to reproduce PR #2014's exact score.
Do not require 8xH100, FlashAttention 3, lrzip, custom CUDA kernels, Triton kernels, or Hugging Face CaseOps data.
Do not implement full GPTQ, full LQER, full per-group compression, SmearGate, SparseAttnGate, or Muon in the initial version.
Do not optimize for leaderboard performance before the teaching path is complete.
Do not hide complexity inside a monolithic `train_gpt.py`.

## Expected Repository Output

Implement the repo with this top-level structure:

```text
mini-parameter-golf/
  AGENTS.md
  README.md
  pyproject.toml
  requirements.txt
  requirements-dev.txt
  Makefile
  configs/
    baseline.yaml
    progressive_context.yaml
    gqa.yaml
    quant_int8.yaml
    quant_lowrank.yaml
    ttt_lora.yaml
  src/
    mini_pgolf/
      __init__.py
      config.py
      data.py
      tokenizer.py
      model.py
      attention.py
      train.py
      evaluate.py
      metrics.py
      size_report.py
      quantization.py
      lowrank.py
      lora.py
      ttt.py
      checkpoint.py
      utils.py
  scripts/
    train.py
    evaluate.py
    quantize.py
    run_ttt.py
    report_size.py
    make_tiny_corpus.py
  tests/
    test_tokenizer.py
    test_attention.py
    test_model_shapes.py
    test_progressive_context.py
    test_quantization.py
    test_lora_ttt.py
    test_size_report.py
  docs/
    00_project_brief.md
    01_architecture.md
    02_implementation_plan.md
    03_tutorial_plan.md
    04_quality_checks.md
    05_dependency_policy.md
    06_experiment_matrix.md
```

## Implementation Priorities

Implement in phases. Do not jump directly to advanced features.

### Phase 0: Repo Skeleton and Checks

Create the package skeleton, `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `Makefile`, and `AGENTS.md`.

Add commands:

```bash
make setup
make format
make lint
make typecheck
make test
make smoke
```

Use `ruff`, `black`, `mypy`, and `pytest`.

### Phase 1: Baseline Tiny LM

Implement:

- tiny byte or character tokenizer;
- tiny corpus generator;
- causal Transformer with RMSNorm, RoPE, tied embeddings, MLP, and causal attention;
- config dataclasses or Pydantic models;
- train loop;
- eval loop;
- cross-entropy and BPB metrics;
- checkpoint save/load;
- deterministic smoke test.

Target model:

```yaml
vocab_size: 256
n_layers: 4
d_model: 256
n_heads: 4
n_kv_heads: 4
mlp_ratio: 4
max_seq_len: 512
dropout: 0.0
tie_embeddings: true
```

### Phase 2: GQA and Progressive Context Growth

Implement:

- attention with `n_heads >= n_kv_heads`;
- repeat/interleave KV heads to query-head count;
- progressive context schedule, for example:
  - `128@0.20`
  - `256@0.60`
  - `512@1.00`
- training loop that samples or crops batches according to the active sequence length;
- logs that report current progress, current sequence length, tokens/sec, loss, BPB, and model size.

### Phase 3: Model Size Accounting and Compression

Implement:

- exact parameter count;
- fp32/fp16/bfloat16 estimated storage;
- state dict byte count;
- artifact zip byte count;
- naive symmetric int8 quantization;
- per-channel int8 quantization for linear weights;
- fake int4 quantization for teaching;
- loss-before/loss-after quantization comparison.

### Phase 4: Low-Rank Quantization Error Correction

Implement a simplified LQER-like teaching module:

```text
W_quant = dequantize(quantize(W))
E = W - W_quant
E ≈ A @ B
W_effective = W_quant + A @ B
```

Use truncated SVD or `torch.linalg.svd` on small matrices. The implementation is for pedagogy, not for production-scale GPTQ.

### Phase 5: Score-First Test-Time LoRA

Implement a minimal test-time adaptation loop:

1. Freeze the base model.
2. Attach LoRA modules to selected projections.
3. For each document:
   - reset LoRA weights;
   - score a chunk first without updating;
   - then update LoRA on that same chunk;
   - use updated LoRA for later chunks.
4. Add ablations:
   - no TTT;
   - LoRA on Q/V;
   - LoRA on K/O/MLP;
   - LoRA on all projections.

Default teaching setting should use `no_qv`, inspired by PR #2014.

### Phase 6: Tutorial Docs and Classroom Exercises

Add runnable tutorials:

- `docs/lesson_01_tiny_lm.md`
- `docs/lesson_02_progressive_context.md`
- `docs/lesson_03_gqa_kv_cost.md`
- `docs/lesson_04_quantization.md`
- `docs/lesson_05_lowrank_error_correction.md`
- `docs/lesson_06_score_first_ttt.md`

Each lesson must contain:

- learning objective;
- concept overview;
- command to run;
- expected output;
- discussion questions;
- extension exercise.

## Acceptance Criteria

The repo is acceptable when:

1. `make format lint typecheck test` passes.
2. `make smoke` trains for a few iterations on a tiny synthetic corpus.
3. `scripts/report_size.py` reports parameter count and estimated artifact size.
4. `scripts/quantize.py` can quantize a checkpoint and report loss change.
5. `scripts/run_ttt.py` can run score-first LoRA TTT on a tiny validation set.
6. README includes a clear 60-minute teaching path.
7. AGENTS.md gives coding-agent operating rules.
8. No hard dependency requires CUDA-only kernels.
9. CPU smoke tests pass in under a few minutes on a normal machine.
10. GPU acceleration is optional, not mandatory.

## Style Requirements

- Write Python 3.11+.
- Use PyTorch 2.3+.
- Use type hints throughout.
- Prefer small functions and explicit tensor shape comments.
- Avoid clever metaprogramming.
- Avoid monolithic scripts.
- Use deterministic seeds in tests.
- Keep docs clear for students who know basic PyTorch but not model compression.

## Final Deliverables

Open a pull request containing:

1. code implementation;
2. tests;
3. config files;
4. tutorial docs;
5. README;
6. AGENTS.md;
7. a short implementation summary describing completed phases, skipped items, and known limitations.

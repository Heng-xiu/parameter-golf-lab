# Implementation Plan

## Phase 0: Repository Foundation

### Deliverables

- package skeleton;
- `pyproject.toml`;
- `requirements.txt`;
- `requirements-dev.txt`;
- `Makefile`;
- `AGENTS.md`;
- empty tests that validate importability.

### Acceptance Criteria

- `python -m mini_pgolf` or package import works;
- `make lint`, `make typecheck`, and `make test` run;
- no CUDA-only dependency.

## Phase 1: Baseline Tiny LM

### Deliverables

- byte tokenizer;
- tiny synthetic corpus script;
- dataset and batch sampler;
- Transformer block;
- training loop;
- evaluation loop;
- checkpoint save/load;
- BPB metric.

### Acceptance Criteria

- `make smoke` trains for a few iterations;
- model forward pass shape is tested;
- train loss decreases on a tiny overfit sample.

## Phase 2: GQA and Progressive Context

### Deliverables

- GQA attention implementation;
- tests for MHA, GQA, and MQA shapes;
- progressive context schedule parser;
- training loop integration;
- config `progressive_context.yaml`.

### Acceptance Criteria

- schedule parser tests pass;
- logs show active sequence length;
- baseline and progressive-context configs both train.

## Phase 3: Size Accounting and Quantization

### Deliverables

- size report utility;
- artifact zip reporting;
- int8 tensor-wise quantization;
- per-channel int8 linear quantization;
- fake int4 quantization;
- quantization script.

### Acceptance Criteria

- quantize/dequantize tests pass;
- script reports original size, quantized size estimate, and loss delta;
- docs explain why low-bit storage and forward speed are separate topics.

## Phase 4: Low-Rank Error Correction

### Deliverables

- truncated-SVD low-rank correction;
- wrapper module for corrected linear weights;
- reconstruction test;
- config `quant_lowrank.yaml`.

### Acceptance Criteria

- low-rank correction reduces reconstruction error on toy matrix;
- loss-after-correction is reported separately from loss-after-quantization.

## Phase 5: Score-First LoRA TTT

### Deliverables

- LoRA linear module;
- injection helpers;
- trainable-parameter filtering;
- score-first document loop;
- target masks: `no_qv`, `qv_only`, `all`, `ko_mlp`;
- config `ttt_lora.yaml`.

### Acceptance Criteria

- tests prove LoRA is not updated before scoring the current chunk;
- `scripts/run_ttt.py` reports no-TTT vs TTT loss;
- docs warn about future-token leakage.

## Phase 6: Tutorial Documentation

### Deliverables

- six lesson docs;
- experiment matrix;
- README teaching path;
- troubleshooting section.

### Acceptance Criteria

- a student can follow the README path without reading source code first;
- each lesson has a runnable command and expected output;
- repo explains which upstream PR #2014 ideas are included, simplified, or omitted.

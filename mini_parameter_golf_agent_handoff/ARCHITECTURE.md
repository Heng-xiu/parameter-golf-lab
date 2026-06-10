# Architecture Specification

## System Overview

The repo is organized as a small package plus runnable scripts.

```text
src/mini_pgolf/
  config.py        # typed config dataclasses
  tokenizer.py     # byte/char tokenizer
  data.py          # dataset and batch sampling
  attention.py     # MHA/GQA attention, RoPE, causal masks
  model.py         # tiny causal Transformer
  train.py         # train loop functions
  evaluate.py      # evaluation functions
  metrics.py       # CE, BPB, perplexity helpers
  size_report.py   # parameter and artifact accounting
  quantization.py  # int8/per-channel/fake-int4 quantization
  lowrank.py       # low-rank error correction
  lora.py          # LoRA modules and injection helpers
  ttt.py           # score-first test-time LoRA loop
  checkpoint.py    # save/load helpers
  utils.py         # seeds, devices, logging
```

## Baseline Model

Default baseline:

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
norm: rmsnorm
positional_encoding: rope
```

## Attention

Support:

- standard multi-head attention: `n_heads == n_kv_heads`;
- grouped-query attention: `n_heads > n_kv_heads`;
- optional multi-query attention: `n_kv_heads == 1`.

Shape contract:

```text
input x:        [batch, seq, d_model]
q projection:  [batch, seq, n_heads, head_dim]
k projection:  [batch, seq, n_kv_heads, head_dim]
v projection:  [batch, seq, n_kv_heads, head_dim]
attention q:   [batch, n_heads, seq, head_dim]
attention k/v: [batch, n_heads, seq, head_dim] after KV repeat
output:        [batch, seq, d_model]
```

## Progressive Context Growth

Implement a schedule parser:

```text
128@0.20,256@0.60,512@1.00
```

Interpretation:

- use sequence length 128 until progress < 0.20;
- use 256 until progress < 0.60;
- use 512 until progress <= 1.00.

Progress may be based on step count in the teaching repo.

## Quantization

Implement progressively:

1. tensor-wise symmetric int8;
2. per-channel symmetric int8 for linear weights;
3. fake int4;
4. low-rank residual correction.

Quantized modules do not need custom kernels. It is acceptable to dequantize for forward pass in the teaching version.

## Low-Rank Error Correction

For a weight matrix `W`:

```text
Wq = dequantize(quantize(W))
E = W - Wq
E ≈ A @ B
Weff = Wq + A @ B
```

Use truncated SVD in the teaching implementation.

## Score-First TTT

The TTT loop must guarantee:

1. score current chunk before updating LoRA;
2. update LoRA only after scoring;
3. reset LoRA state per document;
4. log per-document and aggregate loss;
5. allow adapter target masks:
   - `no_qv`
   - `qv_only`
   - `all`
   - `ko_mlp`

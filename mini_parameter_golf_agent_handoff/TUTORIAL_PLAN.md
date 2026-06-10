# Tutorial Plan

## Lesson 1: Build a Tiny Causal LM

### Objective

Students implement or inspect a minimal causal Transformer and train it on a tiny corpus.

### Concepts

- tokenization;
- input-target shift;
- causal attention;
- cross-entropy;
- BPB.

### Command

```bash
python scripts/make_tiny_corpus.py --out data/tiny.txt
python scripts/train.py --config configs/baseline.yaml
```

### Discussion

- Why is BPB useful when tokenizer choices differ?
- What changes when the vocabulary is byte-level?

## Lesson 2: Add Progressive Context Growth

### Objective

Students compare fixed context length with progressive context length.

### Concepts

- context length as a compute budget;
- curriculum over sequence length;
- long-context adaptation.

### Command

```bash
python scripts/train.py --config configs/progressive_context.yaml
```

### Discussion

- Why not train at the longest context from step 1?
- What metrics show whether longer context helped?

## Lesson 3: GQA and KV-Cost Control

### Objective

Students compare MHA, GQA, and MQA.

### Concepts

- query heads vs KV heads;
- KV cache;
- long-context inference cost;
- shape transformations in attention.

### Command

```bash
python scripts/train.py --config configs/gqa.yaml
```

### Discussion

- What is lost when reducing KV heads?
- Why do long-context models care about KV cache?

## Lesson 4: Quantization

### Objective

Students quantize a checkpoint and measure size/loss trade-offs.

### Concepts

- symmetric int8;
- per-channel quantization;
- fake int4;
- dequantized forward pass;
- quantization error.

### Command

```bash
python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8
python scripts/evaluate.py --checkpoint outputs/baseline/checkpoint.int8.pt
```

### Discussion

- Why can storage shrink while inference speed does not improve?
- Which tensors are most sensitive to quantization?

## Lesson 5: Low-Rank Error Correction

### Objective

Students repair quantization error with a low-rank residual.

### Concepts

- quantization residual;
- truncated SVD;
- low-rank approximation;
- parameter-size trade-off.

### Command

```bash
python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8-lowrank --rank 4
```

### Discussion

- Why can a small low-rank update recover disproportionate loss?
- How does rank affect storage?

## Lesson 6: Score-First Test-Time LoRA

### Objective

Students apply test-time adaptation without leaking future-token information.

### Concepts

- frozen base model;
- LoRA adapters;
- score-before-update;
- document-local adaptation;
- target mask ablations.

### Command

```bash
python scripts/run_ttt.py --checkpoint outputs/baseline/checkpoint.pt --config configs/ttt_lora.yaml
```

### Discussion

- Why must scoring happen before updating?
- Why might `no_qv` be more stable than adapting every projection?

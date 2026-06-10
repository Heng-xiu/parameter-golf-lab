# mini-parameter-golf

A teaching-oriented mini language-modeling repository inspired by Parameter Golf.

This repo helps students build a tiny causal Transformer, train it under context and size constraints, compress it, and test score-first LoRA adaptation.

## What Students Learn

- Causal language modeling
- Bits-per-byte evaluation
- Progressive context growth
- Grouped-query attention
- Model size accounting
- Quantization
- Low-rank error correction
- Score-first test-time training

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
make setup
make smoke
```

## Run the Baseline

```bash
python scripts/make_tiny_corpus.py --out data/tiny.txt
python scripts/train.py --config configs/baseline.yaml
python scripts/evaluate.py --checkpoint outputs/baseline/checkpoint.pt
python scripts/report_size.py --checkpoint outputs/baseline/checkpoint.pt
```

## Run Progressive Context Growth

```bash
python scripts/train.py --config configs/progressive_context.yaml
```

## Quantize a Checkpoint

```bash
python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8
```

## Run Score-First TTT

```bash
python scripts/run_ttt.py --checkpoint outputs/baseline/checkpoint.pt --config configs/ttt_lora.yaml
```

## Teaching Path

1. Lesson 1: Tiny LM
2. Lesson 2: Progressive context
3. Lesson 3: GQA and KV cost
4. Lesson 4: Quantization
5. Lesson 5: Low-rank error correction
6. Lesson 6: Score-first TTT

## Upstream Inspiration

The original Parameter Golf challenge targets a 16MB language-model artifact and bits-per-byte validation. PR #2014 demonstrated that progressive context growth and score-first TTT can be powerful in a compact model. This repo adapts those ideas into a smaller, more teachable system.

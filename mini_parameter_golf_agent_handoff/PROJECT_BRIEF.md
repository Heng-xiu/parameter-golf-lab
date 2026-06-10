# Project Brief: Mini Parameter Golf

## One-Sentence Description

`mini-parameter-golf` is a small PyTorch teaching repo that helps students build, train, compress, and adapt a tiny causal language model under explicit size and context-length constraints.

## Inspiration

The project is inspired by OpenAI's Parameter Golf challenge, where the goal is to train a language model that fits in a 16MB artifact and is evaluated by tokenizer-agnostic bits per byte. It is also inspired by the PR #2014 lineage, which used progressive context growth and score-first test-time training to improve a compact model.

This repo is not a leaderboard reproduction. It is a teaching adaptation.

## Teaching Goals

Students should learn:

1. how a causal Transformer is assembled;
2. why artifact size matters;
3. how BPB differs from token-level loss;
4. why long-context training is a compute-budget problem;
5. how GQA reduces KV cost;
6. how quantization changes model quality and storage;
7. how low-rank correction can repair quantization error;
8. how score-first TTT avoids future-token leakage;
9. how to measure trade-offs instead of relying on intuition.

## Target Audience

Students should know:

- basic Python;
- basic PyTorch tensors and autograd;
- the idea of language modeling;
- simple training loops.

They do not need prior experience with GPTQ, LQER, FlashAttention, Muon, or distributed training.

## Hardware Target

Default experience:

- CPU smoke tests;
- laptop or Colab GPU for short training;
- no distributed training required.

Optional:

- single CUDA GPU for faster experiments.

## Success Definition

A student should be able to run:

```bash
make setup
make smoke
python scripts/train.py --config configs/baseline.yaml
python scripts/evaluate.py --checkpoint outputs/baseline/checkpoint.pt
python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8
python scripts/run_ttt.py --checkpoint outputs/baseline/checkpoint.pt --config configs/ttt_lora.yaml
```

and understand the loss, BPB, model size, compression ratio, and TTT behavior.

# mini-parameter-golf

A teaching-oriented mini language-modeling repository inspired by Parameter Golf and PR #2014's progressive context-growth lineage.

This is not a leaderboard reproduction. It is a small, typed, tested PyTorch repo for learning how compact language models are trained, measured, compressed, and adapted.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
make setup
make smoke
```

Expected smoke output includes `loss=...`, `bpb=...`, a saved checkpoint, and a size report.

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

The training log includes `seq_len=...`, showing the active context length.

## Quantize a Checkpoint

```bash
python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8
python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8-lowrank --rank 4
```

The teaching quantizer dequantizes for forward passes. It teaches storage/error trade-offs, not inference-speed kernels.

## Run Score-First TTT

```bash
python scripts/run_ttt.py --checkpoint outputs/baseline/checkpoint.pt --config configs/ttt_lora.yaml
```

Logs state `scored_before_update=true` for each chunk. This is the key anti-leakage rule: score the current chunk before adapting on it.

## Optional Hugging Face Chat Corpus

```bash
python -m pip install -r requirements-hf.txt
python scripts/export_hf_chat_corpus.py --out data/hf_chat.txt --max-examples 1200
python scripts/train.py --config configs/hf_chat_mps.yaml --device mps
python scripts/chat.py --checkpoint outputs/hf_chat_mps/checkpoint.pt --device mps --allowed-text data/hf_chat.txt --chat-template --stop-at-user
```

## Visual Tutorial Site

```bash
make site-setup
make site-build
make site-test
make site-dev
```

Then open the Vite URL printed by `make site-dev`.

## Teaching Path

1. [Lesson 1: Tiny LM](docs/lessons/01_tiny_lm.md)
2. [Lesson 2: Progressive Context](docs/lessons/02_progressive_context.md)
3. [Lesson 3: GQA](docs/lessons/03_gqa.md)
4. [Lesson 4: Quantization](docs/lessons/04_quantization.md)
5. [Lesson 5: Low-Rank Correction](docs/lessons/05_lowrank.md)
6. [Lesson 6: Score-First TTT](docs/lessons/06_ttt.md)
7. [Lesson 7: Chat Quality](docs/lessons/07_chat_quality.md)

See [Experiment Matrix](docs/experiment_matrix.md) for suggested comparisons.

## Included, Simplified, Omitted

Included: byte-level BPB, tiny causal Transformer, GQA/MQA shape mechanics, progressive context schedules, size accounting, int8/fake-int4 quantization, SVD residual correction, score-first LoRA TTT.

Simplified: no custom quantized kernels, no full GPTQ, no full LQER, no competition tokenizer tricks, no leaderboard artifact pipeline.

Omitted for the baseline: CUDA-only kernels, FlashAttention, Triton, distributed training, Muon, and competition-specific compression.

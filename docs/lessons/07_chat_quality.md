# Lesson 7: Why Chat Needs Dialogue Data

FineWeb teaches general text continuation. It does not teach a model to respond as an assistant. For chat behavior, train on examples that use the same prompt format you will use at inference time.

Create a tiny dialogue corpus:

```bash
python scripts/make_tiny_chat_corpus.py --out data/tiny_chat.txt
```

Train on Apple Silicon MPS:

```bash
python scripts/train.py --config configs/chat_tiny.yaml --device mps --max-steps 1200
```

Chat with the dialogue prompt template:

```bash
python scripts/chat.py \
  --checkpoint outputs/chat_tiny_mps/checkpoint.pt \
  --device mps \
  --allowed-text data/tiny_chat.txt \
  --chat-template \
  --stop-at-user
```

Expected result: the model should produce replies that resemble the tiny training answers. It still will not be a general assistant, because the corpus is tiny and repetitive.

To improve quality further:

- add more varied `User:` and `Assistant:` examples;
- increase `d_model`, `n_layers`, and `max_seq_len`;
- train longer;
- hold out some dialogue examples for evaluation;
- avoid judging chat quality from FineWeb-only checkpoints.

## Optional: Hugging Face Chat Data

The tiny built-in corpus is only for proving the format. For a better classroom demo, export a small Hugging Face chat dataset:

```bash
python -m pip install -r requirements-hf.txt
python scripts/export_hf_chat_corpus.py \
  --dataset HuggingFaceTB/everyday-conversations-llama3.1-2k \
  --split train_sft \
  --out data/hf_chat.txt \
  --max-examples 1200
```

Train a larger mini model:

```bash
python scripts/train.py --config configs/hf_chat_mps.yaml --device mps
```

Chat with the result:

```bash
python scripts/chat.py \
  --checkpoint outputs/hf_chat_mps/checkpoint.pt \
  --device mps \
  --allowed-text data/hf_chat.txt \
  --chat-template \
  --stop-at-user
```

This is still a tiny byte-level model. Expect simple, short replies, not frontier assistant quality.

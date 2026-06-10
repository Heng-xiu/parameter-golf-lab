# Quality Checks and Type-Checking Policy

## Required Tools

Use:

- `black` for formatting;
- `ruff` for linting;
- `mypy` for static type checks;
- `pytest` for tests.

## Make Targets

```makefile
setup:
	python -m pip install -r requirements.txt -r requirements-dev.txt

format:
	black src tests scripts
	ruff check --fix src tests scripts

lint:
	ruff check src tests scripts

typecheck:
	mypy src

test:
	pytest -q

smoke:
	python scripts/make_tiny_corpus.py --out data/tiny.txt
	python scripts/train.py --config configs/baseline.yaml --max-steps 20
	python scripts/evaluate.py --checkpoint outputs/baseline/checkpoint.pt
```

## mypy Policy

Start with a practical strictness level:

```toml
[tool.mypy]
python_version = "3.11"
packages = ["mini_pgolf"]
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_return_any = true
warn_unused_ignores = true
```

Allow carefully scoped exceptions for PyTorch-heavy code only if needed.

## ruff Policy

Suggested rules:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "C4", "ARG"]
ignore = ["E501"]
```

## Test Matrix

Required tests:

| Area | Required test |
|---|---|
| Tokenizer | encode/decode round trip |
| Data | batch shape and target shift |
| Attention | MHA/GQA/MQA output shapes |
| RoPE | shape preservation and deterministic output |
| Model | forward logits shape |
| Metrics | BPB sanity check |
| Context schedule | boundary values |
| Quantization | dequantized tensor shape and error sanity |
| Low-rank correction | reconstruction error improves |
| LoRA | only LoRA parameters are trainable |
| TTT | score-first update ordering |

## CI Recommendation

Add GitHub Actions later:

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install -r requirements.txt -r requirements-dev.txt
      - run: make lint
      - run: make typecheck
      - run: make test
```

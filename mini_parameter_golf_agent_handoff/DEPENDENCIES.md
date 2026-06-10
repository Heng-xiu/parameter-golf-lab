# Dependency and Version Policy

## Runtime Dependencies

Use conservative, common dependencies:

```text
python>=3.11
torch>=2.3,<2.6
numpy>=1.26,<3
pyyaml>=6.0,<7
tqdm>=4.66,<5
typing_extensions>=4.10
```

Optional:

```text
safetensors>=0.4,<0.5
brotli>=1.1,<2
```

Do not require:

- FlashAttention;
- Triton;
- CUDA custom extensions;
- lrzip;
- distributed training;
- Hugging Face datasets;
- SentencePiece;
- Weights & Biases.

These can be added later behind optional extras, but not in the teaching baseline.

## Development Dependencies

```text
black>=24.0,<25
ruff>=0.5,<1
mypy>=1.10,<2
pytest>=8.0,<9
pytest-cov>=5.0,<6
```

## Suggested `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mini-parameter-golf"
version = "0.1.0"
description = "A teaching-oriented mini Parameter Golf repository."
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "C4", "ARG"]
ignore = ["E501"]

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

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

## Dependency Philosophy

Dependencies should support teaching and reproducibility. Avoid dependencies that obscure model mechanics or make installation fragile.

The default path must run without GPU-specific packages.

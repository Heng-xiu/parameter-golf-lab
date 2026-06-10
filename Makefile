.PHONY: setup format lint typecheck test smoke site-setup site-build site-test site-dev

VENV ?= .venv
SYSTEM_PYTHON ?= /opt/homebrew/bin/python3.11
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
BLACK := $(VENV)/bin/black
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest

setup:
	$(SYSTEM_PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(PIP) install -e .

format:
	$(BLACK) src tests scripts
	$(RUFF) check --fix src tests scripts

lint:
	$(RUFF) check src tests scripts

typecheck:
	$(MYPY) src

test:
	$(PYTEST) -q

smoke:
	$(PYTHON) scripts/make_tiny_corpus.py --out data/tiny.txt
	$(PYTHON) scripts/train.py --config configs/baseline.yaml --max-steps 20
	$(PYTHON) scripts/evaluate.py --checkpoint outputs/baseline/checkpoint.pt
	$(PYTHON) scripts/report_size.py --checkpoint outputs/baseline/checkpoint.pt

site-setup:
	cd tutorial-site && npm install

site-build:
	cd tutorial-site && npm run build

site-test:
	cd tutorial-site && npm test

site-dev:
	cd tutorial-site && npm run dev

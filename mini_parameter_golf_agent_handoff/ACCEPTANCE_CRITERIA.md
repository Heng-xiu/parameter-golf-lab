# Acceptance Criteria

## Repository-Level Done Definition

The implementation is complete when:

- the repo installs from scratch;
- all tests pass;
- type checks pass;
- smoke training runs;
- baseline tutorial path works;
- quantization scripts produce a before/after report;
- TTT script demonstrates score-first behavior;
- docs explain all major concepts.

## Required Commands

```bash
make setup
make format
make lint
make typecheck
make test
make smoke
```

## Functional Acceptance Tests

### Baseline Training

```bash
python scripts/train.py --config configs/baseline.yaml --max-steps 50
```

Expected:

- loss is logged;
- BPB is logged;
- checkpoint is saved.

### Evaluation

```bash
python scripts/evaluate.py --checkpoint outputs/baseline/checkpoint.pt
```

Expected:

- cross-entropy is reported;
- BPB is reported.

### Size Report

```bash
python scripts/report_size.py --checkpoint outputs/baseline/checkpoint.pt
```

Expected:

- parameter count;
- fp32 estimated size;
- fp16 estimated size;
- actual checkpoint bytes.

### Quantization

```bash
python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8
```

Expected:

- quantized checkpoint or artifact is written;
- loss delta is reported if eval data is available.

### Low-Rank Correction

```bash
python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8-lowrank --rank 4
```

Expected:

- low-rank metadata is written;
- reconstruction error improves over int8-only on at least one tested matrix.

### TTT

```bash
python scripts/run_ttt.py --checkpoint outputs/baseline/checkpoint.pt --config configs/ttt_lora.yaml
```

Expected:

- no-TTT and TTT metrics are shown;
- logs state that each chunk was scored before update;
- LoRA reset occurs per document.

## Documentation Acceptance

Docs must explain:

- what was simplified from the upstream competition version;
- what each lesson teaches;
- how to run each experiment;
- expected output;
- known limitations.

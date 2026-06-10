# Experiment Matrix

| Experiment | Command | Watch |
|---|---|---|
| Baseline | `python scripts/train.py --config configs/baseline.yaml` | loss, BPB |
| GQA | `python scripts/train.py --config configs/gqa.yaml` | same quality with fewer KV heads |
| Progressive context | `python scripts/train.py --config configs/progressive_context.yaml` | `seq_len` log changes |
| Int8 quantization | `python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8` | size report and loss delta |
| Low-rank correction | `python scripts/quantize.py --checkpoint outputs/baseline/checkpoint.pt --method int8-lowrank --rank 4` | reconstruction error |
| TTT target mask | edit `configs/ttt_lora.yaml` target mask | no-TTT vs TTT loss |

Expected output is intentionally noisy on the tiny corpus. The goal is to see the measurement path, not to claim benchmark quality.

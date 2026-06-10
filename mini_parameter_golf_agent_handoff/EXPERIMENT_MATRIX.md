# Experiment Matrix

Use this file as the classroom experiment board.

## Baseline Experiments

| Experiment | Config | Variable | Expected lesson |
|---|---|---|---|
| B0 | `baseline.yaml` | default tiny LM | establish baseline loss/BPB |
| B1 | edited baseline | `n_layers=2,4,6` | depth vs size |
| B2 | edited baseline | `d_model=128,256,384` | width vs size |
| B3 | edited baseline | tied vs untied embeddings | storage vs quality |

## Context Experiments

| Experiment | Config | Variable | Expected lesson |
|---|---|---|---|
| C0 | fixed 128 | short context only | fast but limited |
| C1 | fixed 512 | long context from start | expensive early |
| C2 | progressive | 128→256→512 | context curriculum trade-off |

## Attention Experiments

| Experiment | Config | Variable | Expected lesson |
|---|---|---|---|
| A0 | MHA | `n_kv_heads=n_heads` | full KV capacity |
| A1 | GQA | `n_kv_heads=n_heads/2` | lower KV cost |
| A2 | MQA | `n_kv_heads=1` | maximum KV sharing |

## Compression Experiments

| Experiment | Config | Variable | Expected lesson |
|---|---|---|---|
| Q0 | int8 tensor-wise | one scale per tensor | simple but coarse |
| Q1 | int8 per-channel | one scale per output channel | better reconstruction |
| Q2 | fake int4 | lower bit width | more loss regression |
| Q3 | int8 + low-rank | rank 2/4/8 | residual correction trade-off |

## TTT Experiments

| Experiment | Config | Variable | Expected lesson |
|---|---|---|---|
| T0 | no TTT | frozen model | baseline |
| T1 | Q/V LoRA | adapt attention Q/V | may be unstable |
| T2 | K/O/MLP LoRA | `no_qv` style | often more stable |
| T3 | all LoRA | more degrees of freedom | risk of over-adaptation |

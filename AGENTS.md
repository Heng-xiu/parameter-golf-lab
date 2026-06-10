# AGENTS.md

This repository is a teaching adaptation of Parameter Golf ideas. Keep changes small, typed, tested, and CPU-smoke-testable.

## Rules

- Use the Markdown files in `mini_parameter_golf_agent_handoff/` as source-of-truth specs.
- Prefer readable PyTorch over clever kernels.
- Do not add CUDA-only dependencies, FlashAttention, Triton, distributed training, GPTQ, full LQER, Muon, or competition-specific compression in the baseline.
- Run `make format`, `make lint`, `make typecheck`, `make test`, and `make smoke` before calling work complete.
- Keep docs runnable: every lesson should include a command and expected output shape.

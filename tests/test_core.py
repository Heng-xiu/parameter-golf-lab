from __future__ import annotations

import torch

from mini_pgolf.attention import CausalSelfAttention, apply_rope
from mini_pgolf.config import ModelConfig, TTTConfig
from mini_pgolf.context import active_seq_len, parse_context_schedule
from mini_pgolf.data import ByteBatcher
from mini_pgolf.lora import freeze_non_lora, inject_lora, lora_parameters
from mini_pgolf.lowrank import lowrank_residual
from mini_pgolf.metrics import bpb_from_loss
from mini_pgolf.model import TinyCausalTransformer
from mini_pgolf.quantization import dequantize, quantize_per_channel, quantize_symmetric
from mini_pgolf.tokenizer import ByteTokenizer
from mini_pgolf.ttt import run_score_first_ttt


def test_tokenizer_round_trip() -> None:
    tok = ByteTokenizer()
    text = "hello byte golf"
    assert tok.decode(tok.encode(text)) == text


def test_batch_shape_and_shift() -> None:
    tokens = torch.arange(40)
    x, y = ByteBatcher(tokens, batch_size=3, seed=1).batch(8)
    assert x.shape == (3, 8)
    assert y.shape == (3, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_attention_modes_and_rope_shape() -> None:
    x = torch.randn(2, 8, 32)
    for kv_heads in (4, 2, 1):
        attn = CausalSelfAttention(d_model=32, n_heads=4, n_kv_heads=kv_heads)
        assert attn(x).shape == x.shape
    rope_in = torch.randn(2, 4, 8, 8)
    assert torch.equal(apply_rope(rope_in), apply_rope(rope_in))
    assert apply_rope(rope_in).shape == rope_in.shape


def test_model_forward_shape() -> None:
    model = TinyCausalTransformer(ModelConfig(d_model=32, n_layers=1, n_heads=4, n_kv_heads=2))
    logits = model(torch.randint(0, 256, (2, 12)))
    assert logits.shape == (2, 12, 256)


def test_bpb_sanity() -> None:
    assert bpb_from_loss(0.0) == 0.0
    assert bpb_from_loss(0.6931471805599453) == 1.0


def test_context_schedule_boundaries() -> None:
    stages = parse_context_schedule("32@0.20,64@0.60,128@1.00")
    assert active_seq_len(stages, 0.0) == 32
    assert active_seq_len(stages, 0.2) == 64
    assert active_seq_len(stages, 0.8) == 128
    assert active_seq_len(stages, 1.0) == 128


def test_quantization_shape_and_error() -> None:
    x = torch.randn(4, 8)
    q = quantize_symmetric(x)
    dq = dequantize(q)
    assert dq.shape == x.shape
    assert torch.mean((x - dq).abs()).item() < 0.02
    assert dequantize(quantize_per_channel(x)).shape == x.shape


def test_lowrank_reduces_reconstruction_error() -> None:
    torch.manual_seed(0)
    base = torch.randn(12, 8)
    weight = torch.round(base * 3) / 3 + 0.03 * torch.randn(12, 8)
    q_weight = dequantize(quantize_per_channel(weight))
    corrected, a, b = lowrank_residual(weight, rank=4)
    assert torch.linalg.matrix_norm(weight - (corrected + a @ b)) < torch.linalg.matrix_norm(
        weight - q_weight
    )


def test_lora_only_lora_parameters_trainable() -> None:
    model = TinyCausalTransformer(ModelConfig(d_model=32, n_layers=1, n_heads=4, n_kv_heads=2))
    assert inject_lora(model, rank=2, alpha=4.0, target_mask="qv_only") > 0
    freeze_non_lora(model)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable)
    assert len(lora_parameters(model)) == len(trainable)


def test_ttt_scores_before_update() -> None:
    tokens = torch.arange(0, 140, dtype=torch.long) % 64
    model = TinyCausalTransformer(
        ModelConfig(vocab_size=256, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=16)
    )
    cfg = TTTConfig(
        chunk_len=8, max_documents=2, target_mask="qv_only", steps_per_chunk=1, device="cpu"
    )
    result = run_score_first_ttt(model, tokens, cfg)
    assert result.chunks_scored_before_update > 0
    assert result.no_ttt_loss > 0
    assert result.ttt_loss > 0

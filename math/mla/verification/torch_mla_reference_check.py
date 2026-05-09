#!/usr/bin/env python3
"""Independent Torch reference checks for the local DeepSeek-V2 MLA variants."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "math" / "mla" / "deepseekv2-profile"
sys.path.insert(0, str(PROFILE))

from mla.impl import AttentionAbsorbed, AttentionBaseline, AttentionCacheCompressed  # noqa: E402


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    input_dtype = hidden.dtype
    x = hidden.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight.to(device=hidden.device, dtype=torch.float32) * x.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotary_tables(seq_len: int, dim: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_deepseek_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    cos_pos = cos[position_ids].unsqueeze(1)
    sin_pos = sin[position_ids].unsqueeze(1)
    bsz, heads, seq_len, dim = x.shape
    x_permuted = x.view(bsz, heads, seq_len, dim // 2, 2).transpose(4, 3).reshape(bsz, heads, seq_len, dim)
    return (x_permuted * cos_pos) + (rotate_half(x_permuted) * sin_pos)


def shape(tensor: torch.Tensor) -> list[int]:
    return list(tensor.shape)


def assert_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise AssertionError(f"{name} expected {expected}, got {actual}")


def reference_materialized(
    attn: torch.nn.Module,
    hidden_q_BLD: torch.Tensor,
    hidden_kv_BMD: torch.Tensor,
    q_position_ids_BL: torch.Tensor,
    kv_position_ids_BM: torch.Tensor,
) -> dict[str, torch.Tensor]:
    bsz, q_len, hidden_size = hidden_q_BLD.shape
    kv_len = hidden_kv_BMD.shape[1]
    h = attn.num_heads
    n = attn.qk_nope_head_dim
    r = attn.qk_rope_head_dim
    v = attn.v_head_dim
    q_rank = attn.q_lora_rank
    c_rank = attn.kv_lora_rank
    t_dim = attn.q_head_dim
    y_dim = n + v

    q_down_BLQ = F.linear(hidden_q_BLD, attn.q_a_proj.weight, attn.q_a_proj.bias)
    q_latent_BLQ = rms_norm(q_down_BLQ, attn.q_a_layernorm.weight)
    q_packed_BLT = F.linear(q_latent_BLQ, attn.q_b_proj.weight, attn.q_b_proj.bias)
    q_full_BHLT = q_packed_BLT.view(bsz, q_len, h, t_dim).transpose(1, 2)
    q_nope_BHLN, q_rope_BHLR = torch.split(q_full_BHLT, [n, r], dim=-1)

    kv_raw_BMX = F.linear(hidden_kv_BMD, attn.kv_a_proj_with_mqa.weight, attn.kv_a_proj_with_mqa.bias)
    compressed_kv_BMC, k_rope_raw_BMR = torch.split(kv_raw_BMX, [c_rank, r], dim=-1)
    compressed_kv_norm_BMC = rms_norm(compressed_kv_BMC, attn.kv_a_layernorm.weight)
    k_rope_B1MR = k_rope_raw_BMR.view(bsz, kv_len, 1, r).transpose(1, 2)

    max_position = int(torch.stack([q_position_ids_BL.max(), kv_position_ids_BM.max()]).max().item()) + 1
    cos, sin = rotary_tables(max_position, r, hidden_q_BLD.device, hidden_q_BLD.dtype)
    q_rope_after_rope_BHLR = apply_deepseek_rope(q_rope_BHLR, cos, sin, q_position_ids_BL)
    k_rope_after_rope_B1MR = apply_deepseek_rope(k_rope_B1MR, cos, sin, kv_position_ids_BM)

    kv_packed_BMY = F.linear(compressed_kv_norm_BMC, attn.kv_b_proj.weight, attn.kv_b_proj.bias)
    kv_full_BHMY = kv_packed_BMY.view(bsz, kv_len, h, y_dim).transpose(1, 2)
    k_nope_BHMN, value_BHMV = torch.split(kv_full_BHMY, [n, v], dim=-1)

    query_BHLT = torch.cat([q_nope_BHLN, q_rope_after_rope_BHLR], dim=-1)
    key_BHMT = torch.cat([k_nope_BHMN, k_rope_after_rope_B1MR.expand(bsz, h, kv_len, r)], dim=-1)
    scores_BHLM = torch.matmul(query_BHLT, key_BHMT.transpose(2, 3)) * float(attn.q_head_dim ** -0.5)
    scores_decomposed_BHLM = (
        torch.matmul(q_nope_BHLN, k_nope_BHMN.transpose(2, 3))
        + torch.matmul(q_rope_after_rope_BHLR, k_rope_after_rope_B1MR.transpose(2, 3))
    ) * float(attn.q_head_dim ** -0.5)
    weights_BHLM = torch.softmax(scores_BHLM, dim=-1, dtype=torch.float32).to(query_BHLT.dtype)
    context_BHLV = torch.matmul(weights_BHLM, value_BHMV)
    context_flat_BLHV = context_BHLV.transpose(1, 2).contiguous()
    context_flat_BLD = context_flat_BLHV.reshape(bsz, q_len, h * v)
    output_BLD = F.linear(context_flat_BLD, attn.o_proj.weight, attn.o_proj.bias)

    assert_shape("q_latent_BLQ", q_latent_BLQ, (bsz, q_len, q_rank))
    assert_shape("q_full_BHLT", q_full_BHLT, (bsz, h, q_len, t_dim))
    assert_shape("q_nope_BHLN", q_nope_BHLN, (bsz, h, q_len, n))
    assert_shape("q_rope_after_rope_BHLR", q_rope_after_rope_BHLR, (bsz, h, q_len, r))
    assert_shape("kv_raw_BMX", kv_raw_BMX, (bsz, kv_len, c_rank + r))
    assert_shape("compressed_kv_norm_BMC", compressed_kv_norm_BMC, (bsz, kv_len, c_rank))
    assert_shape("k_rope_after_rope_B1MR", k_rope_after_rope_B1MR, (bsz, 1, kv_len, r))
    assert_shape("kv_full_BHMY", kv_full_BHMY, (bsz, h, kv_len, y_dim))
    assert_shape("k_nope_BHMN", k_nope_BHMN, (bsz, h, kv_len, n))
    assert_shape("value_BHMV", value_BHMV, (bsz, h, kv_len, v))
    assert_shape("query_BHLT", query_BHLT, (bsz, h, q_len, t_dim))
    assert_shape("key_BHMT", key_BHMT, (bsz, h, kv_len, t_dim))
    assert_shape("scores_BHLM", scores_BHLM, (bsz, h, q_len, kv_len))
    assert_shape("weights_BHLM", weights_BHLM, (bsz, h, q_len, kv_len))
    assert_shape("context_BHLV", context_BHLV, (bsz, h, q_len, v))
    assert_shape("output_BLD", output_BLD, (bsz, q_len, hidden_size))

    kv_b_weight_HYC = attn.kv_b_proj.weight.view(h, y_dim, c_rank)
    k_weight_HNC = kv_b_weight_HYC[:, :n, :]
    v_weight_HVC = kv_b_weight_HYC[:, n:, :]
    q_absorbed_BHLC = torch.einsum("hnc,bhln->bhlc", k_weight_HNC, q_nope_BHLN)
    scores_absorbed_BHLM = (
        torch.matmul(q_absorbed_BHLC, compressed_kv_norm_BMC.unsqueeze(1).transpose(2, 3))
        + torch.matmul(q_rope_after_rope_BHLR, k_rope_after_rope_B1MR.transpose(2, 3))
    ) * float(attn.q_head_dim ** -0.5)
    context_latent_BHLC = torch.einsum("bhlm,bmc->bhlc", weights_BHLM, compressed_kv_norm_BMC)
    context_absorbed_BHLV = torch.einsum("bhlc,hvc->bhlv", context_latent_BHLC, v_weight_HVC)

    return {
        "q_latent_BLQ": q_latent_BLQ,
        "q_full_BHLT": q_full_BHLT,
        "q_nope_BHLN": q_nope_BHLN,
        "q_rope_after_rope_BHLR": q_rope_after_rope_BHLR,
        "kv_raw_BMX": kv_raw_BMX,
        "compressed_kv_norm_BMC": compressed_kv_norm_BMC,
        "k_rope_after_rope_B1MR": k_rope_after_rope_B1MR,
        "kv_full_BHMY": kv_full_BHMY,
        "k_nope_BHMN": k_nope_BHMN,
        "value_BHMV": value_BHMV,
        "query_BHLT": query_BHLT,
        "key_BHMT": key_BHMT,
        "scores_BHLM": scores_BHLM,
        "scores_decomposed_BHLM": scores_decomposed_BHLM,
        "scores_absorbed_BHLM": scores_absorbed_BHLM,
        "weights_BHLM": weights_BHLM,
        "context_BHLV": context_BHLV,
        "context_absorbed_BHLV": context_absorbed_BHLV,
        "context_flat_BLD": context_flat_BLD,
        "output_BLD": output_BLD,
    }


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().detach().cpu().item())


def max_rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = torch.linalg.vector_norm((a - b).detach().float())
    denom = torch.linalg.vector_norm(a.detach().float()).clamp_min(1e-12)
    return float((diff / denom).cpu().item())


def make_attention(cls: type[torch.nn.Module], device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    kwargs = {
        "hidden_size": 16,
        "num_attention_heads": 4,
        "q_lora_rank": 5,
        "qk_rope_head_dim": 2,
        "kv_lora_rank": 6,
        "v_head_dim": 3,
        "qk_nope_head_dim": 3,
        "max_position_embeddings": 32,
        "torch_dtype": dtype,
        "attention_bias": False,
    }
    attn = cls(**kwargs).to(device)
    if hasattr(attn, "softmax_scale") and isinstance(attn.softmax_scale, torch.Tensor):
        attn.softmax_scale = attn.softmax_scale.to(device)
    return attn


def run_check(device_name: str) -> dict[str, Any]:
    torch.manual_seed(20260509)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    device = torch.device(device_name)
    dtype = torch.float32
    bsz, q_len, kv_len, hidden_size = 2, 3, 5, 16
    hidden_q_BLD = torch.randn((bsz, q_len, hidden_size), device=device, dtype=dtype)
    hidden_kv_BMD = torch.randn((bsz, kv_len, hidden_size), device=device, dtype=dtype)
    q_position_ids_BL = torch.tensor([[2, 3, 4], [4, 5, 6]], device=device, dtype=torch.long)
    kv_position_ids_BM = torch.arange(0, kv_len, device=device, dtype=torch.long).unsqueeze(0).repeat(bsz, 1)

    baseline = make_attention(AttentionBaseline, device, dtype)
    baseline.eval()
    with torch.no_grad():
        local_output_BLD = baseline(hidden_q_BLD, hidden_kv_BMD, q_position_ids_BL, kv_position_ids_BM)
        ref = reference_materialized(baseline, hidden_q_BLD, hidden_kv_BMD, q_position_ids_BL, kv_position_ids_BM)

        cache_compressed = make_attention(AttentionCacheCompressed, device, dtype)
        cache_compressed.load_state_dict(baseline.state_dict())
        cache_compressed.eval()
        compressed_cache_BMX = cache_compressed.compress_kv(hidden_kv_BMD, kv_position_ids_BM)
        compressed_output_BLD = cache_compressed(hidden_q_BLD, q_position_ids_BL, compressed_cache_BMX)

        absorbed = make_attention(AttentionAbsorbed, device, dtype)
        absorbed.load_state_dict(baseline.state_dict())
        absorbed.eval()
        absorbed_output_BLD = absorbed(hidden_q_BLD, hidden_kv_BMD, q_position_ids_BL, kv_position_ids_BM)

    checks = {
        "reference_vs_local_baseline_max_abs": max_abs(ref["output_BLD"], local_output_BLD),
        "reference_vs_local_baseline_rel_l2": max_rel_l2(ref["output_BLD"], local_output_BLD),
        "score_cat_vs_decomposed_max_abs": max_abs(ref["scores_BHLM"], ref["scores_decomposed_BHLM"]),
        "score_cat_vs_absorbed_max_abs": max_abs(ref["scores_BHLM"], ref["scores_absorbed_BHLM"]),
        "context_materialized_vs_absorbed_max_abs": max_abs(ref["context_BHLV"], ref["context_absorbed_BHLV"]),
        "cache_compressed_vs_baseline_max_abs": max_abs(compressed_output_BLD, local_output_BLD),
        "cache_compressed_vs_baseline_rel_l2": max_rel_l2(local_output_BLD, compressed_output_BLD),
        "absorbed_impl_vs_baseline_max_abs": max_abs(absorbed_output_BLD, local_output_BLD),
        "absorbed_impl_vs_baseline_rel_l2": max_rel_l2(local_output_BLD, absorbed_output_BLD),
    }
    tolerances = {
        "reference_vs_local_baseline_max_abs": 1e-5,
        "score_cat_vs_decomposed_max_abs": 1e-6,
        "score_cat_vs_absorbed_max_abs": 1e-5,
        "context_materialized_vs_absorbed_max_abs": 1e-5,
        "cache_compressed_vs_baseline_max_abs": 1e-5,
        "absorbed_impl_vs_baseline_max_abs": 1e-5,
    }
    failures = [
        f"{name}={checks[name]:.6g} > {limit:.6g}"
        for name, limit in tolerances.items()
        if checks[name] > limit
    ]

    shapes = {name: shape(tensor) for name, tensor in ref.items() if name.endswith(tuple(["BLQ", "BHLT", "BHLN", "BHLR", "BMX", "BMC", "B1MR", "BHMY", "BHMN", "BHMV", "BHMT", "BHLM", "BHLV", "BLD"]))}
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "device": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" and torch.cuda.is_available() else None,
        "dimensions": {"B": bsz, "L": q_len, "M": kv_len, "D": hidden_size, "H": 4, "N": 3, "R": 2, "V": 3, "Q": 5, "C": 6, "T": 5, "X": 8, "Y": 6},
        "checks": checks,
        "tolerances": tolerances,
        "shapes": shapes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    result = run_check(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

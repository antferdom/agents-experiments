#!/usr/bin/env python3
"""Dummy-weight MLA shape trace for the pinned vLLM tensor contract.

This script does not import vLLM kernels.  It follows the equations and flattened
token layouts documented in the pinned vLLM source:

  q_c      -> q_nope, q_pe
  kv_c     -> k_nope, v
  compute-friendly MHA over [q_nope, q_pe], [k_nope, k_pe], v
  data-movement-friendly MQA over [(W_UK)^T q_nope, q_pe], [kv_c, k_pe], kv_c
  final latent up-projection through W_UV

The purpose is to pdb/print tensor shapes with dummy weights without downloading
official DeepSeek weights or building vLLM native kernels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def dims(name: str) -> dict[str, int]:
    if name == "small":
        return {
            "Sq": 2,
            "Skv": 3,
            "D": 16,
            "H": 2,
            "Q": 5,
            "C": 6,
            "N": 3,
            "R": 2,
            "V": 3,
        }
    if name == "profile":
        return {
            "Sq": 2,
            "Skv": 3,
            "D": 5120,
            "H": 128,
            "Q": 1536,
            "C": 512,
            "N": 128,
            "R": 64,
            "V": 128,
        }
    raise ValueError(f"unknown dims preset: {name}")


def shape(x: torch.Tensor) -> list[int]:
    return list(x.shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", choices=["small", "profile"], default="small")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pdb", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    d = dims(args.dims)
    Sq, Skv, H = d["Sq"], d["Skv"], d["H"]
    Q, C, N, R, V = d["Q"], d["C"], d["N"], d["R"], d["V"]
    T = N + R

    torch.manual_seed(args.seed)
    dtype = torch.float32

    q_c = torch.randn(Sq, Q, device=device, dtype=dtype)
    kv_c = torch.randn(Skv, C, device=device, dtype=dtype)
    q_b = torch.randn(Q, H * T, device=device, dtype=dtype) / math.sqrt(Q)
    w_uk = torch.randn(C, H, N, device=device, dtype=dtype) / math.sqrt(C)
    w_uv = torch.randn(C, H, V, device=device, dtype=dtype) / math.sqrt(C)
    k_pe = torch.randn(Skv, R, device=device, dtype=dtype)

    # Use a nonzero RoPE-shaped query suffix but do not implement RoPE here.  The
    # vLLM algebra only needs the post-RoPE tensors q_pe and k_pe for this check.
    q = (q_c @ q_b).view(Sq, H, T)
    q_nope, q_pe = q.split([N, R], dim=-1)

    # vLLM compute-friendly path: materialize per-head k_nope and v.
    k_nope = torch.einsum("mc,chn->mhn", kv_c, w_uk)
    v = torch.einsum("mc,chv->mhv", kv_c, w_uv)
    scores_nope_materialized = torch.einsum("shn,mhn->shm", q_nope, k_nope)
    scores_rope = torch.einsum("shr,mr->shm", q_pe, k_pe)
    scores_materialized = (scores_nope_materialized + scores_rope) / math.sqrt(T)
    weights = torch.softmax(scores_materialized, dim=-1)
    out_materialized = torch.einsum("shm,mhv->shv", weights, v)

    # vLLM data-movement-friendly path: absorb W_UK into q_nope, attend to kv_c,
    # and apply W_UV after the latent attention reduction.
    q_abs = torch.einsum("shn,chn->shc", q_nope, w_uk)
    scores_nope_absorbed = torch.einsum("shc,mc->shm", q_abs, kv_c)
    scores_absorbed = (scores_nope_absorbed + scores_rope) / math.sqrt(T)
    weights_absorbed = torch.softmax(scores_absorbed, dim=-1)
    latent_out = torch.einsum("shm,mc->shc", weights_absorbed, kv_c)
    out_absorbed = torch.einsum("shc,chv->shv", latent_out, w_uv)

    if args.pdb:
        breakpoint()

    report = {
        "device": str(device),
        "torch": torch.__version__,
        "dims_preset": args.dims,
        "dimensions": {**d, "T": T, "X": C + R, "Y": N + V},
        "shapes": {
            "q_c_SQ": shape(q_c),
            "kv_c_SkvC": shape(kv_c),
            "q_SH_T": shape(q),
            "q_nope_SHN": shape(q_nope),
            "q_pe_SHR": shape(q_pe),
            "k_pe_SkvR": shape(k_pe),
            "w_uk_CHN": shape(w_uk),
            "w_uv_CHV": shape(w_uv),
            "k_nope_SkvHN": shape(k_nope),
            "v_SkvHV": shape(v),
            "q_abs_SHC": shape(q_abs),
            "scores_materialized_SHSkv": shape(scores_materialized),
            "scores_absorbed_SHSkv": shape(scores_absorbed),
            "latent_out_SHC": shape(latent_out),
            "out_materialized_SHV": shape(out_materialized),
            "out_absorbed_SHV": shape(out_absorbed),
            "cache_token_X": [C + R],
        },
        "errors": {
            "scores_nope_max_abs": float(
                (scores_nope_materialized - scores_nope_absorbed).abs().max().item()
            ),
            "scores_total_max_abs": float(
                (scores_materialized - scores_absorbed).abs().max().item()
            ),
            "out_max_abs": float((out_materialized - out_absorbed).abs().max().item()),
            "out_rel_l2": float(
                torch.linalg.vector_norm(out_materialized - out_absorbed).item()
                / max(torch.linalg.vector_norm(out_materialized).item(), 1e-12)
            ),
        },
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

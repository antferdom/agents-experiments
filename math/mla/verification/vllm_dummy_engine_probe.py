#!/usr/bin/env python3
"""Run an actual vLLM DeepSeek-V2 dummy-weight forward and capture MLA shapes.

The model directory contains only a small one-layer config. vLLM's
`load_format="dummy"` initializes weights randomly, so this does not download
official DeepSeek weights.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def tensor_shape(x: Any) -> list[int] | str:
    shape = getattr(x, "shape", None)
    if shape is None:
        return type(x).__name__
    return list(shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("math/mla/verification/vllm_dummy_deepseek_v2_model"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-model-len", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--attention-backend", default="")
    parser.add_argument("--pdb", action="store_true")
    args = parser.parse_args()

    # Keep the probe local and deterministic.
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if args.attention_backend:
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", args.attention_backend)
    venv_bin = Path(__file__).resolve().parents[3] / ".venv" / "bin"
    if venv_bin.exists():
        os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

    import torch
    from vllm import LLM, SamplingParams

    records: list[dict[str, Any]] = []

    try:
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention

        original_forward = MLAAttention.forward

        def traced_forward(self, *pos_args, **kwargs):
            names = ["q", "kv_c_normed", "k_pe"]
            rec: dict[str, Any] = {
                "module": type(self).__name__,
                "input_shapes": {},
                "kwargs": {},
                "num_heads": getattr(self, "num_heads", None),
                "kv_lora_rank": getattr(self, "kv_lora_rank", None),
                "qk_nope_head_dim": getattr(self, "qk_nope_head_dim", None),
                "qk_rope_head_dim": getattr(self, "qk_rope_head_dim", None),
                "v_head_dim": getattr(self, "v_head_dim", None),
                "head_size": getattr(self, "head_size", None),
            }
            for i, value in enumerate(pos_args):
                key = names[i] if i < len(names) else f"arg_{i}"
                rec["input_shapes"][key] = tensor_shape(value)
            for key, value in kwargs.items():
                rec["kwargs"][key] = tensor_shape(value)
            if args.pdb:
                breakpoint()
            out = original_forward(self, *pos_args, **kwargs)
            rec["output_shape"] = tensor_shape(out)
            records.append(rec)
            return out

        MLAAttention.forward = traced_forward
    except Exception as exc:
        records.append({"hook_error": f"{type(exc).__name__}: {exc}"})

    llm = LLM(
        model=str(args.model_dir),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="float16",
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        disable_custom_all_reduce=True,
        trust_remote_code=False,
    )

    sampling = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate([[1, 3, 4, 5]], sampling, use_tqdm=False)

    report = {
        "vllm_version": __import__("vllm").__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "model_dir": str(args.model_dir),
        "load_format": "dummy",
        "prompt_token_ids": [1, 3, 4, 5],
        "generated_token_ids": [list(out.outputs[0].token_ids) for out in outputs],
        "mla_records": records,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Two-head MLA arithmetic example for manual checking."""

from __future__ import annotations

import json
import math
from pathlib import Path


def softmax2(a: float, b: float) -> list[float]:
    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    total = ea + eb
    return [ea / total, eb / total]


def main() -> int:
    scale = 1.0 / math.sqrt(3.0)
    # B=1, L=1, M=2, D=2, H=2, Q=1, C=1, N=1, R=2, V=1.
    q_nope = [2.0, -1.0]  # per head
    compressed_kv = [1.0, 2.0]  # two memory tokens
    w_k = [1.0, -1.0]  # per-head non-RoPE key up-projection from C=1 to N=1
    w_v = [2.0, 1.0]  # per-head value up-projection from C=1 to V=1

    k_nope = [[w_k[h] * c for c in compressed_kv] for h in range(2)]
    values = [[w_v[h] * c for c in compressed_kv] for h in range(2)]
    scores = [[q_nope[h] * k * scale for k in k_nope[h]] for h in range(2)]
    weights = [softmax2(*scores[h]) for h in range(2)]
    context = [
        sum(weights[h][m] * values[h][m] for m in range(2))
        for h in range(2)
    ]
    output = context[:]  # o_proj is the 2x2 identity in the example.

    result = {
        "dimensions": {"B": 1, "L": 1, "M": 2, "D": 2, "H": 2, "Q": 1, "C": 1, "N": 1, "R": 2, "V": 1, "T": 3},
        "scale": scale,
        "q_nope_by_head": q_nope,
        "q_rope_by_head": [[0.0, 0.0], [0.0, 0.0]],
        "compressed_kv_by_token": compressed_kv,
        "k_rope_by_token": [[0.0, 0.0], [0.0, 0.0]],
        "w_k_by_head": w_k,
        "w_v_by_head": w_v,
        "k_nope_by_head_token": k_nope,
        "value_by_head_token": values,
        "scores_by_head_token": scores,
        "weights_by_head_token": weights,
        "context_by_head": context,
        "output": output,
    }
    output_path = Path(__file__).with_suffix(".json")
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

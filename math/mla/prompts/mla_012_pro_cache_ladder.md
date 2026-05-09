# Task: TeX-Ready Cache Ladder From MHA/MQA/GQA To MLA

Produce direct TeX-ready mathematical prose. Keep hidden reasoning brief and put the useful derivation in the answer.

We are building `math/mla/mla.tex`, an in-depth derivation of DeepSeek-V2 MLA. Use this dimension key:

```text
B batch, L query length, M memory length, D hidden, H query heads,
G KV groups for GQA, N non-RoPE key/query head dim, R RoPE key/query suffix dim,
V value head dim, Q query compression rank, C KV compression rank,
T=N+R, X=C+R, Y=N+V.
```

DeepSeek-V2 profile values: `D=5120,H=128,N=128,R=64,V=128,Q=1536,C=512,T=192,X=576,Y=256`.

Source-backed MLA facts:

- Materialized MLA key shape is `[B,H,M,T]`, value shape `[B,H,M,V]`.
- Compressed MLA cache stores normalized compressed KV `[B,M,C]` and shared RoPE key suffix `[B,M,R]`, so cache elements per token per layer are `C+R`.
- Local materialized cache accounting is `H*(T+V)`.
- Use shape-suffixed tensor names where helpful.

Required output:

1. A rigorous TeX section deriving MHA, MQA, and GQA cache shapes and cache element counts.
2. A transition explaining why MLA should not be described as “just low-rank projection”; explain the cache object change.
3. A concise table comparing MHA, MQA, GQA, materialized MLA, and compressed MLA cache elements per token per layer.
4. State assumptions and what is not covered by cache element counts.

Avoid unsupported performance claims. Use direct final text, not meta commentary.

# Task: TeX-Ready Materialized MLA Forward Derivation

Produce direct TeX-ready mathematical prose for `math/mla/mla.tex`. Keep hidden reasoning brief and put the derivation in the answer.

Use this accepted notation:

```text
B batch, L query length, M memory length, D hidden, H heads,
N non-RoPE query/key head dim, R RoPE suffix dim, V value head dim,
Q query compression rank, C KV compression rank, T=N+R, X=C+R, Y=N+V.
```

Accepted shape trace:

```text
hidden_q_BLD -> q_latent_BLQ -> q_full_BHLT -> q_nope_BHLN, q_rope_BHLR -> q_rope_after_rope_BHLR
hidden_kv_BMD -> kv_raw_BMX -> compressed_kv_BMC, k_rope_raw_BMR -> compressed_kv_norm_BMC, k_rope_B1MR -> k_rope_after_rope_B1MR -> kv_full_BHMY -> k_nope_BHMN, value_BHMV
query_BHLT @ key_BHMT^T -> scores_BHLM -> weights_BHLM -> context_BHLV -> output_BLD
```

Implementation-backed details:

- Query path is `q_b_proj(q_a_layernorm(q_a_proj(hidden_q)))`, then reshape and split. The query RoPE suffix is not produced by the KV RoPE projection.
- KV path is `kv_a_proj_with_mqa(hidden_kv)`, split into compressed KV and raw shared RoPE key suffix. `kv_a_layernorm` applies to the compressed KV only. `kv_b_proj` maps compressed KV to packed per-head non-RoPE keys and values.
- Key RoPE suffix is shared across heads and broadcast when materializing `key_BHMT`.
- Scores are scaled by `1/sqrt(T)` in the simplified local implementation.

Required output:

1. A full TeX derivation of the materialized MLA forward pass with equations and shapes.
2. Index notation for the score tensor `scores_BHLM`.
3. The non-RoPE/RoPE score decomposition with shapes.
4. A short paragraph explaining how the local implementation operations (`view`, `transpose`, `split`, `cat`, `matmul`, `softmax`, output projection) correspond to the derivation.
5. Explicit assumptions: no mask, dropout disabled, base RoPE for this accepted executable baseline.

Avoid claims about full-model YaRN or masks except as assumptions/non-goals.

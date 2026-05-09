# Task: TeX-Ready Compressed Cache And Projection Absorption Derivation

Produce direct TeX-ready mathematical prose for `math/mla/mla.tex`. Keep hidden reasoning brief and put the derivation in the answer.

Accepted notation:

```text
B batch, L query length, M memory length, D hidden, H heads,
N non-RoPE query/key head dim, R RoPE suffix dim, V value head dim,
Q query compression rank, C KV compression rank, T=N+R, X=C+R, Y=N+V.
```

Source-backed local facts:

- Compressed cache stores `compressed_kv_norm_BMC` and `k_rope_after_rope_BMR`, packed as `[B,M,C+R]`.
- `kv_b_proj.weight` can be viewed as `W_KV[H,Y,C]` with `Y=N+V`, split into `W_K[H,N,C]` and `W_V[H,V,C]`.
- Non-RoPE scores can be computed as:
  `q_absorbed_BHLC = einsum("hnc,bhln->bhlc", W_K, q_nope_BHLN)`
  then dot with `compressed_kv_norm_BMC`.
- Value aggregation can be reordered:
  `context_latent_BHLC = einsum("bhlm,bmc->bhlc", weights_BHLM, compressed_kv_norm_BMC)`
  then `context_BHLV = einsum("bhlc,hvc->bhlv", context_latent_BHLC, W_V)`.
- Local Torch parity checks show absorbed implementation matches materialized baseline within float32 tolerance on a deterministic case.

Required output:

1. A rigorous TeX derivation of compressed-cache MLA.
2. Projection absorption proof in index notation for non-RoPE scores.
3. Projection absorption proof in index notation for value aggregation.
4. Discuss finite precision and operation ordering: algebraic equality over reals vs possible floating point differences.
5. State which assumptions are required for the local parity test and which full-model features remain unverified.

Do not overclaim performance. Explain memory/cache benefit as shape and cache-element accounting.

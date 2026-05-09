# MLA Notation And Shape Registry

Status: accepted for first-version derivation and verification.

## Dimension Key

| Key | Meaning | DeepSeek-V2 profile value |
| --- | --- | ---: |
| `B` | batch size | runtime |
| `L` | query sequence length | runtime |
| `M` | memory / key-value sequence length | runtime |
| `D` | model hidden dimension | 5120 |
| `H` | number of attention heads | 128 |
| `N` | non-RoPE query/key head dimension | 128 |
| `R` | RoPE query/key head dimension | 64 |
| `V` | value head dimension | 128 |
| `Q` | query compression rank | 1536 |
| `C` | key/value compression rank | 512 |
| `T` | full query/key head dimension, `T = N + R` | 192 |
| `X` | packed KV down-projection dimension, `X = C + R` | 576 |
| `Y` | packed KV up-projection per head, `Y = N + V` | 256 |

The first-version verifier also uses small synthetic dimensions with the same equalities: `T = N + R`, `X = C + R`, and `Y = N + V`.

## Layouts

Shape suffixes are part of tensor names whenever practical:

- `BLD`: `[B, L, D]`, query-side hidden states.
- `BMD`: `[B, M, D]`, key/value-side hidden states.
- `BLQ`: `[B, L, Q]`, query latent.
- `BHLD`: `[B, H, L, D]`, head-major query layout with generic last dimension.
- `BHLT`: `[B, H, L, T]`, materialized query.
- `BHLN`: `[B, H, L, N]`, non-RoPE query.
- `BHLR`: `[B, H, L, R]`, RoPE query.
- `BMX`: `[B, M, X]`, raw packed KV down-projection.
- `BMC`: `[B, M, C]`, compressed KV latent.
- `BMR`: `[B, M, R]`, shared KV RoPE payload before head axis.
- `B1MR`: `[B, 1, M, R]`, shared KV RoPE payload with singleton head axis.
- `BHMY`: `[B, H, M, Y]`, decompressed packed KV per head.
- `BHMN`: `[B, H, M, N]`, non-RoPE key.
- `BHMV`: `[B, H, M, V]`, value.
- `BHMT`: `[B, H, M, T]`, materialized key.
- `BHLM`: `[B, H, L, M]`, attention scores or probabilities.
- `BHLV`: `[B, H, L, V]`, per-head context.

## Tensor Names

The accepted first-version names are:

```text
hidden_q_BLD
hidden_kv_BMD
q_latent_BLQ
q_full_BHLT
q_nope_BHLN
q_rope_BHLR
q_rope_after_rope_BHLR
kv_raw_BMX
compressed_kv_BMC
compressed_kv_norm_BMC
k_rope_raw_BMR
k_rope_B1MR
k_rope_after_rope_B1MR
kv_full_BHMY
k_nope_BHMN
value_BHMV
query_BHLT
key_BHMT
scores_BHLM
weights_BHLM
context_BHLV
context_flat_BLD
output_BLD
```

For source code, the suffix follows the local variable when it remains readable, for example `scores_BHLM`. In TeX, the suffix is written in prose or as a subscripted shape annotation rather than as part of every symbol.

## Projection Names

| Code name | Math role | Shape |
| --- | --- | --- |
| `q_a_proj` | query down-projection `W_DQ` | `D -> Q` |
| `q_a_layernorm` | RMSNorm on query latent | `Q -> Q` |
| `q_b_proj` | packed query up-projection `W_UQ` | `Q -> H*T` |
| `kv_a_proj_with_mqa` | packed KV down-projection `W_DKV` plus shared RoPE key projection | `D -> C+R` |
| `kv_a_layernorm` | RMSNorm on compressed KV latent | `C -> C` |
| `kv_b_proj` | packed KV up-projection, first `N` dims for key and next `V` dims for value per head | `C -> H*(N+V)` |
| `o_proj` | output projection | `H*V -> D` |

## RoPE Convention

- Query RoPE and key RoPE are applied only to the `R`-dimensional suffix components.
- The key RoPE component is shared across heads: `k_rope_after_rope_B1MR` is broadcast to `H` heads when building `key_BHMT`.
- The local simplified implementations use base RoPE and the helper transformation:
  1. select `cos[position_ids]` and `sin[position_ids]`;
  2. reshape pairs as `[..., R/2, 2]`;
  3. transpose the last two axes and reshape back;
  4. apply `x * cos + rotate_half(x) * sin`.
- The full model may use YaRN scaling and changes `softmax_scale` when `rope_scaling` is configured. That is outside the first executable parity check.

## Attention And Cache Conventions

- First-version materialized MLA assumes no attention mask and dropout disabled.
- Scores are scaled by `T^{-1/2}` in the simplified local implementations.
- Softmax is upcast to fp32 and cast back to the query dtype afterward.
- Materialized cache stores `key_BHMT` and `value_BHMV`.
- Compressed cache stores `compressed_kv_norm_BMC` and `k_rope_after_rope_BMR`, packed as `[B, M, C+R]`.

## Accepted Shape Trace

```text
hidden_q_BLD
  -> q_latent_BLQ
  -> q_full_BHLT
  -> q_nope_BHLN, q_rope_BHLR
  -> q_rope_after_rope_BHLR

hidden_kv_BMD
  -> kv_raw_BMX
  -> compressed_kv_BMC, k_rope_raw_BMR
  -> compressed_kv_norm_BMC, k_rope_B1MR
  -> k_rope_after_rope_B1MR
  -> kv_full_BHMY
  -> k_nope_BHMN, value_BHMV

q_nope_BHLN + q_rope_after_rope_BHLR
  -> query_BHLT
k_nope_BHMN + broadcast(k_rope_after_rope_B1MR)
  -> key_BHMT
query_BHLT @ transpose(key_BHMT)
  -> scores_BHLM
softmax(scores_BHLM)
  -> weights_BHLM
weights_BHLM @ value_BHMV
  -> context_BHLV
context_BHLV
  -> context_flat_BLD with last dimension H*V
  -> output_BLD
```

Every shape-changing operation in the local baseline (`view`, `transpose`, `split`, assignment/broadcast, `matmul`, `softmax`, `reshape`, output projection) appears in this trace.

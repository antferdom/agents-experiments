# Task: Deep Mathematical Review And Expansion Of MLA Derivation

You are a mathematical reasoning reviewer for a formalization of DeepSeek-V2 Multi-head Latent Attention (MLA). Produce a TeX-ready critique and expansion plan. Do not rely on unstated implementation behavior; distinguish implementation-backed claims, algebraic claims, and unverified explanatory claims.

## Local Evidence Summary

Primary source hierarchy:

1. Local code under `math/mla/deepseekv2-profile`.
2. The independent Torch reference in `math/mla/verification/torch_mla_reference_check.py`.
3. DeepSeek-V2 paper statements as secondary source.
4. Exported article `tools/exports/articles/kexue.fm-archives-10091.md` as explanatory context.

Accepted dimensions:

```text
B: batch size
L: query sequence length
M: memory / key-value sequence length
D: model hidden dimension
H: number of attention heads
N: non-RoPE query/key head dimension
R: RoPE query/key head dimension
V: value head dimension
Q: query compression rank
C: key/value compression rank
T = N + R
X = C + R
Y = N + V
```

DeepSeek-V2 profile values:

```text
D=5120, H=128, N=128, R=64, V=128, Q=1536, C=512, T=192, X=576, Y=256
```

Accepted materialized shape trace:

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
  -> output_BLD
```

Verification result on CUDA B200 with Torch 2.11.0+cu128:

```text
reference_vs_local_baseline_max_abs = 5.960464477539063e-08
score_cat_vs_decomposed_max_abs = 1.1920928955078125e-07
score_cat_vs_absorbed_max_abs = 1.1920928955078125e-07
context_materialized_vs_absorbed_max_abs = 5.960464477539063e-08
cache_compressed_vs_baseline_max_abs = 0.0
absorbed_impl_vs_baseline_max_abs = 5.960464477539063e-08
```

Known first-version limitations:

- Full-model attention masks, past-key-value updates, dropout, and YaRN softmax scaling are not parity-tested yet.
- vLLM/SGLang are not pinned.
- The current `mla.tex` has a cache ladder and materialized/absorbed derivation, but it should become a best-possible in-depth mathematical derivation, not merely an implementation trace.

## Required Output

Return Markdown with these sections:

1. `Verdict`: whether the current first-version derivation is mathematically coherent under its stated assumptions.
2. `Corrections`: any mathematical or shape errors to fix before expanding.
3. `TeX-Ready Expansion`: a rigorous derivation that can be merged into `mla.tex`, covering:
   - MHA, MQA, and GQA cache formulas and shape conventions.
   - MLA query path with low-rank query compression.
   - MLA KV path with joint compression and decoupled RoPE.
   - Materialized MLA as ordinary attention over constructed Q/K/V.
   - Non-RoPE/RoPE score decomposition.
   - Compressed-cache representation.
   - Projection absorption for key scores and value aggregation, preferably in index notation with shapes.
4. `Assumptions And Non-Goals`: masks, dropout, YaRN scaling, finite precision, cache offsets, and external kernels.
5. `Verification Recommendations`: concrete symbolic or Torch checks that would increase confidence.

Be precise about dimensions and avoid claims not implied by the evidence above.

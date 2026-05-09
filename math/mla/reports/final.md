# MLA Formalization Report

Status: accepted mathematical derivation with executable validation.

## Scope

The main artifact is `math/mla/mla.tex`.  It now presents MLA as mathematics:
dimension registry, cache ladder, query/KV projections, materialized attention,
projection absorption, RoPE separation, an external implementation equivalence
check, executable shape assertions, and a two-head hand-checkable example.  The
latest revision was reviewed by GPT-5.5-pro task `mla_018` for step-by-step
readability and mathematical assumptions.

Implementation details are kept in ledgers and verification artifacts rather
than embedded as code in the TeX.

## Evidence

Primary local sources:

- `deepseekv2-profile/mla/impl/baseline.py`
- `deepseekv2-profile/mla/impl/cache_compressed.py`
- `deepseekv2-profile/mla/impl/absorbed.py`
- `deepseekv2-profile/mla/modeling_deepseek.py`

External source pins:

- vLLM `e3b65a5ba069b350120ca7a614a010787d2de867`
- SGLang `a61a14f416c4003809a82112b4a591eec38a0a10`

Runtime validation:

- Torch reference/parity verifier on NVIDIA B200.
- vLLM-style algebra contract trace on small and profile dimensions.
- Actual vLLM `0.20.1` dummy-weight DeepSeek-V2 engine probe.
- Article re-export with LaTeX integrity verification.
- Two-pass `pdflatex` build.

## Accepted Core Algebra

For the DeepSeek-V2 profile:

```text
D=5120, H=128, N=128, R=64, V=128, Q=1536, C=512
T=N+R=192, X=C+R=576, Y=N+V=256
```

The compressed MLA cache stores

```text
[z, rho] in R^{B x M x (C+R)}
```

where `z` is the normalized KV latent and `rho` is the rotated key RoPE suffix.

The materialized form uses

```text
k_nope = z W_K^T
v = z W_V^T
score = <q_nope, k_nope> + <q_rope, rho>
```

The absorbed form uses

```text
q' = q_nope W_K
score = <q', z> + <q_rope, rho>
latent_context = sum_m attention_weight_m z_m
context = latent_context W_V^T
```

The proof rests on finite-dimensional associativity and distributivity:

```text
q_nope (z W_K^T)^T = (q_nope W_K) z^T
sum_m a_m z_m W_V^T = (sum_m a_m z_m) W_V^T
```

The RoPE suffix remains separate because it is position dependent.

## Verification Results

Torch parity:

```text
reference_vs_local_baseline_max_abs = 5.960464477539063e-08
score_cat_vs_decomposed_max_abs = 1.1920928955078125e-07
score_cat_vs_absorbed_max_abs = 1.1920928955078125e-07
context_materialized_vs_absorbed_max_abs = 5.960464477539063e-08
cache_compressed_vs_baseline_max_abs = 0.0
absorbed_impl_vs_baseline_max_abs = 5.960464477539063e-08
```

vLLM-style profile trace:

```text
out_max_abs = 2.6226043701171875e-06
out_rel_l2 = 4.351123351398705e-07
```

vLLM dummy-engine probe:

```text
observed q shape = [S, 128, 192]
observed latent KV shape = [S, 512]
observed RoPE suffix shape = [S, 1, 64]
head_size = 576
pre-output width = 16384 = H V
generated_token_ids = [[233]]
```

Manual example:

```text
output = [3.520736883716041, 1.6404574756806274]
```

TeX:

```text
verification/mla.pdf builds in 12 pages with no overfull hbox warnings.
```

## Limits

The accepted proof covers the dense base MLA algebra.  Training dropout,
full-model mask variants, paged-cache mutation semantics, YaRN-specific scaling,
quantized storage, sparse/NSA paths, and kernel fusion remain separate proof or
test obligations if the scope expands.

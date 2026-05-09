# MLA Formalization State

Status: accepted source-backed, shape-traced, and test-backed mathematical
derivation for the shared DeepSeek-style MLA algebra.  The TeX target now keeps
implementation code out of the derivation; code assertions, runtime probes, and
execution logs live in verification artifacts.

Last updated: 2026-05-09T02:12:00Z

## Objective Restatement

Produce a rigorous formalization of DeepSeek-V2 Multi-head Latent Attention
(MLA), using local implementation code as primary evidence, external
vLLM/SGLang implementations as cross-checks, GPT-5.5-pro only for hard
mathematical reasoning, and executable assertions to validate tensor shapes and
algebraic equalities.

## Model Call Ledger

- `gpt-5.5-pro` calls submitted: 6 of the cap of 20.
- `mla_010_pro_derivation_review`: failed with no text after exhausting its
  output budget as hidden reasoning tokens.
- `mla_012_pro_cache_ladder`: completed; response id
  `resp_00150760af08fd060069fe87b49d288195938e97fa48415d30`; 13,482 text
  characters.
- `mla_013_pro_materialized_mla`: completed; response id
  `resp_06adea5a818b959f0069fe87b49e7081a1b56bcb4d051520d5`; 12,592 text
  characters.
- `mla_014_pro_absorption`: completed; response id
  `resp_04029012afa51dd30069fe87b49e4881a18363f8890e3ee568`; 16,306 text
  characters.
- `mla_015_pro_external_impl_synthesis`: completed; response id
  `resp_0f926c32f805960d0069fe8aaed7a4819db3d2d74bcd70c0ee`; 11,575 text
  characters.
- `mla_018_pro_step_by_step_tex_review`: completed; response id
  `resp_00c085bec66849560069fe8fb5e5208196be667fc5a49e7820`; 40,346 text
  characters.  Its accepted recommendations were integrated into `mla.tex`.

## Environment

- GPU: NVIDIA B200.
- Torch: `2.11.0+cu128`.
- PyTorch CUDA runtime: 12.8.
- vLLM runtime package: `0.20.1`.
- System CUDA being newer than the PyTorch CUDA runtime is acceptable here:
  the wheel bundles the CUDA runtime it was built against and relies on a
  compatible NVIDIA driver.

## Accepted Artifacts

- Mathematical derivation: `mla.tex`.
- Built PDF: `verification/mla.pdf`.
- TeX build summary: `verification/tex_build.md`.
- Completion gates: `completion_gates.md`.
- Source ledger: `source_ledger.md`.
- External implementation pins and notes: `external_sources.md`.
- Notation registry: `notation.md`.
- GPT-5.5-pro prompts/responses: `prompts/mla_012_*`,
  `prompts/mla_013_*`, `prompts/mla_014_*`, `prompts/mla_015_*`,
  `prompts/mla_018_*`, and corresponding files in `responses/`.
- Local Torch verifier: `verification/torch_mla_reference_check.py` and
  `verification/torch_mla_reference_check.json`.
- Manual example verifier: `verification/two_head_manual_example.py` and
  `verification/two_head_manual_example.json`.
- vLLM-style contract verifier:
  `verification/vllm_dummy_mla_shape_trace.py`,
  `verification/vllm_dummy_mla_shape_trace.small.json`, and
  `verification/vllm_dummy_mla_shape_trace.profile.json`.
- Actual vLLM dummy-engine probe:
  `verification/vllm_dummy_deepseek_v2_model/config.json`,
  `verification/vllm_dummy_engine_probe.py`, and
  `verification/vllm_dummy_engine_probe.json`.
- Article re-export:
  `verification/source_exports/kexue.fm-archives-10091.reexport.md`.

## Accepted Mathematical Facts

For the DeepSeek-V2 profile:

```text
D = 5120
H = 128
N = 128
R = 64
V = 128
Q = 1536
C = 512
T = N + R = 192
X = C + R = 576
Y = N + V = 256
```

The query path is

```text
h^q -> q_latent in R^{B x L x Q}
    -> q_full in R^{B x H x L x T}
    -> q_nope in R^{B x H x L x N}, q_rope in R^{B x H x L x R}.
```

The KV path is

```text
h^kv -> u in R^{B x M x (C+R)}
     -> z in R^{B x M x C}, rho in R^{B x M x R}
     -> compressed cache [z, rho] in R^{B x M x (C+R)}.
```

The materialized form uses

```text
k_nope = z W_K^T in R^{B x H x M x N}
v = z W_V^T in R^{B x H x M x V}
scores in R^{B x H x L x M}
output in R^{B x L x D}.
```

The absorbed form uses

```text
q' = q_nope W_K in R^{B x H x L x C}
scores = <q', z> + <q_rope, rho>
latent context in R^{B x H x L x C}
W_V maps latent context to R^{B x H x L x V} by multiplying W_V^T on the right.
```

The equality proof accepted in `mla.tex` is the finite-dimensional identity

```text
q_nope (z W_K^T)^T = (q_nope W_K) z^T
sum_m a_m z_m W_V^T = (sum_m a_m z_m) W_V^T
```

with the RoPE suffix kept separate because it is position dependent.

## Verification Results

Local Torch reference on CUDA/B200:

```text
reference_vs_local_baseline_max_abs = 5.960464477539063e-08
score_cat_vs_decomposed_max_abs = 1.1920928955078125e-07
score_cat_vs_absorbed_max_abs = 1.1920928955078125e-07
context_materialized_vs_absorbed_max_abs = 5.960464477539063e-08
cache_compressed_vs_baseline_max_abs = 0.0
absorbed_impl_vs_baseline_max_abs = 5.960464477539063e-08
```

Manual two-head example:

```text
output = [3.520736883716041, 1.6404574756806274]
```

vLLM-style profile contract trace:

```text
C=512, H=128, N=128, R=64, V=128, T=192, X=576
out_max_abs = 2.6226043701171875e-06
out_rel_l2 = 4.351123351398705e-07
```

Actual vLLM dummy-engine probe:

```text
q shape = [S, 128, 192]
latent KV shape = [S, 512]
RoPE suffix shape = [S, 1, 64]
head_size = 576
pre-output width = 16384 = H V
generated_token_ids = [[233]]
```

TeX build:

```text
pdflatex -interaction=nonstopmode -halt-on-error -output-directory math/mla/verification math/mla/mla.tex
```

Result: passed twice; latest PDF has 12 pages and no overfull hbox warnings.

Article export:

```text
LaTeX integrity: OK (101 fragments)
```

## Rejected Or Limited Claims

- Rejected from the weak draft: query RoPE is not produced by a separate
  KV-side projection.  It is produced by the query low-rank path and then split
  and rotated.
- Not fully parity-tested: training dropout, all attention-mask forms, all
  paged-cache mutation semantics, and YaRN scaling behavior.
- External production details such as quantized storage, sparse/NSA paths,
  page tables, CUDA graphs, and kernel fusion are treated as storage or
  execution layers over the base algebra, not as part of the base proof.

## Open Follow-Up Tasks

- Add full-model mask/cache-offset parity tests against `modeling_deepseek.py`.
- Add a separate proof layer for YaRN-scaled RoPE/softmax scaling if needed.
- Add quantized or paged-cache proof obligations only if the TeX scope expands
  beyond the dense base MLA algebra.

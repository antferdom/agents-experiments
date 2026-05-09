We are formalizing DeepSeek-style Multi-head Latent Attention (MLA) in TeX.
Use the following source-backed facts and produce a rigorous, TeX-ready section
that synthesizes the external implementation comparison. Do not invent API facts.

Notation:
- B batch, L query length, M memory length, D hidden, H heads.
- N non-RoPE q/k head dim, R RoPE suffix dim, V value head dim.
- Q query compression rank, C KV compression rank.
- T=N+R, X=C+R, Y=N+V.
- z_{bmc} is the normalized compressed KV latent of shape B x M x C.
- rho_{bmr} is the rotated key RoPE suffix of shape B x M x R.
- q_nope has shape B x H x L x N and q_rope has shape B x H x L x R.
- W^K_h has shape N x C and W^V_h has shape V x C from kv_b_proj.

Primary local implementation facts:
- materialized path forms k_nope = W^K z and v = W^V z per head, concatenates
  [q_nope, q_rope] and [k_nope, rho], performs ordinary per-head attention, then output projection.
- compressed-cache path caches [z, rho] of width C+R.
- absorbed decode path computes q_abs = (W^K)^T q_nope, attends with key/value latent z plus rho,
  obtains latent context, then applies W^V before output projection.

External vLLM snapshot:
- commit e3b65a5ba069b350120ca7a614a010787d2de867, 2026-05-08.
- vllm/model_executor/layers/attention/mla_attention.py lines 26-63 define MLA:
  single latent vector per-token cache; Lkv=512, P(nope)=128, R=64, V=128.
- lines 66-90 describe compute-friendly path:
  materialize k_nope = kv_c @ W_UK and v = kv_c @ W_UV; attention over [q_nope,q_pe], [k_nope,k_pe], v.
- lines 94-118 describe data-movement-friendly path:
  absorb W_UK into q via ql_nope, attend over [ql_nope,q_pe] against [kv_c,k_pe] with value kv_c,
  then multiply by W_UV.
- lines 951-961 return MLAAttentionSpec with num_kv_heads=1 and head_size=kv_lora_rank+qk_rope_head_dim.
- lines 826-852 view kv_b_proj_weight as (C, H, N+V) and split W_UK/W_UV.

External SGLang snapshot:
- commit a61a14f416c4003809a82112b4a591eec38a0a10, 2026-05-09.
- python/sglang/srt/models/deepseek_v2.py lines 1431-1440 construct attn_mqa with
  head_dim=C+R, num_kv_heads=1, v_head_dim=C.
- lines 1391-1399 construct kv_b_proj from C to H(N+V).
- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
  lines 241-243 split q into q_nope/q_pe and latent_cache into k_pe; lines 320-323 compute
  q_nope_out = q_nope W_K; lines 457-468 attend with q=[q_nope_out,q_pe],
  k=[k_nope,k_pe], value=k_nope (latent); lines 472-592 up-project latent context through W_V
  and then o_proj.
- python/sglang/srt/mem_cache/memory_pool.py lines 1560-1567 allocate one kv_buffer per layer
  with shape (size+page_size, 1, kv_cache_dim); lines 1630-1679 set_mla_kv_buffer writes
  cache_k_nope then cache_k_rope into that single buffer.
- python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py lines 138-183 define
  default MLA kv_cache_dim = kv_lora_rank + qk_rope_head_dim, with FP8/NSA storage exceptions.

Required output:
1. A compact but formal TeX section titled "External implementation check".
2. Explain why both vLLM and SGLang validate the same algebraic object: cache [z,rho] in R^{C+R}.
3. Explain compute-friendly vs data-movement-friendly algorithms and prove their equality at the score and context level.
4. State the limitations: external code has quantization, sparse/NSA, paged-cache and kernel details that do not change the base algebra but should not be folded into the base derivation.
5. Include a concise theorem with equations for equivalence:
   k_h = W^K_h z, v_h = W^V_h z,
   q'_h = (W^K_h)^T q_nope_h,
   softmax(q_nope_h k_h^T + q_rope rho^T) z W^V_h
   equals materialized attention output before o_proj.
6. Avoid broad prose. Prefer definitions, equalities, and shape annotations.

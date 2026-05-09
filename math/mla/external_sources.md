# External MLA Implementation Sources

Status: local source snapshots cloned for comparison only. The clone trees under
`math/mla/external/` are ignored by git; this file records the reproducible pins
and the inspected implementation facts.

## Snapshot Pins

| Project | Local path | Commit | Commit date | Subject |
| --- | --- | --- | --- | --- |
| vLLM | `math/mla/external/vllm` | `e3b65a5ba069b350120ca7a614a010787d2de867` | `2026-05-08T18:03:33-07:00` | `[feat] Add explicit /start_weight_update and /finish_weight_update APIs for weight transfer (#39212)` |
| SGLang | `math/mla/external/sglang` | `a61a14f416c4003809a82112b4a591eec38a0a10` | `2026-05-09T08:52:51+08:00` | `[KDA] Optimize prefill kernels with diagonal and recompute fuse (#24271)` |

## vLLM Findings

Primary files:

- `vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/model_executor/layers/mla.py`
- `vllm/v1/kv_cache_interface.py`

Accepted facts:

- `mla_attention.py` lines 26-63 document the same DeepSeek MLA decomposition used in this project: one latent vector per token in the KV cache, query latent dimension `Lq`, KV latent dimension `Lkv`, non-RoPE dimension `P`, RoPE dimension `R`, and value dimension `V`.
- `mla_attention.py` lines 66-90 define the compute-friendly path: materialize `k_nope = kv_c @ W_UK` and `v = kv_c @ W_UV`, then run ordinary MHA over `[q_nope, q_pe]`, `[k_nope, k_pe]`, and `v`.
- `mla_attention.py` lines 94-118 define the data-movement-friendly path: absorb `W_UK` into the query side, run MQA over `[ql_nope, q_pe]`, `[kv_c, k_pe]`, and value `kv_c`, then apply `W_UV`.
- `mla.py` lines 139-179 compute `q_c`, split `kv_lora` into `kv_c` and `k_pe`, normalize `kv_c`, apply RoPE to the query suffix and `k_pe`, then call `MLAAttention(q, kv_c_normed, k_pe, ...)`.
- `mla_attention.py` lines 826-852 view `kv_b_proj.weight` as `(C, H, N+V)` after transposition and split it into `W_UK` and `W_UV`.
- `mla_attention.py` lines 951-961 returns `MLAAttentionSpec(num_kv_heads=1, head_size=kv_lora_rank + qk_rope_head_dim)`, matching the symbolic cache width `C+R`.
- `kv_cache_interface.py` lines 323-354 defines `MLAAttentionSpec`; its default real page size is `storage_block_size * num_kv_heads * head_size * dtype_size`, with special `fp8_ds_mla` layouts. This supports the base algebra while separating quantized storage details.

## SGLang Findings

Primary files:

- `python/sglang/srt/models/deepseek_v2.py`
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/layers/attention/flashmla_backend.py`
- `python/sglang/srt/layers/attention/cutlass_mla_backend.py`

Accepted facts:

- `deepseek_v2.py` lines 1312-1329 constructs the fused query/KV down-projection and `q_b_proj` when `q_lora_rank` is present.
- `deepseek_v2.py` lines 1391-1399 constructs `kv_b_proj: C -> H(N+V)`.
- `deepseek_v2.py` lines 1431-1440 constructs `attn_mqa` with `head_dim=C+R`, `num_kv_heads=1`, and `v_head_dim=C`. This is the absorbed decode/MQA algebra over the latent cache.
- `forward_mla.py` lines 241-243 split `q` into `q_nope/q_pe` and the latent cache into the `k_pe` suffix.
- `forward_mla.py` lines 320-323 compute the absorbed non-RoPE query `q_nope_out = q_nope W_K`.
- `forward_mla.py` lines 457-468 call `attn_mqa` with query `[q_nope_out, q_pe]`, key `[k_nope, k_pe]`, and latent value `k_nope`.
- `forward_mla.py` lines 472-592 up-project the latent attention output through the value-side weight and then applies `o_proj`.
- `model_runner_kv_cache_mixin.py` lines 138-183 sets the default MLA KV cache dimension to `kv_lora_rank + qk_rope_head_dim`, with FP8/NSA storage exceptions.
- `memory_pool.py` lines 1560-1567 allocates one KV buffer per layer with shape `(size + page_size, 1, kv_cache_dim)`.
- `memory_pool.py` lines 1630-1679 writes `cache_k_nope` followed by `cache_k_rope` into the single MLA buffer; for the base non-NSA layout this is the symbolic `[z,\rho]` cache.
- `flashmla_backend.py` lines 438-466 and `cutlass_mla_backend.py` lines 274-285 pass `k_cache.view(..., kv_cache_dim)` to MLA kernels with latent value width `kv_lora_rank`.

## Cross-Implementation Decision

Both vLLM and SGLang support the same base algebra as the local simplified implementation:

```text
cache token = [normalized KV latent z_C, rotated key RoPE suffix rho_R]
cache width = C + R
decode key/value head count = 1
latent decode value width = C
```

The external implementations add important runtime details that are outside the
base derivation: paged cache tables, chunked prefill, quantized cache formats,
sparse/NSA indexers, CUDA graph constraints, and backend-specific kernels. These
details validate that the algebra is used in production paths, but they are not
folded into the simplified TeX proof unless explicitly stated.

## Runtime vLLM Dummy-Weight Probe

Artifact: `math/mla/verification/vllm_dummy_engine_probe.json`.

Accepted facts from the runtime package `vllm==0.20.1`:

- A one-layer DeepSeek-V2 config was loaded with dummy weights; no official
  model weights were downloaded.
- The engine selected an MLA backend and generated one token from the prompt
  token ids `[1, 3, 4, 5]`.
- The hooked MLA call observed the profile dimensions:
  \(H=128\), \(N=128\), \(R=64\), \(C=512\), \(V=128\), and \(C+R=576\).
- The observed attention input shapes included
  \(S\times H\times(N+R)\), \(S\times C\), and \(S\times1\times R\),
  with concrete profile shapes `[S, 128, 192]`, `[S, 512]`, and
  `[S, 1, 64]`.
- The pre-output width was `16384 = H V`, matching the mathematical output
  projection domain.

# MLA Source Ledger

Status: expanded ledger, created from local implementation sources, the local article export, the DeepSeek-V2 paper HTML, and pinned vLLM/SGLang implementation snapshots.

Trust rule: local code under `deepseekv2-profile` is the primary correctness source for executable tensor flow. Paper and article sources are used for naming, high-level design intent, and explanatory context unless a formula is independently aligned with local code.

## Primary Implementation Sources

| Source | Lines inspected | Trusted for | Notes |
| --- | ---: | --- | --- |
| `deepseekv2-profile/mla/config.json` | 20, 34, 38, 45, 50, 52 | DeepSeek-V2 profile dimensions: `C=512`, `Q=1536`, `N=128`, `R=64`, `V=128`, `H=128`, `D=5120`; dropout disabled in config | `rope_scaling` is present in the full model config; simplified implementations ignore scaling and use base RoPE. |
| `deepseekv2-profile/mla/impl/baseline.py` | 99-188 | Simplified materialized MLA forward pass without mask or past cache | This is the first executable parity target. It exposes Q low-rank path, joint KV path, split/cat materialization, score matmul, fp32 softmax, value aggregation, and output projection. |
| `deepseekv2-profile/mla/modeling_deepseek.py` | 681-902 | Full eager DeepSeekV2 attention path with mask, cache update, dropout, and optional YaRN scaling | This confirms that the full model constructs the same materialized Q/K/V tensors before ordinary attention. It also adds full-model concerns not exercised by the simplified baseline. |
| `deepseekv2-profile/mla/impl/cache_compressed.py` | 123-200 | Compressed-cache interface and what is cached for decode | `compress_kv` caches `kv_a_layernorm(compressed_kv_BMC)` together with already-rotated `k_rope_BMR`, packed as `[B, M, C+R]`. |
| `deepseekv2-profile/mla/impl/absorbed.py` | 123-190 | Projection absorption algebra as implemented | The implementation absorbs the non-RoPE key up-projection into the query side and the value up-projection before the output projection, while keeping the RoPE score term separate. |
| `deepseekv2-profile/mla/test.py` | 42-85 | Existing CUDA-only parity harness across local variants | It compares local optimized variants against `AttentionBaseline`; it was not used directly because this run adds a smaller deterministic reference harness. |
| `deepseekv2-profile/mla/benchmark.py` | 46-84, 133-168 | Cache-size measurement method for decompressed and compressed variants | Cache size is computed from actual tensor `numel() * element_size()`, not symbolic formulas. |

## Secondary Local Sources

| Source | Lines inspected | Trusted for | Notes |
| --- | ---: | --- | --- |
| `deepseekv2-profile/workspace/blog/optimizing-mla.md` | 13-58, 116-148, 162-219, 226-284 | Explanatory derivation and optimization commentary from the local profile repo | Secondary to implementation code. Useful for cache and projection-absorption motivation. |
| `tools/exports/articles/kexue.fm-archives-10091.md` | 17-31, 45-78, 82-203 | MHA/MQA/GQA-to-MLA explanatory ladder and RoPE incompatibility motivation | Exported from `https://kexue.fm/archives/10091`; used as article context, not as implementation truth. |
| `tools/zhihu-to-markdown/export_article.py` | README plus code 560-650 | Reproducible article export and LaTeX integrity checking | Re-run in this cycle; see `verification/source_exports/kexue.fm-archives-10091.reexport.md`. |
| `mla.tex` before this cycle | 1-245 | Weak draft and final integration target | Useful for initial prose and code snippets only. It contained at least one notation issue recorded below. |

## Paper Sources

| Source | Lines inspected | Trusted for | Notes |
| --- | ---: | --- | --- |
| DeepSeek-V2 paper, arXiv HTML `2405.04434v2` | 128-177 | MLA design intent: low-rank joint KV compression, low-rank query compression, decoupled RoPE, and KV cache size concept | The arXiv HTML elides many equation symbols in the text view, so exact formulas are not promoted from the HTML alone. |
| DeepSeek-V2 paper, arXiv HTML Appendix B | 716-731 | Existence and role of full MLA formulas plus statement that `W^{UK}` and `W^{UV}` can be absorbed during inference | Exact symbolic equations are represented in the report using local code notation and checked shapes, not copied from the HTML extraction. |

Paper URL: https://arxiv.org/html/2405.04434v2

## External Implementation Sources

The exact clone pins and line-level notes are recorded in `external_sources.md`.
The clone trees are local comparison artifacts and are ignored by git.

| Source | Lines inspected | Trusted for | Notes |
| --- | ---: | --- | --- |
| vLLM `e3b65a5ba069b350120ca7a614a010787d2de867` | `mla_attention.py` 26-118, 826-852, 951-961; `mla.py` 139-179; `kv_cache_interface.py` 323-354 | Cross-check for compute-friendly materialization, data-movement-friendly absorption, `num_kv_heads=1`, and cache head size `C+R` | Used as independent production implementation evidence, not as a replacement for local code truth. |
| SGLang `a61a14f416c4003809a82112b4a591eec38a0a10` | `deepseek_v2.py` 1312-1440; `forward_mla.py` 241-243, 320-323, 457-592; `model_runner_kv_cache_mixin.py` 138-183; `memory_pool.py` 1560-1679 | Cross-check for `attn_mqa` latent decode, `kv_b_proj: C -> H(N+V)`, and one-buffer cache width `C+R` | Runtime adds paged-cache, quantized, sparse/NSA, and kernel details outside the base algebra. |
| vLLM runtime package `0.20.1` | `verification/vllm_dummy_engine_probe.py` and JSON output | Dummy-weight execution of the DeepSeek-V2 MLA path with captured profile shapes | The probe validates runtime shapes and module dimensions without downloading official weights. |

## Accepted Source Decisions

- Use `deepseekv2-profile/mla/impl/baseline.py` as the executable definition of materialized MLA for the first version.
- Use `deepseekv2-profile/mla/modeling_deepseek.py` only to confirm full-model concerns: attention masks, past-key-value update, dropout, and YaRN softmax scaling.
- Use `deepseekv2-profile/mla/impl/cache_compressed.py` and `impl/absorbed.py` as optimization evidence, with an extra equivalence gate against the baseline.
- Treat the exported article as an explanatory ladder from MHA to MQA/GQA/MLA. It is not allowed to override local implementation tensor flow.
- Promote vLLM/SGLang only for the shared base algebra after exact pins: both independently support a single MLA cache vector `[z,\rho]` of symbolic width `C+R`, with `z` used as the latent decode value before the value-side up-projection. Do not promote backend-specific quantized or sparse layouts into the base proof.

## Conflicts And Corrections

- The weak draft in `mla.tex` described the query RoPE projection as `W^{KR} h_t` in one place. Local implementation and the exported article show that the whole per-head query vector comes from `q_b_proj(q_a_layernorm(q_a_proj(hidden)))`, then splits into non-RoPE and RoPE pieces. Accepted correction: query RoPE is produced from the query low-rank path, not from the KV RoPE projection.
- The full model uses `attention_mask` and may apply YaRN softmax scaling when `rope_scaling` is configured. The simplified local variants omit masks and YaRN scaling. First-version executable tests therefore target the simplified algebra only and explicitly state this assumption.
- The local simplified `apply_rotary_pos_emb` has a layout permutation before applying `rotate_half`. The reference check implements the same transformation to test local parity, while notation keeps this as a RoPE convention rather than a mathematical novelty.
- vLLM and SGLang use production names such as `head_dim`, `v_head_dim`, `Lkv`, `P`, and `kv_lora_rank` in ways that differ from this document's symbols. Accepted mapping: `Lkv = C`, `P = N`, `R = R`, and the absorbed decode attention value width is `C` before multiplying by `W^V`, even though the final materialized value head width is `V`.

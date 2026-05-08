# MLA

We want to perform an in-depth and explanatory formal derivation of **Multi-head Latent Attention (MLA)**, released by DeepSeek-V2. The goal is to show how the algebra unfolds from MHA, MQA, and GQA into MLA while keeping correctness grounded in implementations and executable tests.

This plan applies the long-horizon multi-agent workflow from [../long_horizon_math_multi_agent_design.md](../long_horizon_math_multi_agent_design.md) to this MLA project.

## Objective

Produce a source-backed and test-backed formalization of MLA:

- Explain MHA, MQA, and GQA as baseline attention/cache designs.
- Derive MLA step by step from projection, compression, RoPE split, attention scoring, value aggregation, and output projection.
- Separate baseline MLA algebra from cache optimizations and absorbed-projection optimizations.
- Compare the derivation against real implementation code.
- Build our own small Torch implementation as an independent executable reference.
- Optionally compare against vLLM and SGLang MLA implementations once their exact versions are pinned.
- Use `mla.tex` as the final formalization target, not as the source of truth.

## Correctness Principle

A claim should be accepted only if only it has at least one clear evidence path:

- It is directly supported by inspected implementation code or paper formulas.
- Its tensor shapes check under the accepted notation.
- It is numerically equivalent to an implementation on deterministic test cases.
- It is reviewed as a derivation and marked as not yet executable.

For this project, implementation parity is more important than proof-assistant formalization at the beginning.

## Source Hierarchy

Use sources in this order:

1. Local DeepSeek-V2 implementation under `deepseekv2-profile`.
2. Our own minimal Torch reference implementation derived from the accepted notation.
3. DeepSeek-V2 paper formulas, especially the MLA appendix.
4. vLLM and SGLang implementations, only when pinned to exact commits or local snapshots.
5. Blog posts and translations for explanation, intuition, and optimization context.
6. `mla.tex` as a weak draft reference.

If sources disagree, create an explicit conflict note instead of blending them silently.

## Core Artifacts

The long-horizon run should maintain these project-specific artifacts:

- `state.md`: accepted facts, accepted derivations, blockers, and rejected claims.
- `source_ledger.md`: implementation files, paper sections, external implementation commits, and what each source is trusted for.
- `notation.md`: symbols, tensor layouts, shape suffixes, dimensions, projection names, cache conventions, RoPE conventions, mask conventions, and scaling conventions.
- `tasks.jsonl`: decomposition into source audit, derivation, verification, review, and integration tasks.
- `events.jsonl`: append-only log of submissions, verifier runs, accepted results, and failures.
- `verification/`: shape checks, Torch parity checks, external comparator checks, and TeX build/lint outputs.
- `reports/final.md`: readable derivation with evidence references.
- `mla.tex`: final TeX integration target.

## Workstreams

### 1. Source Audit

Inspect the local implementation before broad derivation. Record:

- Projection modules and their dimensions.
- Forward-pass tensor flow.
- RoPE application and position handling.
- Attention score construction and softmax scaling.
- Value aggregation and output projection.
- Cache representation in baseline and optimized paths.
- Differences between local simplified implementations and the full model implementation.

The source audit should produce a compact source ledger that later prompts can cite.

### 2. Notation And Shape Registry

Define a notation registry before deriving large sections. It should settle:

- Batch, query length, key/value length, hidden size, head count, head dimensions, and compression ranks.
- Tensor layouts such as `[B, L, D]`, `[B, H, L, D]`, and `[B, L, H, D]`.
- Math names for code projections.
- Conventions for RoPE and non-RoPE components.
- Causal masks, scaling, dtype upcasts, dropout-disabled testing, and cache layout.
- A single-letter dimension key and a tensor-name suffix convention.

Every later derivation task should use this registry rather than inventing notation locally.

### 3. Shape Evolution Convention

Shape evolution is a first-class correctness artifact for this project. The formalization should not only state final tensor shapes; it should show how shapes change through every forward-pass operation.

Use a documented dimension key in `notation.md`. A starting key for MLA:

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
T: full query/key head dimension, T = N + R
X: pre-split KV projection dimension, X = C + R
Y: post-KV-decompression packed dimension, Y = N + V
```

When writing reference code, tests, or derivation pseudocode, tensor names should end with a shape suffix built from the dimension key whenever that improves readability. Examples:

```python
hidden_BLD
q_latent_BLQ
q_full_BHLT
q_nope_BHLN
q_rope_BHLR
kv_raw_BMX
compressed_kv_BMC
k_rope_raw_BMR
k_rope_BHMR
kv_full_BHMY
k_nope_BHMN
value_BHMV
query_BHLT
key_BHMT
scores_BHLM
weights_BHLM
context_BHLV
output_BLD
```

The exact letters can be revised during the notation task, but once accepted they should remain stable. Avoid reusing the same letter for two different logical dimensions in the same artifact.

Every derivation section should include a shape trace. For the materialized MLA forward pass, the trace should look structurally like:

```text
hidden_BLD
  -> q_latent_BLQ
  -> q_full_BHLT
  -> q_nope_BHLN, q_rope_BHLR
  -> q_rope_after_rope_BHLR

hidden_BMD
  -> kv_raw_BMX
  -> compressed_kv_BMC, k_rope_raw_BMR
  -> compressed_kv_norm_BMC, k_rope_BHMR
  -> kv_full_BHMY
  -> k_nope_BHMN, value_BHMV

q_nope_BHLN + q_rope_after_rope_BHLR
  -> query_BHLT
k_nope_BHMN + k_rope_BHMR
  -> key_BHMT
query_BHLT @ key_BHMT
  -> scores_BHLM
softmax(scores_BHLM)
  -> weights_BHLM
weights_BHLM @ value_BHMV
  -> context_BHLV
context_BHLV
  -> output_BLD
```

This shape trace should be aligned against implementation operations such as `view`, `transpose`, `split`, `cat`, `matmul`, `einsum`, RoPE application, masking, softmax, and output projection. Any shape-changing operation in code must appear in the corresponding derivation trace.

### 4. Derivation Ladder

Build the formalization in narrow layers:

1. Standard MHA forward pass and KV-cache accounting.
2. MQA as shared K/V across query heads.
3. GQA as grouped K/V sharing.
4. MLA query projection path.
5. MLA joint KV compression path.
6. RoPE and non-RoPE score decomposition.
7. Materialized MLA as ordinary attention over constructed Q/K/V tensors.
8. Compressed-cache MLA and what must be cached for correctness.
9. Absorbed-projection MLA and the assumptions needed for equivalence.
10. Cache-size comparison across MHA, MQA, GQA, materialized MLA, and compressed MLA.

Each layer should have explicit shapes, a shape trace, source references, and a verification plan.

### 5. Independent Torch Reference

Create a small Torch implementation that follows the accepted equations directly. It should be deliberately boring:

- No fused kernels.
- No hidden layout tricks.
- Deterministic seeds.
- Dropout disabled.
- Intermediate tensors exposed for comparison.
- Shape-suffixed intermediate names where readable.
- Assertions that check expected shapes after each forward-pass step.
- Small dimensions supported for fast CPU tests.
- Full-profile dimensions supported when practical.

This implementation is the bridge between derivation and real code. If the derivation is correct, the reference should match the local implementation under aligned assumptions.

### 6. Implementation Parity

Parity tests should compare:

- Our Torch reference against the local simplified baseline.
- Our Torch reference against the local full attention implementation where practical.
- Local optimized variants against the local baseline.
- Optional vLLM/SGLang implementations against the local reference after version pinning.

Minimum checks:

- Query path equality.
- KV path equality.
- RoPE equality.
- Shape evolution equality at named intermediate checkpoints.
- Attention score equality.
- Attention probability equality.
- Pre-output-projection value aggregation equality.
- Final output equality.
- Cache-size accounting equality.

Stress cases should vary batch size, query length, key/value length, position offsets, dtype, and mask behavior.

### 7. External Implementations

vLLM and SGLang should be treated as independent comparators, not initial truth sources.

Before using either:

- Pin the repository and commit.
- Record relevant files/functions in `source_ledger.md`.
- Identify their cache layout, RoPE convention, scaling, dtype assumptions, and kernel path.
- Decide whether the comparison is algebraic, numerical, or only structural.

If an external implementation disagrees with the local baseline, preserve the mismatch as a finding and decide whether it is a layout difference, optimization difference, bug, or unsupported comparison.

## Task Types

Use the generic task state machine, but prefer MLA-specific task kinds:

- `source_audit`: inspect source code, papers, and external implementations.
- `notation`: define symbols, dimensions, and layouts.
- `shape_trace`: document forward-pass shape evolution from implementation and derivation.
- `derivation`: produce a checked algebraic section.
- `shape_check`: verify tensor compatibility.
- `torch_reference`: implement or refine independent Torch equations.
- `parity_test`: compare outputs or intermediate tensors.
- `review`: critique derivations and tests.
- `integration`: update `mla.tex` or final reports.
- `repair`: fix rejected derivations or failed tests.

## Acceptance Gates

A derivation section is accepted only if:

- Its source facts are cited.
- Its shapes are explicit and consistent.
- Its shape trace covers every forward-pass shape-changing operation.
- Its tensor names follow the accepted shape-suffix convention where practical.
- Its assumptions are listed.
- Its test plan is concrete.
- Its parity tests pass when executable.
- Any untested part is labeled as reviewed but not verified.

Optimization sections need an extra gate: they must state exactly what is being optimized and whether the optimization is mathematically equivalent, numerically approximate, layout-only, or performance-only.

## First Long-Horizon Cycle

Start with a small cycle before launching many model jobs:

1. Create the MLA goal directory.
2. Build `source_ledger.md` from local implementation files and the DeepSeek-V2 paper.
3. Build `notation.md`.
4. Define the dimension key and tensor shape-suffix convention.
5. Derive only the query projection path with a full shape trace.
6. Implement or sketch the matching Torch reference path with shape assertions.
7. Run a parity check against local code.
8. Promote the result only if source, shape, trace, and parity evidence are recorded.

After this smoke test works, expand to the full MHA/MQA/GQA-to-MLA derivation and cache optimization sections.

## References

- `mla.tex` is an incomplete TeX formalization of MLA. Use it as a weak reference and final integration target.
- `deepseekv2-profile` has local implementation code and local MLA variants.
- [Extreme trade-off between caching and performance: from MHA, MQA, GQA to MLA, Su Jianlin](https://kexue.fm/archives/10091)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model, Appendix B formulas of MLA](https://arxiv.org/pdf/2405.04434v2)
- [Follow-up MLA discussion, Su Jianlin](https://kexue.fm/archives/10907)

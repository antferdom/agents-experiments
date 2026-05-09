# MLA Experiment Conclusions

Status: paused after a coherent first completed pass.  Further refinement can
continue from the accepted artifacts and completion gates.

## Experiment Setup

The experiment used `math/mla/plan_mla.md` as the project plan and
`math/long_horizon_math_multi_agent_design.md` as the operating protocol.  The
target artifact was `math/mla/mla.tex`: a source-backed, test-backed, and
step-by-step mathematical derivation of DeepSeek-style Multi-head Latent
Attention.

Primary evidence came from the local DeepSeek-V2 profile under
`math/mla/deepseekv2-profile`.  The external comparison used shallow local
snapshots of vLLM and SGLang, pinned in `external_sources.md`.  The external
trees are intentionally ignored by git; only their commit pins and inspected
facts are tracked.

The executable validation ran on the GPU VM:

- GPU: NVIDIA B200.
- Torch: `2.11.0+cu128`.
- PyTorch CUDA runtime: 12.8.
- vLLM runtime package: `0.20.1`.
- System CUDA is newer than the PyTorch wheel runtime, which is acceptable
  because the wheel bundles the CUDA runtime it was built against and relies on
  the installed NVIDIA driver for compatibility.

The final TeX build produced `math/mla/verification/mla.pdf`, 12 pages, with no
LaTeX errors, no rerun warning, and no overfull hbox warnings.

## What We Assumed Would Work

The core assumption was that a reliable mathematical derivation should be built
from three layers, in this order:

1. inspected implementation facts;
2. explicit tensor-shape notation and algebra;
3. executable tensor checks on deterministic cases.

That assumption held.  The strongest workflow was to first define a stable
dimension registry, then write independent Torch checks that expose named
intermediate tensors, then promote only claims that were either direct source
facts, algebraic consequences, or verified parity results.

The second assumption was that GPT-5.5-pro is most useful as a referee and
derivation expander, not as the source of truth.  That also held.  The useful
calls were narrow: cache ladder, materialized forward pass, projection
absorption, external implementation synthesis, and final step-by-step TeX
review.  The failed broad derivation review was a good counterexample: it spent
the output budget on hidden reasoning and produced no usable text.

The third assumption was that production implementations could validate the
shape contract without requiring official model weights.  That held.  vLLM
could run a one-layer DeepSeek-V2 config with dummy weights, capture actual MLA
runtime shapes, and generate a token.  This was enough to validate the
mathematical shape contract without downloading very large HF weights.

## What Worked Well

The final derivation is easiest to trust because it has separate artifacts for
different kinds of evidence:

- `mla.tex`: mathematical statement and derivation only.
- `notation.md`: dimension and layout registry.
- `source_ledger.md`: local source evidence and accepted source decisions.
- `external_sources.md`: vLLM/SGLang pins and external findings.
- `verification/*.py` and `verification/*.json`: executable tensor checks.
- `completion_gates.md`: explicit stop criteria.
- `tasks.jsonl` and `events.jsonl`: durable run history.

The GPU tensor verification was especially useful for this derivation because
MLA correctness is mostly about preserving many shape-sensitive equalities:

- local reference vs baseline;
- materialized score vs decomposed score;
- materialized score vs absorbed score;
- materialized context vs delayed value up-projection;
- compressed cache vs baseline;
- vLLM-style materialized vs absorbed contract at both small and profile
  dimensions.

The small manual two-head example was also valuable.  It forces the derivation
to be checkable by hand and prevents the TeX from becoming only a high-level
shape proof.

## What To Update In The Long-Horizon Protocol

1. Add an explicit `completion_gates.md` early for mathematical projects.
   The run improved once "done" meant concrete gates: source grounding, stable
   notation, no code listings in TeX, executable shape checks, clean PDF build,
   and valid ledgers.

2. Treat model calls as auditable proof-review tasks with narrow scope.
   Broad prompts are risky and can waste budget.  Better tasks ask for one
   lemma, one proof obligation, one external comparison, or one referee pass.

3. Keep final TeX mathematical, not implementation-driven.
   Implementation details belong in ledgers and verification artifacts.  The
   TeX should state assumptions, equations, shapes, validation contracts, and
   limits.

4. Require a shape registry before major derivation work.
   MLA is easy to confuse because `N`, `R`, `C`, `T`, `X`, and `V` interact in
   several layouts.  A stable registry prevented notation drift.

5. Promote GPU tensor computation to a first-class verifier.
   For neural-network math, a deterministic Torch checker with named
   intermediates is often the practical bridge between implementation and
   formal derivation.  It should run small synthetic dimensions first, then
   profile-shaped smoke checks when memory allows.

6. Use dummy-weight runtime probes for production frameworks.
   Running vLLM with a one-layer dummy DeepSeek-V2 config gave real runtime
   evidence without the cost and risk of cloning official weights.  This should
   be a standard comparator pattern for large-model architecture derivations.

7. Record failed reasoning calls, not just successful ones.
   The failed GPT-5.5-pro call showed that output budgeting and prompt scope
   matter.  Keeping it in `events.jsonl` and `tasks.jsonl` made the budget and
   process honest.

8. Separate dense base algebra from production runtime layers.
   Quantization, paging, sparse/NSA attention, CUDA graphs, fused kernels, and
   YaRN scaling should not be folded into the base proof without their own
   proof or parity task.

## Current Pause Point

The current pause point is a completed first version, not a dead end.  The most
natural future refinements are:

- add full-model mask and cache-offset parity tests against
  `modeling_deepseek.py`;
- add a separate YaRN/softmax-scale proof layer;
- add explicit proof obligations for paged or quantized cache layouts;
- improve the TeX exposition further if a human read-through finds unclear
  steps.

The project can resume from `completion_gates.md`, `state.md`, and `mla.tex`
without relying on chat history.

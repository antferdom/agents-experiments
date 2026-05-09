# Task: make `mla.tex` maximally easy to follow as a step-by-step mathematical derivation

You are reviewing and improving a TeX derivation of DeepSeek-style Multi-head
Latent Attention (MLA).

## Goal

Produce guidance and TeX-ready content that makes the derivation easy to follow
step by step while remaining rigorous and evidence-backed.

The user explicitly requested:

- `math/mla/mla.tex` should be the final mathematical derivation target.
- Do **not** put code implementations in `mla.tex`.
- Use code assertions, tests, and execution only to validate the mathematics.
- Use GPT-5.5-pro as needed for mathematical reasoning.
- Include a small hand-checkable two-head example.

## Acceptance Gates From `plan_mla.md`

A section is acceptable only if:

- source facts are cited or traceable to the source ledger;
- shapes are explicit and consistent;
- shape evolution covers the forward-pass operations at the mathematical level;
- assumptions are listed;
- tests or executable checks are concrete;
- parity tests pass when executable;
- untested parts are labeled as limits, not promoted as proved.

## Current Status

The current `mla.tex` is appended below.  It already builds and has verifier
artifacts, but it can be improved pedagogically.  The desired final document
should read like a derivation ladder:

1. define dimensions and evidence boundary;
2. derive MHA, MQA, GQA cache accounting;
3. define the MLA projections and split into non-RoPE/RoPE parts;
4. derive the compressed KV cache object;
5. materialize ordinary attention from the MLA projections;
6. decompose the score into non-RoPE and RoPE terms;
7. prove projection absorption for the non-RoPE key side;
8. prove delayed value up-projection;
9. explain why RoPE cannot be absorbed into the same static linear map;
10. state how vLLM/SGLang validate the same base algebra without embedding
    implementation code;
11. give executable shape assertions and validation results;
12. give a small two-head hand-checkable forward pass;
13. close with limits.

## Requested Output

Return three parts:

1. **Structural Critique**: identify unclear ordering, missing intermediate
   lemmas, confusing notation, or claims that need assumptions.
2. **TeX-Ready Replacement Sections**: provide improved TeX for any sections
   you recommend replacing or inserting.  Keep this mathematical.  Do not
   include code listings or implementation pseudocode.
3. **Completion Checklist**: a short checklist for deciding whether the final
   `mla.tex` is done.

Be concrete: use the notation \(B,L,M,D,H,N,R,V,Q,C,T,X,Y\), and preserve the
cache object \([z,\rho]\in\mathbb{R}^{B\times M\times(C+R)}\).  Be explicit
about row-vector versus column-vector orientation when needed; avoid ambiguity
in \(W^K_h\) and \(W^V_h\).  Keep the hand example small and manually
checkable.

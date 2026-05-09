## 1. Structural Critique

The current `mla.tex` is already mathematically close, but its pedagogy can be improved by making the document read more like a derivation ladder and less like a collection of related facts. The main issues I would fix are:

1. **Evidence boundary is too broad and not operational enough.**
   The current section names the source ledger and external source files, but it does not clearly distinguish:
   - pure mathematical consequences of definitions,
   - source facts from the DeepSeek-V2 profile / local implementation,
   - external implementation facts from vLLM/SGLang,
   - executable validation facts.
   Add an explicit traceability convention early so every nontrivial factual claim is either proved in the text or traceable to `source_ledger.md`, `external_sources.md`, or verification artifacts.

2. **Assumptions are scattered or implicit.**
   The proof relies on assumptions that should be stated before the derivation:
   - one-layer algebra;
   - deterministic inference path;
   - dropout disabled;
   - same additive mask if a mask is present;
   - base RoPE or identical RoPE transform in both paths;
   - static, position-independent \(W^K_h,W^V_h\);
   - bias-free up-projection for exact logit equality as written, or explicit handling of static biases;
   - real-arithmetic equality versus finite-precision parity;
   - attention scale remains \(1/\sqrt{T}\), not \(1/\sqrt{C+R}\), in the absorbed path.

3. **Row-vector versus column-vector orientation needs to be fixed once.**
   The current notation writes
   \[
   q' = (W^K_h)^\top q^{\mathrm n},
   \]
   which treats \(q^{\mathrm n}\) as a column vector and \(W^K_h\in\mathbb R^{N\times C}\). That is fine, but the document should explicitly say:
   - tensors store vectors along the last axis;
   - in mathematical equations, those vectors are column vectors;
   - framework row-major / row-vector linear-layer storage is not the mathematical orientation;
   - \(W^K_h z\in\mathbb R^N\), \(W^V_h z\in\mathbb R^V\).

4. **Projection definitions should remove ambiguity in \(W^K_h\) and \(W^V_h\).**
   The current text states
   \[
   W^{KV}\in\mathbb R^{H\times(N+V)\times C}
   \]
   and splits it, but it would be clearer to define the packed coordinate rule:
   \[
   (P_{kv,b}(z))_{(h-1)Y+j}
   =
   \sum_{c=1}^C W^{KV}_{h,j,c}z_c.
   \]
   Then define
   \[
   W^K_{h,n,c}=W^{KV}_{h,n,c},
   \qquad
   W^V_{h,\nu,c}=W^{KV}_{h,N+\nu,c}.
   \]
   This eliminates possible transpose confusion.

5. **Cache accounting should mention the shared-RoPE semi-materialized view.**
   The current MLA cache section says ordinary per-head materialization would cost
   \[
   H(T+V).
   \]
   That is true for an ordinary MHA-shaped cache that duplicates the RoPE suffix across heads. But MLA’s RoPE suffix \(\rho\) is shared across heads, so a useful intermediate comparison is:
   \[
   H(N+V)+R.
   \]
   This is not the compressed MLA cache, but it helps readers understand exactly what is being avoided.

6. **Materialized attention should be introduced first as ordinary attention.**
   The current `Materialized MLA Forward Pass` section begins directly with the decomposed score:
   \[
   \langle q^{\mathrm n},k^{\mathrm n}\rangle
   +
   \langle q^{\mathrm r},\rho\rangle.
   \]
   Pedagogically, first define the full ordinary vectors
   \[
   Q=[q^{\mathrm n},q^{\mathrm r}],
   \qquad
   K=[k^{\mathrm n},\rho],
   \]
   then define ordinary attention
   \[
   \tilde r=\langle Q,K\rangle,
   \]
   and only afterward decompose the score. This makes the derivation ladder clearer.

7. **Projection absorption and delayed value up-projection should be separate lemmas.**
   The current `Projection Absorption` section proves both key-side absorption and value-side delay. They are two different linearity arguments:
   - key-side absorption moves \(W^K_h\) from the key side into the query side before softmax;
   - value-side delay moves \(W^V_h\) after the memory weighted sum.
   Separating them makes it easier to check the assumptions and shapes for each.

8. **RoPE non-absorption needs a stronger argument.**
   The current section says RoPE is position dependent. That is true, but the rigorous statement should be:
   if a static matrix \(A\) absorbed RoPE for all positions, then for all \(p^q,p^{kv}\),
   \[
   A=\mathcal R_{p^q}^{\top}\mathcal R_{p^{kv}},
   \]
   which is impossible for non-degenerate RoPE because the right-hand side depends on relative position. This also clarifies that the relative-position identity
   \[
   \langle \mathcal R_i x,\mathcal R_j y\rangle
   =
   x^\top \mathcal R_{j-i}y
   \]
   does not provide a single static map.

9. **External implementation discussion mixes evidence, theorem, and runtime observations.**
   The vLLM/SGLang section should be reorganized into:
   - source-backed facts;
   - the mathematical contract those implementations validate;
   - what the runtime probe observed;
   - what remains outside the proof.
   Avoid making backend-specific layouts look like part of the base mathematical theorem.

10. **Executable validation should be described as contracts, not implementation.**
    The current shape assertions are good, but the final document should explicitly say the scripts are used only to validate the math, not to define it. It should also list exact verification artifacts or point to the ledger/run logs for exact artifact names.

11. **The two-head example should also show the absorbed path.**
    The current hand example checks materialized attention, but the key pedagogical point is that the absorbed path gives the same answer. Add:
    \[
    q'_h=(W^K_h)^\top q^{\mathrm n}_h,
    \qquad
    g_h=\sum_m a_{h,m}z_m,
    \qquad
    o_h=W^V_h g_h.
    \]
    This makes the example a manual proof of both sides.

12. **Limits should be sharper.**
    The current `Open Limits` section is good, but should explicitly distinguish:
    - algebra that extends immediately if the same mask is added to both logits;
    - production features not proved here: dropout, YaRN variants, quantized/paged/sparse/NSA layouts, cache mutation semantics, backend-specific softmax scaling, and bitwise floating-point identity.

---

## 2. TeX-Ready Replacement Sections

Below is a drop-in replacement body for the main derivation sections. It uses the existing preamble from your appended `mla.tex`. I would replace everything from `\section{Evidence Boundary}` through `\section{Open Limits}` with the following.

```tex
\section{Evidence Boundary and Traceability}

This document derives the base algebra of DeepSeek-style Multi-head Latent
Attention (MLA) for one attention layer.  The final mathematical target is the
compressed cache object
\[
  \cache_{\mathrm{mla}}=[z,\rho]\in\R^{B\times M\times(C+R)}
\]
and the proof that the materialized attention computation and the absorbed
latent-cache computation produce the same attention output.

\paragraph{Traceability convention.}
A claim in this document is one of four kinds:
\begin{enumerate}
  \item a \emph{definition} introduced in this document;
  \item a \emph{mathematical consequence} proved from the definitions;
  \item a \emph{source fact} traceable to
        \path{math/mla/source_ledger.md},
        \path{math/mla/deepseekv2-profile}, or
        \path{math/mla/notation.md};
  \item an \emph{external or executable validation fact} traceable to
        \path{math/mla/external_sources.md} or the run logs under
        \path{math/mla/verification}.
\end{enumerate}
The TeX derivation does not embed implementation code.  Code assertions,
runtime probes, and tests are used only to validate the mathematical contracts
stated here.

\paragraph{Algebraic target.}
The executable target for the core equivalence is the simplified local algebra:
dropout disabled, deterministic inference, no stochastic training behavior, and
base RoPE.  If an additive attention mask
\(\mu_{b,\ell,m}\) is present and is added to both forms of the same logits,
the algebra below is unchanged after replacing each score \(s_{b,h,\ell,m}\) by
\(s_{b,h,\ell,m}+\mu_{b,\ell,m}\).

\paragraph{What is proved.}
Conditional on the tensors and projections defined below, the document proves
in real arithmetic that:
\[
  \text{materialize } (K,V) \text{ then attend}
  \quad\equiv\quad
  \text{attend to } [z,\rho] \text{ with absorbed queries, then up-project}.
\]
The proof is independent of a particular kernel, cache page layout, CUDA graph
constraint, or quantized storage format.

\paragraph{What is validated.}
The verification artifacts check shapes, score decomposition, projection
absorption, delayed value up-projection, compressed-cache parity, and selected
vLLM/SGLang-style contracts.  These tests catch mistakes in the mathematical
model and in tensor reshaping, but untested production details are listed as
limits at the end.

\section{Dimensions, Orientation, and Assumptions}

For an integer \(n\), write
\[
  [n]=\{1,\ldots,n\}.
\]

\subsection{Dimensions}

\begin{center}
\begin{tabular}{lll}
\toprule
Symbol & Meaning & DeepSeek-V2 profile value \\
\midrule
\(B\) & batch size & runtime \\
\(L\) & query sequence length & runtime \\
\(M\) & memory / key-value sequence length & runtime \\
\(D\) & model hidden dimension & \(5120\) \\
\(H\) & number of query heads & \(128\) \\
\(N\) & non-RoPE query/key head dimension & \(128\) \\
\(R\) & RoPE query/key suffix dimension & \(64\) \\
\(V\) & materialized value head dimension & \(128\) \\
\(Q\) & query compression rank & \(1536\) \\
\(C\) & KV compression rank & \(512\) \\
\(T\) & full query/key head width, \(T=N+R\) & \(192\) \\
\(X\) & packed KV down-projection width, \(X=C+R\) & \(576\) \\
\(Y\) & packed KV up-projection width per head, \(Y=N+V\) & \(256\) \\
\bottomrule
\end{tabular}
\end{center}

\paragraph{Trace.}
The numeric profile values in the table are source facts traceable to the local
DeepSeek-V2 profile and the source ledger.  The symbols \(B,L,M\) are runtime
sequence/batch quantities and are not fixed by the profile.

\subsection{Orientation convention}

All tensor shapes are written in the order used by the derivation.  A tensor
such as \(z\in\R^{B\times M\times C}\) stores one \(C\)-dimensional vector on
its last axis.  In algebraic equations, last-axis vectors are interpreted as
column vectors.  Thus, for each head \(h\),
\[
  W^K_h\in\R^{N\times C},
  \qquad
  W^V_h\in\R^{V\times C},
\]
and for \(z_m\in\R^C\),
\[
  W^K_h z_m\in\R^N,
  \qquad
  W^V_h z_m\in\R^V.
\]
Framework storage conventions for linear layers may use row-major weights or
row-vector multiplication.  The matrices above are the abstract mathematical
maps, not a claim about memory layout.

\subsection{Assumptions used by the proof}

\begin{enumerate}
  \item \textbf{Single-layer algebra.}
        All equations are for one attention layer.  Layer indices are omitted.

  \item \textbf{Last-axis maps.}
        \(P_{q,a},P_{q,b},P_{kv,a},P_{kv,b},P_o,\LN_q,\LN_{kv}\)
        act independently on the last axis of their inputs.

  \item \textbf{Deterministic normalization.}
        The layer normalizations are evaluated identically in both paths.
        They need not be linear because the absorption proof starts after
        \(q^{\mathrm n},q^{\mathrm r},z,\rho\) have already been produced.

  \item \textbf{Static KV up-projections.}
        \(W^K_h\) and \(W^V_h\) are position-independent learned linear maps.
        The local DeepSeek-style profile is treated as bias-free for these
        up-projections in the exact logit equality written below.  If a
        different model has static key/value biases, the key bias contributes
        \(q^{\mathrm n\top}b^K_h\), which is constant over memory index \(m\)
        for fixed \((b,h,\ell)\) and therefore cancels in softmax; a value bias
        may be added after the delayed value projection because attention
        probabilities sum to one.

  \item \textbf{RoPE convention.}
        For position \(p\), write
        \[
          \RoPE_p(x)=\mathcal R_p x,
          \qquad
          \mathcal R_p\in\R^{R\times R}.
        \]
        Base RoPE has \(\mathcal R_p\) block-orthogonal and position dependent.
        The proof requires only that the same RoPE transform is used in both
        the materialized and absorbed computations.  RoPE variants such as YaRN
        are outside the base theorem unless checked separately.

  \item \textbf{Attention scale.}
        The softmax scale is inherited from the materialized query/key head
        width:
        \[
          \frac{1}{\sqrt{T}}=\frac{1}{\sqrt{N+R}}.
        \]
        In the absorbed path the score can be written as a dot product against
        \([z,\rho]\in\R^{C+R}\), but the scale remains \(1/\sqrt{T}\), not
        \(1/\sqrt{C+R}\), except in accidental toy cases where \(C=N\).

  \item \textbf{Mask and dropout.}
        The core verifier uses no attention mask and dropout is disabled.  If a
        shared additive mask is added to both forms of the logits, the equality
        of logits before masking implies equality after masking.  Training
        dropout is not included in the theorem.

  \item \textbf{Arithmetic.}
        The symbolic equalities are real-arithmetic equalities.  Executable
        parity is expected only up to floating-point evaluation order and dtype
        effects.
\end{enumerate}

\section{Cache Accounting for MHA, MQA, GQA, and MLA}

For one layer, define the scalar cache count per memory token per batch element
as
\[
  e(\cache)=\frac{\elts(\cache)}{BM}.
\]
This count ignores dtype bytes, page padding, metadata, and backend-specific
storage overhead.  Bytes are obtained only after multiplying by dtype size and
accounting for the backend layout.

\subsection{MHA}

Standard multi-head attention stores a key and value stream for every query
head:
\[
  K^{\mathrm{mha}}\in\R^{B\times H\times M\times T},
  \qquad
  V^{\mathrm{mha}}\in\R^{B\times H\times M\times V}.
\]
Therefore
\[
\begin{aligned}
  e(\cache_{\mathrm{mha}})
  &=
  \frac{BHMT+BHMV}{BM} \\
  &=H(T+V).
\end{aligned}
\]
For the DeepSeek-V2 profile this is
\[
  128(192+128)=40960
\]
scalars per token per layer.

\subsection{MQA}

Multi-query attention keeps \(H\) query heads but shares one key/value stream:
\[
  K^{\mathrm{mqa}}\in\R^{B\times 1\times M\times T},
  \qquad
  V^{\mathrm{mqa}}\in\R^{B\times 1\times M\times V}.
\]
Thus
\[
  e(\cache_{\mathrm{mqa}})=T+V.
\]
For the profile values this is
\[
  192+128=320.
\]

\subsection{GQA}

Grouped-query attention has \(G\) key/value groups serving \(H\) query heads.
Assume \(G\mid H\), and define the group serving head \(h\) by
\[
  g(h)=1+\left\lfloor\frac{h-1}{H/G}\right\rfloor .
\]
The cache is
\[
  K^{\mathrm{gqa}}\in\R^{B\times G\times M\times T},
  \qquad
  V^{\mathrm{gqa}}\in\R^{B\times G\times M\times V}.
\]
Therefore
\[
  e(\cache_{\mathrm{gqa}})=G(T+V).
\]
The endpoints are MQA when \(G=1\) and MHA when \(G=H\).

\subsection{MLA}

MLA can produce head-specific non-RoPE keys and values, but it does not cache
them in the compressed representation.

If one materializes an ordinary MHA-shaped cache that duplicates the RoPE
suffix across all heads, the scalar count is again
\[
  H(T+V)=H(N+R+V).
\]
Because the MLA RoPE suffix \(\rho\) is shared across heads, an intermediate
semi-materialized cache that stores head-specific non-RoPE keys and values but
stores \(\rho\) only once would have count
\[
  H(N+V)+R.
\]
For the profile values this is
\[
  128(128+128)+64=32832.
\]
The actual compressed MLA cache stores only the normalized KV latent and the
shared rotated key RoPE suffix:
\[
  \cache_{\mathrm{mla}}
  =
  [z,\rho]
  \in\R^{B\times M\times(C+R)}.
\]
Thus
\[
  e(\cache_{\mathrm{mla}})=C+R.
\]
For the DeepSeek-V2 profile this is
\[
  512+64=576
\]
scalars per token per layer.  This is larger than the MQA scalar count
\(320\) for the same dimensions, but MLA retains head-specific key and value
up-projections through \(W^K_h\) and \(W^V_h\), rather than forcing a single
materialized KV head.

\section{MLA Projection Definitions}

Let
\[
  h^q\in\R^{B\times L\times D},
  \qquad
  h^{kv}\in\R^{B\times M\times D}.
\]
For self-attention, \(h^q\) and \(h^{kv}\) come from the same layer input but
may represent different query and memory positions during cached decoding.

The learned maps corresponding to the local DeepSeek-style MLA profile are
\[
  P_{q,a}:\R^D\to\R^Q,
  \qquad
  P_{q,b}:\R^Q\to\R^{HT},
\]
\[
  P_{kv,a}:\R^D\to\R^{C+R},
  \qquad
  P_{kv,b}:\R^C\to\R^{H(N+V)}=\R^{HY},
  \qquad
  P_o:\R^{HV}\to\R^D.
\]
The layer normalizations act on the last dimension:
\[
  \LN_q:\R^Q\to\R^Q,
  \qquad
  \LN_{kv}:\R^C\to\R^C.
\]

\paragraph{Packed KV up-projection orientation.}
For \(z\in\R^C\), write
\[
  \bar w=P_{kv,b}(z)\in\R^{HY}.
\]
Define the abstract packed weight tensor
\[
  W^{KV}\in\R^{H\times Y\times C}
\]
by the coordinate rule
\[
  \bar w_{(h-1)Y+j}
  =
  \sum_{c=1}^C W^{KV}_{h,j,c}z_c,
  \qquad
  h\in[H],\ j\in[Y].
\]
The key and value slices are then unambiguous:
\[
  W^K_{h,n,c}=W^{KV}_{h,n,c},
  \qquad
  n\in[N],
\]
\[
  W^V_{h,\nu,c}=W^{KV}_{h,N+\nu,c},
  \qquad
  \nu\in[V].
\]
Hence
\[
  W^K_h\in\R^{N\times C},
  \qquad
  W^V_h\in\R^{V\times C}.
\]

\paragraph{Trace.}
The projection names, split points, and profile dimensions are local source
facts traceable through \path{math/mla/source_ledger.md}.  The derivation uses
only their mathematical domains, codomains, and split structure.

\section{Query Path: Low-Rank Query and RoPE Split}

The query low-rank path is
\[
  q^{\mathrm{lat}}
  =
  \LN_q(P_{q,a}(h^q))
  \in\R^{B\times L\times Q}.
\]
The query up-projection gives
\[
  \bar q
  =
  P_{q,b}(q^{\mathrm{lat}})
  \in\R^{B\times L\times HT}.
\]
Viewing the packed last dimension as \(H\) heads of width \(T\),
\[
  q^{\mathrm{full}}_{b,h,\ell,t}
  =
  \bar q_{b,\ell,(h-1)T+t},
  \qquad
  q^{\mathrm{full}}\in\R^{B\times H\times L\times T}.
\]
Split the head dimension into a non-RoPE prefix and a RoPE suffix:
\[
  q^{\mathrm{full}}=[q^{\mathrm n},q^{\mathrm r,raw}],
\]
where
\[
  q^{\mathrm n}\in\R^{B\times H\times L\times N},
  \qquad
  q^{\mathrm r,raw}\in\R^{B\times H\times L\times R}.
\]
For query position id \(p^q_{b,\ell}\), RoPE acts only on the suffix:
\[
  q^{\mathrm r}_{b,h,\ell,:}
  =
  \RoPE_{p^q_{b,\ell}}
  \!\left(q^{\mathrm r,raw}_{b,h,\ell,:}\right)
  \in\R^R.
\]
The full materialized query vector for ordinary attention is
\[
  Q_{b,h,\ell,:}
  =
  [q^{\mathrm n}_{b,h,\ell,:},q^{\mathrm r}_{b,h,\ell,:}]
  \in\R^T.
\]

The query RoPE suffix is produced by the query path and then rotated.  It is
not produced by the KV RoPE projection.

\section{KV Path: Compressed Latent and Shared RoPE Suffix}

The packed KV down-projection is
\[
  u=P_{kv,a}(h^{kv})\in\R^{B\times M\times(C+R)}.
\]
Split it as
\[
  u=[c,k^{\mathrm r,raw}],
  \qquad
  c\in\R^{B\times M\times C},
  \qquad
  k^{\mathrm r,raw}\in\R^{B\times M\times R}.
\]
Normalize the latent part:
\[
  z=\LN_{kv}(c)\in\R^{B\times M\times C}.
\]
For KV position id \(p^{kv}_{b,m}\), rotate the RoPE suffix:
\[
  \rho_{b,m,:}
  =
  \RoPE_{p^{kv}_{b,m}}
  \!\left(k^{\mathrm r,raw}_{b,m,:}\right)
  \in\R^R.
\]
The compressed cache token is exactly
\[
  \cache_{b,m,:}
  =
  [z_{b,m,:},\rho_{b,m,:}]
  \in\R^{C+R},
\]
and the layer cache is
\[
  \cache_{\mathrm{mla}}
  =
  [z,\rho]\in\R^{B\times M\times(C+R)}.
\]
No per-head \(k^{\mathrm n}\) or \(v\) tensor is part of the compressed cache.

\section{Materialized Attention from the MLA Projections}

A first way to run the attention is to reconstruct ordinary per-head key and
value tensors from the compressed objects.  For each head \(h\),
\[
  k^{\mathrm n}_{b,h,m,:}
  =
  W^K_h z_{b,m,:}
  \in\R^N,
\]
or in coordinates,
\[
  k^{\mathrm n}_{b,h,m,n}
  =
  \sum_{c=1}^C W^K_{h,n,c}z_{b,m,c}.
\]
Similarly,
\[
  v_{b,h,m,:}
  =
  W^V_h z_{b,m,:}
  \in\R^V,
\]
or
\[
  v_{b,h,m,\nu}
  =
  \sum_{c=1}^C W^V_{h,\nu,c}z_{b,m,c}.
\]
The full materialized key vector is
\[
  K_{b,h,m,:}
  =
  [k^{\mathrm n}_{b,h,m,:},\rho_{b,m,:}]
  \in\R^T.
\]
The RoPE suffix \(\rho_{b,m,:}\) is broadcast across heads.

Ordinary attention then uses the full query and key vectors:
\[
  \tilde r_{b,h,\ell,m}
  =
  \left\langle Q_{b,h,\ell,:},K_{b,h,m,:}\right\rangle .
\]
The scaled score is
\[
  s_{b,h,\ell,m}
  =
  \frac{\tilde r_{b,h,\ell,m}}{\sqrt{T}}.
\]
If a shared additive mask \(\mu_{b,\ell,m}\) is present, use
\(s_{b,h,\ell,m}+\mu_{b,\ell,m}\) in both the materialized and absorbed forms.
The simplified local verifier sets \(\mu=0\).

Attention probabilities are
\[
  a_{b,h,\ell,m}
  =
  \frac{\exp(s_{b,h,\ell,m})}
       {\sum_{j=1}^M \exp(s_{b,h,\ell,j})}.
\]
The materialized context is
\[
  o_{b,h,\ell,\nu}
  =
  \sum_{m=1}^M a_{b,h,\ell,m}v_{b,h,m,\nu}
  \in\R^{B\times H\times L\times V}.
\]
Flatten heads in increasing head order:
\[
  x_{b,\ell,(h-1)V+\nu}
  =
  o_{b,h,\ell,\nu},
  \qquad
  x\in\R^{B\times L\times HV}.
\]
The final output is
\[
  y_{b,\ell,:}=P_o(x_{b,\ell,:})\in\R^D.
\]

\section{Score Decomposition}

The full materialized score decomposes because both \(Q\) and \(K\) are
concatenations of a non-RoPE part and a RoPE part:
\[
  Q_{b,h,\ell,:}
  =
  [q^{\mathrm n}_{b,h,\ell,:},q^{\mathrm r}_{b,h,\ell,:}],
\]
\[
  K_{b,h,m,:}
  =
  [k^{\mathrm n}_{b,h,m,:},\rho_{b,m,:}].
\]
Therefore
\[
\begin{aligned}
  \tilde r_{b,h,\ell,m}
  &=
  \left\langle
    [q^{\mathrm n}_{b,h,\ell,:},q^{\mathrm r}_{b,h,\ell,:}],
    [k^{\mathrm n}_{b,h,m,:},\rho_{b,m,:}]
  \right\rangle \\
  &=
  \left\langle
    q^{\mathrm n}_{b,h,\ell,:},
    k^{\mathrm n}_{b,h,m,:}
  \right\rangle
  +
  \left\langle
    q^{\mathrm r}_{b,h,\ell,:},
    \rho_{b,m,:}
  \right\rangle .
\end{aligned}
\]
This is the first key separation: the non-RoPE term contains the learned
head-specific map \(W^K_h\), while the RoPE term contains the position-dependent
rotated suffix \(\rho\).

\section{Key-Side Projection Absorption}

The non-RoPE key term can be evaluated without materializing
\(k^{\mathrm n}\).  Define the absorbed query latent
\[
  q'_{b,h,\ell,:}
  =
  (W^K_h)^\top q^{\mathrm n}_{b,h,\ell,:}
  \in\R^C,
\]
or in coordinates,
\[
  q'_{b,h,\ell,c}
  =
  \sum_{n=1}^N q^{\mathrm n}_{b,h,\ell,n}W^K_{h,n,c}.
\]
Then, for each \((b,h,\ell,m)\),
\[
\begin{aligned}
  \left\langle
    q^{\mathrm n}_{b,h,\ell,:},
    k^{\mathrm n}_{b,h,m,:}
  \right\rangle
  &=
  \left(q^{\mathrm n}_{b,h,\ell,:}\right)^\top
  W^K_h z_{b,m,:} \\
  &=
  \left((W^K_h)^\top q^{\mathrm n}_{b,h,\ell,:}\right)^\top
  z_{b,m,:} \\
  &=
  \left\langle q'_{b,h,\ell,:},z_{b,m,:}\right\rangle .
\end{aligned}
\]
Equivalently, in coordinate form,
\[
\begin{aligned}
  \sum_{n=1}^N q^{\mathrm n}_{b,h,\ell,n}
      \sum_{c=1}^C W^K_{h,n,c}z_{b,m,c}
  &=
  \sum_{c=1}^C
      \left(\sum_{n=1}^N
        q^{\mathrm n}_{b,h,\ell,n}W^K_{h,n,c}
      \right)
      z_{b,m,c}.
\end{aligned}
\]

Thus the absorbed unscaled score is
\[
  \tilde r^{\mathrm{abs}}_{b,h,\ell,m}
  =
  \left\langle q'_{b,h,\ell,:},z_{b,m,:}\right\rangle
  +
  \left\langle q^{\mathrm r}_{b,h,\ell,:},\rho_{b,m,:}\right\rangle .
\]
Equivalently,
\[
  \tilde r^{\mathrm{abs}}_{b,h,\ell,m}
  =
  \left\langle
    [q'_{b,h,\ell,:},q^{\mathrm r}_{b,h,\ell,:}],
    [z_{b,m,:},\rho_{b,m,:}]
  \right\rangle .
\]
The dot product in this last display has width \(C+R\), but the softmax scale is
still \(1/\sqrt{T}\).  By the calculation above,
\[
  \tilde r^{\mathrm{abs}}_{b,h,\ell,m}
  =
  \tilde r_{b,h,\ell,m}
\]
exactly in real arithmetic.

\section{Delayed Value Up-Projection}

Because the absorbed and materialized logits are identical, they produce the
same attention probabilities:
\[
  a^{\mathrm{abs}}_{b,h,\ell,m}
  =
  a_{b,h,\ell,m}.
\]
Define the latent context
\[
  g_{b,h,\ell,c}
  =
  \sum_{m=1}^M a_{b,h,\ell,m}z_{b,m,c}
  \in\R^{B\times H\times L\times C}.
\]
Then the value up-projection can be delayed until after the weighted memory
sum:
\[
\begin{aligned}
  \sum_{c=1}^C W^V_{h,\nu,c}g_{b,h,\ell,c}
  &=
  \sum_{c=1}^C W^V_{h,\nu,c}
      \sum_{m=1}^M a_{b,h,\ell,m}z_{b,m,c} \\
  &=
  \sum_{m=1}^M a_{b,h,\ell,m}
      \sum_{c=1}^C W^V_{h,\nu,c}z_{b,m,c} \\
  &=
  \sum_{m=1}^M a_{b,h,\ell,m}v_{b,h,m,\nu} \\
  &=
  o_{b,h,\ell,\nu}.
\end{aligned}
\]
Thus the absorbed path produces the same pre-output context as the materialized
path:
\[
  W^V_h g_{b,h,\ell,:}
  =
  o_{b,h,\ell,:}.
\]
After flattening heads in the same order and applying the same output
projection \(P_o\), both paths produce the same \(y\), up to floating-point
evaluation order.

\section{Why the RoPE Suffix Is Not Absorbed into the Same Static Map}

The key-side absorption above works because \(W^K_h\) is a static,
position-independent linear map:
\[
  \left(q^{\mathrm n}\right)^\top W^K_h z
  =
  \left((W^K_h)^\top q^{\mathrm n}\right)^\top z.
\]
RoPE is different.  Let
\[
  q^{\mathrm r}=\mathcal R_{p^q}q^{\mathrm r,raw},
  \qquad
  \rho=\mathcal R_{p^{kv}}k^{\mathrm r,raw}.
\]
Then
\[
\begin{aligned}
  \left\langle q^{\mathrm r},\rho\right\rangle
  &=
  \left(q^{\mathrm r,raw}\right)^\top
  \mathcal R_{p^q}^{\top}
  \mathcal R_{p^{kv}}
  k^{\mathrm r,raw}.
\end{aligned}
\]
For a non-degenerate RoPE, the matrix
\[
  \mathcal R_{p^q}^{\top}\mathcal R_{p^{kv}}
\]
depends on the relative position.  If a single static matrix \(A\) absorbed the
RoPE term for all positions, then for all raw vectors \(x,y\) and all positions
\(i,j\),
\[
  x^\top A y
  =
  x^\top \mathcal R_i^\top\mathcal R_j y.
\]
Therefore
\[
  A=\mathcal R_i^\top\mathcal R_j
\]
for all \(i,j\).  Taking \(i=j\) gives \(A=I\), while taking two positions with
a nonzero relative RoPE rotation gives \(A\ne I\), a contradiction.

The familiar relative-position identity,
\[
  \left\langle
    \mathcal R_i x,
    \mathcal R_j y
  \right\rangle
  =
  x^\top\mathcal R_i^\top\mathcal R_j y,
\]
does not provide a single static absorption matrix; it provides a
position-dependent bilinear form.  Therefore the MLA cache keeps the rotated
RoPE key suffix explicitly:
\[
  \rho_{b,m,:}
  =
  \RoPE_{p^{kv}_{b,m}}
  \!\left(k^{\mathrm r,raw}_{b,m,:}\right),
\]
and the compressed cache is not merely \(z\in\R^C\).  It is
\[
  [z,\rho]\in\R^{B\times M\times(C+R)}.
\]

\section{External Implementation Validation Without Embedding Code}

The symbolic proof above is independent of any specific serving backend.  The
vLLM and SGLang checks are used as external validation that production-oriented
implementations optimize the same base algebra.

\begin{center}
\begin{tabular}{p{0.24\linewidth}p{0.34\linewidth}p{0.32\linewidth}}
\toprule
Source class & Source-backed fact & Mathematical role \\
\midrule
Local profile and source ledger &
The profile dimensions are
\(D=5120,H=128,N=128,R=64,V=128,Q=1536,C=512\).
The local projection split is
\(P_{kv,a}:\R^D\to\R^{C+R}\) followed by
\(P_{kv,b}:\R^C\to\R^{H(N+V)}\). &
Fixes the dimensions and projection domains/codomains used in the derivation. \\
\addlinespace
Pinned vLLM snapshot &
The compute-friendly path materializes non-RoPE keys and values from the latent;
the data-movement-friendly path uses absorbed queries and a semantic cache
width \(C+R\). &
Validates the same two algebraic forms:
materialized attention and absorbed latent-cache attention. \\
\addlinespace
Pinned SGLang snapshot &
The absorbed MLA decode shape is represented with one KV stream, head dimension
\(C+R\), and value latent dimension \(C\), with latent part followed by RoPE
suffix in the MLA buffer. &
Validates the cache object
\([z,\rho]\in\R^{B\times M\times(C+R)}\)
and delayed value up-projection contract. \\
\addlinespace
Runtime vLLM dummy probe &
With dummy weights and a one-layer profile, the observed attention-related
shapes were
\(\R^{S\times128\times192}\),
\(\R^{S\times512}\), and
\(\R^{S\times1\times64}\);
the pre-output width was \(128\cdot128=16384=HV\). &
Checks that the profile-level tensor shapes match the derivation without
requiring official weights. \\
\bottomrule
\end{tabular}
\end{center}

\paragraph{External mathematical contract.}
For fixed \((b,h,\ell)\), suppress those indices and let
\[
  \alpha_m
  =
  \softmax_m\!\left(
    \frac{
      (q^{\mathrm n})^\top W^K_h z_m
      +
      (q^{\mathrm r})^\top\rho_m
    }{\sqrt{T}}
  \right).
\]
The compute-friendly materialized output is
\[
  o^{\mathrm{mat}}_\nu
  =
  \sum_{m=1}^M \alpha_m
  \sum_{c=1}^C W^V_{h,\nu,c}z_{m,c}.
\]
The data-movement-friendly absorbed output is
\[
  o^{\mathrm{abs}}_\nu
  =
  \sum_{c=1}^C W^V_{h,\nu,c}
  \sum_{m=1}^M \alpha_m z_{m,c}.
\]
By exchanging finite sums,
\[
  o^{\mathrm{mat}}_\nu=o^{\mathrm{abs}}_\nu.
\]
The logits defining \(\alpha_m\) are identical because
\[
  (q^{\mathrm n})^\top W^K_h z_m
  =
  \left((W^K_h)^\top q^{\mathrm n}\right)^\top z_m.
\]
This is the base algebra optimized by the external implementations.  Paged
cache tables, quantized storage, sparse/NSA indexers, CUDA graph constraints,
and specialized kernels alter storage or execution strategy, not these
equalities.

\section{Executable Shape Assertions and Validation Results}

The executable checks assert the following mathematical shape evolution.  The
checks are not implementation definitions; they validate the tensor contracts
used in the derivation.

\subsection{Shape ladder}

The query path is
\[
\begin{array}{rcl}
  h^q\in\R^{B\times L\times D}
  &\longmapsto&
  q^{\mathrm{lat}}\in\R^{B\times L\times Q}
  \longmapsto
  \bar q\in\R^{B\times L\times HT}
  \\[2pt]
  &\longmapsto&
  q^{\mathrm{full}}\in\R^{B\times H\times L\times T}
  \longmapsto
  q^{\mathrm n}\in\R^{B\times H\times L\times N},
  \quad
  q^{\mathrm r}\in\R^{B\times H\times L\times R}.
\end{array}
\]

The KV down path and cache construction are
\[
\begin{array}{rcl}
  h^{kv}\in\R^{B\times M\times D}
  &\longmapsto&
  u\in\R^{B\times M\times(C+R)}
  \\[2pt]
  &\longmapsto&
  c\in\R^{B\times M\times C},
  \quad
  k^{\mathrm r,raw}\in\R^{B\times M\times R}
  \\[2pt]
  &\longmapsto&
  z\in\R^{B\times M\times C},
  \quad
  \rho\in\R^{B\times M\times R}
  \\[2pt]
  &\longmapsto&
  [z,\rho]\in\R^{B\times M\times(C+R)}.
\end{array}
\]

The materialized path asserts
\[
\begin{array}{rcl}
  z
  &\longmapsto&
  k^{\mathrm n}\in\R^{B\times H\times M\times N},
  \quad
  v\in\R^{B\times H\times M\times V},
  \\[2pt]
  [q^{\mathrm n},q^{\mathrm r}]
  &\in&
  \R^{B\times H\times L\times T},
  \\[2pt]
  [k^{\mathrm n},\rho]
  &\in&
  \R^{B\times H\times M\times T},
  \\[2pt]
  s
  &\in&
  \R^{B\times H\times L\times M},
  \\[2pt]
  o
  &\in&
  \R^{B\times H\times L\times V},
  \qquad
  y\in\R^{B\times L\times D}.
\end{array}
\]

The absorbed path asserts
\[
\begin{array}{rcl}
  q'=(W^K_h)^\top q^{\mathrm n}
  &\in&
  \R^{B\times H\times L\times C},
  \\[2pt]
  [q',q^{\mathrm r}]
  &\text{attends against}&
  [z,\rho]\in\R^{B\times M\times(C+R)},
  \\[2pt]
  s^{\mathrm{abs}}
  &\in&
  \R^{B\times H\times L\times M},
  \\[2pt]
  g
  &\in&
  \R^{B\times H\times L\times C},
  \\[2pt]
  W^V_h g
  &\in&
  \R^{B\times H\times L\times V},
  \qquad
  y\in\R^{B\times L\times D}.
\end{array}
\]
In both paths the score scale is \(1/\sqrt{T}\).

\subsection{Deterministic algebra verifier}

The main deterministic verifier ran on an NVIDIA B200 GPU with
\texttt{torch==2.11.0+cu128}.  The checked small shape was
\[
  B=2,\ L=3,\ M=5,\ D=16,\ H=4,\ N=3,\ R=2,\ V=3,\ Q=5,\ C=6.
\]
All checked equalities passed:

\begin{center}
\begin{tabular}{lr}
\toprule
Check & Max absolute error \\
\midrule
Independent reference vs local baseline & \(5.96\cdot 10^{-8}\) \\
Concatenated score vs decomposed score & \(1.19\cdot 10^{-7}\) \\
Concatenated score vs absorbed score & \(1.19\cdot 10^{-7}\) \\
Materialized context vs absorbed context & \(5.96\cdot 10^{-8}\) \\
Compressed-cache implementation vs baseline & \(0\) \\
Absorbed implementation vs baseline & \(5.96\cdot 10^{-8}\) \\
\bottomrule
\end{tabular}
\end{center}

The standalone vLLM-oriented contract trace checks both a small shape and the
profile shape.  On the profile shape, the output max absolute difference was
\(2.62\cdot 10^{-6}\), consistent with large float32 reductions.  The actual
vLLM dummy-engine probe loaded a one-layer DeepSeek-V2-style config, captured
the MLA tensor shapes described above, and generated one token with dummy
weights; no official weights were downloaded.

The article export was reproduced with
\path{tools/zhihu-to-markdown/export_article.py}; LaTeX integrity passed with
101 fragments.  The TeX build result is tracked in
\path{math/mla/verification/tex_build.md}.

\section{A Two-Head Manual Forward Pass}

This example is intentionally small enough to check by hand.  It starts after
the projections have produced \(q^{\mathrm n}\), \(q^{\mathrm r}\), \(z\), and
\(\rho\).  Therefore the hidden states and projection matrices before that
point need not be specified.

Let
\[
  B=1,\quad L=1,\quad M=2,\quad D=2,\quad H=2,
\]
\[
  Q=C=1,\quad N=V=1,\quad R=2.
\]
Then
\[
  T=N+R=3,
  \qquad
  C+R=3.
\]
Choose zero RoPE suffixes so that the RoPE-shaped tensors are present but do
not affect the arithmetic:
\[
  q^{\mathrm r}_{h=1}=q^{\mathrm r}_{h=2}=(0,0),
  \qquad
  \rho_1=\rho_2=(0,0).
\]
The compressed cache tokens are therefore
\[
  \cache_{1}=[z_1,\rho_1]=[1,0,0],
  \qquad
  \cache_{2}=[z_2,\rho_2]=[2,0,0].
\]
Let the non-RoPE query scalars be
\[
  q^{\mathrm n}_{h=1}=2,
  \qquad
  q^{\mathrm n}_{h=2}=-1.
\]
Choose per-head scalar up-projections
\[
  W^K_1=1,\quad W^V_1=2,
  \qquad
  W^K_2=-1,\quad W^V_2=1.
\]

\subsection*{Materialized calculation}

For head \(1\),
\[
  k^{\mathrm n}_{1}=(1,2),
  \qquad
  v_{1}=(2,4).
\]
The unscaled scores are
\[
  (2\cdot 1,\ 2\cdot 2)=(2,4),
\]
and the scaled scores are
\[
  s_1=
  \left(\frac{2}{\sqrt3},\frac{4}{\sqrt3}\right)
  \approx
  (1.1547,2.3094).
\]
Thus
\[
  a_1=\softmax(s_1)\approx(0.2396316,0.7603684).
\]
The materialized output for head \(1\) is
\[
  o_1
  =
  0.2396316\cdot 2
  +
  0.7603684\cdot 4
  \approx
  3.5207369.
\]

For head \(2\),
\[
  k^{\mathrm n}_{2}=(-1,-2),
  \qquad
  v_{2}=(1,2).
\]
The unscaled scores are
\[
  ((-1)(-1),\ (-1)(-2))=(1,2),
\]
and the scaled scores are
\[
  s_2=
  \left(\frac{1}{\sqrt3},\frac{2}{\sqrt3}\right)
  \approx
  (0.5774,1.1547).
\]
Thus
\[
  a_2=\softmax(s_2)\approx(0.3595425,0.6404575).
\]
The materialized output for head \(2\) is
\[
  o_2
  =
  0.3595425\cdot 1
  +
  0.6404575\cdot 2
  \approx
  1.6404575.
\]

\subsection*{Absorbed calculation}

The absorbed query latents are
\[
  q'_{h=1}
  =
  (W^K_1)^\top q^{\mathrm n}_{h=1}
  =
  1\cdot 2
  =
  2,
\]
\[
  q'_{h=2}
  =
  (W^K_2)^\top q^{\mathrm n}_{h=2}
  =
  (-1)(-1)
  =
  1.
\]
Therefore the absorbed unscaled scores are
\[
  q'_{h=1}(z_1,z_2)=(2,4),
  \qquad
  q'_{h=2}(z_1,z_2)=(1,2),
\]
which are the same scores as in the materialized calculation.  Hence the
attention probabilities are the same.

For head \(1\), the latent context is
\[
  g_1
  =
  0.2396316\cdot 1
  +
  0.7603684\cdot 2
  \approx
  1.7603684.
\]
Delayed value up-projection gives
\[
  W^V_1 g_1
  =
  2\cdot 1.7603684
  \approx
  3.5207369.
\]
For head \(2\),
\[
  g_2
  =
  0.3595425\cdot 1
  +
  0.6404575\cdot 2
  \approx
  1.6404575,
\]
and
\[
  W^V_2 g_2
  =
  1\cdot 1.6404575
  \approx
  1.6404575.
\]
Thus both paths give the same pre-output context:
\[
  (o_1,o_2)\approx(3.5207369,1.6404575).
\]
With identity output projection \(P_o:\R^2\to\R^2\),
\[
  y\approx(3.5207369,1.6404575)\in\R^{1\times1\times2}.
\]
The executable hand-check artifact
\path{math/mla/verification/two_head_manual_example.py} records the same
calculation and outputs
\[
  [3.520736883716041,\;1.6404574756806274].
\]

\section{Open Limits}

The theorem above proves equality of the shared base MLA algebra under the
listed assumptions.  The following are limits, not proved claims:

\begin{itemize}
  \item \textbf{Training dropout.}
        Dropout is disabled in the algebraic verifier.  Stochastic training
        behavior is not covered.

  \item \textbf{RoPE variants such as YaRN.}
        If both paths apply exactly the same deterministic RoPE variant, the
        same style of proof should apply after redefining \(\mathcal R_p\).
        This document does not separately prove every YaRN or scaling variant.

  \item \textbf{All production mask forms.}
        The proof extends immediately to a shared additive mask added to both
        logits.  Backend-specific mask broadcasting, paged-cache mask mutation,
        and unusual sparse masks require separate checks.

  \item \textbf{Paged cache mutation semantics.}
        The mathematical cache object is
        \([z,\rho]\in\R^{B\times M\times(C+R)}\).  Production page tables,
        block allocators, and in-place cache update semantics are storage and
        runtime concerns not proved here.

  \item \textbf{Quantized, FP8, sparse, NSA, or backend-specialized layouts.}
        These may preserve the same conceptual algebra, but each layout has its
        own rounding, packing, indexing, and kernel assumptions.  They are not
        promoted to proved parity by the base derivation.

  \item \textbf{Bitwise floating-point identity.}
        The symbolic proof is exact in real arithmetic.  Executable parity is
        expected only within tolerances because materialized and absorbed paths
        may change reduction order and intermediate dtype behavior.

  \item \textbf{External implementation completeness.}
        The vLLM and SGLang checks validate that the same base cache and
        absorption equations appear in pinned implementation snapshots.  They
        do not constitute a full proof of every production execution path.
\end{itemize}
```

---

## 3. Completion Checklist

Use this checklist before declaring the final `math/mla/mla.tex` done.

- [ ] **Source traceability:** Every source-dependent fact is traceable to `source_ledger.md`, `external_sources.md`, the profile directory, or a verification artifact.
- [ ] **No code in TeX:** `mla.tex` contains equations, prose, paths, and result summaries, but no implementation listings or pseudocode.
- [ ] **Dimensions explicit:** \(B,L,M,D,H,N,R,V,Q,C,T,X,Y\) are all defined once, and every tensor shape uses these symbols consistently.
- [ ] **Orientation explicit:** The document clearly states that last-axis vectors are treated as column vectors in equations, with \(W^K_h\in\mathbb R^{N\times C}\) and \(W^V_h\in\mathbb R^{V\times C}\).
- [ ] **Assumptions listed early:** The assumptions include deterministic inference, no dropout, same mask if present, base/same RoPE, static up-projections, bias handling, real arithmetic, and scale \(1/\sqrt T\).
- [ ] **Cache ladder complete:** MHA, MQA, GQA, semi-materialized MLA, and compressed MLA cache counts are all derived with scalar counts per token per layer.
- [ ] **Cache object preserved:** The compressed MLA cache is always written as
  \[
  [z,\rho]\in\mathbb R^{B\times M\times(C+R)}.
  \]
- [ ] **Derivation ladder intact:** The document proceeds through query path, KV path, materialized attention, score decomposition, key absorption, delayed value up-projection, RoPE non-absorption, external validation, executable checks, hand example, and limits.
- [ ] **Score scaling correct:** The absorbed path explicitly states that the dot product may have width \(C+R\), but the attention scale remains \(1/\sqrt{T}\).
- [ ] **RoPE separation rigorous:** The document proves that a single static absorption matrix cannot replace position-dependent RoPE except in degenerate/fixed-position cases.
- [ ] **Executable checks concrete:** Shape assertions and parity results are listed, with exact artifact paths or source-ledger references for the run logs.
- [ ] **Parity numbers current:** The max-error table and vLLM/SGLang probe summaries match the latest executed artifacts.
- [ ] **Two-head example manually checkable:** The example includes both materialized and absorbed calculations, shows the cache tokens, computes \(q'\), computes \(g\), and matches the executable script output.
- [ ] **Limits labeled:** Untested production details are listed as limits and are not described as proved.
- [ ] **TeX builds cleanly:** `mla.tex` compiles successfully, and the build result is recorded under `math/mla/verification/tex_build.md` or the current equivalent artifact.

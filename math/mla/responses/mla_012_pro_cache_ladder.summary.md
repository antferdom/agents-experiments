\section{A cache ladder from MHA/MQA/GQA to MLA}
\label{sec:mla-cache-ladder}

We count scalar cache elements, not bytes.  For one layer, define
\[
  e(\mathcal C)
  \;=\;
  \frac{\operatorname{elts}(\mathcal C)}{B M},
\]
the number of cached scalar elements per memory token per batch element.  Thus a cached tensor
\(A_{B,K,M,d}\) contributes
\[
  \frac{B K M d}{B M}=K d
\]
elements per token per layer, and a cached tensor \(A_{B,M,d}\) contributes \(d\).
The total scalar cache size for one layer is \(B M e(\mathcal C)\); the total over all cached
layers is obtained by multiplying by the number of layers.

Throughout this section the key/query head width is
\[
  T=N+R,
\]
where \(N\) is the non-RoPE key/query head dimension and \(R\) is the RoPE key/query suffix
dimension.  The value head dimension is \(V\).  We also use
\[
  X=C+R,
  \qquad
  Y=N+V.
\]
For the DeepSeek-V2 profile,
\[
  D=5120,\quad H=128,\quad N=128,\quad R=64,\quad V=128,\quad Q=1536,\quad C=512,
\]
so
\[
  T=N+R=192,\qquad
  X=C+R=576,\qquad
  Y=N+V=256,
\]
and
\[
  T+V=(N+R)+V=Y+R=320.
\]

\paragraph{Multi-head attention (MHA).}

In standard MHA, every query head has its own cached key head and its own cached value head.
For a query block of length \(L\), write the query tensor as
\[
  Q^{\mathrm{mha}}_{B,H,L,T}.
\]
The persistent KV cache for one layer is
\[
  K^{\mathrm{mha}}_{B,H,M,T},
  \qquad
  V^{\mathrm{mha}}_{B,H,M,V}.
\]
For head \(h\), query position \(\ell\), and memory position \(m\), the attention score is computed
from the \(h\)-th cached key head,
\[
  S^{\mathrm{mha}}_{b,h,\ell,m}
  =
  \left\langle
    Q^{\mathrm{mha}}_{b,h,\ell,:},
    K^{\mathrm{mha}}_{b,h,m,:}
  \right\rangle,
\]
up to the usual attention scale and mask, which do not affect cache shape.  The value aggregation
uses \(V^{\mathrm{mha}}_{b,h,m,:}\in\mathbb R^V\).

Therefore the MHA cache element count per token per layer is
\[
\begin{aligned}
  e(\mathcal C_{\mathrm{mha}})
  &=
  \frac{
    \operatorname{elts}\!\left(K^{\mathrm{mha}}_{B,H,M,T}\right)
    +
    \operatorname{elts}\!\left(V^{\mathrm{mha}}_{B,H,M,V}\right)
  }{B M} \\[2mm]
  &=
  \frac{B H M T + B H M V}{B M} \\[1mm]
  &=
  H(T+V).
\end{aligned}
\]
For the DeepSeek-V2 dimensions this materialized MHA-style count is
\[
  H(T+V)=128(192+128)=128\cdot 320=40960.
\]

\paragraph{Multi-query attention (MQA).}

In MQA, all \(H\) query heads share a single cached key head and a single cached value head.
The query tensor is still head-specific,
\[
  Q^{\mathrm{mqa}}_{B,H,L,T},
\]
but the cached tensors have a singleton KV-head axis:
\[
  K^{\mathrm{mqa}}_{B,1,M,T},
  \qquad
  V^{\mathrm{mqa}}_{B,1,M,V}.
\]
Equivalently, the single key/value stream is broadcast to all query heads during attention:
\[
  S^{\mathrm{mqa}}_{b,h,\ell,m}
  =
  \left\langle
    Q^{\mathrm{mqa}}_{b,h,\ell,:},
    K^{\mathrm{mqa}}_{b,1,m,:}
  \right\rangle.
\]
Thus the cache count is
\[
\begin{aligned}
  e(\mathcal C_{\mathrm{mqa}})
  &=
  \frac{
    \operatorname{elts}\!\left(K^{\mathrm{mqa}}_{B,1,M,T}\right)
    +
    \operatorname{elts}\!\left(V^{\mathrm{mqa}}_{B,1,M,V}\right)
  }{B M} \\[2mm]
  &=
  \frac{B M T + B M V}{B M} \\[1mm]
  &=
  T+V.
\end{aligned}
\]
For the DeepSeek-V2 dimensions this is
\[
  T+V=192+128=320.
\]

\paragraph{Grouped-query attention (GQA).}

GQA interpolates between MHA and MQA.  There are \(H\) query heads but only \(G\) distinct
KV groups.  Assume for simplicity that \(G\mid H\), so each KV group serves \(H/G\) query heads.
Let
\[
  g(h)=1+\left\lfloor\frac{h-1}{H/G}\right\rfloor,
  \qquad
  h\in\{1,\ldots,H\},
\]
be the KV-group index used by query head \(h\).  The query tensor is
\[
  Q^{\mathrm{gqa}}_{B,H,L,T},
\]
while the persistent KV cache is
\[
  K^{\mathrm{gqa}}_{B,G,M,T},
  \qquad
  V^{\mathrm{gqa}}_{B,G,M,V}.
\]
The score for query head \(h\) uses only the key cache of group \(g(h)\):
\[
  S^{\mathrm{gqa}}_{b,h,\ell,m}
  =
  \left\langle
    Q^{\mathrm{gqa}}_{b,h,\ell,:},
    K^{\mathrm{gqa}}_{b,g(h),m,:}
  \right\rangle.
\]
The cache element count is therefore
\[
\begin{aligned}
  e(\mathcal C_{\mathrm{gqa}})
  &=
  \frac{
    \operatorname{elts}\!\left(K^{\mathrm{gqa}}_{B,G,M,T}\right)
    +
    \operatorname{elts}\!\left(V^{\mathrm{gqa}}_{B,G,M,V}\right)
  }{B M} \\[2mm]
  &=
  \frac{B G M T + B G M V}{B M} \\[1mm]
  &=
  G(T+V).
\end{aligned}
\]
MHA and MQA are the endpoint cases:
\[
  G=H \implies e(\mathcal C_{\mathrm{gqa}})=H(T+V)=e(\mathcal C_{\mathrm{mha}}),
\]
and
\[
  G=1 \implies e(\mathcal C_{\mathrm{gqa}})=T+V=e(\mathcal C_{\mathrm{mqa}}).
\]
For the DeepSeek-V2 dimensions the GQA count is
\[
  G(T+V)=G(192+128)=320G.
\]

\paragraph{Why MLA is not merely ``a low-rank projection''.}

It is tempting to describe MLA only as a low-rank factorization of key and value projections.
That description is insufficient for cache accounting.  A low-rank factorization is a statement
about model weights; a KV cache is a statement about the runtime object that is persisted across
future decoding steps.

For example, suppose a conventional attention layer computes head-specific materialized keys and
values through a factorization
\[
  k_{h,m}=U^K_h c_m,
  \qquad
  v_{h,m}=U^V_h c_m,
\]
where \(c_m\in\mathbb R^C\).  If the implementation still writes the materialized tensors
\[
  K_{B,H,M,T},
  \qquad
  V_{B,H,M,V}
\]
into the persistent cache, then the cache count is still \(H(T+V)\).  The factorization of the
projection weights has not, by itself, changed the cache object.

MLA changes the cache object.  Instead of persistently caching per-head materialized keys and
values, DeepSeek-V2 MLA caches a normalized compressed KV vector and a shared RoPE key suffix:
\[
  \widetilde C^{\mathrm{KV}}_{B,M,C},
  \qquad
  K^R_{B,M,R}.
\]
The per-head non-RoPE content keys and values are represented implicitly through learned
up-projections.  The shared RoPE key suffix is stored once per memory token, not once per query
head, in the compressed cache.

\paragraph{Materialized MLA tensors.}

Let
\[
  \widetilde c^{\mathrm{KV}}_{b,m}\in\mathbb R^C
\]
denote the normalized compressed KV vector for memory token \(m\), and let
\[
  k^R_{b,m}\in\mathbb R^R
\]
denote the shared RoPE key suffix.  For each query head \(h\), let
\[
  U^K_h\in\mathbb R^{N\times C},
  \qquad
  U^V_h\in\mathbb R^{V\times C}
\]
be the head-specific key-content and value up-projections.  Equivalently, the combined up-projection
has output width
\[
  N+V=Y
\]
per head and width \(H Y\) across all heads.

The materialized non-RoPE content key and value for head \(h\) are
\[
  k^C_{b,h,m}
  =
  U^K_h \widetilde c^{\mathrm{KV}}_{b,m}
  \in\mathbb R^N,
  \qquad
  v^{\mathrm{mla}}_{b,h,m}
  =
  U^V_h \widetilde c^{\mathrm{KV}}_{b,m}
  \in\mathbb R^V.
\]
The corresponding fully materialized MLA key is the concatenation of the per-head content key and
the shared RoPE key suffix:
\[
  k^{\mathrm{mla,mat}}_{b,h,m}
  =
  \begin{bmatrix}
    k^C_{b,h,m}\\
    k^R_{b,m}
  \end{bmatrix}
  \in\mathbb R^{N+R}
  =
  \mathbb R^T.
\]
Therefore the materialized MLA tensors have shapes
\[
  K^{\mathrm{mla,mat}}_{B,H,M,T},
  \qquad
  V^{\mathrm{mla,mat}}_{B,H,M,V}.
\]
If these materialized tensors are stored as the cache, the local materialized cache accounting is
\[
\begin{aligned}
  e(\mathcal C_{\mathrm{mla,mat}})
  &=
  \frac{
    \operatorname{elts}\!\left(K^{\mathrm{mla,mat}}_{B,H,M,T}\right)
    +
    \operatorname{elts}\!\left(V^{\mathrm{mla,mat}}_{B,H,M,V}\right)
  }{B M} \\[2mm]
  &=
  H(T+V).
\end{aligned}
\]
For the DeepSeek-V2 profile this is again
\[
  H(T+V)=128(192+128)=40960.
\]

\paragraph{Compressed MLA cache.}

The compressed MLA cache persists the pair
\[
  \mathcal C_{\mathrm{mla,cmp}}
  =
  \left(
    \widetilde C^{\mathrm{KV}}_{B,M,C},
    K^R_{B,M,R}
  \right).
\]
Its element count is
\[
\begin{aligned}
  e(\mathcal C_{\mathrm{mla,cmp}})
  &=
  \frac{
    \operatorname{elts}\!\left(\widetilde C^{\mathrm{KV}}_{B,M,C}\right)
    +
    \operatorname{elts}\!\left(K^R_{B,M,R}\right)
  }{B M} \\[2mm]
  &=
  \frac{B M C + B M R}{B M} \\[1mm]
  &=
  C+R \\[1mm]
  &=
  X.
\end{aligned}
\]
For the DeepSeek-V2 profile,
\[
  e(\mathcal C_{\mathrm{mla,cmp}})
  =
  C+R
  =
  512+64
  =
  576.
\]

The algebraic reason this compressed object can represent the materialized attention inputs is that
the head-specific up-projections can be consumed without being stored in the persistent cache.
For a query decomposed as
\[
  q^{\mathrm{mla}}_{b,h,\ell}
  =
  \begin{bmatrix}
    q^C_{b,h,\ell}\\
    q^R_{b,h,\ell}
  \end{bmatrix},
  \qquad
  q^C_{b,h,\ell}\in\mathbb R^N,\quad
  q^R_{b,h,\ell}\in\mathbb R^R,
\]
the materialized MLA score can be written as
\[
\begin{aligned}
  S^{\mathrm{mla}}_{b,h,\ell,m}
  &=
  \left\langle
    q^C_{b,h,\ell},
    U^K_h \widetilde c^{\mathrm{KV}}_{b,m}
  \right\rangle
  +
  \left\langle
    q^R_{b,h,\ell},
    k^R_{b,m}
  \right\rangle \\[1mm]
  &=
  \left\langle
    (U^K_h)^{\top} q^C_{b,h,\ell},
    \widetilde c^{\mathrm{KV}}_{b,m}
  \right\rangle
  +
  \left\langle
    q^R_{b,h,\ell},
    k^R_{b,m}
  \right\rangle.
\end{aligned}
\]
Thus the content-key part of the score can be evaluated against the cached compressed vector
\(\widetilde c^{\mathrm{KV}}_{b,m}\), while the RoPE part is evaluated against the cached shared
suffix \(k^R_{b,m}\).  Similarly, after attention probabilities
\(P^{\mathrm{mla}}_{b,h,\ell,m}\) are formed over memory positions \(m\), the value aggregation can
be written as
\[
\begin{aligned}
  z^{\mathrm{mla}}_{b,h,\ell}
  &=
  \sum_{m=1}^{M}
    P^{\mathrm{mla}}_{b,h,\ell,m}
    \,
    U^V_h \widetilde c^{\mathrm{KV}}_{b,m} \\[1mm]
  &=
  U^V_h
  \left(
    \sum_{m=1}^{M}
      P^{\mathrm{mla}}_{b,h,\ell,m}
      \,
      \widetilde c^{\mathrm{KV}}_{b,m}
  \right).
\end{aligned}
\]
This is a change in the persistent cache representation: the cache stores the compressed latent
and the shared RoPE suffix, not the \(H\)-fold materialized key/value tensors.

\paragraph{Cache element comparison.}

The following table compares scalar cache elements per token per layer.  The DeepSeek-V2 profile
uses \(T=192\), \(V=128\), \(H=128\), and \(X=C+R=576\).

\begin{center}
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{l l l l}
\hline
Scheme
&
Persistent cached tensors
&
Elements per token per layer
&
DeepSeek-V2 profile
\\
\hline
MHA
&
\(K^{\mathrm{mha}}_{B,H,M,T},\; V^{\mathrm{mha}}_{B,H,M,V}\)
&
\(H(T+V)\)
&
\(128(192+128)=40960\)
\\

MQA
&
\(K^{\mathrm{mqa}}_{B,1,M,T},\; V^{\mathrm{mqa}}_{B,1,M,V}\)
&
\(T+V\)
&
\(192+128=320\)
\\

GQA
&
\(K^{\mathrm{gqa}}_{B,G,M,T},\; V^{\mathrm{gqa}}_{B,G,M,V}\)
&
\(G(T+V)\)
&
\(G(192+128)=320G\)
\\

MLA, materialized
&
\(K^{\mathrm{mla,mat}}_{B,H,M,T},\; V^{\mathrm{mla,mat}}_{B,H,M,V}\)
&
\(H(T+V)\)
&
\(128(192+128)=40960\)
\\

MLA, compressed
&
\(\widetilde C^{\mathrm{KV}}_{B,M,C},\; K^R_{B,M,R}\)
&
\(C+R=X\)
&
\(512+64=576\)
\\
\hline
\end{tabular}
\end{center}

For this profile, a fully materialized \(H\)-head key/value cache stores \(40960\) scalar elements
per token per layer, while the compressed MLA cache stores \(576\) scalar elements per token per
layer.  This is only a cache-element statement; it is not a statement about latency, throughput, or
model quality.

\paragraph{Assumptions and exclusions.}

The counts above use the following assumptions.

\begin{itemize}
  \item The unit is a scalar cache element.  If all cached tensors use \(s\) bytes per scalar and
  there is no padding or metadata, then the byte count for one layer is
  \[
    s\,B M\,e(\mathcal C).
  \]
  Quantization, mixed cache dtypes, per-block scales, page tables, allocator metadata, and alignment
  padding must be added separately.

  \item The counts are for the persistent autoregressive KV cache of one layer.  They do not count
  projection weights, output projection weights, RoPE tables, attention masks, residual streams,
  MLP activations, layer-normalization statistics, optimizer state, or training-time saved
  activations.

  \item The query length \(L\) affects the size of query tensors and attention workspaces, but it
  does not change the number of persistent cache elements added per memory token.  The table
  therefore has no \(L\) factor.

  \item The hidden dimension \(D\) and the query compression rank \(Q\) affect the projection
  parameterization and transient query-side tensors.  They are not persistent KV-cache widths in
  the compressed MLA cache, so they do not appear in \(C+R\).

  \item Temporary materialization is distinct from persistent caching.  If an implementation
  temporarily forms \(K^{\mathrm{mla,mat}}_{B,H,M,T}\) or
  \(V^{\mathrm{mla,mat}}_{B,H,M,V}\) inside a kernel and then discards it, that workspace is not
  counted as persistent cache.  If an implementation stores those materialized tensors across
  decoding steps, then the materialized accounting \(H(T+V)\) applies.

  \item The GQA count assumes the cache is stored compactly with \(G\) KV groups.  If an
  implementation physically expands a GQA cache to \(H\) key/value heads and persists that expanded
  representation, the stored-cache count becomes \(H(T+V)\), regardless of the mathematical sharing.

  \item The compressed MLA count assumes the persistent DeepSeek-V2 MLA cache consists of the
  normalized compressed KV tensor \(\widetilde C^{\mathrm{KV}}_{B,M,C}\) and the shared RoPE key
  suffix \(K^R_{B,M,R}\).  Additional implementation-specific cached side data must be counted
  separately.
\end{itemize}

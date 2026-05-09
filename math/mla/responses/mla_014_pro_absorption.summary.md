\subsection{Compressed-cache MLA and projection absorption}
\label{sec:mla-compressed-cache}

For an integer \(K\), write \([K]=\{1,\ldots,K\}\).  We use
\[
B=\text{batch size},\qquad
L=\text{query length},\qquad
M=\text{memory length},\qquad
D=\text{hidden size},
\]
\[
H=\text{number of heads},\qquad
N=\text{non-RoPE query/key head dimension},\qquad
R=\text{RoPE suffix dimension},\qquad
V=\text{value head dimension},
\]
\[
Q=\text{query compression rank},\qquad
C=\text{KV compression rank},
\]
and the derived dimensions
\[
T=N+R,\qquad X=C+R,\qquad Y=N+V.
\]
The hidden dimension \(D\) and query-compression rank \(Q\) are upstream of the cache identity below: they determine how the query tensors are produced.  The projection-absorption argument only assumes that the compared implementations use the same already-computed tensors
\[
q^{\mathrm{nope}}\in\mathbb{R}^{B\times H\times L\times N},
\qquad
q^{\mathrm{rope}}\in\mathbb{R}^{B\times H\times L\times R},
\]
where \(q^{\mathrm{rope}}\) denotes the query RoPE suffix after applying RoPE at the query positions.

The compressed MLA cache stores the normalized latent KV tensor
\[
z_{b,m,c}
=
\mathtt{compressed\_kv\_norm\_BMC}_{b,m,c},
\qquad
z\in\mathbb{R}^{B\times M\times C},
\]
and the rotated key RoPE suffix
\[
\rho_{b,m,r}
=
\mathtt{k\_rope\_after\_rope\_BMR}_{b,m,r},
\qquad
\rho\in\mathbb{R}^{B\times M\times R}.
\]
These are packed in the last dimension as a cache tensor of shape
\[
B\times M\times (C+R)=B\times M\times X
\]
by
\[
\mathrm{cache}_{b,m,j}
=
\begin{cases}
z_{b,m,j}, & 1\leq j\leq C,\\[2mm]
\rho_{b,m,j-C}, & C<j\leq C+R.
\end{cases}
\]
Thus the persistent cache does not store materialized per-head non-RoPE keys or materialized per-head values.

The weight \(\mathtt{kv\_b\_proj.weight}\) is viewed as
\[
W^{KV}\in\mathbb{R}^{H\times Y\times C},
\qquad
Y=N+V,
\]
and is split along the \(Y\)-axis into
\[
W^K\in\mathbb{R}^{H\times N\times C},
\qquad
W^V\in\mathbb{R}^{H\times V\times C},
\]
with index definitions
\[
W^K_{h,n,c}=W^{KV}_{h,n,c},
\qquad
W^V_{h,\nu,c}=W^{KV}_{h,N+\nu,c},
\]
for \(h\in[H]\), \(n\in[N]\), \(\nu\in[V]\), and \(c\in[C]\).

\paragraph{Materialized MLA reference form.}
If the non-RoPE keys and values were materialized from the compressed latent \(z\), they would be
\[
k^{\mathrm{nope}}_{b,h,m,n}
=
\sum_{c=1}^{C} W^K_{h,n,c}\,z_{b,m,c},
\qquad
k^{\mathrm{nope}}\in\mathbb{R}^{B\times H\times M\times N},
\]
and
\[
v^{\mathrm{mat}}_{b,h,m,\nu}
=
\sum_{c=1}^{C} W^V_{h,\nu,c}\,z_{b,m,c},
\qquad
v^{\mathrm{mat}}\in\mathbb{R}^{B\times H\times M\times V}.
\]
The full key for head \(h\) at memory position \(m\) consists of the non-RoPE component \(k^{\mathrm{nope}}_{b,h,m,:}\in\mathbb{R}^{N}\) and the shared RoPE suffix \(\rho_{b,m,:}\in\mathbb{R}^{R}\).  Hence the materialized attention score is
\[
S^{\mathrm{mat}}_{b,h,l,m}
=
S^{\mathrm{nope,mat}}_{b,h,l,m}
+
S^{\mathrm{rope}}_{b,h,l,m},
\]
where
\[
S^{\mathrm{nope,mat}}_{b,h,l,m}
=
\sum_{n=1}^{N}
q^{\mathrm{nope}}_{b,h,l,n}\,
k^{\mathrm{nope}}_{b,h,m,n},
\]
and
\[
S^{\mathrm{rope}}_{b,h,l,m}
=
\sum_{r=1}^{R}
q^{\mathrm{rope}}_{b,h,l,r}\,
\rho_{b,m,r}.
\]
Let \(\alpha\) denote the attention scale; the common unmodified choice is \(\alpha=T^{-1/2}\), but the derivation only requires that the same scalar scale be used in both implementations.  Let \(\mu_{b,h,l,m}\in\mathbb{R}\cup\{-\infty\}\) be the additive attention mask, including any causal or padding mask.  Assuming at least one unmasked memory position for each \((b,h,l)\), the materialized attention weights are
\[
P^{\mathrm{mat}}_{b,h,l,m}
=
\frac{
\exp\!\left(\alpha S^{\mathrm{mat}}_{b,h,l,m}+\mu_{b,h,l,m}\right)
}{
\sum_{m'=1}^{M}
\exp\!\left(\alpha S^{\mathrm{mat}}_{b,h,l,m'}+\mu_{b,h,l,m'}\right)
}.
\]
The materialized attention output before any final output projection is
\[
o^{\mathrm{mat}}_{b,h,l,\nu}
=
\sum_{m=1}^{M}
P^{\mathrm{mat}}_{b,h,l,m}\,
v^{\mathrm{mat}}_{b,h,m,\nu}.
\]

\paragraph{Compressed-cache MLA form.}
The compressed-cache implementation avoids materializing \(k^{\mathrm{nope}}\) and \(v^{\mathrm{mat}}\).  First absorb the non-RoPE key projection into the query side:
\[
\bar q_{b,h,l,c}
=
\sum_{n=1}^{N}
W^K_{h,n,c}\,
q^{\mathrm{nope}}_{b,h,l,n},
\qquad
\bar q\in\mathbb{R}^{B\times H\times L\times C}.
\]
This is the contraction \texttt{hnc,bhln->bhlc}.  The non-RoPE score is then computed directly against the compressed latent cache:
\[
\widehat S^{\mathrm{nope}}_{b,h,l,m}
=
\sum_{c=1}^{C}
\bar q_{b,h,l,c}\,
z_{b,m,c},
\]
which is the contraction \texttt{bhlc,bmc->bhlm}.  The RoPE score is unchanged:
\[
S^{\mathrm{rope}}_{b,h,l,m}
=
\sum_{r=1}^{R}
q^{\mathrm{rope}}_{b,h,l,r}\,
\rho_{b,m,r}.
\]
The compressed-cache score is therefore
\[
\widehat S_{b,h,l,m}
=
\widehat S^{\mathrm{nope}}_{b,h,l,m}
+
S^{\mathrm{rope}}_{b,h,l,m}.
\]
Using the same scale \(\alpha\) and mask \(\mu\), define
\[
\widehat P_{b,h,l,m}
=
\frac{
\exp\!\left(\alpha \widehat S_{b,h,l,m}+\mu_{b,h,l,m}\right)
}{
\sum_{m'=1}^{M}
\exp\!\left(\alpha \widehat S_{b,h,l,m'}+\mu_{b,h,l,m'}\right)
}.
\]

For values, aggregate in latent \(C\)-space first:
\[
\lambda_{b,h,l,c}
=
\sum_{m=1}^{M}
\widehat P_{b,h,l,m}\,
z_{b,m,c},
\qquad
\lambda\in\mathbb{R}^{B\times H\times L\times C}.
\]
This is the contraction \texttt{bhlm,bmc->bhlc}.  Then apply the value projection:
\[
\widehat o_{b,h,l,\nu}
=
\sum_{c=1}^{C}
\lambda_{b,h,l,c}\,
W^V_{h,\nu,c},
\qquad
\widehat o\in\mathbb{R}^{B\times H\times L\times V}.
\]
This final projection is the contraction \texttt{bhlc,hvc->bhlv}.

\paragraph{Projection absorption for non-RoPE scores.}
Over the real numbers, all sums are finite, so distributivity, commutativity, and associativity give, for every
\((b,h,l,m)\in[B]\times[H]\times[L]\times[M]\),
\[
\begin{aligned}
S^{\mathrm{nope,mat}}_{b,h,l,m}
&=
\sum_{n=1}^{N}
q^{\mathrm{nope}}_{b,h,l,n}\,
k^{\mathrm{nope}}_{b,h,m,n}
\\
&=
\sum_{n=1}^{N}
q^{\mathrm{nope}}_{b,h,l,n}
\left(
\sum_{c=1}^{C}
W^K_{h,n,c}\,
z_{b,m,c}
\right)
\\
&=
\sum_{n=1}^{N}
\sum_{c=1}^{C}
q^{\mathrm{nope}}_{b,h,l,n}\,
W^K_{h,n,c}\,
z_{b,m,c}
\\
&=
\sum_{c=1}^{C}
\left(
\sum_{n=1}^{N}
W^K_{h,n,c}\,
q^{\mathrm{nope}}_{b,h,l,n}
\right)
z_{b,m,c}
\\
&=
\sum_{c=1}^{C}
\bar q_{b,h,l,c}\,
z_{b,m,c}
\\
&=
\widehat S^{\mathrm{nope}}_{b,h,l,m}.
\end{aligned}
\]
The RoPE term is identical in both forms because both use the same after-RoPE query suffix \(q^{\mathrm{rope}}\) and the same cached after-RoPE key suffix \(\rho\).  Therefore
\[
\widehat S_{b,h,l,m}
=
S^{\mathrm{mat}}_{b,h,l,m}
\]
for all indices in exact real arithmetic.  Consequently, with the same scale and additive mask,
\[
\widehat P_{b,h,l,m}
=
P^{\mathrm{mat}}_{b,h,l,m}
\]
for all indices, because both are obtained by applying the same softmax to the same score vector over \(m\).

\paragraph{Projection absorption for value aggregation.}
Using the equality \(\widehat P=P^{\mathrm{mat}}\) established above, the materialized value aggregation satisfies, for every
\((b,h,l,\nu)\in[B]\times[H]\times[L]\times[V]\),
\[
\begin{aligned}
o^{\mathrm{mat}}_{b,h,l,\nu}
&=
\sum_{m=1}^{M}
P^{\mathrm{mat}}_{b,h,l,m}\,
v^{\mathrm{mat}}_{b,h,m,\nu}
\\
&=
\sum_{m=1}^{M}
P^{\mathrm{mat}}_{b,h,l,m}
\left(
\sum_{c=1}^{C}
W^V_{h,\nu,c}\,
z_{b,m,c}
\right)
\\
&=
\sum_{m=1}^{M}
\sum_{c=1}^{C}
P^{\mathrm{mat}}_{b,h,l,m}\,
W^V_{h,\nu,c}\,
z_{b,m,c}
\\
&=
\sum_{c=1}^{C}
W^V_{h,\nu,c}
\left(
\sum_{m=1}^{M}
P^{\mathrm{mat}}_{b,h,l,m}\,
z_{b,m,c}
\right)
\\
&=
\sum_{c=1}^{C}
W^V_{h,\nu,c}\,
\lambda_{b,h,l,c}
\\
&=
\widehat o_{b,h,l,\nu}.
\end{aligned}
\]
Thus the compressed-cache form and the materialized form are algebraically identical over \(\mathbb{R}\), provided that they use the same \(q^{\mathrm{nope}}\), \(q^{\mathrm{rope}}\), compressed latent cache \(z\), RoPE cache \(\rho\), projection weights \(W^K,W^V\), attention scale, and mask.

\paragraph{Cache-element accounting.}
The compressed cache stores
\[
z\in\mathbb{R}^{B\times M\times C}
\qquad\text{and}\qquad
\rho\in\mathbb{R}^{B\times M\times R},
\]
packed as \(B\times M\times (C+R)=B\times M\times X\).  Hence its persistent cache element count is
\[
E_{\mathrm{comp}}
=
BM(C+R)
=
BMX.
\]
A materialized MLA cache using the same head-shared RoPE suffix would store
\[
k^{\mathrm{nope}}\in\mathbb{R}^{B\times H\times M\times N},
\qquad
v^{\mathrm{mat}}\in\mathbb{R}^{B\times H\times M\times V},
\qquad
\rho\in\mathbb{R}^{B\times M\times R},
\]
for an element count
\[
E_{\mathrm{mat,shared\ rope}}
=
BM(HN+HV+R)
=
BM(HY+R).
\]
Relative to this materialized MLA cache, the element difference is
\[
E_{\mathrm{mat,shared\ rope}}-E_{\mathrm{comp}}
=
BM(HY-C),
\]
so this is a reduction exactly when \(C<HY\).  If instead a baseline stored the RoPE suffix separately for every head as part of a fully materialized per-head key, the corresponding key/value cache shape would be
\[
B\times H\times M\times (N+R)
\quad\text{and}\quad
B\times H\times M\times V,
\]
with element count
\[
E_{\mathrm{mat,replicated\ rope}}
=
BMH(N+R+V)
=
BMH(T+V).
\]
The compressed-cache ratio against the head-shared-RoPE materialized form is
\[
\frac{E_{\mathrm{comp}}}{E_{\mathrm{mat,shared\ rope}}}
=
\frac{C+R}{HY+R},
\]
and against the fully per-head materialized form is
\[
\frac{E_{\mathrm{comp}}}{E_{\mathrm{mat,replicated\ rope}}}
=
\frac{C+R}{H(T+V)}.
\]
These are shape and element-count statements only.  Actual byte counts multiply by dtype size and may include alignment, paging, padding, or metadata overhead.  These identities also do not by themselves imply a runtime improvement: runtime depends on kernel fusion, memory bandwidth, tensor-core or vector-unit utilization, contraction order, batching, masking, cache layout, and other implementation details.

\paragraph{Finite precision and operation ordering.}
The equalities above are exact algebraic equalities over the real numbers.  Floating-point implementations perform rounded arithmetic, and the absorbed and materialized forms generally use different operation orderings.  For example, for some inputs one can have
\[
\operatorname{fl}\!\left(
\sum_{n=1}^{N}
q_n\,
\operatorname{fl}\!\left(\sum_{c=1}^{C} W^K_{n,c}z_c\right)
\right)
\neq
\operatorname{fl}\!\left(
\sum_{c=1}^{C}
\operatorname{fl}\!\left(\sum_{n=1}^{N} W^K_{n,c}q_n\right)
z_c
\right),
\]
even though the two expressions are equal over \(\mathbb{R}\).  The left expression corresponds to materializing the key projection first and then dotting with the query; the right expression corresponds to absorbing the key projection into the query first and then dotting with the latent cache.

The same issue occurs for value aggregation:
\[
\operatorname{fl}\!\left(
\sum_{m=1}^{M}
P_m\,
\operatorname{fl}\!\left(\sum_{c=1}^{C}W^V_{\nu,c}z_{m,c}\right)
\right)
\neq
\operatorname{fl}\!\left(
\sum_{c=1}^{C}
W^V_{\nu,c}\,
\operatorname{fl}\!\left(\sum_{m=1}^{M}P_m z_{m,c}\right)
\right)
\]
in general.  The left expression materializes values before attention aggregation; the right expression aggregates latents before applying \(W^V\).

Therefore the absorbed implementation should not be expected to be bitwise identical to the materialized implementation unless the same contraction order, accumulation precision, fused-multiply-add behavior, math mode, and rounding points are forced.  Differences can arise from non-associativity of floating-point addition, mixed precision, bfloat16 or float16 inputs, TF32 tensor-core execution, different \(\mathtt{einsum}\) or GEMM contraction paths, fused softmax kernels, and different casting points.  Small score differences may also be amplified by the exponential in the softmax, especially when competing memory positions have close logits.  In practice, a deterministic float32 comparison is therefore expected to use absolute and relative tolerances rather than bitwise equality.

\paragraph{Assumptions behind the local parity check.}
A local Torch parity check that compares the absorbed implementation with a materialized baseline on a deterministic case and observes agreement within float32 tolerance is evidence for this local algebraic rewrite under the following assumptions.

\begin{enumerate}
\item The same tensors \(q^{\mathrm{nope}}\), \(q^{\mathrm{rope}}\), \(z=\mathtt{compressed\_kv\_norm\_BMC}\), and \(\rho=\mathtt{k\_rope\_after\_rope\_BMR}\) are used in both branches.

\item The compressed cache is sliced in the same order in both branches: the first \(C\) coordinates are \(z\), and the final \(R\) coordinates are \(\rho\), giving packed shape \(B\times M\times(C+R)\).

\item The weight \(\mathtt{kv\_b\_proj.weight}\) is viewed with the implementation's intended layout as \(W^{KV}\in\mathbb{R}^{H\times Y\times C}\), with \(Y=N+V\), and is split exactly as
\[
W^K=W^{KV}_{:,\;1:N,\;:},
\qquad
W^V=W^{KV}_{:,\;N+1:N+V,\;:}.
\]

\item The materialized baseline forms
\[
k^{\mathrm{nope}}_{b,h,m,n}
=
\sum_{c=1}^{C}W^K_{h,n,c}z_{b,m,c}
\]
and
\[
v^{\mathrm{mat}}_{b,h,m,\nu}
=
\sum_{c=1}^{C}W^V_{h,\nu,c}z_{b,m,c}
\]
from the same \(z\), \(W^K\), and \(W^V\) used by the absorbed branch.

\item There are no unaccounted projection biases, extra scales, extra normalizations, quantization corrections, or layout transpositions in one branch but not the other.  If such terms exist in a full implementation, they must be included consistently, or their algebraic effect must be handled separately.

\item The RoPE tensors are already in the same after-RoPE convention in both branches: query positions, memory positions, offsets, cosine/sine tables, interleaving convention, and the head-shared shape \(B\times M\times R\) for \(\rho\) agree.

\item Both branches use the same attention scale \(\alpha\), the same additive mask \(\mu\), the same softmax dimension over memory index \(m\), and the same handling of masked positions.

\item Dropout and other training-time randomness are disabled, or else the same post-softmax randomness is applied in both branches.  The deterministic local parity check corresponds to inference-style attention.

\item The comparison uses compatible floating-point settings: dtype, accumulation precision, TF32 enablement or disablement, deterministic-kernel settings, and tolerance thresholds are the same or intentionally controlled.  The reported match is a tolerance-based float32 match, not a proof of bitwise identity.
\end{enumerate}

\paragraph{Full-model features not verified by the local parity check.}
The local parity check verifies only the exercised local attention rewrite, starting from already-formed \(q^{\mathrm{nope}}\), \(q^{\mathrm{rope}}\), \(z\), \(\rho\), and \(W^{KV}\).  By itself, it does not verify the following full-model features.

\begin{enumerate}
\item The upstream \(D\)-dimensional hidden-state projections that produce the \(Q\)-rank query latent and the final \(q^{\mathrm{nope}}\) and \(q^{\mathrm{rope}}\) tensors.

\item The upstream KV compression path that produces \(\mathtt{compressed\_kv\_norm\_BMC}\), including any normalization, scaling, activation, dtype cast, or projection from hidden states.

\item The projection and RoPE path that produces \(\mathtt{k\_rope\_after\_rope\_BMR}\), including position-index offsets during incremental decoding.

\item End-to-end equality of a full transformer layer, including output projection after attention, residual connections, layer normalization, MLP or MoE blocks, and subsequent layers.

\item Cache append and update semantics across multiple autoregressive decoding steps, including prefill-versus-decode behavior, variable sequence lengths, padding, causal masking edge cases, and paged-cache or block-cache layouts.

\item Behavior under other dtypes or math modes, including float16, bfloat16, quantized weights, quantized KV caches, TF32, tensor parallelism, sharded weights, fused attention kernels, and hardware-specific accumulation paths.

\item Training-mode behavior, including dropout, backward gradients, optimizer interactions, activation checkpointing, and nondeterministic kernels.

\item Performance characteristics.  The derivation explains why the persistent cache can be represented with \(BM(C+R)\) elements and why the absorbed contractions are algebraically valid; it does not by itself establish lower latency, higher throughput, or lower end-to-end memory traffic for a particular implementation.
\end{enumerate}

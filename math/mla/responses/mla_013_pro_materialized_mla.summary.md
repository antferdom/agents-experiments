\subsection{Materialized MLA forward pass}
\label{sec:materialized-mla-forward}

Let
\[
T=N+R,\qquad X=C+R,\qquad Y=N+V,
\]
where \(B\) is the batch size, \(L\) is the query length, \(M\) is the memory length, \(D\) is the hidden size, \(H\) is the number of heads, \(N\) is the non-RoPE query/key head dimension, \(R\) is the RoPE suffix dimension, \(V\) is the value head dimension, \(Q\) is the query compression rank, and \(C\) is the KV compression rank.  The inputs are
\[
\mathbf h^q \equiv \texttt{hidden\_q\_BLD}\in\mathbb R^{B\times L\times D},
\qquad
\mathbf h^{kv} \equiv \texttt{hidden\_kv\_BMD}\in\mathbb R^{B\times M\times D}.
\]
All learned projections below are applied independently at each leading batch/sequence position.  Let
\[
P_{q,a}:\mathbb R^D\to\mathbb R^Q,\quad
P_{q,b}:\mathbb R^Q\to\mathbb R^{H T},\quad
P_{kv,a}:\mathbb R^D\to\mathbb R^X,\quad
P_{kv,b}:\mathbb R^C\to\mathbb R^{H Y},\quad
P_o:\mathbb R^{H V}\to\mathbb R^D
\]
denote the learned projections corresponding to \texttt{q\_a\_proj}, \texttt{q\_b\_proj}, \texttt{kv\_a\_proj\_with\_mqa}, \texttt{kv\_b\_proj}, and the output projection, respectively.  The layer normalizations \(\operatorname{LN}_{q,a}\) and \(\operatorname{LN}_{kv,a}\) act on the last dimension only.

Let \(\rho_p:\mathbb R^R\to\mathbb R^R\) denote the base RoPE rotation at position id \(p\).  For even \(R\), this can be written pairwise as
\[
\begin{pmatrix}
[\rho_p(x)]_{2i-1}\\
[\rho_p(x)]_{2i}
\end{pmatrix}
=
\begin{pmatrix}
\cos(p\omega_i) & -\sin(p\omega_i)\\
\sin(p\omega_i) & \cos(p\omega_i)
\end{pmatrix}
\begin{pmatrix}
x_{2i-1}\\
x_{2i}
\end{pmatrix},
\qquad i=1,\ldots,R/2,
\]
with the base-RoPE frequency schedule \(\{\omega_i\}\) used by the executable baseline.  Let \(p^q_\ell\) and \(p^k_m\) be the query and key position ids used for query position \(\ell\) and memory position \(m\).

\paragraph{Query path.}
The query path is
\[
\texttt{q\_b\_proj(q\_a\_layernorm(q\_a\_proj(hidden\_q)))}.
\]
In tensor notation,
\begin{align}
\mathbf q^{\mathrm{lat}}
&=
\operatorname{LN}_{q,a}\!\left(P_{q,a}(\mathbf h^q)\right)
&&\in \mathbb R^{B\times L\times Q}
&&\equiv \texttt{q\_latent\_BLQ},
\\
\overline{\mathbf q}
&=
P_{q,b}(\mathbf q^{\mathrm{lat}})
&&\in \mathbb R^{B\times L\times (H T)}.
\end{align}
The packed last dimension of \(\overline{\mathbf q}\) is viewed as \(H\) heads of width \(T\) and transposed to head-major layout:
\[
\mathbf q^{\mathrm{full}}_{b h \ell t}
=
\overline{\mathbf q}_{b,\ell,(h-1)T+t},
\qquad
\mathbf q^{\mathrm{full}}\in\mathbb R^{B\times H\times L\times T}
\equiv \texttt{q\_full\_BHLT}.
\]
Splitting the last dimension into the non-RoPE prefix of size \(N\) and the RoPE suffix of size \(R\) gives
\begin{align}
\mathbf q^{\mathrm n}_{b h \ell n}
&=
\mathbf q^{\mathrm{full}}_{b h \ell n},
&& n=1,\ldots,N,
\\
\mathbf q^{\mathrm{r,raw}}_{b h \ell r}
&=
\mathbf q^{\mathrm{full}}_{b h \ell,N+r},
&& r=1,\ldots,R,
\end{align}
with
\[
\mathbf q^{\mathrm n}\in\mathbb R^{B\times H\times L\times N}
\equiv \texttt{q\_nope\_BHLN},
\qquad
\mathbf q^{\mathrm{r,raw}}\in\mathbb R^{B\times H\times L\times R}
\equiv \texttt{q\_rope\_BHLR}.
\]
Base RoPE is then applied to the query RoPE suffix:
\[
\mathbf q^{\mathrm r}_{b h \ell,:}
=
\rho_{p^q_\ell}\!\left(\mathbf q^{\mathrm{r,raw}}_{b h \ell,:}\right),
\qquad
\mathbf q^{\mathrm r}\in\mathbb R^{B\times H\times L\times R}
\equiv \texttt{q\_rope\_after\_rope\_BHLR}.
\]
The materialized query used in attention is the concatenation
\[
\mathbf q
=
\left[\mathbf q^{\mathrm n},\mathbf q^{\mathrm r}\right]_{\mathrm{last}}
\in\mathbb R^{B\times H\times L\times T}
\equiv \texttt{query\_BHLT}.
\]
The query RoPE suffix \(\mathbf q^{\mathrm{r,raw}}\) is produced by the query projection path above; it is not produced by the KV RoPE projection.

\paragraph{KV path.}
The KV path begins with the shared MQA-style projection
\[
\mathbf z^{kv}
=
P_{kv,a}(\mathbf h^{kv})
\in\mathbb R^{B\times M\times X}
\equiv \texttt{kv\_raw\_BMX},
\qquad X=C+R.
\]
This tensor is split along the last dimension into the compressed KV latent and the raw shared RoPE key suffix:
\begin{align}
\mathbf c^{\mathrm{raw}}_{b m c}
&=
\mathbf z^{kv}_{b m c},
&& c=1,\ldots,C,
\\
\mathbf k^{\mathrm{r,raw}}_{b m r}
&=
\mathbf z^{kv}_{b m,C+r},
&& r=1,\ldots,R.
\end{align}
Thus
\[
\mathbf c^{\mathrm{raw}}\in\mathbb R^{B\times M\times C}
\equiv \texttt{compressed\_kv\_BMC},
\qquad
\mathbf k^{\mathrm{r,raw}}\in\mathbb R^{B\times M\times R}
\equiv \texttt{k\_rope\_raw\_BMR}.
\]
The KV layer normalization applies only to the compressed KV latent:
\[
\mathbf c
=
\operatorname{LN}_{kv,a}\!\left(\mathbf c^{\mathrm{raw}}\right)
\in\mathbb R^{B\times M\times C}
\equiv \texttt{compressed\_kv\_norm\_BMC}.
\]
The raw shared RoPE key suffix bypasses this layer normalization.  It is first given a singleton head axis,
\[
\mathbf k^{\mathrm{r,in}}_{b,1,m,r}
=
\mathbf k^{\mathrm{r,raw}}_{b m r},
\qquad
\mathbf k^{\mathrm{r,in}}\in\mathbb R^{B\times 1\times M\times R}
\equiv \texttt{k\_rope\_B1MR},
\]
and then base RoPE is applied:
\[
\mathbf k^{\mathrm r}_{b,1,m,:}
=
\rho_{p^k_m}\!\left(\mathbf k^{\mathrm{r,in}}_{b,1,m,:}\right),
\qquad
\mathbf k^{\mathrm r}\in\mathbb R^{B\times 1\times M\times R}
\equiv \texttt{k\_rope\_after\_rope\_B1MR}.
\]
The compressed KV latent is expanded into packed per-head non-RoPE keys and values:
\begin{align}
\overline{\mathbf u}^{kv}
&=
P_{kv,b}(\mathbf c)
&&\in\mathbb R^{B\times M\times (H Y)},
\\
\mathbf u^{kv}_{b h m y}
&=
\overline{\mathbf u}^{kv}_{b,m,(h-1)Y+y}
&&\in\mathbb R^{B\times H\times M\times Y}
&&\equiv \texttt{kv\_full\_BHMY}.
\end{align}
Splitting the last dimension of \(\mathbf u^{kv}\) into the non-RoPE key part of size \(N\) and the value part of size \(V\) gives
\begin{align}
\mathbf k^{\mathrm n}_{b h m n}
&=
\mathbf u^{kv}_{b h m n},
&& n=1,\ldots,N,
\\
\mathbf v_{b h m v}
&=
\mathbf u^{kv}_{b h m,N+v},
&& v=1,\ldots,V,
\end{align}
with
\[
\mathbf k^{\mathrm n}\in\mathbb R^{B\times H\times M\times N}
\equiv \texttt{k\_nope\_BHMN},
\qquad
\mathbf v\in\mathbb R^{B\times H\times M\times V}
\equiv \texttt{value\_BHMV}.
\]
The RoPE key suffix is shared across heads.  When materializing the full key tensor, it is broadcast from head dimension \(1\) to head dimension \(H\):
\[
\mathbf k^{\mathrm{r},H}_{b h m r}
=
\mathbf k^{\mathrm r}_{b,1,m,r},
\qquad
\mathbf k^{\mathrm{r},H}\in\mathbb R^{B\times H\times M\times R}.
\]
The materialized key is then
\[
\mathbf k
=
\left[\mathbf k^{\mathrm n},\mathbf k^{\mathrm{r},H}\right]_{\mathrm{last}}
\in\mathbb R^{B\times H\times M\times T}
\equiv \texttt{key\_BHMT}.
\]

\paragraph{Scores and attention weights.}
The materialized attention scores are computed by multiplying
\[
\texttt{query\_BHLT}
\quad\text{against}\quad
\texttt{key\_BHMT}^{\mathsf T}
\]
over the last dimension \(T\), using the simplified local scale \(1/\sqrt{T}\):
\[
\mathbf S
=
\frac{1}{\sqrt{T}}\,
\mathbf q\,\mathbf k^{\mathsf T}
\in\mathbb R^{B\times H\times L\times M}
\equiv \texttt{scores\_BHLM},
\]
where \(\mathbf k^{\mathsf T}\) denotes transposition of the last two axes of \(\mathbf k\), i.e. \(B\times H\times M\times T\) becomes \(B\times H\times T\times M\).

Equivalently, in index notation,
\[
\texttt{scores\_BHLM}[b,h,\ell,m]
=
\mathbf S_{b h \ell m}
=
\frac{1}{\sqrt{T}}
\sum_{t=1}^{T}
\mathbf q_{b h \ell t}\,
\mathbf k_{b h m t},
\]
for
\[
b=1,\ldots,B,\qquad
h=1,\ldots,H,\qquad
\ell=1,\ldots,L,\qquad
m=1,\ldots,M.
\]

With no mask, the attention weights are the softmax over the memory axis:
\[
\mathbf a_{b h \ell m}
=
\frac{\exp(\mathbf S_{b h \ell m})}
{\sum_{j=1}^{M}\exp(\mathbf S_{b h \ell j})},
\qquad
\mathbf a\in\mathbb R^{B\times H\times L\times M}
\equiv \texttt{weights\_BHLM}.
\]

\paragraph{Non-RoPE/RoPE score decomposition.}
Because the query and key tensors are concatenations of disjoint non-RoPE and RoPE feature blocks,
\[
\mathbf q
=
\left[\mathbf q^{\mathrm n},\mathbf q^{\mathrm r}\right]_{\mathrm{last}},
\qquad
\mathbf k
=
\left[\mathbf k^{\mathrm n},\mathbf k^{\mathrm{r},H}\right]_{\mathrm{last}},
\]
the score tensor decomposes as
\[
\mathbf S
=
\mathbf S^{\mathrm n}
+
\mathbf S^{\mathrm r},
\qquad
\mathbf S^{\mathrm n},\mathbf S^{\mathrm r}\in\mathbb R^{B\times H\times L\times M}.
\]
The non-RoPE contribution is
\[
\mathbf S^{\mathrm n}_{b h \ell m}
=
\frac{1}{\sqrt{T}}
\sum_{n=1}^{N}
\mathbf q^{\mathrm n}_{b h \ell n}\,
\mathbf k^{\mathrm n}_{b h m n},
\qquad
\mathbf q^{\mathrm n}\in\mathbb R^{B\times H\times L\times N},
\quad
\mathbf k^{\mathrm n}\in\mathbb R^{B\times H\times M\times N}.
\]
In batched matrix form,
\[
\mathbf S^{\mathrm n}
=
\frac{1}{\sqrt{T}}\,
\mathbf q^{\mathrm n}\,
(\mathbf k^{\mathrm n})^{\mathsf T}
\in\mathbb R^{B\times H\times L\times M}.
\]
The RoPE contribution is
\[
\mathbf S^{\mathrm r}_{b h \ell m}
=
\frac{1}{\sqrt{T}}
\sum_{r=1}^{R}
\mathbf q^{\mathrm r}_{b h \ell r}\,
\mathbf k^{\mathrm{r},H}_{b h m r}
=
\frac{1}{\sqrt{T}}
\sum_{r=1}^{R}
\mathbf q^{\mathrm r}_{b h \ell r}\,
\mathbf k^{\mathrm r}_{b,1,m,r},
\]
where
\[
\mathbf q^{\mathrm r}\in\mathbb R^{B\times H\times L\times R},
\qquad
\mathbf k^{\mathrm r}\in\mathbb R^{B\times 1\times M\times R},
\qquad
\mathbf k^{\mathrm{r},H}\in\mathbb R^{B\times H\times M\times R}.
\]
In batched matrix form with head broadcast,
\[
\mathbf S^{\mathrm r}
=
\frac{1}{\sqrt{T}}\,
\mathbf q^{\mathrm r}\,
(\mathbf k^{\mathrm{r},H})^{\mathsf T}
\in\mathbb R^{B\times H\times L\times M}.
\]
Thus the full materialized score computation is
\[
\boxed{
\mathbf S_{b h \ell m}
=
\frac{1}{\sqrt{T}}
\left(
\sum_{n=1}^{N}
\mathbf q^{\mathrm n}_{b h \ell n}\,
\mathbf k^{\mathrm n}_{b h m n}
+
\sum_{r=1}^{R}
\mathbf q^{\mathrm r}_{b h \ell r}\,
\mathbf k^{\mathrm r}_{b,1,m,r}
\right)
}
\]
with output shape \(B\times H\times L\times M\).

\paragraph{Context and output projection.}
The attention context is the weighted sum of values over the memory axis:
\[
\mathbf c^{\mathrm{attn}}_{b h \ell v}
=
\sum_{m=1}^{M}
\mathbf a_{b h \ell m}\,
\mathbf v_{b h m v},
\qquad
\mathbf c^{\mathrm{attn}}\in\mathbb R^{B\times H\times L\times V}
\equiv \texttt{context\_BHLV}.
\]
The context is transposed back to \(B\times L\times H\times V\), flattened over the head and value dimensions, and projected to the hidden dimension:
\[
\overline{\mathbf c}^{\mathrm{attn}}_{b,\ell,(h-1)V+v}
=
\mathbf c^{\mathrm{attn}}_{b h \ell v},
\qquad
\overline{\mathbf c}^{\mathrm{attn}}\in\mathbb R^{B\times L\times (H V)}.
\]
The final output is
\[
\mathbf o
=
P_o\!\left(\overline{\mathbf c}^{\mathrm{attn}}\right)
\in\mathbb R^{B\times L\times D}
\equiv \texttt{output\_BLD}.
\]

\paragraph{Correspondence with the local implementation.}
The local implementation's \texttt{view} followed by \texttt{transpose(1, 2)} implements the reshapes
\[
B\times L\times (H T)\to B\times H\times L\times T
\]
for queries and
\[
B\times M\times (H Y)\to B\times H\times M\times Y
\]
for expanded KV.  The \texttt{split} operations implement the decompositions
\[
T=N+R,\qquad X=C+R,\qquad Y=N+V,
\]
namely \(\texttt{q\_full\_BHLT}\mapsto(\texttt{q\_nope\_BHLN},\texttt{q\_rope\_BHLR})\), \(\texttt{kv\_raw\_BMX}\mapsto(\texttt{compressed\_kv\_BMC},\texttt{k\_rope\_raw\_BMR})\), and \(\texttt{kv\_full\_BHMY}\mapsto(\texttt{k\_nope\_BHMN},\texttt{value\_BHMV})\).  The \texttt{cat} operations materialize \(\texttt{query\_BHLT}\) from the non-RoPE query prefix and the RoPE-rotated query suffix, and materialize \(\texttt{key\_BHMT}\) from the non-RoPE keys and the head-broadcast RoPE key suffix.  The call \texttt{matmul(query, key.transpose(-2, -1))} evaluates the indexed score formula above, including the multiplication over the \(T=N+R\) feature axis; multiplication by \(1/\sqrt{T}\) applies the executable-baseline score scale.  The \texttt{softmax(..., dim=-1)} operation produces \(\texttt{weights\_BHLM}\) over the memory index \(m\).  The second \texttt{matmul(weights, value)} produces \(\texttt{context\_BHLV}\).  Finally, \texttt{transpose} and \texttt{view}/\texttt{reshape} flatten \(H\) and \(V\) into \(H V\), after which the output projection maps \(B\times L\times (H V)\) to \(\texttt{output\_BLD}\).

\paragraph{Assumptions and non-goals.}
This materialized executable baseline uses no attention mask, disables dropout, and applies base RoPE to the RoPE suffixes.  The softmax is therefore taken over all \(M\) memory positions, and the attention weights are used directly in the value matmul.  The score scale is exactly \(1/\sqrt{T}\).  This derivation is only for the accepted local baseline and does not model full-model YaRN behavior or mask variants.

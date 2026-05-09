```tex
\section{External implementation check}
\label{sec:external-implementation-check}

\paragraph{Conventions.}
For a fixed batch element \(b\), write the cached latent and RoPE suffix as row-token
matrices
\[
  Z_b := [z_{bm:}]_{m=1}^{M}\in \mathbb{R}^{M\times C},
  \qquad
  \rho_b := [\rho_{bm:}]_{m=1}^{M}\in \mathbb{R}^{M\times R}.
\]
For head \(h\), write
\[
  Q^{\mathrm n}_{bh}:=q_{\mathrm{nope},bh}\in \mathbb{R}^{L\times N},
  \qquad
  Q^{\mathrm r}_{bh}:=q_{\mathrm{rope},bh}\in \mathbb{R}^{L\times R},
\]
where \(\mathrm n\) denotes the non-RoPE part and \(\mathrm r\) denotes the RoPE
suffix.  The projection blocks from \(\mathrm{kv\_b\_proj}\) are treated as
column maps
\[
  W^K_h\in \mathbb{R}^{N\times C},
  \qquad
  W^V_h\in \mathbb{R}^{V\times C}.
\]
Thus implementation row-products use \((W^K_h)^\top\in\mathbb{R}^{C\times N}\)
and \((W^V_h)^\top\in\mathbb{R}^{C\times V}\).  The canonical compressed MLA
cache object is
\[
  \mathcal{C}^{\mathrm{MLA}}_{bm}
  :=
  [\,z_{bm:},\rho_{bm:}\,]\in \mathbb{R}^{C+R}
  =
  \mathbb{R}^{X},
  \qquad
  \mathcal{C}^{\mathrm{MLA}}_{b}
  =
  [\,Z_b,\rho_b\,]\in \mathbb{R}^{M\times X}.
\]

\paragraph{Source-backed comparison.}
The two external implementations expose the same algebraic cache
\([z,\rho]\in\mathbb{R}^{C+R}\), up to physical layout.

\begin{itemize}
\item \textbf{vLLM, commit \texttt{e3b65a5ba069b350120ca7a614a010787d2de867}
      \((2026\text{-}05\text{-}08)\).}
      In
      \texttt{vllm/model\_executor/layers/attention/mla\_attention.py}:
      \begin{itemize}
      \item Lines \(26\)--\(63\) define MLA with a single latent vector per-token
            cache.  The source constants are
            \[
              L_{\mathrm{kv}}=512,\qquad
              P_{\mathrm{nope}}=128,\qquad
              R=64,\qquad
              V=128,
            \]
            corresponding in the present notation to
            \(C=512\), \(N=128\), \(R=64\), \(V=128\)
            \((L_{\mathrm{kv}}\) there is the KV LoRA rank, not the query
            length \(L)\).
      \item Lines \(66\)--\(90\) describe the compute-friendly path:
            \[
              \texttt{kv\_c @ W\_UK}
              \;\equiv\;
              Z_b (W^K_h)^\top,
              \qquad
              \texttt{kv\_c @ W\_UV}
              \;\equiv\;
              Z_b (W^V_h)^\top,
            \]
            followed by ordinary attention over
            \[
              [\,Q^{\mathrm n}_{bh},Q^{\mathrm r}_{bh}\,]\in\mathbb{R}^{L\times T},
              \qquad
              [\,Z_b(W^K_h)^\top,\rho_b\,]\in\mathbb{R}^{M\times T},
              \qquad
              Z_b(W^V_h)^\top\in\mathbb{R}^{M\times V}.
            \]
      \item Lines \(94\)--\(118\) describe the data-movement-friendly path:
            absorb \(W_UK\) into the query,
            attend with
            \[
              [\,Q^{\mathrm n}_{bh}W^K_h,Q^{\mathrm r}_{bh}\,]
              \in\mathbb{R}^{L\times X}
              \quad\text{against}\quad
              [\,Z_b,\rho_b\,]\in\mathbb{R}^{M\times X},
            \]
            use value \(Z_b\in\mathbb{R}^{M\times C}\), and then multiply by
            \(W_UV\), i.e. by \((W^V_h)^\top\) in row convention.
      \item Lines \(826\)--\(852\) view
            \[
              \texttt{kv\_b\_proj\_weight}
              \quad\text{as}\quad
              (C,H,N+V)=(C,H,Y),
            \]
            and split the last dimension into the stored row-product blocks
            \(W_UK\) and \(W_UV\), i.e. into
            \((W^K_h)^\top\) and \((W^V_h)^\top\) for each head \(h\).
      \item Lines \(951\)--\(961\) return an
            \(\texttt{MLAAttentionSpec}\) with
            \[
              \texttt{num\_kv\_heads}=1,
              \qquad
              \texttt{head\_size}
              =
              \texttt{kv\_lora\_rank}
              +
              \texttt{qk\_rope\_head\_dim}
              =
              C+R=X.
            \]
      \end{itemize}

\item \textbf{SGLang, commit \texttt{a61a14f416c4003809a82112b4a591eec38a0a10}
      \((2026\text{-}05\text{-}09)\).}
      \begin{itemize}
      \item In \texttt{python/sglang/srt/models/deepseek\_v2.py},
            lines \(1391\)--\(1399\) construct
            \[
              \texttt{kv\_b\_proj}:\mathbb{R}^{C}\to
              \mathbb{R}^{H(N+V)}=\mathbb{R}^{HY}.
            \]
      \item In the same file, lines \(1431\)--\(1440\) construct
            \(\texttt{attn\_mqa}\) with
            \[
              \texttt{head\_dim}=C+R=X,
              \qquad
              \texttt{num\_kv\_heads}=1,
              \qquad
              \texttt{v\_head\_dim}=C.
            \]
            Thus the attention kernel consumes one KV head whose key is
            \([z,\rho]\) and whose value is the latent \(z\).
      \item In
            \texttt{python/sglang/srt/models/deepseek\_common/attention\_forward\_methods/forward\_mla.py},
            lines \(241\)--\(243\) split \(q\) into \(q_{\mathrm{nope}}\) and
            \(q_{\mathrm{pe}}\), and split the latent cache to obtain
            \(k_{\mathrm{pe}}\).  Lines \(320\)--\(323\) compute
            \[
              q_{\mathrm{nope,out}}
              =
              q_{\mathrm{nope}} W_K
              \;\equiv\;
              Q^{\mathrm n}_{bh}W^K_h.
            \]
            Lines \(457\)--\(468\) attend with
            \[
              q=[\,q_{\mathrm{nope,out}},q_{\mathrm{pe}}\,],
              \qquad
              k=[\,k_{\mathrm{nope}},k_{\mathrm{pe}}\,],
              \qquad
              v=k_{\mathrm{nope}},
            \]
            where \(k_{\mathrm{nope}}\) is the \(C\)-wide latent component.
            Lines \(472\)--\(592\) up-project the latent context through \(W_V\)
            and then apply \(\texttt{o\_proj}\).
      \item In
            \texttt{python/sglang/srt/mem\_cache/memory\_pool.py},
            lines \(1560\)--\(1567\) allocate one KV buffer per layer with
            physical shape
            \[
              (\texttt{size}+\texttt{page\_size},\,1,\,
              \texttt{kv\_cache\_dim}),
            \]
            and lines \(1630\)--\(1679\) make
            \(\texttt{set\_mla\_kv\_buffer}\) write
            \[
              [\,\texttt{cache\_k\_nope},\texttt{cache\_k\_rope}\,]
              \;\equiv\;
              [\,z,\rho\,]
            \]
            into that single buffer.
      \item In
            \texttt{python/sglang/srt/model\_executor/model\_runner\_kv\_cache\_mixin.py},
            lines \(138\)--\(183\) define the default MLA cache width
            \[
              \texttt{kv\_cache\_dim}
              =
              \texttt{kv\_lora\_rank}
              +
              \texttt{qk\_rope\_head\_dim}
              =
              C+R=X,
            \]
            with FP8/NSA storage exceptions.
      \end{itemize}
\end{itemize}

Consequently, modulo paging and storage format,
\[
  \mathcal{C}^{\mathrm{vLLM}}_{bm}
  =
  \mathcal{C}^{\mathrm{SGLang}}_{bm}
  =
  \mathcal{C}^{\mathrm{local}}_{bm}
  =
  [\,z_{bm:},\rho_{bm:}\,]\in\mathbb{R}^{C+R}.
\]

\paragraph{Compute-friendly versus data-movement-friendly forms.}
For fixed \(b,h\), suppress \(b,h\) where unambiguous and set
\[
  Z:=Z_b\in\mathbb{R}^{M\times C},
  \qquad
  \rho:=\rho_b\in\mathbb{R}^{M\times R},
  \qquad
  Q^{\mathrm n}:=Q^{\mathrm n}_{bh}\in\mathbb{R}^{L\times N},
  \qquad
  Q^{\mathrm r}:=Q^{\mathrm r}_{bh}\in\mathbb{R}^{L\times R}.
\]
The two algorithms are:
\[
\begin{array}{rcll}
\text{compute-friendly:}
&
K_h^{\mathrm n}
&:=&
Z(W^K_h)^\top
\in\mathbb{R}^{M\times N},
\\[2pt]
&
V_h
&:=&
Z(W^V_h)^\top
\in\mathbb{R}^{M\times V},
\\[2pt]
&
S_h^{\mathrm{cf}}
&:=&
Q^{\mathrm n}(K_h^{\mathrm n})^\top
+
Q^{\mathrm r}\rho^\top
\in\mathbb{R}^{L\times M},
\\[2pt]
&
O_h^{\mathrm{cf}}
&:=&
\operatorname{softmax}_{\mathrm{row}}(S_h^{\mathrm{cf}})\,V_h
\in\mathbb{R}^{L\times V};
\\[8pt]
\text{data-movement-friendly:}
&
Q'_h
&:=&
Q^{\mathrm n}W^K_h
\in\mathbb{R}^{L\times C},
\\[2pt]
&
S_h^{\mathrm{dm}}
&:=&
Q'_h Z^\top
+
Q^{\mathrm r}\rho^\top
\in\mathbb{R}^{L\times M},
\\[2pt]
&
G_h
&:=&
\operatorname{softmax}_{\mathrm{row}}(S_h^{\mathrm{dm}})\,Z
\in\mathbb{R}^{L\times C},
\\[2pt]
&
O_h^{\mathrm{dm}}
&:=&
G_h(W^V_h)^\top
\in\mathbb{R}^{L\times V}.
\end{array}
\]
Equivalently, the compute-friendly attention has query/key width
\(T=N+R\), while the data-movement-friendly attention has query/key width
\(X=C+R\) and value width \(C\).

\paragraph{Theorem (external MLA equivalence).}
Fix \(b,h\).  For each memory token \(m\) and query token \(\ell\), using
column-vector notation for individual tokens,
\[
  k_{hm}=W^K_h z_{bm}\in\mathbb{R}^{N},
  \qquad
  v_{hm}=W^V_h z_{bm}\in\mathbb{R}^{V},
  \qquad
  q'_{bh\ell}=(W^K_h)^\top q_{\mathrm{nope},bh\ell}\in\mathbb{R}^{C}.
\]
Then the materialized path and the absorbed path have identical logits,
identical attention weights, and identical per-head context before
\(\texttt{o\_proj}\).  In row-matrix form,
\[
  S_h^{\mathrm{cf}}
  =
  Q^{\mathrm n}(K_h^{\mathrm n})^\top
  +
  Q^{\mathrm r}\rho^\top
  =
  Q^{\mathrm n}W^K_h Z^\top
  +
  Q^{\mathrm r}\rho^\top
  =
  Q'_h Z^\top
  +
  Q^{\mathrm r}\rho^\top
  =
  S_h^{\mathrm{dm}}.
\]
Hence
\[
  A_h
  :=
  \operatorname{softmax}_{\mathrm{row}}(S_h^{\mathrm{cf}})
  =
  \operatorname{softmax}_{\mathrm{row}}(S_h^{\mathrm{dm}})
  \in\mathbb{R}^{L\times M}.
\]
At the context level,
\[
  O_h^{\mathrm{cf}}
  =
  A_h V_h
  =
  A_h Z(W^V_h)^\top
  =
  G_h(W^V_h)^\top
  =
  O_h^{\mathrm{dm}},
  \qquad
  G_h:=A_hZ.
\]
Equivalently, per query token,
\[
  \sum_{m=1}^{M} A_{h,\ell m} v_{hm}
  =
  \sum_{m=1}^{M} A_{h,\ell m} W^V_h z_{bm}
  =
  W^V_h
  \left(
    \sum_{m=1}^{M} A_{h,\ell m} z_{bm}
  \right).
\]
Thus, using the common implementation shorthand in which
\(zW^V_h\) denotes the row-product \(Z(W^V_h)^\top\),
\[
  \operatorname{softmax}_{\mathrm{row}}
  \!\left(
    q_{\mathrm{nope},h} k_h^\top
    +
    q_{\mathrm{rope},h}\rho^\top
  \right)
  z W^V_h
  =
  \operatorname{softmax}_{\mathrm{row}}
  \!\left(
    q'_h z^\top
    +
    q_{\mathrm{rope},h}\rho^\top
  \right)
  z W^V_h
  =
  O_h^{\mathrm{cf}}
  =
  O_h^{\mathrm{dm}}.
\]
With the declared mathematical shape \(W^V_h\in\mathbb{R}^{V\times C}\),
the fully explicit row-matrix term is
\[
  zW^V_h \equiv Z(W^V_h)^\top\in\mathbb{R}^{M\times V}.
\]
The same proof applies with any common scalar attention scale and any common
additive attention mask inserted into both score matrices.  After concatenating
the \(H\) equal per-head contexts and applying the shared output projection,
the post-\(\texttt{o\_proj}\) outputs are therefore equal as well.
\hfill\(\mathrm{q.e.d.}\)

\paragraph{Limitations of the external check.}
The cited external code also contains implementation features that are not part
of the base real-valued MLA derivation:
\[
  \text{quantized storage/compute, e.g. FP8},
  \qquad
  \text{sparse or NSA modes},
  \qquad
  \text{paged-cache layout and page indexing},
  \qquad
  \text{kernel fusion, tiling, and scheduling}.
\]
These features may change storage dtype, physical address calculation, enabled
attention pattern, or numerical kernel realization.  They do not change the
dense base algebra above:
\[
  \mathcal{C}^{\mathrm{MLA}}_{bm}=[\,z_{bm:},\rho_{bm:}\,]\in\mathbb{R}^{C+R},
  \qquad
  k_h=W^K_hz,
  \qquad
  v_h=W^V_hz,
  \qquad
  q'_h=(W^K_h)^\top q_{\mathrm{nope},h}.
\]
Accordingly, quantization, NSA/sparsity, paging, and kernel details should be
modeled as separate storage, masking, or numerical-kernel layers, not folded
into the base MLA equivalence derivation.
```

# cacheDB v1 Example Query-Expanded Lookup

The retrieval interface is `rg` over `cachedb/index.jsonl`. Intended v1 agent workflow is:

1. Search `cachedb/index.jsonl` directly with `rg`.
2. Decompose the natural-language prompt into keyword queries.
3. Intersect those queries to find the right implementation neighborhood.
4. If the local cache narrows the stack but does not contain the direct template, use GitHub code search next.
5. Use broader web search only after local cache and GitHub search have already reduced the space.

- **Prompt:** `Implement vector addition in cuteDSL for B200`

## Why The Exact Prompt Fails

Natural-language prompts are too specific for a flat grep index. v1 is path- and keyword-driven, not semantic.

```bash
rg -i 'implement vector addition in cuteDSL for B200' cachedb/index.jsonl | wc -l
```

```text
0
```

This is expected. The agent has to decompose the prompt.

## Prompt Decomposition Into Queries

Break the prompt into orthogonal concepts:

| Concept | Search terms |
|---|---|
| Operation | `vector addition,vector_add,vec-add` |
| Implementation stack | `CuTeDSL, cutedsl, CuTe DSL, cutlass-dsl, CuTe, cute` |
| Target hardware | `B200, Blackwell, GB200, SM100` |
| Nearby implementation seeds | `hello.py`, `simple reduction`, `dense_gemm.py`, `dense_gemm_persistent.py`, `fp16_gemm_0.py`, `quack` |

The hardware expansion matters because material about B200 is often filed under broader Blackwell names such as `Blackwell`, `SM100`, or sometimes nearby `GB200`.

## Stage 1: Local cacheDB lookup with `rg`

### Bucket sizes

```bash
rg -i 'vector addition|vector_add|vec-add|vec add|vadd' cachedb/index.jsonl | wc -l
rg -i 'CuTeDSL|cutedsl|CuTe DSL|cutlass-dsl|CuTe|cute' cachedb/index.jsonl | wc -l
rg -i 'B200|Blackwell|GB200|SM100' cachedb/index.jsonl | wc -l
```

```text
4
210
388
```

This already shows why decomposition works better than the literal prompt. The vocabulary exists in the index, but not as one exact sentence.

### Useful intersections

```bash
rg -i 'vector addition|vector_add|vec-add|vec add|vadd' cachedb/index.jsonl | \
  rg -i 'CuTeDSL|cutedsl|CuTe DSL|cutlass-dsl|CuTe|cute' | wc -l

rg -i 'vector addition|vector_add|vec-add|vec add|vadd' cachedb/index.jsonl | \
  rg -i 'B200|Blackwell|GB200|SM100' | wc -l

rg -i 'B200|Blackwell|GB200|SM100' cachedb/index.jsonl | \
  rg -i 'CuTeDSL|cutedsl|CuTe DSL|cutlass-dsl|CuTe|cute' | wc -l
```

```text
0
1
42
```

Interpretation:

- The local cache has no direct `vector addition + CuTe` bookmark.
- The local cache does have one direct `vector addition + B200` hit.
- The local cache has a strong `Blackwell/B200 + CuTe` neighborhood with 42 rows.

That is enough to route the agent into the right stack even though it does not yet have the exact final template.

## Representative local hits

### Operation-side hits

Query:

```bash
rg -i 'vector addition|vector_add|vec-add|vec add|vadd' cachedb/index.jsonl | \
  jq -r '.title + " || " + .path + " || " + .url'
```

Representative hits:

- `Vector Addition Worklog on a B200 | Chloe Chia`
  `Computational Research > Artificial Intelligence > Software > AI Compilers & PL > Kernels > vector addition`
  `https://chloechiaw.github.io/vec-add/`
- `learn-cuda/01_vector_addition at main · gau-nernst/learn-cuda`
  `Computational Research > Artificial Intelligence > Software > AI Compilers & PL > Kernels > vector addition`
  `https://github.com/gau-nernst/learn-cuda/tree/main/01_vector_addition`
- `learn-cuda/01_vector_addition/main.py at main · gau-nernst/learn-cuda`
  `Computational Research > Artificial Intelligence > Software > AI Compilers & PL > PyTorch > pybind > torch-cpp_extension > cpp_extension-load_inline`
  `https://github.com/gau-nernst/learn-cuda/blob/main/01_vector_addition/main.py`

### Stack-and-hardware hits

Query:

```bash
rg -i 'B200|Blackwell|GB200|SM100' cachedb/index.jsonl | \
  rg -i 'CuTeDSL|cutedsl|CuTe DSL|cutlass-dsl|CuTe|cute' | \
  jq -r '.title + " || " + .path + " || " + .url'
```

Representative hits:

- `cutlass/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py at main · NVIDIA/cutlass`
  `Computational Research > Artificial Intelligence > Software > AI Compilers & PL > Kernels > GEMM > cutlass-gemm > cute-gemm > cute-gemm-blackwell`
  `https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py`
- `cutlass/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py at main · NVIDIA/cutlass`
  `Computational Research > Artificial Intelligence > Software > AI Compilers & PL > CUDA > CUTLASS > cutlass-dsl`
  `https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py`
- `cutlass/examples/python/CuTeDSL/blackwell/dense_gemm.py at 8825e8be4f0ee9b55ce6198271b7a16d6b473f02 · NVIDIA/cutlass`
  `Computational Research > Artificial Intelligence > Software > AI Compilers & PL > CUDA > CUTLASS > cutlass-dsl > cutlass-4.2`
  `https://github.com/NVIDIA/cutlass/blob/8825e8be4f0ee9b55ce6198271b7a16d6b473f02/examples/python/CuTeDSL/blackwell/dense_gemm.py`
- `quack/quack/gemm_sm100.py at main · Dao-AILab/quack`
  `Computational Research > Artificial Intelligence > Software > AI Compilers & PL > Kernels > GEMM > gemm-blackwell > gemm-blackwell-cute`
  `https://github.com/Dao-AILab/quack/blob/main/quack/gemm_sm100.py`
- `Blackwell Pipelining with CuTeDSL | simons blog`
  `Computational Research > Artificial Intelligence > Software > AI Compilers & PL > CUDA > CUTLASS > cutlass-dsl`
  `https://veitner.bearblog.dev/blackwell-pipelining-with-cutedsl/`

## What the local cache tells the agent

After the local pass, the agent should conclude:

1. `CuTeDSL + Blackwell` is the right implementation stack.
2. `B200` should be expanded to broader Blackwell identifiers like `SM100`.
3. There is no direct cached `vector-add-in-CuTeDSL` bookmark.
4. The best next step is not generic web search. It is GitHub code search in authoritative repos such as `NVIDIA/cutlass`.

## Stage 2: GitHub code search follow-up

Once the local cache has narrowed the problem to `CUTLASS/CuTeDSL/Blackwell`, GitHub search can recover direct code templates.

### Query: `elementwise_add.py` in `NVIDIA/cutlass`

This is the most important follow-up because the local cache does not contain a direct CuTeDSL vector-add example.

GitHub code search result:

- `examples/python/CuTeDSL/ampere/elementwise_add.py`
  `https://github.com/NVIDIA/cutlass/blob/08185b9c3e90510ee2b656662ed0d53b06d28157/examples/python/CuTeDSL/ampere/elementwise_add.py`

This is not B200-specific, but it is a direct CuTeDSL elementwise-add template. It is a strong starting point for porting the operation shape to Blackwell-era conventions.

### Query: `fp16_gemm_0 CuTeDSL blackwell` in `NVIDIA/cutlass`

GitHub code search results:

- `examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py`
  `https://github.com/NVIDIA/cutlass/blob/08185b9c3e90510ee2b656662ed0d53b06d28157/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py`
- `examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py`
  `https://github.com/NVIDIA/cutlass/blob/08185b9c3e90510ee2b656662ed0d53b06d28157/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py`

These give current Blackwell launch structure, scheduling, and runtime conventions.

### Query: `dense_gemm.py` in `NVIDIA/cutlass`

GitHub code search results:

- `examples/python/CuTeDSL/blackwell/dense_gemm.py`
  `https://github.com/NVIDIA/cutlass/blob/08185b9c3e90510ee2b656662ed0d53b06d28157/examples/python/CuTeDSL/blackwell/dense_gemm.py`
- `examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py`
  `https://github.com/NVIDIA/cutlass/blob/08185b9c3e90510ee2b656662ed0d53b06d28157/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py`

These are not vector-add kernels, but they are authoritative Blackwell CuTeDSL examples for memory movement, launch structure, and kernel organization.

## Practical synthesis

For this prompt, the local-first and GitHub-follow-up workflow should be:

1. Use local `rg` to identify the stack: `vector addition`, `CuTeDSL`, `Blackwell/B200`.
2. Notice that the local cache has:
   - one direct `vector addition + B200` hit
   - zero direct `vector addition + CuTe` hits
   - a large `Blackwell + CuTe` neighborhood
3. Use that neighborhood to select the right authoritative repo: `NVIDIA/cutlass`.
4. Use GitHub code search to fetch:
   - a direct operation template: `elementwise_add.py`
   - Blackwell execution templates: `fp16_gemm_0.py`, `dense_gemm.py`, `dense_gemm_persistent.py`
5. Combine them into the implementation plan:
   - start from `elementwise_add.py` for the elementwise structure
   - adapt it using the Blackwell CuTeDSL examples for architecture-specific conventions

## Why this example is a good v1 demonstration

This prompt is a useful stress test for cacheDB v1 because it shows both strengths and limits:

- v1 does not need a semantic search engine to be useful.
- The local cache can still route the agent into the correct technical neighborhood.
- The missing direct template is surfaced as a gap, not hidden.
- GitHub search is the right second stage because it is more precise than broad web search for code retrieval.

In short: the local cache did not answer the full prompt directly, but it successfully reduced the search from an open-ended web problem to `CUTLASS/CuTeDSL/Blackwell`, after which GitHub code search recovered the direct implementation seeds.

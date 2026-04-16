# cacheDB: Ligweight web local index from curated bookmarks

## Motivation

Code agents (like Claude Code) default to expensive `web_search` tool calls. To create a local **index engine** for code agents as lightweight caching retrival. Agent then calls this tiny curated web cacheDB before using `web_search` tool calling. A mechanism to extend plain web search with high signal information for a given prompt.

We have been creating a systematic long curated bookmark index using **Chrome** in first place to faciliate rapid retrival of resources about complex AI research and computing. Web bookmarks are exported as `html` single file. The bookmarks are organized in a rich tree hiearchy. The tree structure itself carries semantic meaning to improve indexing and taxonomy. A bookmark filed under `Computational Research > Kernels > FlashAttention-3` is far more discoverable than its title alone.

Another example, we have the node `cuda` and then within cuda node `cuda-debugging` or Cuda related topics like `cutlass` (but without the prefix to not be verbose although potential incosistent).

cacheDB converts this into a local index that agents query *before* web search to save cost and to surface curated, high-quality links that generic search misses.

Inspired by [tinygrad's universal cache DB](https://github.com/tinygrad/tinygrad/blob/c04f3eaa/tinygrad/helpers.py#L366).

## Versioned Roadmap

| Version | Scope | Core Output |
|---------|-------|-------------|
| **v1**  | Parser + grep-optimized flat file | `index.jsonl` - one enriched record per line, searchable via `grep`/`ripgrep` |
| **v2** | SQLite KV store + FTS5 ranked search | `index.db` - BM25-ranked queries, sibling expansion, Python API |
| **v3** | Semantic search + query expansion | Embedding-based retrieval, LLM-assisted query rewriting, multi-source ingestion |

## v1: Parser that emits a grep-optimized flat file

### Target

Single Python script that parses `bookmarks.html` and emits `index.jsonl` - a flat file where each line is a JSON record enriched with tree-path context. Agents use `grep`/`ripgrep` directly on this file. No database, no dependencies beyond stdlib.

### File Structure

```text
agents-experiments/cachedb/
├── data/
│   └── bookmarks.html       # Input (gitignored)
├── cachedb.py               # Single-file v1: parser + enrichment + export
├── index.jsonl              # Generated output (gitignored)
├── README.md                # Project overview and roadmap
└── tests/
    └── test_cachedb.py      # Parser + enrichment tests
```

### Data Model: One JSONL Record

Each line of `index.jsonl` is a compact JSON object - four fields:

```json
{"url": "https://github.com/NVIDIA/cudnn-frontend/blob/...",
"title": "cudnn-frontend/benchmark/sdpa_benchmark_training/README.md ...",
"path": "Computational Research > GPU > NVGPU > Blackwell > GB300 > gb300-benchmarking",
"key": "Computational Research GPU NVGPU Blackwell GB300 gb300-benchmarking gb300 benchmarking cudnn-frontend/benchmark/sdpa_benchmark_training/README.md github.com NVIDIA cudnn-frontend"}
```

- **`url`** - the link the agent acts on
- **`title`** - human-readable label for relevance judgment
- **`path`** - tree information so the agent sees where in the taxonomy a hit lives
- **`key`** - flat superset of `path + title + url hints`, collapsed into one searchable string. Ripgrep matches against this single field. Redundancy is the point: no structured queries needed for v1

**`key` construction** - `key = path segments + hyphen splits + leaf boost + title + github org/repo`:

1. Every path segment added verbatim, plus hyphenated terms split (`gb300-benchmarking` -> also `gb300`, `benchmarking`)
2. Leaf folder (innermost) repeated once for boost
3. Full title appended
4. For GitHub URLs: org and repo name (`NVIDIA`, `cudnn-frontend`) - meaningful search terms not always in the title

This means a bookmark filed under `GB300 > gb300-benchmarking` whose title never mentions "GB300" still matches `rg -i "GB300"` because the key embeds the entire ancestor path.

### Parser Algorithm (`cachedb.py`)

Line-by-line state machine (stdlib only - no HTML parser needed, the Netscape format is rigid):

- Maintain `path_stack: list[str]`
- `<H3 FOLDED ...>name</H3>` -> push name onto stack
- `</DL>` -> pop from stack
- `<DT><A HREF="url">title</A>` -> emit record with current stack as path
- Multi-line titles: if line has `<DT><A` but no `</A>`, accumulate until `</A>` found
- `html.unescape()` on all text (handles 557 HTML entities like `&amp;`)

Edge cases from data exploration:
- One H3 has extra `id` attribute: `<H3 FOLDED id="com.apple.ReadingList">`
- Titles up to ~3,800 chars (full tweet embeds from X/Twitter)
- 14 multi-line title cases
- Generic titles ("Notion", "File not found") - kept as-is, tree path compensates

### Key Construction: `build_key(bookmark)`

```python
def build_key(path: list[str], title: str, url: str) -> str:
    parts = []
    for seg in path:
        parts.append(seg)
        if '-' in seg:
            parts.extend(seg.split('-'))
    # Boost leaf folder
    if path:
        leaf = path[-1]
        parts.append(leaf)
        if '-' in leaf:
            parts.extend(leaf.split('-'))
    # Title
    parts.append(title)
    # URL hints
    domain = extract_domain(url)
    if domain:
        parts.append(domain)
    gh_org, gh_repo = extract_github_info(url)
    if gh_org:
        parts.extend([gh_org, gh_repo])
    return ' '.join(parts)
```

### CLI

```bash
# Build the index
python cachedb.py build

# Agent usage (ripgrep)
rg -i "GB300.*benchmark" index.jsonl
rg -i "FlashAttention" index.jsonl
rg -i "profiling.*kernel|kernel.*profiling" index.jsonl

# Pretty-print matched JSONL records for easier reading
rg -i "GB300.*benchmark|benchmark.*GB300" index.jsonl | jq .
# Or only show selected fields
rg -i "GB300.*benchmark|benchmark.*GB300" index.jsonl | jq '{url, path}'

# Multi-term (agent composes grep pipeline)
rg -i "GB300|FlashAttention|profiling|benchmark|correctness" index.jsonl
rg -i "GB300|FlashAttention|profiling|benchmark|correctness" index.jsonl | jq .

# Stats
python cachedb.py stats
```

### Current Stats

Snapshot from:

```bash
python3 cachedb.py stats cachedb/index.jsonl
```

```text
Bookmarks:  15097
Max depth:  12
Avg depth:  6.7
Top domains:
  github.com: 5116
  x.com: 1555
  youtube.com: 1166
  arxiv.org: 1081
  twitter.com: 290
  huggingface.co: 260
  pytorch.org: 189
  gist.github.com: 184
  docs.nvidia.com: 172
  zhuanlan.zhihu.com: 162
```

### Why `key` Field Beats Grep on Raw HTML

**Problem with raw grep:** `rg "ncu" bookmarks.html` returns lines with "ncu" in the URL or title, but you can't tell that it lives under `cuda-profiling > ncu` unless you manually trace the surrounding `<DL>`/`<H3>` tags. And bookmarks *inside* the `GB300` folder whose titles don't mention "GB300" are invisible to `rg "GB300" bookmarks.html`.

**With `index.jsonl`:** `rg "GB300" index.jsonl` finds every record filed under the GB300 subtree, because the `key` field includes all ancestor folder names. Each result is self-contained - one JSON line with URL, title, full path, and enriched key.

## v1 Implementation Steps

1. **`cachedb.py` - parser**
   - State machine: `parse_bookmarks(html_path) -> list[dict]`
   - Handle: multi-line titles, HTML entities, H3 attribute variants
   - Filter out root sections with no meaningful path (Favourites, Bookmarks Menu)

2. **`cachedb.py` - enrichment**
   - `build_key()` - construct the enriched searchable text per bookmark
   - `extract_domain()`, `extract_github_info()` - URL parsing helpers

3. **`cachedb.py` - export**
   - Write `index.jsonl` - one JSON record per line, sorted by path depth (deepest first)
   - Print summary stats on completion

4. **`tests/test_cachedb.py`**
   - Test parser against known line numbers from data exploration
   - Test key construction for representative bookmarks
   - Test that the validation query terms hit expected records

5. **Validation**
   - Run `rg -i "GB300|FlashAttention|profiling|benchmark|correctness" index.jsonl`
   - Verify results include entries from all expected sections (see table below)

### Validation: Target Query Coverage

Query: *"implementing a kernel for GB300, FlashAttention style, with proper profiling, benchmarking and numerical correctness tests"*

| grep term | Expected sections that surface via `key` field |
|---|---|
| `GB300` | `GB300 > gb300-benchmarking`, `GB300 > gb300-numa-affinity`, GB300 softmax SFU entries |
| `FlashAttention` | `FlashAttention-2`, `FlashAttention-3`, `FlashAttention-4`, `FlashAttention-triton` |
| `profiling` | `cuda-profiling > ncu/nsight`, `triton-profiling > proton`, `torch-profiling` |
| `benchmark` | `kernels-benchmarking > Tritonbench`, `cuda-benchmarking`, `gb300-benchmarking` |
| `correctness` | `FACTO` framework, `Liger-Kernel` correctness test, `kernels-tests` |

## v2 Roadmap

- **SQLite + FTS5 backend** (`index.db`) - BM25-ranked multi-term relevance instead of file-order grep results; porter stemming so "benchmark" matches "benchmarking"
- **Python API** - `BookmarkStore` with dict-like KV interface
- **Sibling expansion** - a hit inside `cuda-profiling > ncu` surfaces the entire `ncu` folder
- **Query preprocessing** - stop-word removal, camelCase split, prefix wildcards; one query spans 6+ folders instead of 6+ separate grep commands
- **CLI** - `python -m cachedb search "query"` with ranked output

## v3 Roadmap

- Embedding-based semantic search (sentence-transformers or similar)
- LLM-assisted query expansion (decompose complex queries into sub-queries)
- Multi-source ingestion (not just Safari - browser extensions, Pocket, Raindrop, etc.)
- Agent integration: MCP tool server or Claude Code hook for automatic local-first search

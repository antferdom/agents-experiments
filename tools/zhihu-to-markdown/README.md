# Article Exporter

`tools/zhihu-to-markdown/export_article.py` exports blog/article HTML into
Markdown that is easier for models to read. It is dependency-free and uses only
the Python standard library.

The exporter is useful for pages with math-heavy content, especially articles
that include inline LaTeX and display equations.

## Basic Usage

Export a live URL:

```bash
./tools/zhihu-to-markdown/export_article.py https://kexue.fm/archives/10091 --verify-latex
```

By default, the output is written to:

```text
tools/exports/articles/<host-path>.md
```

For the example above, the output path is:

```text
tools/exports/articles/kexue.fm-archives-10091.md
```

Export from saved HTML instead of fetching the page:

```bash
./tools/zhihu-to-markdown/export_article.py https://kexue.fm/archives/10091 \
  --input-html /path/to/article.html \
  --verify-latex
```

Write to a specific output file:

```bash
./tools/zhihu-to-markdown/export_article.py https://kexue.fm/archives/10091 \
  --output tools/exports/articles/mla.md \
  --verify-latex
```

Use a specific article-body selector when automatic extraction is not enough:

```bash
./tools/zhihu-to-markdown/export_article.py https://example.com/post \
  --selector '#PostContent' \
  --verify-latex
```

## LaTeX Handling

The exporter preserves inline math expressions such as:

```text
$\boldsymbol{x}_i\in\mathbb{R}^d$
```

For display math, source wrappers such as:

```latex
\begin{equation}
\begin{gathered}
...
\end{gathered}
\end{equation}
```

are converted to Markdown-renderable dollar blocks:

```markdown
$$
...
$$
```

This is intentional. The source environment wrappers are useful in HTML/MathJax
contexts, but most Markdown renderers handle display math more reliably when it
is fenced with `$$`.

Labels and references are preserved inside the math payload when present, for
example:

```latex
\label{eq:mla-mha}
```

and inline references such as:

```text
$\eqref{eq:mla-mha}$
```

## Verification

Use `--verify-latex` whenever the article contains math:

```bash
./tools/zhihu-to-markdown/export_article.py https://kexue.fm/archives/10091 --verify-latex
```

The verifier extracts LaTeX fragments from the source article body and from the
exported Markdown, then compares their normalized math payloads. This allows the
exporter to convert display wrappers like `\begin{equation}` to `$$` while still
checking that the mathematical content itself is preserved.

A successful run prints:

```text
LaTeX integrity: OK (<n> fragments)
```

For `https://kexue.fm/archives/10091`, the verified result was:

```text
LaTeX integrity: OK (101 fragments)
```

## Supported Page Patterns

The exporter tries these article selectors by default:

```text
#PostContent
.PostContent
article
main
.RichText.ztext.Post-RichText
.RichText.ztext
.post-content
.entry-content
```

This covers the tested `kexue.fm` article layout and common Zhihu article body
markup. The original Zhihu browser userscript that motivated the local utility
is available at:

```text
https://github.com/RustyPiano/zhihu-to-markdown.git
```

## Notes

- The utility may need network access when fetching a live URL.
- Use `--input-html` when a site blocks direct command-line fetching or when you
  want a fully reproducible export from a saved page snapshot.
- The exporter intentionally skips common non-article sections such as comments,
  sharing widgets, citation boxes, and donation blocks.
- It is designed for model consumption, so preserving text, links, headings, and
  LaTeX integrity is prioritized over pixel-perfect page reproduction.

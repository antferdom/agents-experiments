#!/usr/bin/env python3
"""Export a blog/article page to Markdown.

This is intentionally dependency-free: it uses the Python standard library so
agents can run it in a fresh checkout without installing parser packages.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import http.cookiejar
import html.parser
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "exports" / "articles"

DEFAULT_CONTENT_SELECTORS = (
    "#PostContent",
    ".PostContent",
    "article",
    "main",
    ".RichText.ztext.Post-RichText",
    ".RichText.ztext",
    ".post-content",
    ".entry-content",
)

DEFAULT_TITLE_SELECTORS = (
    ".PostHead h1",
    "h1.Post-Title",
    ".QuestionHeader-title",
    "article h1",
    "h1",
    "title",
)

SKIP_TAGS = {"script", "style", "noscript", "iframe", "form", "button", "input"}
MARKDOWN_DISPLAY_ENVIRONMENTS = {
    "equation",
    "equation*",
    "gather",
    "gather*",
    "gathered",
}
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
SKIP_IDS = {
    "content_tips",
    "pay",
    "QR",
    "how_to_cite",
    "share",
    "tools",
    "comments",
    "PostComment",
}
SKIP_CLASSES = {"tools", "respond", "comment_content", "tableofcontents"}


@dataclasses.dataclass
class Node:
    tag: str | None = None
    attrs: dict[str, str] = dataclasses.field(default_factory=dict)
    children: list["Node"] = dataclasses.field(default_factory=list)
    text: str = ""

    @property
    def is_text(self) -> bool:
        return self.tag is None

    def attr(self, key: str, default: str = "") -> str:
        return self.attrs.get(key, default)

    def classes(self) -> set[str]:
        return set(self.attr("class").split())


class TreeBuilder(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag.lower(), {k.lower(): v or "" for k, v in attrs})
        self.stack[-1].children.append(node)

        if tag.lower() not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag.lower(), {k.lower(): v or "" for k, v in attrs})
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data: str) -> None:
        if data:
            self.stack[-1].children.append(Node(text=data))


def parse_html(raw_html: str) -> Node:
    parser = TreeBuilder()
    parser.feed(raw_html)
    parser.close()
    return parser.root


def walk(node: Node) -> list[Node]:
    nodes = [node]
    for child in node.children:
        nodes.extend(walk(child))
    return nodes


def simple_selector_parts(selector: str) -> tuple[str | None, str | None, set[str]]:
    tag = None
    ident = None
    classes: set[str] = set()
    pos = 0

    tag_match = re.match(r"^[a-zA-Z][\w-]*", selector)
    if tag_match:
        tag = tag_match.group(0).lower()
        pos = len(tag)

    while pos < len(selector):
        marker = selector[pos]
        pos += 1
        end = pos
        while end < len(selector) and selector[end] not in ".#":
            end += 1
        value = selector[pos:end]
        if marker == "#":
            ident = value
        elif marker == ".":
            classes.add(value)
        pos = end

    return tag, ident, classes


def node_matches(node: Node, selector: str) -> bool:
    if node.is_text:
        return False

    tag, ident, classes = simple_selector_parts(selector)
    if tag and node.tag != tag:
        return False
    if ident and node.attr("id") != ident:
        return False
    if classes and not classes.issubset(node.classes()):
        return False
    return True


def iter_descendants(node: Node) -> list[Node]:
    result: list[Node] = []
    for child in node.children:
        result.append(child)
        result.extend(iter_descendants(child))
    return result


def find_first(root: Node, selector: str) -> Node | None:
    parts = selector.split()
    current = [root]

    for part in parts:
        matches: list[Node] = []
        for node in current:
            matches.extend(d for d in iter_descendants(node) if node_matches(d, part))
        current = matches
        if not current:
            return None

    return current[0] if current else None


def find_first_of(root: Node, selectors: tuple[str, ...]) -> Node | None:
    for selector in selectors:
        found = find_first(root, selector)
        if found:
            return found
    return None


def plain_text(node: Node) -> str:
    if node.is_text:
        return node.text
    if should_skip(node):
        return ""
    pieces = [plain_text(child) for child in node.children]
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def should_skip(node: Node) -> bool:
    if node.is_text:
        return False
    if node.tag in SKIP_TAGS:
        return True
    if node.attr("id") in SKIP_IDS:
        return True
    if node.classes() & SKIP_CLASSES:
        return True
    text = plain_text_without_skip(node)
    if node.tag == "p" and (
        text.startswith("转载到请包括本文地址")
        or text.startswith("更详细的转载事宜请参考")
    ):
        return True
    return False


def plain_text_without_skip(node: Node) -> str:
    if node.is_text:
        return node.text
    return re.sub(r"\s+", " ", "".join(plain_text_without_skip(c) for c in node.children)).strip()


def escape_markdown_text(text: str) -> str:
    return text.replace("\xa0", " ")


def render_children(node: Node, base_url: str) -> str:
    return "".join(render_node(child, base_url) for child in node.children)


def render_text_with_breaks(node: Node) -> str:
    if node.is_text:
        return node.text
    if should_skip(node):
        return ""
    if node.tag == "br":
        return "\n"
    return "".join(render_text_with_breaks(child) for child in node.children)


def render_node(node: Node, base_url: str) -> str:
    if node.is_text:
        return escape_markdown_text(node.text)
    if should_skip(node):
        return ""

    tag = node.tag or ""

    if tag == "span" and "ztext-math" in node.classes():
        tex = node.attr("data-tex").replace("\\bm", "\\boldsymbol").strip()
        if not tex:
            return ""
        if tex.endswith(r"\\"):
            return markdown_display_math(tex.removesuffix(r"\\").strip())
        return f"${tex}$"
    if tag == "br":
        return "\n"
    if tag in {"p", "div", "section"}:
        content = render_children(node, base_url).strip()
        return f"\n\n{content}\n\n" if content else ""
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        content = render_children(node, base_url).strip()
        if not content:
            return ""
        level = int(tag[1])
        return f"\n\n{'#' * level} {content}\n\n"
    if tag == "a":
        text = render_children(node, base_url).strip()
        href = node.attr("href")
        if text == "#" and href.startswith("#"):
            return ""
        if not text:
            return ""
        if not href or href.startswith("javascript:"):
            return text
        return f"[{text}]({urllib.parse.urljoin(base_url, href)})"
    if tag in {"strong", "b"}:
        content = render_children(node, base_url).strip()
        return f"**{content}**" if content else ""
    if tag in {"em", "i"}:
        content = render_children(node, base_url).strip()
        return f"*{content}*" if content else ""
    if tag == "code":
        content = plain_text_without_skip(node)
        return f"`{content}`"
    if tag == "pre":
        content = plain_text_without_skip(node).strip("\n")
        return f"\n\n```\n{content}\n```\n\n"
    if tag == "blockquote":
        content = normalize_markdown(render_children(node, base_url))
        quoted = "\n".join(f"> {line}" if line else ">" for line in content.splitlines())
        return f"\n\n{quoted}\n\n"
    if tag == "ul":
        return render_list(node, base_url, ordered=False)
    if tag == "ol":
        return render_list(node, base_url, ordered=True)
    if tag == "li":
        return render_children(node, base_url).strip()
    if tag == "img":
        src = node.attr("src") or node.attr("data-src") or node.attr("data-original")
        alt = node.attr("alt")
        return f"![{alt}]({urllib.parse.urljoin(base_url, src)})" if src else ""
    if tag == "hr":
        return "\n\n---\n\n"
    if tag == "table":
        return render_table(node, base_url)

    return render_children(node, base_url)


def direct_children(node: Node, tag: str) -> list[Node]:
    return [child for child in node.children if not child.is_text and child.tag == tag]


def render_list(node: Node, base_url: str, ordered: bool) -> str:
    items = []
    for index, child in enumerate(direct_children(node, "li"), start=1):
        content = normalize_markdown(render_children(child, base_url))
        if not content:
            continue
        prefix = f"{index}. " if ordered else "- "
        lines = content.splitlines()
        first = f"{prefix}{lines[0]}"
        rest = [f"  {line}" if line else "" for line in lines[1:]]
        items.append("\n".join([first, *rest]))
    return f"\n\n{os.linesep.join(items)}\n\n" if items else ""


def render_table(node: Node, base_url: str) -> str:
    rows: list[list[str]] = []
    for tr in [n for n in walk(node) if not n.is_text and n.tag == "tr"]:
        cells = []
        for cell in [c for c in tr.children if not c.is_text and c.tag in {"th", "td"}]:
            value = normalize_inline(render_children(cell, base_url))
            cells.append(value.replace("|", r"\|"))
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    body = padded[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n\n" + "\n".join(lines) + "\n\n"


def normalize_inline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = compact_tex_blocks(text)
    parts = []
    for is_tex, segment in split_latex_segments(text):
        if is_tex:
            parts.append(segment)
            continue
        segment = re.sub(r"[ \t]+\n", "\n", segment)
        segment = re.sub(r"\n[ \t]+", "\n", segment)
        segment = re.sub(r"[ \t]{2,}", " ", segment)
        segment = re.sub(r"\n{3,}", "\n\n", segment)
        parts.append(segment)
    return "".join(parts).strip()


def compact_tex_blocks(text: str) -> str:
    lines = text.splitlines()
    compacted: list[str] = []
    math_depth = 0

    for line in lines:
        stripped = line.strip()
        begins = len(re.findall(r"\\begin\{[^}]+\}", stripped))
        ends = len(re.findall(r"\\end\{[^}]+\}", stripped))
        inside_math = math_depth > 0

        if stripped or not inside_math:
            compacted.append(line)

        math_depth = max(0, math_depth + begins - ends)

    return "\n".join(compacted)


def convert_display_math_to_dollars(text: str) -> str:
    parts = []
    cursor = 0
    for fragment in extract_latex_fragments(text, include_spans=True):
        if fragment.kind not in {"display", "environment"}:
            continue

        env_name = latex_environment_name(fragment.text)
        if fragment.kind == "environment" and env_name not in MARKDOWN_DISPLAY_ENVIRONMENTS:
            continue

        if fragment.start > cursor:
            parts.append(text[cursor : fragment.start])
        parts.append(markdown_display_math(fragment.text))
        cursor = fragment.end

    if cursor == 0:
        return text
    parts.append(text[cursor:])
    return "".join(parts)


def latex_environment_name(text: str) -> str | None:
    match = re.match(r"\s*\\begin\{([^}]+)\}", text)
    return match.group(1) if match else None


def markdown_display_math(text: str) -> str:
    payload = latex_payload(text)
    return f"\n\n$$\n{payload}\n$$\n\n"


def latex_payload(text: str) -> str:
    text = text.strip()

    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2]
    elif text.startswith("\\[") and text.endswith("\\]"):
        text = text[2:-2]

    for env in sorted(MARKDOWN_DISPLAY_ENVIRONMENTS, key=len, reverse=True):
        escaped = re.escape(env)
        text = re.sub(rf"\s*\\begin\{{{escaped}\}}\s*", "\n", text)
        text = re.sub(rf"\s*\\end\{{{escaped}\}}\s*", "\n", text)

    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def split_latex_segments(text: str) -> list[tuple[bool, str]]:
    fragments = extract_latex_fragments(text, include_spans=True)
    if not fragments:
        return [(False, text)]

    segments: list[tuple[bool, str]] = []
    cursor = 0
    for fragment in fragments:
        if fragment.start > cursor:
            segments.append((False, text[cursor : fragment.start]))
        segments.append((True, text[fragment.start : fragment.end]))
        cursor = fragment.end
    if cursor < len(text):
        segments.append((False, text[cursor:]))
    return segments


@dataclasses.dataclass(frozen=True)
class LatexFragment:
    kind: str
    start: int
    end: int
    text: str


def extract_latex_fragments(text: str, include_spans: bool = False) -> list[LatexFragment]:
    fragments: list[LatexFragment] = []
    index = 0
    length = len(text)

    while index < length:
        if text.startswith("\\begin{", index):
            env_match = re.match(r"\\begin\{([^}]+)\}", text[index:])
            if env_match:
                env = env_match.group(1)
                close = f"\\end{{{env}}}"
                end = text.find(close, index + env_match.end())
                if end != -1:
                    end += len(close)
                    fragments.append(LatexFragment("environment", index, end, text[index:end]))
                    index = end
                    continue

        if text.startswith("\\[", index):
            end = text.find("\\]", index + 2)
            if end != -1:
                end += 2
                fragments.append(LatexFragment("display", index, end, text[index:end]))
                index = end
                continue

        if text.startswith("\\(", index):
            end = text.find("\\)", index + 2)
            if end != -1:
                end += 2
                fragments.append(LatexFragment("inline", index, end, text[index:end]))
                index = end
                continue

        if text[index] == "$" and not is_escaped(text, index):
            if text.startswith("$$", index):
                end = find_unescaped(text, "$$", index + 2)
                if end != -1:
                    end += 2
                    fragments.append(LatexFragment("display", index, end, text[index:end]))
                    index = end
                    continue
            else:
                end = find_single_dollar_end(text, index + 1)
                if end != -1:
                    end += 1
                    fragments.append(LatexFragment("inline", index, end, text[index:end]))
                    index = end
                    continue

        index += 1

    if include_spans:
        return fragments
    return [dataclasses.replace(fragment, start=0, end=len(fragment.text)) for fragment in fragments]


def is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def find_unescaped(text: str, needle: str, start: int) -> int:
    cursor = start
    while True:
        found = text.find(needle, cursor)
        if found == -1:
            return -1
        if not is_escaped(text, found):
            return found
        cursor = found + len(needle)


def find_single_dollar_end(text: str, start: int) -> int:
    cursor = start
    while True:
        found = text.find("$", cursor)
        if found == -1:
            return -1
        if not is_escaped(text, found) and not text.startswith("$$", found):
            return found
        cursor = found + 1


def canonical_latex(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = latex_payload(text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def verify_latex_integrity(html_text: str, markdown: str, url: str, selector: str | None) -> tuple[bool, str]:
    root = parse_html(html_text)
    content = find_first(root, selector) if selector else find_first_of(root, DEFAULT_CONTENT_SELECTORS)
    if content is None:
        return False, "Could not find source content for LaTeX verification."

    source_text = render_text_with_breaks(content)
    source_fragments = extract_latex_fragments(source_text)
    markdown_fragments = extract_latex_fragments(markdown)
    source_canon = [canonical_latex(fragment.text) for fragment in source_fragments]
    markdown_canon = [canonical_latex(fragment.text) for fragment in markdown_fragments]

    if source_canon == markdown_canon:
        return True, f"LaTeX integrity: OK ({len(source_canon)} fragments)"

    if len(source_canon) != len(markdown_canon):
        return (
            False,
            f"LaTeX integrity: FAILED (source has {len(source_canon)} fragments; markdown has {len(markdown_canon)})",
        )

    for index, (source, exported) in enumerate(zip(source_canon, markdown_canon), start=1):
        if source != exported:
            return (
                False,
                "LaTeX integrity: FAILED "
                f"(fragment {index} differs)\nSOURCE:\n{source}\n\nEXPORTED:\n{exported}",
            )

    return False, "LaTeX integrity: FAILED"


def extract_title(root: Node, html_text: str) -> str:
    title_node = find_first_of(root, DEFAULT_TITLE_SELECTORS)
    if title_node:
        title = plain_text(title_node)
    else:
        match = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.I | re.S)
        title = re.sub(r"\s+", " ", match.group(1)).strip() if match else "Untitled"

    title = re.sub(r"\s+-\s+科学空间\|Scientific Spaces$", "", title)
    title = re.sub(r"\s*-\s*知乎$", "", title)
    return title or "Untitled"


def extract_metadata(root: Node) -> dict[str, str]:
    metadata: dict[str, str] = {}

    submitted = find_first(root, ".submitted")
    if submitted:
        text = plain_text(submitted)
        author_match = re.search(r"By\s+(.+?)\s*\|", text)
        date_match = re.search(r"\|\s*(\d{4}-\d{2}-\d{2})\s*\|", text)
        if author_match:
            metadata["author"] = author_match.group(1).strip()
        if date_match:
            metadata["published"] = date_match.group(1)

    return metadata


def default_output_path(url: str, out_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.replace("www.", "").replace(":", "-")
    path = parsed.path.strip("/") or "index"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{host}-{path}").strip("-")
    return out_dir / f"{slug}.md"


def frontmatter_value(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def article_to_markdown(html_text: str, url: str, selector: str | None = None) -> str:
    root = parse_html(html_text)
    content = find_first(root, selector) if selector else find_first_of(root, DEFAULT_CONTENT_SELECTORS)
    if content is None:
        raise ValueError(
            "Could not find article content. Pass --selector with a CSS-like selector such as '#PostContent'."
        )

    title = extract_title(root, html_text)
    metadata = extract_metadata(root)
    body = convert_display_math_to_dollars(normalize_markdown(render_children(content, url)))

    fields = {
        "title": title,
        "source": url,
        **metadata,
        "exported_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
    }
    header = "\n".join(f"{key}: {frontmatter_value(value)}" for key, value in fields.items())

    return normalize_markdown(f"---\n{header}\n---\n\n# {title}\n\n{body}") + "\n"


def is_redirect_stub(html_text: str) -> bool:
    return "window.location.href" in html_text and len(html_text) < 1000


def redirect_target(html_text: str, base_url: str) -> str | None:
    match = re.search(r"window\.location\.href\s*=\s*['\"]([^'\"]+)['\"]", html_text)
    if not match:
        return None
    return urllib.parse.urljoin(base_url, match.group(1))


def fetch_url(url: str) -> str:
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    current = url
    last_html = ""

    for _ in range(4):
        request = urllib.request.Request(
            current,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            },
        )
        try:
            with opener.open(request, timeout=30) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()

        html_text = raw.decode("utf-8", errors="replace")
        last_html = html_text
        target = redirect_target(html_text, current) if is_redirect_stub(html_text) else None
        if not target:
            return html_text
        current = target

    return last_html


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Source article URL. Used for fetching and resolving relative links.")
    parser.add_argument(
        "--input-html",
        type=Path,
        help="Read HTML from a saved file instead of fetching the URL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Markdown output path. Defaults to tools/exports/articles/<host-path>.md.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for default output paths.",
    )
    parser.add_argument(
        "--selector",
        help="CSS-like selector for the article body, for example '#PostContent'.",
    )
    parser.add_argument(
        "--verify-latex",
        action="store_true",
        help="Fail if LaTeX fragments in the output differ from the source article.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    html_text = args.input_html.read_text(encoding="utf-8") if args.input_html else fetch_url(args.url)
    markdown = article_to_markdown(html_text, args.url, args.selector)

    output = args.output or default_output_path(args.url, args.out_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(display_path(output))
    if args.verify_latex:
        ok, message = verify_latex_integrity(html_text, markdown, args.url, args.selector)
        stream = sys.stdout if ok else sys.stderr
        print(message, file=stream)
        if not ok:
            return 1
    return 0


def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return path


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

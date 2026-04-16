# cacheDB v1: Parse Chrome bookmarks HTML into a grep-optimized JSONL index
import argparse
import html
import json
import re
from urllib.parse import urlparse
from pathlib import Path

WD = Path(__file__).parent.absolute()
DEFAULT_HTML = WD / "data" / "bookmarks.html"
DEFAULT_INDEX = WD / "index.jsonl"

# Sections to skip — no meaningful tree context
SKIP_SECTIONS = {"Favourites", "Bookmarks Menu", "Tab Group Favourites", "Reading List"}

# Date-like folder names (dd-mm-yyyy) used as root containers — strip from path
RE_DATE_FOLDER = re.compile(r"^\d{2}-\d{2}-\d{4}$")

# Regex patterns for the Netscape bookmark format
RE_H3 = re.compile(r"<H3 FOLDED[^>]*>(.+?)</H3>", re.IGNORECASE)
RE_LINK_FULL = re.compile(r'<DT><A HREF="([^"]*)">(.*?)</A>', re.IGNORECASE | re.DOTALL)
RE_LINK_OPEN = re.compile(r'<DT><A HREF="([^"]*)">(.*)', re.IGNORECASE)
RE_LINK_CLOSE = re.compile(r"(.*?)</A>", re.IGNORECASE)
RE_DL_CLOSE = re.compile(r"</DL>", re.IGNORECASE)


def extract_domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    if host.startswith("www."):
        host = host[4:]
    return host


def extract_github_info(url: str) -> tuple[str, str]:
    """Return (org, repo) for GitHub URLs."""
    parsed = urlparse(url)
    if parsed.hostname not in ("github.com", "www.github.com"):
        return "", ""
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def build_key(path: list[str], title: str, url: str) -> str:
    parts = []
    for seg in path:
        parts.append(seg)
        if "-" in seg:
            parts.extend(seg.split("-"))
    # Boost leaf folder
    if path:
        leaf = path[-1]
        parts.append(leaf)
        if "-" in leaf:
            parts.extend(leaf.split("-"))
    # Title
    parts.append(title)
    # GitHub org/repo as search terms (e.g. "NVIDIA", "cutlass")
    gh_org, gh_repo = extract_github_info(url)
    if gh_org:
        parts.extend([gh_org, gh_repo])
    return " ".join(parts)


def _clean_path(path_stack: list[str]) -> list[str]:
    """Filter out date-like root folders from path."""
    return [seg for seg in path_stack if not RE_DATE_FOLDER.match(seg)]


def parse_bookmarks(html_path: str) -> list[dict]:
    """Parse Netscape bookmarks HTML into a list of bookmark dicts."""
    with open(html_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    bookmarks = []
    path_stack: list[str] = []
    skip_depth = 0  # > 0 means we're inside a skipped section
    accumulating = False
    accum_url = ""
    accum_title_parts: list[str] = []

    for line in lines:
        # Check for folder open
        m = RE_H3.search(line)
        if m:
            name = html.unescape(m.group(1).strip())
            if skip_depth > 0:
                skip_depth += 1
            elif name in SKIP_SECTIONS:
                skip_depth = 1
            else:
                path_stack.append(name)
            continue

        # Check for DL close (folder end)
        if RE_DL_CLOSE.search(line):
            if skip_depth > 0:
                skip_depth -= 1
            elif path_stack:
                path_stack.pop()
            continue

        # Skip everything inside skipped sections
        if skip_depth > 0:
            continue

        # Handle multi-line title accumulation
        if accumulating:
            m_close = RE_LINK_CLOSE.search(line)
            if m_close:
                accum_title_parts.append(m_close.group(1))
                title = html.unescape(" ".join(accum_title_parts))
                path = _clean_path(path_stack)
                bookmarks.append({
                    "url": accum_url,
                    "title": title,
                    "path": " > ".join(path),
                    "key": build_key(path, title, accum_url),
                })
                accumulating = False
            else:
                accum_title_parts.append(line.strip())
            continue

        # Check for complete link on one line
        m = RE_LINK_FULL.search(line)
        if m:
            url = m.group(1)
            title = html.unescape(m.group(2).strip())
            path = _clean_path(path_stack)
            bookmarks.append({
                "url": url,
                "title": title,
                "path": " > ".join(path),
                "key": build_key(path, title, url),
            })
            continue

        # Check for link opening without close (multi-line title)
        m = RE_LINK_OPEN.search(line)
        if m:
            accum_url = m.group(1)
            accum_title_parts = [m.group(2).strip()]
            accumulating = True
            continue

    return bookmarks


def build_index(html_path: str, output_path: str = "index.jsonl") -> list[dict]:
    """Parse bookmarks and write JSONL index, sorted deepest-first."""
    bookmarks = parse_bookmarks(html_path)
    bookmarks.sort(key=lambda b: (-b["path"].count(">"), b["path"]))

    with open(output_path, "w", encoding="utf-8") as f:
        for bm in bookmarks:
            f.write(json.dumps(bm, ensure_ascii=False) + "\n")

    return bookmarks


def print_stats(bookmarks: list[dict]) -> None:
    """Print summary stats."""
    if not bookmarks:
        print("No bookmarks found.")
        return

    depths = [b["path"].count(">") + 1 for b in bookmarks]
    domains: dict[str, int] = {}
    for b in bookmarks:
        d = extract_domain(b["url"])
        if d:
            domains[d] = domains.get(d, 0) + 1

    top_domains = sorted(domains.items(), key=lambda x: -x[1])[:10]

    print(f"Bookmarks:  {len(bookmarks)}")
    print(f"Max depth:  {max(depths)}")
    print(f"Avg depth:  {sum(depths) / len(depths):.1f}")
    print(f"Top domains:")
    for d, c in top_domains:
        print(f"  {d}: {c}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build grep-optimized JSONL index"
    )
    parser.add_argument(
        "command",
        choices=("build", "stats"),
        help="Command to run.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "For `build`: [html_path] [output_path]. "
            f"For `stats`: [index_path]. Defaults are {DEFAULT_HTML} and {DEFAULT_INDEX}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)

    if args.command == "build":
        if len(args.paths) > 2:
            make_parser().error(
                "build accepts at most 2 positional paths: [html_path] [output_path]"
            )
        html_path = args.paths[0] if len(args.paths) >= 1 else DEFAULT_HTML
        output_path = args.paths[1] if len(args.paths) >= 2 else DEFAULT_INDEX
        bookmarks = build_index(html_path, output_path)
        print(f"Wrote {len(bookmarks)} records to {output_path}")
        print_stats(bookmarks)
        return 0

    if len(args.paths) > 1:
        make_parser().error("stats accepts at most 1 positional path: [index_path]")
    index_path = args.paths[0] if args.paths else DEFAULT_INDEX

    with open(index_path, "r", encoding="utf-8") as f:
        bookmarks = [json.loads(line) for line in f if line.strip()]
    print_stats(bookmarks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

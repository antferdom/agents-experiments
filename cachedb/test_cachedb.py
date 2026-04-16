import json
import os
import tempfile

import pytest

from cachedb import (
    build_key,
    extract_domain,
    extract_github_info,
    parse_bookmarks,
    build_index,
    _clean_path,
)

BOOKMARKS_HTML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "bookmarks.html"
)
HAS_BOOKMARKS = os.path.exists(BOOKMARKS_HTML)


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

def test_extract_domain():
    assert extract_domain("https://github.com/NVIDIA/cutlass") == "github.com"
    assert extract_domain("https://www.youtube.com/watch?v=abc") == "youtube.com"
    assert extract_domain("https://arxiv.org/pdf/2301.00001") == "arxiv.org"
    assert extract_domain("") == ""


def test_extract_github_info():
    assert extract_github_info("https://github.com/NVIDIA/cutlass/blob/main/README.md") == ("NVIDIA", "cutlass")
    assert extract_github_info("https://github.com/pytorch/pytorch") == ("pytorch", "pytorch")
    assert extract_github_info("https://arxiv.org/pdf/2301.00001") == ("", "")
    assert extract_github_info("https://github.com/") == ("", "")


def test_clean_path():
    assert _clean_path(["13-04-2026", "Computational Research", "GPU"]) == ["Computational Research", "GPU"]
    assert _clean_path(["Computational Research", "GPU"]) == ["Computational Research", "GPU"]
    assert _clean_path(["01-01-2025"]) == []
    assert _clean_path([]) == []


# ---------------------------------------------------------------------------
# Unit tests: build_key
# ---------------------------------------------------------------------------

def test_build_key_basic():
    key = build_key(
        ["Computational Research", "GPU", "NVGPU"],
        "CUDA Toolkit",
        "https://developer.nvidia.com/cuda-toolkit",
    )
    assert "Computational Research" in key
    assert "GPU" in key
    assert "NVGPU" in key  # leaf boost
    assert "CUDA Toolkit" in key
    assert "developer.nvidia.com" not in key  # domain not added to key


def test_build_key_hyphen_split():
    key = build_key(
        ["gb300-benchmarking"],
        "Some Title",
        "https://example.com",
    )
    assert "gb300-benchmarking" in key
    assert "gb300" in key
    assert "benchmarking" in key


def test_build_key_github():
    key = build_key(
        ["Kernels"],
        "cutlass README",
        "https://github.com/NVIDIA/cutlass/blob/main/README.md",
    )
    assert "NVIDIA" in key
    assert "cutlass" in key
    assert "github.com" not in key  # domain not added to key


# ---------------------------------------------------------------------------
# Parser tests: synthetic HTML
# ---------------------------------------------------------------------------

MINIMAL_HTML = """\
<!DOCTYPE NETSCAPE-Bookmark-file-1>
<HTML>
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<Title>Bookmarks</Title>
<H1>Bookmarks</H1>
<DT><H3 FOLDED>Favourites</H3>
<DL><p>
    <DT><A HREF="https://www.google.com">Google</A>
</DL><p>
<DT><H3 FOLDED>13-04-2026</H3>
<DL><p>
    <DT><H3 FOLDED>Research</H3>
    <DL><p>
        <DT><H3 FOLDED>cuda-profiling</H3>
        <DL><p>
            <DT><A HREF="https://github.com/NVIDIA/ncu">NCU Tool</A>
            <DT><A HREF="https://x.com/someone/status/123">Multi-line
title here</A>
        </DL><p>
        <DT><A HREF="https://arxiv.org/pdf/1234">Some &amp; Paper</A>
    </DL><p>
</DL><p>
"""


@pytest.fixture(scope="module")
def minimal_bookmarks(tmp_path_factory):
    path = tmp_path_factory.mktemp("cachedb") / "minimal.html"
    path.write_text(MINIMAL_HTML)
    return parse_bookmarks(str(path))


def test_parse_skips_favourites(minimal_bookmarks):
    urls = [b["url"] for b in minimal_bookmarks]
    assert "https://www.google.com" not in urls


def test_parse_strips_date_from_path(minimal_bookmarks):
    ncu = [b for b in minimal_bookmarks if "NCU" in b["title"]][0]
    assert "13-04-2026" not in ncu["path"]
    assert ncu["path"] == "Research > cuda-profiling"


def test_parse_multi_line_title(minimal_bookmarks):
    multi = [b for b in minimal_bookmarks if "Multi-line" in b["title"]]
    assert len(multi) == 1
    assert "title here" in multi[0]["title"]


def test_parse_html_entities(minimal_bookmarks):
    paper = [b for b in minimal_bookmarks if "Paper" in b["title"]][0]
    assert "&" in paper["title"]
    assert "&amp;" not in paper["title"]


def test_build_index_writes_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(MINIMAL_HTML)
        f.flush()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as out:
            build_index(f.name, out.name)
            out_path = out.name
    os.unlink(f.name)
    with open(out_path) as fj:
        lines = fj.readlines()
    os.unlink(out_path)
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        assert {"url", "title", "path", "key"} <= record.keys()


# ---------------------------------------------------------------------------
# Integration: real bookmarks.html
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_BOOKMARKS, reason="bookmarks.html not present")
class TestRealBookmarks:
    @pytest.fixture(scope="class")
    def bookmarks(self):
        return parse_bookmarks(BOOKMARKS_HTML)

    def test_no_date_in_path(self, bookmarks):
        for b in bookmarks:
            assert not b["path"].startswith("13-04-2026")
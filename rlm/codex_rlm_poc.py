#!/usr/bin/env python3
"""Tiny RLM-style proof of concept using Codex as the recursive LM call.

The Python process owns symbolic control flow: target selection, looping,
timeouts, result capture, and JSONL storage. Each child Codex process is only a
read-only semantic leaf call over one file.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("poc-results.jsonl")

MAX_BYTES = 60_000
TIMEOUT_SECONDS = 240


def predicate(path: Path) -> bool:
    """Deterministic file predicate controlled by the symbolic runtime."""
    return path.suffix == ".md" and path.stat().st_size <= MAX_BYTES


def candidate_files() -> list[Path]:
    seeds = [
        ROOT / "README.md",
        ROOT / "docs" / "codex.md",
        ROOT / "rlm" / "README.md",
    ]
    return [path for path in seeds if path.is_file() and predicate(path)]


def parse_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def codex_call(path: Path) -> dict[str, Any]:
    rel = path.relative_to(ROOT)
    prompt = (
        f"Read {rel}. Output only compact JSON with keys path, purpose, "
        "and rlm_relevance. Do not modify any files."
    )

    with tempfile.NamedTemporaryFile(prefix="codex-rlm-", suffix=".txt") as final:
        completed = subprocess.run(
            [
                "codex",
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--cd",
                str(ROOT),
                "-c",
                'model_reasoning_effort="low"',
                "-o",
                final.name,
                prompt,
            ],
            cwd=ROOT,
            text=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        final.seek(0)
        final_text = final.read().decode("utf-8", errors="replace").strip()

    result = {
        "path": str(rel),
        "returncode": completed.returncode,
        "final_json": parse_json(final_text),
    }
    if completed.returncode != 0 or result["final_json"] is None:
        result["raw_text"] = final_text
        result["stderr_tail"] = completed.stderr[-1200:]
    return result


def main() -> int:
    targets = candidate_files()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", encoding="utf-8") as out:
        for path in targets:
            result = codex_call(path)
            out.write(json.dumps(result, sort_keys=True) + "\n")
            out.flush()
            print(json.dumps({"finished": result["path"], "returncode": result["returncode"]}))

    print(json.dumps({"results": str(OUT.relative_to(ROOT)), "count": len(targets)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

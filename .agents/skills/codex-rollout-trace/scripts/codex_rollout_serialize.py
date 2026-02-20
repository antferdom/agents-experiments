#!/usr/bin/env python3
"""Build a markdown conversation trace from Codex rollout JSONL files.

This tool supports:
1) Rollout mode:
   codex_rollout_serialize.py --thread-id <thread-id> [--output <path>]
2) Notify-hook mode:
   codex_rollout_serialize.py --from-notify '<json payload>'
3) Replay mode:
   codex_rollout_serialize.py --replay-input <path-to-replay.jsonl>

Optional outputs:
- Replay JSONL export (`--emit-replay` or `--replay-output`)
- Agent Trace JSONL export (`--emit-agent-trace` or `--agent-trace-output`)
- Terminal preview via Rich (`--preview-rich`)

Refs:
  - https://agent-trace.dev/
  - https://agent-trace.dev/specification/trace-record
  - https://agent-trace.dev/specification/trace-fields
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from rich.console import Console
from rich.markdown import Markdown

MAX_MESSAGE_CHARS = 20000
MAX_TOOL_TEXT_CHARS = 6000

AGENT_TRACE_SPEC_VERSION = "0.1.0"
AGENT_TRACE_TOOL_NAME = "codex-rollout-trace"
AGENT_TRACE_TOOL_VERSION = "0.1.0"

TURN_RENDER_SPECS: list[tuple[str, str, int]] = [
    ("User", "user_messages", MAX_MESSAGE_CHARS),
    ("Assistant", "assistant_messages", MAX_MESSAGE_CHARS),
    ("Reasoning", "reasoning_summaries", MAX_TOOL_TEXT_CHARS),
    ("Tool Calls", "tool_calls", MAX_TOOL_TEXT_CHARS),
    ("Tool Outputs", "tool_outputs", MAX_TOOL_TEXT_CHARS),
    ("Compaction Events", "compactions", MAX_MESSAGE_CHARS),
    ("Warnings", "warnings", MAX_TOOL_TEXT_CHARS),
    ("Errors", "errors", MAX_TOOL_TEXT_CHARS),
]


@dataclass
class TurnRecord:
    index: int
    user_messages: list[str] = field(default_factory=list)
    assistant_messages: list[str] = field(default_factory=list)
    reasoning_summaries: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    tool_outputs: list[str] = field(default_factory=list)
    compactions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class SessionMeta:
    session_id: str | None = None
    timestamp: str | None = None
    cwd: str | None = None
    model_provider: str | None = None


@dataclass
class RenderResult:
    text: str
    turn_line_ranges: dict[int, tuple[int, int]]


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[truncated]"


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def as_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = as_text(item).strip()
        if text:
            items.append(text)
    return items


def extract_message_text(message_payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for content_item in message_payload.get("content", []):
        item_type = content_item.get("type")
        if item_type in {"input_text", "output_text", "text"}:
            text = as_text(content_item.get("text")).strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()


def extract_reasoning_text(reasoning_payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in reasoning_payload.get("summary", []):
        if item.get("type") == "summary_text":
            text = as_text(item.get("text")).strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()


def ensure_turn(turns: list[TurnRecord], current: TurnRecord | None) -> TurnRecord:
    if current is not None:
        return current
    turn = TurnRecord(index=len(turns) + 1)
    turns.append(turn)
    return turn


def parse_rollout(rollout_path: Path) -> tuple[SessionMeta, list[TurnRecord]]:
    meta = SessionMeta()
    turns: list[TurnRecord] = []
    current_turn: TurnRecord | None = None

    with rollout_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_type = record.get("type")
            payload = record.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}

            if record_type == "session_meta":
                meta.session_id = as_text(payload.get("id")).strip() or meta.session_id
                meta.timestamp = as_text(payload.get("timestamp")).strip() or meta.timestamp
                meta.cwd = as_text(payload.get("cwd")).strip() or meta.cwd
                meta.model_provider = (
                    as_text(payload.get("model_provider")).strip() or meta.model_provider
                )
                continue

            if record_type == "compacted":
                message = as_text(payload.get("message")).strip()
                if message:
                    current_turn = ensure_turn(turns, current_turn)
                    current_turn.compactions.append(message)
                continue

            if record_type == "event_msg":
                event_type = payload.get("type")
                if event_type == "warning":
                    message = as_text(payload.get("message")).strip()
                    if message:
                        current_turn = ensure_turn(turns, current_turn)
                        current_turn.warnings.append(message)
                elif event_type == "error":
                    message = as_text(payload.get("message")).strip()
                    if message:
                        current_turn = ensure_turn(turns, current_turn)
                        current_turn.errors.append(message)
                elif event_type == "agent_reasoning":
                    message = as_text(payload.get("text")).strip()
                    if message:
                        current_turn = ensure_turn(turns, current_turn)
                        current_turn.reasoning_summaries.append(message)
                elif event_type == "context_compacted":
                    current_turn = ensure_turn(turns, current_turn)
                    current_turn.compactions.append("Context compacted")
                continue

            if record_type != "response_item":
                continue

            response_type = payload.get("type")
            if response_type == "message":
                role = as_text(payload.get("role")).strip()
                text = extract_message_text(payload)
                if not text:
                    continue
                if role == "user":
                    current_turn = TurnRecord(index=len(turns) + 1)
                    current_turn.user_messages.append(text)
                    turns.append(current_turn)
                elif role == "assistant":
                    current_turn = ensure_turn(turns, current_turn)
                    current_turn.assistant_messages.append(text)
                continue

            if response_type == "reasoning":
                summary = extract_reasoning_text(payload)
                if summary:
                    current_turn = ensure_turn(turns, current_turn)
                    current_turn.reasoning_summaries.append(summary)
                continue

            if response_type in {"function_call", "custom_tool_call", "web_search_call"}:
                name = (
                    as_text(payload.get("name")).strip()
                    or as_text(payload.get("tool_name")).strip()
                    or response_type
                )
                args = as_text(payload.get("arguments")).strip()
                call_text = f"{name}: {args}" if args else name
                current_turn = ensure_turn(turns, current_turn)
                current_turn.tool_calls.append(call_text)
                continue

            if response_type == "function_call_output":
                output = as_text(payload.get("output")).strip()
                if output:
                    current_turn = ensure_turn(turns, current_turn)
                    current_turn.tool_outputs.append(output)
                continue

    return meta, turns


def parse_replay(replay_path: Path) -> tuple[str | None, SessionMeta, list[TurnRecord], str | None]:
    thread_id: str | None = None
    meta = SessionMeta()
    turns: list[TurnRecord] = []
    last_turn_id: str | None = None

    with replay_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue

            record_type = as_text(record.get("type")).strip()
            if record_type == "session":
                thread_id = as_text(record.get("thread_id")).strip() or thread_id
                last_turn_id = as_text(record.get("last_turn_id")).strip() or last_turn_id

                session_payload = record.get("session", {})
                if isinstance(session_payload, dict):
                    meta.session_id = (
                        as_text(session_payload.get("session_id")).strip() or meta.session_id
                    )
                    meta.timestamp = (
                        as_text(session_payload.get("timestamp")).strip() or meta.timestamp
                    )
                    meta.cwd = as_text(session_payload.get("cwd")).strip() or meta.cwd
                    meta.model_provider = (
                        as_text(session_payload.get("model_provider")).strip()
                        or meta.model_provider
                    )
                continue

            if record_type != "turn":
                continue

            index = record.get("turn_index")
            if isinstance(index, int):
                turn_index = index
            elif isinstance(index, str) and index.isdigit():
                turn_index = int(index)
            else:
                turn_index = len(turns) + 1
            turn = TurnRecord(index=turn_index)
            turn.user_messages = as_text_list(record.get("user_messages"))
            turn.assistant_messages = as_text_list(record.get("assistant_messages"))
            turn.reasoning_summaries = as_text_list(record.get("reasoning_summaries"))
            turn.tool_calls = as_text_list(record.get("tool_calls"))
            turn.tool_outputs = as_text_list(record.get("tool_outputs"))
            turn.compactions = as_text_list(record.get("compactions"))
            turn.warnings = as_text_list(record.get("warnings"))
            turn.errors = as_text_list(record.get("errors"))
            turns.append(turn)

    return thread_id, meta, turns, last_turn_id


def find_rollout_for_thread(codex_home: Path, thread_id: str) -> Path | None:
    pattern = f"rollout-*-{thread_id}.jsonl"
    candidates: list[Path] = []
    for subdir in ("sessions", "archived_sessions"):
        root = codex_home / subdir
        if root.is_dir():
            candidates.extend(root.rglob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


TURN_HEADING_PATTERN = re.compile(r"^### Turn (\d+)$")


def iter_turn_sections(turn: TurnRecord) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    for title, attribute_name, max_chars in TURN_RENDER_SPECS:
        values = getattr(turn, attribute_name)
        if not values:
            continue
        sections.append((title, [truncate(value, max_chars) for value in values]))
    return sections


def compute_turn_line_ranges(lines: list[str]) -> dict[int, tuple[int, int]]:
    headings: list[tuple[int, int]] = []
    for line_no, line in enumerate(lines, start=1):
        match = TURN_HEADING_PATTERN.match(line)
        if match:
            headings.append((int(match.group(1)), line_no))

    ranges: dict[int, tuple[int, int]] = {}
    for idx, (turn_index, start_line) in enumerate(headings):
        if idx + 1 < len(headings):
            end_line = headings[idx + 1][1] - 1
        else:
            end_line = len(lines)
        ranges[turn_index] = (start_line, end_line)
    return ranges


def render_markdown(
    *,
    thread_id: str,
    source_path_display: str | None,
    source_kind: str,
    meta: SessionMeta,
    turns: list[TurnRecord],
    last_turn_id: str | None,
) -> RenderResult:
    lines: list[str] = [
        "# Codex Conversation Trace",
        "",
        f"- Thread ID: `{thread_id}`",
    ]
    if source_path_display is not None:
        lines.append(f"- Source ({source_kind}): `{source_path_display}`")
    lines.append(f"- Generated (UTC): `{now_utc_iso()}`")
    if last_turn_id:
        lines.append(f"- Last completed turn: `{last_turn_id}`")
    lines.extend(["", "## Session", ""])

    if meta.timestamp:
        lines.append(f"- Started: `{meta.timestamp}`")
    if meta.cwd:
        lines.append(f"- CWD: `{meta.cwd}`")
    if meta.model_provider:
        lines.append(f"- Model provider: `{meta.model_provider}`")
    if meta.session_id:
        lines.append(f"- Session ID: `{meta.session_id}`")
    lines.append("")

    if not turns:
        lines.extend(["_No user/assistant turn content found in source._", ""])
        return RenderResult(text="\n".join(lines), turn_line_ranges={})

    lines.extend(["## Timeline", ""])
    for turn in turns:
        lines.extend([f"### Turn {turn.index}", ""])
        for title, entries in iter_turn_sections(turn):
            lines.extend([f"#### {title}", ""])
            for entry in entries:
                lines.extend(["```text", entry, "```", ""])

    return RenderResult(
        text="\n".join(lines),
        turn_line_ranges=compute_turn_line_ranges(lines),
    )


def preview_markdown_rich(markdown_text: str) -> None:
    console = Console()
    console.print(Markdown(markdown_text))


def resolve_codex_home(explicit: str | None) -> Path:
    return Path(explicit or os.environ.get("CODEX_HOME") or "~/.codex").expanduser().resolve()


def parse_notify_payload(payload_raw: str) -> dict[str, Any] | None:
    if not payload_raw:
        return None
    try:
        value = json.loads(payload_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value


def resolve_path_with_thread(raw_path: str, *, cwd_path: Path, thread_id: str) -> Path:
    templated = raw_path.replace("{thread_id}", thread_id)
    path = Path(templated).expanduser()
    if not path.is_absolute():
        path = cwd_path / path
    return path.resolve()


def best_relative_path(path: Path, *, repo_root: Path | None, cwd_path: Path) -> str:
    for base in (repo_root, cwd_path):
        if base is None:
            continue
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.name


def get_git_info(cwd_path: Path) -> tuple[Path | None, dict[str, str] | None]:
    try:
        root_cmd = subprocess.run(
            ["git", "-C", str(cwd_path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None, None
    if root_cmd.returncode != 0:
        return None, None

    root_text = root_cmd.stdout.strip()
    if not root_text:
        return None, None
    repo_root = Path(root_text).resolve()

    revision_cmd = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if revision_cmd.returncode != 0:
        return repo_root, None
    revision = revision_cmd.stdout.strip()
    if not revision:
        return repo_root, None
    return repo_root, {"type": "git", "revision": revision}


def build_turn_urn(thread_id: str, turn_index: int) -> str:
    return f"urn:codex:thread:{quote(thread_id, safe='')}:turn:{turn_index}"


def build_session_urn(session_id: str) -> str:
    return f"urn:codex:session:{quote(session_id, safe='')}"


def build_replay_urn(thread_id: str, turn_index: int) -> str:
    return f"urn:codex:replay:{quote(thread_id, safe='')}:turn:{turn_index}"


def build_replay_records(
    *,
    thread_id: str,
    source_path_display: str | None,
    source_kind: str,
    meta: SessionMeta,
    turns: list[TurnRecord],
    last_turn_id: str | None,
) -> list[dict[str, Any]]:
    session_payload = {
        "session_id": meta.session_id,
        "timestamp": meta.timestamp,
        "cwd": meta.cwd,
        "model_provider": meta.model_provider,
    }
    session_payload = {k: v for k, v in session_payload.items() if v}

    session_record: dict[str, Any] = {
        "type": "session",
        "thread_id": thread_id,
        "generated_utc": now_utc_iso(),
        "source_kind": source_kind,
        "session": session_payload,
    }
    if source_path_display is not None:
        session_record["source_path"] = source_path_display
    if last_turn_id:
        session_record["last_turn_id"] = last_turn_id

    records: list[dict[str, Any]] = [session_record]
    for turn in turns:
        records.append(
            {
                "type": "turn",
                "turn_index": turn.index,
                "user_messages": turn.user_messages,
                "assistant_messages": turn.assistant_messages,
                "reasoning_summaries": turn.reasoning_summaries,
                "tool_calls": turn.tool_calls,
                "tool_outputs": turn.tool_outputs,
                "compactions": turn.compactions,
                "warnings": turn.warnings,
                "errors": turn.errors,
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def build_agent_trace_record(
    *,
    thread_id: str,
    markdown_path: Path,
    replay_path: Path | None,
    source_path: Path | None,
    source_kind: str,
    meta: SessionMeta,
    turns: list[TurnRecord],
    render_result: RenderResult,
    cwd_path: Path,
    last_turn_id: str | None,
) -> dict[str, Any]:
    repo_root, vcs_info = get_git_info(cwd_path)
    markdown_rel = best_relative_path(markdown_path, repo_root=repo_root, cwd_path=cwd_path)

    markdown_conversations: list[dict[str, Any]] = []
    for turn in turns:
        line_range = render_result.turn_line_ranges.get(turn.index)
        if line_range is None:
            continue
        start_line, end_line = line_range
        turn_blob = "\n".join(
            turn.user_messages
            + turn.assistant_messages
            + turn.reasoning_summaries
            + turn.tool_calls
            + turn.tool_outputs
            + turn.compactions
            + turn.warnings
            + turn.errors
        )
        range_payload: dict[str, Any] = {
            "start_line": start_line,
            "end_line": end_line,
        }
        if turn_blob:
            range_payload["content_hash"] = (
                f"sha256:{hashlib.sha256(turn_blob.encode('utf-8')).hexdigest()}"
            )

        conversation: dict[str, Any] = {
            "url": build_turn_urn(thread_id, turn.index),
            "contributor": {"type": "ai"},
            "ranges": [range_payload],
        }
        related: list[dict[str, str]] = []
        if meta.session_id:
            related.append({"type": "session", "url": build_session_urn(meta.session_id)})
        if replay_path is not None:
            related.append({"type": "replay", "url": build_replay_urn(thread_id, turn.index)})
        if related:
            conversation["related"] = related
        markdown_conversations.append(conversation)

    files_payload: list[dict[str, Any]] = [
        {
            "path": markdown_rel,
            "conversations": markdown_conversations,
        }
    ]

    replay_rel: str | None = None
    if replay_path is not None:
        replay_rel = best_relative_path(replay_path, repo_root=repo_root, cwd_path=cwd_path)
        replay_conversations: list[dict[str, Any]] = []
        # Replay format writes one session line, then one line per turn.
        for replay_line, turn in enumerate(turns, start=2):
            replay_conversations.append(
                {
                    "url": build_turn_urn(thread_id, turn.index),
                    "contributor": {"type": "ai"},
                    "ranges": [
                        {
                            "start_line": replay_line,
                            "end_line": replay_line,
                        }
                    ],
                }
            )
        files_payload.append(
            {
                "path": replay_rel,
                "conversations": replay_conversations,
            }
        )

    metadata: dict[str, Any] = {
        "source": "codex_rollout_serialize",
        "thread_id": thread_id,
        "source_kind": source_kind,
    }
    if meta.session_id:
        metadata["session_id"] = meta.session_id
    if meta.model_provider:
        metadata["model_provider"] = meta.model_provider
    if meta.timestamp:
        metadata["session_started"] = meta.timestamp
    if last_turn_id:
        metadata["last_turn_id"] = last_turn_id
    if source_path is not None:
        metadata["source_path"] = best_relative_path(
            source_path, repo_root=repo_root, cwd_path=cwd_path
        )
    if replay_rel:
        metadata["replay_path"] = replay_rel

    record: dict[str, Any] = {
        "version": AGENT_TRACE_SPEC_VERSION,
        "id": str(uuid.uuid4()),
        "timestamp": now_utc_iso(),
        "tool": {
            "name": AGENT_TRACE_TOOL_NAME,
            "version": AGENT_TRACE_TOOL_VERSION,
        },
        "files": files_payload,
        "metadata": metadata,
    }
    if vcs_info:
        record["vcs"] = vcs_info
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  1) Live notify hook (main /compact-safe trace path):\n"
            "     codex_rollout_serialize.py --from-notify --emit-replay --emit-agent-trace <notify-json>\n"
            "\n"
            "  2) Backfill from rollout by thread id:\n"
            "     codex_rollout_serialize.py --thread-id <thread-id>\n"
            "\n"
            "  3) Rebuild from replay snapshot:\n"
            "     codex_rollout_serialize.py --replay-input .codex-traces/<thread-id>.replay.jsonl\n"
        ),
    )
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--rollout", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--codex-home", default=None)
    parser.add_argument("--from-notify", action="store_true")
    parser.add_argument("--emit-replay", action="store_true")
    parser.add_argument("--replay-output", default=None)
    parser.add_argument("--replay-input", default=None)
    parser.add_argument("--emit-agent-trace", action="store_true")
    parser.add_argument("--agent-trace-output", default=None)
    parser.add_argument("--preview-rich", action="store_true")
    parser.add_argument("notify_payload", nargs="?", default=None)
    args = parser.parse_args()

    notify_payload: dict[str, Any] | None = None
    if args.from_notify:
        notify_payload = parse_notify_payload(args.notify_payload or "")
        if notify_payload is None:
            return 0

    cwd = args.cwd
    if not cwd and notify_payload:
        cwd = as_text(notify_payload.get("cwd")).strip() or None
    if not cwd:
        cwd = os.getcwd()
    cwd_path = Path(cwd).expanduser().resolve()

    thread_id = args.thread_id
    if not thread_id and notify_payload:
        thread_id = (
            as_text(notify_payload.get("thread-id")).strip()
            or as_text(notify_payload.get("thread_id")).strip()
            or None
        )

    meta = SessionMeta()
    turns: list[TurnRecord] = []
    source_path: Path | None = None
    source_kind: str = "rollout"
    last_turn_id: str | None = None
    replay_input_path: Path | None = None

    if args.replay_input:
        replay_input_candidate = Path(args.replay_input).expanduser()
        if not replay_input_candidate.is_absolute():
            replay_input_candidate = cwd_path / replay_input_candidate
        replay_input_path = replay_input_candidate.resolve()
        if not replay_input_path.is_file():
            print(
                f"codex_rollout_serialize: replay input not found: {replay_input_path}",
                file=sys.stderr,
            )
            return 1

        replay_thread_id, meta, turns, replay_last_turn_id = parse_replay(replay_input_path)
        if not thread_id:
            thread_id = replay_thread_id
        if not thread_id:
            print("codex_rollout_serialize: missing thread id", file=sys.stderr)
            return 1
        if replay_last_turn_id:
            last_turn_id = replay_last_turn_id
        source_path = replay_input_path
        source_kind = "replay"
    else:
        if not thread_id:
            print("codex_rollout_serialize: missing thread id", file=sys.stderr)
            return 1

        codex_home = resolve_codex_home(args.codex_home)
        source_path = Path(args.rollout).expanduser().resolve() if args.rollout else None
        if source_path is None:
            source_path = find_rollout_for_thread(codex_home, thread_id)
        if source_path is None or not source_path.is_file():
            return 0

        meta, turns = parse_rollout(source_path)
        source_kind = "rollout"

    if notify_payload:
        notify_turn = as_text(notify_payload.get("turn-id")).strip()
        if notify_turn:
            last_turn_id = notify_turn

    source_path_display: str | None = None
    if source_path is not None:
        source_path_display = best_relative_path(
            source_path, repo_root=None, cwd_path=cwd_path
        )

    output_path = (
        resolve_path_with_thread(args.output, cwd_path=cwd_path, thread_id=thread_id)
        if args.output
        else cwd_path.joinpath(".codex-traces", f"{thread_id}.md").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    render_result = render_markdown(
        thread_id=thread_id,
        source_path_display=source_path_display,
        source_kind=source_kind,
        meta=meta,
        turns=turns,
        last_turn_id=last_turn_id,
    )
    output_path.write_text(render_result.text, encoding="utf-8")
    if args.preview_rich:
        preview_markdown_rich(render_result.text)

    replay_output_path: Path | None = None
    if args.replay_output:
        replay_output_path = resolve_path_with_thread(
            args.replay_output, cwd_path=cwd_path, thread_id=thread_id
        )
    elif args.emit_replay:
        replay_output_path = cwd_path.joinpath(".codex-traces", f"{thread_id}.replay.jsonl")

    if replay_output_path is not None:
        replay_records = build_replay_records(
            thread_id=thread_id,
            source_path_display=source_path_display,
            source_kind=source_kind,
            meta=meta,
            turns=turns,
            last_turn_id=last_turn_id,
        )
        write_jsonl(replay_output_path, replay_records, append=False)

    agent_trace_output_path: Path | None = None
    if args.agent_trace_output:
        agent_trace_output_path = resolve_path_with_thread(
            args.agent_trace_output, cwd_path=cwd_path, thread_id=thread_id
        )
    elif args.emit_agent_trace:
        agent_trace_output_path = cwd_path.joinpath(".agent-trace", "traces.jsonl").resolve()

    if agent_trace_output_path is not None:
        replay_reference_path = replay_output_path or replay_input_path
        agent_trace_record = build_agent_trace_record(
            thread_id=thread_id,
            markdown_path=output_path,
            replay_path=replay_reference_path,
            source_path=source_path,
            source_kind=source_kind,
            meta=meta,
            turns=turns,
            render_result=render_result,
            cwd_path=cwd_path,
            last_turn_id=last_turn_id,
        )
        write_jsonl(agent_trace_output_path, [agent_trace_record], append=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

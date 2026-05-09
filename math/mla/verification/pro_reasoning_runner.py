#!/usr/bin/env python3
"""Small Responses API runner for recorded GPT-5.5-pro MLA reasoning calls."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


API_URL = "https://api.openai.com/v1/responses"


def request_json(method: str, url: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.request(method, url, headers=headers, json=payload, timeout=120)
    try:
        data = response.json()
    except Exception:
        data = {"raw_text": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"API {method} {url} failed with {response.status_code}: {json.dumps(data)[:2000]}")
    return data


def extract_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    chunks.append(content["text"])
    if chunks:
        return "\n\n".join(chunks)
    if data.get("output_text"):
        return str(data["output_text"])
    return json.dumps(data, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument(
        "--append-file",
        type=Path,
        action="append",
        default=[],
        help="Append a file's contents to the prompt as labeled context.",
    )
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--response-id", help="Resume polling an existing background response instead of submitting.")
    parser.add_argument("--model", default="gpt-5.5-pro")
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--reasoning-effort", choices=["medium", "high", "xhigh"], default="medium")
    parser.add_argument("--text-verbosity", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()

    if args.response_id:
        response_id = args.response_id
        submitted = request_json("GET", f"{API_URL}/{response_id}")
        print(json.dumps({"event": "resumed", "response_id": response_id, "status": submitted.get("status")}))
    else:
        prompt = args.prompt.read_text(encoding="utf-8")
        for path in args.append_file:
            prompt += (
                f"\n\n--- BEGIN APPENDED FILE: {path} ---\n"
                f"{path.read_text(encoding='utf-8')}"
                f"\n--- END APPENDED FILE: {path} ---\n"
            )
        payload = {
            "model": args.model,
            "input": prompt,
            "background": True,
            "max_output_tokens": args.max_output_tokens,
            "reasoning": {"effort": args.reasoning_effort},
            "text": {"verbosity": args.text_verbosity},
            "metadata": {"task_id": args.task_id, "project": "math/mla"},
        }
        submitted = request_json("POST", API_URL, payload=payload)
        response_id = submitted["id"]
        print(json.dumps({"event": "submitted", "response_id": response_id, "status": submitted.get("status")}))

    deadline = time.time() + args.timeout_seconds
    current = submitted
    while current.get("status") not in {"completed", "failed", "cancelled", "incomplete"}:
        if time.time() > deadline:
            break
        time.sleep(args.poll_interval)
        current = request_json("GET", f"{API_URL}/{response_id}")
        print(json.dumps({"event": "polled", "response_id": response_id, "status": current.get("status")}))
        sys.stdout.flush()

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_output.write_text(extract_text(current).rstrip() + "\n", encoding="utf-8")

    status = current.get("status")
    if status != "completed":
        print(json.dumps({"event": "terminal_non_completed", "response_id": response_id, "status": status}), file=sys.stderr)
        return 1
    print(json.dumps({"event": "completed", "response_id": response_id, "status": status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

---
name: codex-rollout-trace
description: Persist Codex conversation state into a markdown timeline derived from rollout JSONL files. Use when you need long-horizon traceability, a readable markdown artifact, notify-hook trace refresh, replay snapshots, or Agent Trace JSONL export.
metadata:
  short-description: Persist rollout traces to markdown
---

# Codex Rollout Trace Skill

Persist Codex conversation state into a markdown timeline derived from rollout JSONL files.

Use this skill when you need:
- Long-horizon traceability across many turns.
- A readable `.md` artifact even when `/compact` is used.
- Automated per-turn trace refresh via Codex `notify` hook.
- Optional Agent Trace JSONL export (RFC `0.1.0`) for debugging/replay workflows.
- Replay snapshots that can regenerate markdown without direct access to Codex session folders.

## Files

- `scripts/codex_rollout_serialize.py`: main executable.
- `assets/notify.template.toml`: template notify stanza for `config.toml`.
- `examples.md`: copy-paste CLI test commands.

## Inputs

- `thread-id` (direct mode), or
- Codex notify payload (notify-hook mode).

## Usage

### Direct mode

```bash
python3 scripts/codex_rollout_serialize.py --thread-id <thread-id>
```

Default output:

```text
<cwd>/.codex-traces/<thread-id>.md
```

### Direct mode with optional exports

```bash
python3 scripts/codex_rollout_serialize.py \
  --thread-id <thread-id> \
  --emit-replay \
  --emit-agent-trace
```

Default optional outputs:

```text
<cwd>/.codex-traces/<thread-id>.replay.jsonl
<cwd>/.agent-trace/traces.jsonl
```

You can also customize output paths and use `{thread_id}` templates:

```bash
python3 scripts/codex_rollout_serialize.py \
  --thread-id <thread-id> \
  --replay-output ".codex-traces/{thread_id}.replay.jsonl" \
  --agent-trace-output ".agent-trace/{thread_id}.jsonl"
```

### Notify-hook mode

Add a notify command to your Codex config using a template path:

```toml
notify = [
  "python3",
  "<AGENTS_EXPERIMENTS_ROOT>/.agents/skills/codex-rollout-trace/scripts/codex_rollout_serialize.py",
  "--from-notify",
  "--emit-replay",
  "--emit-agent-trace"
]
```

If Codex is always started from repository root, a relative path works too:

```toml
notify = [
  "python3",
  ".agents/skills/codex-rollout-trace/scripts/codex_rollout_serialize.py",
  "--from-notify",
  "--emit-replay",
  "--emit-agent-trace"
]
```

Codex appends a JSON payload as the final argument. The script reads that payload and updates the trace for the matching thread.

### Replay mode

Regenerate markdown from a saved replay snapshot:

```bash
python3 scripts/codex_rollout_serialize.py \
  --replay-input .codex-traces/<thread-id>.replay.jsonl
```

### Rich preview

Preview the generated markdown in terminal:

```bash
python3 scripts/codex_rollout_serialize.py \
  --thread-id <thread-id> \
  --preview-rich
```

## Notes

- Rollout JSONL remains the source of truth.
- Markdown output truncates large fields for readability/performance.
- Agent Trace export appends one record per script run (JSONL log).
- `rich` is a required dependency for this script.

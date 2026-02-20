# Codex Rollout Trace CLI Examples

## Setup

```bash
SCRIPT="/opt/agents-experiments/.agents/skills/codex-rollout-trace/scripts/codex_rollout_serialize.py"
THREAD_ID="019c7ac3-0267-7110-afe8-8c9cfc8b1079"
ROLLOUT="/root/.codex/sessions/2026/02/20/rollout-2026-02-20T11-15-26-019c7ac3-0267-7110-afe8-8c9cfc8b1079.jsonl"
OUT="/tmp/codex-trace-live"
mkdir -p "$OUT"
```

## Basic Trace

```bash
python3 "$SCRIPT" --thread-id "$THREAD_ID" --rollout "$ROLLOUT" --cwd "$OUT"
```

Custom markdown output:

```bash
python3 "$SCRIPT" --thread-id "$THREAD_ID" --rollout "$ROLLOUT" --cwd "$OUT" --output ".codex-traces/{thread_id}.custom.md"
```

## Replay Export

Default replay output:

```bash
python3 "$SCRIPT" --thread-id "$THREAD_ID" --rollout "$ROLLOUT" --cwd "$OUT" --emit-replay
```

Custom replay output:

```bash
python3 "$SCRIPT" --thread-id "$THREAD_ID" --rollout "$ROLLOUT" --cwd "$OUT" --replay-output ".codex-traces/{thread_id}.replay.custom.jsonl"
```

## Agent Trace Export

Default Agent Trace output:

```bash
python3 "$SCRIPT" --thread-id "$THREAD_ID" --rollout "$ROLLOUT" --cwd "$OUT" --emit-agent-trace
```

Custom Agent Trace output:

```bash
python3 "$SCRIPT" --thread-id "$THREAD_ID" --rollout "$ROLLOUT" --cwd "$OUT" --agent-trace-output ".agent-trace/{thread_id}.jsonl"
```

## Emit Everything

```bash
python3 "$SCRIPT" --thread-id "$THREAD_ID" --rollout "$ROLLOUT" --cwd "$OUT" --emit-replay --emit-agent-trace
```

## Replay Restore

Rebuild markdown from replay:

```bash
python3 "$SCRIPT" --replay-input "$OUT/.codex-traces/${THREAD_ID}.replay.jsonl" --cwd "$OUT" --output ".codex-traces/{thread_id}.from-replay.md"
```

Replay with rich preview:

```bash
python3 "$SCRIPT" --replay-input "$OUT/.codex-traces/${THREAD_ID}.replay.jsonl" --cwd "$OUT" --preview-rich
```

## Notify-Hook Simulation

```bash
python3 "$SCRIPT" --from-notify --emit-replay --emit-agent-trace '{"thread-id":"019c7ac3-0267-7110-afe8-8c9cfc8b1079","turn-id":"manual-test","cwd":"/tmp/codex-trace-live"}'
```

## Discover Rollout by Thread ID

```bash
python3 "$SCRIPT" --thread-id "$THREAD_ID" --codex-home "$HOME/.codex" --cwd "$OUT"
```

## Troubleshooting

If you get:

```text
can't find '__main__' module in '/path/to/scripts'
```

You passed a directory to `python3` instead of the script file.

Correct:

```bash
python3 ./codex_rollout_serialize.py --thread-id "$THREAD_ID" --rollout "$ROLLOUT" --cwd "$OUT"
```
